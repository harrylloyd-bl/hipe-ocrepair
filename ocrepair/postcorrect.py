#!/usr/bin/env python3
"""
OCR post-correction via Hugging Face Inference API.

Requires HF_TOKEN with Inference Providers permission (prompted if not set).
Optionally pass --cheapest to route to the lowest-cost provider.

Usage:
  export HF_TOKEN=your_token
  python ocr_postcorrect_hf.py input.json --model qwen3.5-397b-a17b -j corrected.json -c corrections.csv
  python ocr_postcorrect_hf.py input.json --model mistral3 -j out.json -c out.csv
  python ocr_postcorrect_hf.py --list-models
  python ocr_postcorrect_hf.py input.json --run-all --output-dir ./results
  python ocr_postcorrect_hf.py input.jsonl --jsonl -j out.jsonl -c out.csv --model mistral3
  python ocr_postcorrect_hf.py input.jsonl -j out.jsonl -c out.csv --model mistral3 --batch-size 10
"""

import argparse
from datetime import datetime
import getpass
import json
import logging
import os
import re
import sys
import time

from huggingface_hub import InferenceClient

try:
    from dotenv import load_dotenv
    if "HF_TOKEN" not in os.environ:
        load_dotenv()
except ImportError:
    pass

if not os.path.exists("logs"):
    os.makedirs("logs")

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"logs/{datetime.now().strftime("%Y%m%d_%H%M")}.log", encoding="utf8", level=logging.DEBUG)

# Lookup table mapping friendly short names to full Hugging Face model IDs.
# This lets you type e.g. --model mistral3 instead of the full path.
MODEL_REGISTRY = {
    "qwen3.5-397b-a17b": "Qwen/Qwen3.5-397B-A17B",
    "qwen3-235b-a22b-thinking-2507": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "mistral3": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "gemma3": "google/gemma-3-27b-it",
    "deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
}

MODEL_CONTEXT_WINDOW: dict[str, int] = {
    "google/gemma-3-27b-it": 32_768,
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": 131_072,
    "Qwen/Qwen3.5-397B-A17B": 131_072,
    "Qwen/Qwen3-235B-A22B-Thinking-2507": 131_072,
    "deepseek-ai/DeepSeek-V3.2": 131_072,
}
DEFAULT_CONTEXT_WINDOW = 32_768
CONTEXT_SAFETY_MARGIN = 256  # tokens reserved for framing overhead

def _estimate_tokens(text: str) -> int:
    """Rough token count: ~3.5 characters per token for mixed JSON/English text."""
    return max(1, int(len(text) / 3.5))


def _compute_max_tokens(model_id: str, messages: list[dict]) -> int:
    """Calculate the maximum output tokens available for a given model and input."""
    base_id = model_id.split(":")[0]
    ctx = MODEL_CONTEXT_WINDOW.get(base_id, DEFAULT_CONTEXT_WINDOW)
    input_text = "".join(m.get("content", "") for m in messages)
    input_tokens = _estimate_tokens(input_text)
    available = ctx - input_tokens - CONTEXT_SAFETY_MARGIN
    return max(512, available)


SYSTEM_PROMPT = """You are an expert in historical OCR post-correction for newspapers and books from the 17th to 20th century in English, French and German.

You will receive a JSON array. Each element has:
- "document_id": a unique identifier
- "language": the language of the text
- "ocr_hypothesis": the raw OCR text to correct

Your task:
1. Correct OCR errors in "ocr_hypothesis" and write the result into a new field "ocr_postcorrection_output"
2. Do NOT alter person names or place names
3. Correct only within the given "language" — do not translate
4. Preserve or restore accented characters correctly
5. Fix four classes of common historical OCR errors:
- Over-segmentation: words broken across lines; hyphens replaced with spaces e.g. "before the follow ing morning" -> "before the follow-ing morning"
- Under-segmentation: missing spaces between words e.g. "Rich oldSwarth LAND" -> "Rich old Swarth LAND"
- Misrecognized character: e.g. "aſter" -> "after"; "ar.d" -> "and"
- Missing character: e.g. "er a good deal" -> "After a good deal"
6. If too ambiguous, copy ocr_hypothesis unchanged

Return a JSON array where each element has exactly: "document_id", "ocr_postcorrection_output"
Return ONLY valid JSON. No explanation, no preamble, no markdown fences.

After the JSON array, output a CSV corrections log.
Marker line: CORRECTIONS_CSV
Columns: document_id,ocr_hypothesis_snippet,ocr_postcorrection_snippet,error_type,confidence,notes
- error_type: over_segmentation, misrecognized_char, under_segmentation, missing_character, wrong_word, uncertain, no_change, or custom:label
- confidence: high, medium, or low
- One row per document_id (use no_change if unchanged)
- Do not wrap the CSV in markdown fences
"""


API_TIMEOUT = 600  # seconds per request before giving up


def get_client() -> InferenceClient:
    """Create and return a Hugging Face InferenceClient, authenticated with HF_TOKEN."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = getpass.getpass(
            "HF_TOKEN not found in environment. Enter your Hugging Face token: "
        )
        if not token.strip():
            raise SystemExit(
                "No token provided. Create one at https://huggingface.co/settings/tokens "
                "with 'Make calls to Inference Providers'."
            )
    return InferenceClient(api_key=token.strip(), timeout=API_TIMEOUT)


MAX_RETRIES = 3
RETRY_BACKOFF = [30, 60, 120]  # seconds to wait between retries


def chat_completion(client, model: str, messages: list, **kwargs):
    """Send a chat completion request with automatic retries on timeout.

    Returns (content, finish_reason).  finish_reason is typically "stop"
    (complete) or "length" (output was truncated by max_tokens).
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            choice = result.choices[0]
            return choice.message.content, getattr(choice, "finish_reason", "stop")
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(
                kw in err_str
                for kw in ("timeout", "gateway", "502", "503", "504", "529",
                           "prematurely", "connection", "reset", "broken pipe")
            )
            if not is_retryable or attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(
                f"  Timeout/server error (attempt {attempt + 1}/{MAX_RETRIES + 1}), "
                f"retrying in {wait}s: {e}",
                file=sys.stderr,
            )
            time.sleep(wait)


def resolve_model(name_or_id: str) -> str:
    """
    Turn a short name like 'mistral3' into the full HF model ID.
    If the input doesn't match any short name, return it unchanged
    (so you can also pass a full model ID directly).
    """
    # TODO warn if ID not in MODEL_REGISTRY keys or values
    key = name_or_id.strip().lower()            # normalise to lowercase for lookup
    return MODEL_REGISTRY.get(key, name_or_id.strip())  # fallback to original if not found


def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) if present."""
    s = text.strip()
    if s.startswith("```json"):
        s = s.lstrip("```json")
    if s.endswith("```"):
        s = s.rstrip("```")

    # Remove any residual leading/trailing linebreaks
    s = s.strip("\n")

    return s


def _try_parse_json(raw: str):
    """Parse JSON, handling cases where the model returns multiple objects
    instead of a proper array (Extra data error), and attempting repair
    on truncated output."""
    raw = _strip_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as first_err:
        print(f"  (initial json.loads failed: {first_err})", file=sys.stderr)

    # Model may have returned one JSON object per line instead of an array
    objects = []
    for line in raw.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            objects.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if objects:
        return objects
    else:
        raise json.JSONDecodeError("Could not parse model JSON output", raw, 0)


def _json_value_start(text: str) -> int | None:
    """Index of the first ``[`` or ``{`` that can start a JSON value."""
    for i, ch in enumerate(text):
        if ch in "[{":
            return i
    return None


def _split_json_from_rest(text: str) -> tuple[str, str]:
    """Split *text* into the first complete JSON value and the remainder.

    Uses :meth:`json.JSONDecoder.raw_decode` so delimiters follow JSON
    string/escape rules (unlike naive bracket counting).
    If parsing fails, returns *text* unchanged and an empty rest string so
    callers can fall back to repair heuristics.
    """
    text = text.strip()
    if not text:
        return "", ""
    start = _json_value_start(text)
    if start is None:
        return text, ""
    try:
        _obj, end = json.JSONDecoder().raw_decode(text, start)
        return text[start:end].strip(), text[end:].strip()
    except json.JSONDecodeError:
        return text, ""


_FENCED_BLOCK_RE = re.compile(
    r"```[a-zA-Z]*\s*\n(.*?)```", re.DOTALL
)


def parse_response(response: str):
    """Split model output into (json_raw, csv_section).

    Handles any combination of markers, markdown fences, and raw output.
    Returns the raw JSON string — use _try_parse_json() to parse it.
    """
    # Step 1: if there's an explicit CORRECTIONS_CSV marker, split on it
    if "CORRECTIONS_CSV" in response:
        parts = response.split("CORRECTIONS_CSV", 1)
        stripped = _strip_fences(parts[0].strip())
        json_raw, json_tail = _split_json_from_rest(stripped)
        rest = parts[1].strip() if len(parts) > 1 else ""
        merged = "\n".join(x for x in (json_tail, rest) if x)
        if merged:
            csv_section = _strip_fences(merged.strip())

        return json_raw, csv_section

    # Step 2: fenced blocks; split JSON from CSV with raw_decode (first fence
    # may still contain JSON+CSV if the model used two ``` regions).
    blocks = _FENCED_BLOCK_RE.findall(response)
    if len(blocks) >= 2:
        json_raw, rest0 = _split_json_from_rest(blocks[0].strip())
        merged = "\n".join(x for x in (rest0, blocks[1].strip()) if x)
        csv_section = ""
        if merged:
            csv_section = _strip_fences(merged)

        return json_raw, csv_section

    # Single block or no blocks: extract content, then split JSON / CSV
    if blocks:
        content = blocks[0].strip()
        last_fence_end = response.rfind("```")
        if last_fence_end != -1:
            trailing = response[last_fence_end + 3:].strip()
            if trailing:
                content = content + "\n" + trailing
    else:
        content = _strip_fences(response.strip())

    json_raw, rest = _split_json_from_rest(content)

    csv_section = ""
    if rest:
        csv_section = _strip_fences(rest)

    return json_raw, csv_section


def _csv_body_lines(csv_section: str) -> tuple[str | None, list[str]]:
    """Split CSV text into header (first line) and data lines."""
    lines = [ln for ln in csv_section.strip().splitlines() if ln.strip()]
    if not lines:
        return None, []
    return lines[0], lines[1:]


def main():
    # ── Argument parser ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="OCR post-correction via Hugging Face Inference API"
    )
    # Positional: path to JSON or JSONL file containing OCR tokens
    parser.add_argument(
        "input_json",
        nargs="?",                                     # optional so --list-models works without it
        type=str,
        default=None,
        help="Path to JSON or JSONL file with OCR tokens (ocr_hypothesis per token)",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Read input as JSON Lines (one JSON object per line). Outputs are JSONL when using -j.",
    )
    # -j: where to write the corrected JSON (same structure as input)
    parser.add_argument(
        "-j", "--output-json",
        type=str,
        default=None,
        metavar="FILE",
        help="Write corrected JSON to FILE (same structure as input, with ocr_postcorrection_output filled)",
    )
    # -c: where to write the corrections CSV log
    parser.add_argument(
        "-c", "--output-csv",
        type=str,
        default=None,
        metavar="FILE",
        help="Write corrections log CSV to FILE (token_id, ocr_hypothesis, ocr_postcorrection_output, error_type, confidence, uncertain_chars, notes)",
    )
    # -o: optional dump of the full raw model response (for debugging)
    parser.add_argument(
        "-o", "--output-raw",
        type=argparse.FileType("w", encoding="utf-8"),
        default=None,
        metavar="FILE",
        help="Optionally write full raw model output to FILE (for debugging); normally use -j and -c only",
    )
    # --model: which model to use (short name or full HF ID)
    parser.add_argument(
        "--model",
        default=os.environ.get("OCR_HF_MODEL", ""),   # can also be set via env var
        help="Model: short name (e.g. qwen3.5-397b-a17b, mistral3, gemma3, deepseek-v3.2) or full HF ID.",
    )
    # --list-models: print the registry and exit
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print registered test models and exit.",
    )
    # --run-all: loop through every model in MODEL_REGISTRY
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all registered test models; write <short_name>.json and <short_name>.csv in output_dir.",
    )
    # --output-dir: folder for --run-all results
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for --run-all outputs (default: current dir).",
    )
    parser.add_argument(
        "--cheapest",
        action="store_true",
        help="Append :cheapest to model ID so HF routes to the lowest-cost provider",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="For JSONL input: send this many records per API call (default: 1). "
             "Increase to 3-5 if your records are small. Set to 0 for all in one call.",
    )
    parser.add_argument(
        "--batches",
        type=str,
        default=None,
        help="Comma-separated list of 1-indexed batch numbers to run "
             "(e.g. --batches 1,5,10,20,21,23). Skipped batches pass through uncorrected.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Override max output tokens (default: auto-computed from model context "
             "window and input size). Set to 0 for automatic.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    args = parser.parse_args()  # parse command-line arguments into args namespace

    # ── --list-models: just print the registry and exit ──────────────────
    if args.list_models:
        print("Registered test models (use --model <short_name> or full HF ID):")
        for short, full in MODEL_REGISTRY.items():
            print(f"  {short}\n    -> {full}")
        return  # nothing else to do

    # ── Validate required arguments ──────────────────────────────────────
    if args.input_json is None:
        parser.error("input_json is required (or use --list-models)")

    if not args.model and not args.run_all:
        parser.error("--model is required (or set OCR_HF_MODEL, or use --run-all). Use --list-models to see options.")

    input_path = args.input_json
    use_jsonl = args.jsonl or input_path.endswith(".jsonl")
    batch_size = args.batch_size
    only_batches = None
    if args.batches:
        only_batches = set(int(b.strip()) for b in args.batches.split(",") if b.strip())

    # ── helpers ──────────────────────────────────────────────────────────

    def _load_input(path: str, jsonl: bool):
        """Load input. JSONL → list of dicts; JSON → single object/array."""
        with open(path, encoding="utf-8") as f:
            if jsonl:
                records = []
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Skip line {line_no}: invalid JSON: {e}", file=sys.stderr)
                return records
            return json.load(f)

    def _slim_extract(records):
        """Extract lightweight payload: only document_id + language + ocr_hypothesis text."""
        slim_records = []
        for rec in records:
            doc_id = rec.get("document_metadata", {}).get("document_id", "unknown")
            lang = rec.get("document_metadata", {}).get("language", "en")
            hyp = rec.get("ocr_hypothesis", {})
            hyp_text = hyp.get("transcription_unit", "") if isinstance(hyp, dict) else str(hyp)
            slim_records.append({
                "document_id": doc_id,
                "language": lang,
                "ocr_hypothesis": hyp_text,
            })
        return slim_records

    def _slim_reinject(original_records, corrections):
        """Merge model corrections back into original records by document_id order."""
        correction_map = {}
        for corr in corrections:
            doc_id = corr.get("document_id")
            text = corr.get("ocr_postcorrection_output", "")
            if doc_id:
                correction_map[doc_id] = text

        merged = []
        for rec in original_records:
            rec = json.loads(json.dumps(rec))  # deep copy
            doc_id = rec.get("document_metadata", {}).get("document_id")
            corrected_text = correction_map.get(doc_id)
            if corrected_text is None and corrections:
                idx = len(merged)
                if idx < len(corrections):
                    corrected_text = corrections[idx].get("ocr_postcorrection_output", "")

            if corrected_text is not None:
                if isinstance(rec.get("ocr_postcorrection_output"), dict):
                    rec["ocr_postcorrection_output"]["transcription_unit"] = corrected_text
                else:
                    rec["ocr_postcorrection_output"] = corrected_text
            merged.append(rec)
        return merged

    def _build_messages(payload):
        user_content = json.dumps(payload, ensure_ascii=False, indent=0)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _run_model(client, model_id, payload, max_tokens, temperature):
        """Single API call. Automatically computes max_tokens from the model's
        context window and the actual input size.
        Returns (json_raw, csv_section, response).
        """
        messages = _build_messages(payload)
        input_est = _estimate_tokens("".join(m["content"] for m in messages))
        auto_max = _compute_max_tokens(model_id, messages)
        effective_max = min(max_tokens, auto_max) if max_tokens else auto_max
        base_id = model_id.split(":")[0]
        ctx = MODEL_CONTEXT_WINDOW.get(base_id, DEFAULT_CONTEXT_WINDOW)
        print(
            f"    context_window={ctx}, ~{input_est} input tokens, "
            f"max_output={effective_max}",
            file=sys.stderr,
        )

        try:
            logging.info(messages)
            response, finish_reason = chat_completion(
                client, model_id, messages,
                max_tokens=effective_max, temperature=temperature,
            )
        except Exception as e:
            if "maximum context length" in str(e) or "bad_request" in str(e).lower():
                print(
                    f"  Context window exceeded at max_tokens={effective_max}.",
                    file=sys.stderr,
                )
            raise

        output_est = _estimate_tokens(response)
        json_raw, csv_section, _ = parse_response(response)

        fr_from_api = (finish_reason or "").lower()
        truncated_by_length = fr_from_api == "length"
        truncated_by_heuristic = False
        if not truncated_by_length and json_raw:
            stripped = json_raw.rstrip()
            if stripped and stripped[-1] not in ("}", "]"):
                truncated_by_heuristic = True

        truncated = truncated_by_length or truncated_by_heuristic
        has_csv = bool(csv_section.strip())
        json_tail = json_raw.rstrip()[-80:] if json_raw else "(empty)"

        print(
            f"    finish_reason={finish_reason}, ~{output_est} output tokens, "
            f"has_csv={has_csv}",
            file=sys.stderr,
        )
        print(f"    json tail: ...{json_tail!r}", file=sys.stderr)

        if truncated:
            reason = "finish_reason=length" if truncated_by_length else "heuristic (json doesn't end with } or ])"
            print(
                f"  Truncation detected via {reason}. Attempting JSON repair...",
                file=sys.stderr,
            )

        return json_raw, csv_section, response

    def _make_batches(records, size):
        """Split a list into batches. size=0 means one batch with everything."""
        if size <= 0:
            return [records]
        return [records[i:i + size] for i in range(0, len(records), size)]

    def _run_batched(client, model_id, records, max_tokens, temperature,
                     label="", only_batches=None,
                     output_json=None, output_csv=None, is_jsonl=True):
        """Run model on batches of slim payloads; re-inject corrections into
        the original records. Returns (all_results, merged_csv, raw_responses).

        Output is written **incrementally** after each batch so that progress
        is saved even if the process is killed mid-run.

        If *only_batches* is a set of 1-indexed batch numbers, only those
        batches are sent to the API; the rest are skipped (their records pass
        through uncorrected).
        """
        send_records = _slim_extract(records)
        slim_batches = _make_batches(send_records, batch_size)
        orig_batches = _make_batches(records, batch_size)
        n_calls = len(slim_batches)
        if only_batches:
            run_count = sum(1 for i in range(1, n_calls + 1) if i in only_batches)
            print(
                f"  {label}{len(records)} records in {n_calls} batch(es) "
                f"(batch_size={batch_size or 'all'}), running {run_count} selected batch(es): "
                f"{sorted(only_batches)}",
                file=sys.stderr,
            )
        else:
            print(
                f"  {label}{len(records)} records in {n_calls} batch(es) "
                f"(batch_size={batch_size or 'all'})",
                file=sys.stderr,
            )

        all_corrections = []
        csv_header = None
        csv_data_lines = []
        raw_responses = []

        json_fh = open(output_json, "w", encoding="utf-8") if output_json else None
        csv_fh = open(output_csv, "w", encoding="utf-8", newline="") if output_csv else None

        def _flush_json_batch(recs):
            if not json_fh:
                return
            for rec in recs:
                json_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            json_fh.flush()

        def _flush_csv_rows(header, rows):
            if not csv_fh:
                return
            if header and csv_fh.tell() == 0:
                csv_fh.write(header + "\n")
            for row in rows:
                csv_fh.write(row + "\n")
            csv_fh.flush()

        failed = 0
        try:
            for batch_idx, (slim_batch, orig_batch) in enumerate(
                zip(slim_batches, orig_batches), 1
            ):
                if only_batches and batch_idx not in only_batches:
                    _flush_json_batch(orig_batch)
                    continue
                if batch_idx > 1:
                    time.sleep(3)
                print(
                    f"  Batch {batch_idx}/{n_calls} ({len(slim_batch)} records)...",
                    file=sys.stderr,
                )
                last_response = ""
                try:
                    json_raw, csv_section, last_response = _run_model(
                        client, model_id, slim_batch, max_tokens, temperature,
                    )
                    raw_responses.append(last_response)

                    result = _try_parse_json(json_raw)
                    batch_corrections = result if isinstance(result, list) else [result]
                    all_corrections.extend(batch_corrections)

                    corrected = _slim_reinject(orig_batch, batch_corrections)
                    _flush_json_batch(corrected)

                    hdr, rows = _csv_body_lines(csv_section)
                    if hdr and csv_header is None:
                        csv_header = hdr
                    csv_data_lines.extend(rows)
                    _flush_csv_rows(hdr, rows)

                except Exception as e:
                    failed += 1
                    print(
                        f"  Batch {batch_idx} failed (skipping): {e}",
                        file=sys.stderr,
                    )
                    if last_response:
                        preview = last_response[:300].replace("\n", "\\n")
                        print(f"    Raw response preview: {preview}", file=sys.stderr)
                    _flush_json_batch(orig_batch)
        finally:
            if json_fh:
                json_fh.close()
                print(f"  Output saved to {output_json}", file=sys.stderr)
            if csv_fh:
                csv_fh.close()
                if output_csv:
                    print(f"  Output saved to {output_csv}", file=sys.stderr)

        if failed:
            print(
                f"  {failed}/{n_calls} batch(es) failed",
                file=sys.stderr,
            )

        merged_csv = ""
        if csv_header:
            merged_csv = csv_header + "\n" + "\n".join(csv_data_lines)
            if not merged_csv.endswith("\n"):
                merged_csv += "\n"

        all_results = _slim_reinject(records, all_corrections)
        return all_results, merged_csv, raw_responses

    def _write_json(path, result, is_jsonl):
        with open(path, "w", encoding="utf-8") as f:
            if is_jsonl and isinstance(result, list):
                for obj in result:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                json.dump(result, f, ensure_ascii=False, indent=2)

    # ── Load input ───────────────────────────────────────────────────────
    payload = _load_input(input_path, use_jsonl)
    n_records = len(payload) if isinstance(payload, list) else 1
    print(f"Loaded {n_records} record(s) from {input_path}", file=sys.stderr)

    # ── --run-all branch ─────────────────────────────────────────────────
    if args.run_all:
        out_dir = args.output_dir or os.getcwd()
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=False)

        client = get_client()

        for short_name, full_id in MODEL_REGISTRY.items():
            mid = f"{full_id}:cheapest" if args.cheapest else full_id
            ext = "jsonl" if use_jsonl else "json"
            json_path = os.path.join(out_dir, f"hipe-ocrepair-bench_v0.9_{short_name}.{ext}")
            csv_path = os.path.join(out_dir, f"hipe-ocrepair-bench_v0.9_{short_name}.csv")
            print(f"Running {short_name} ({mid}) -> {json_path}, {csv_path}", file=sys.stderr)

            try:
                if use_jsonl and isinstance(payload, list):
                    all_results, merged_csv, _ = _run_batched(
                        client, mid, payload, args.max_tokens, args.temperature,
                        label=f"[{short_name}] ",
                        only_batches=only_batches,
                        output_json=json_path,
                        output_csv=csv_path,
                    )
                else:
                    json_raw, csv_section, _ = _run_model(
                        client, mid, payload, args.max_tokens, args.temperature,
                    )
                    _write_json(json_path, _try_parse_json(json_raw), is_jsonl=False)
                    with open(csv_path, "w", encoding="utf-8", newline="") as f:
                        f.write(csv_section)
                        if csv_section and not csv_section.endswith("\n"):
                            f.write("\n")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  Could not parse response: {e}", file=sys.stderr)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)

        return  # done with --run-all

    # ── Single-model branch ──────────────────────────────────────────────

    model_id = resolve_model(args.model)
    short_name = args.model.strip().lower()
    if args.cheapest and ":" not in model_id:
        model_id = f"{model_id}:cheapest"

    # If --output-dir is set, auto-generate -j and -c paths with model name
    if args.output_dir:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        ext = "jsonl" if use_jsonl else "json"
        if not args.output_json:
            args.output_json = os.path.join(out_dir, f"{short_name}.{ext}")
        if not args.output_csv:
            args.output_csv = os.path.join(out_dir, f"{short_name}.csv")

    client = get_client()

    if not any([args.output_json, args.output_csv, args.output_raw]):
        print(
            "No output requested: use -j (JSON/JSONL), -c (CSV), and/or -o (raw). "
            "Example: -j out.jsonl -c corrections.csv",
            file=sys.stderr,
        )

    try:
        if use_jsonl and isinstance(payload, list):
            all_results, merged_csv, raw_responses = _run_batched(
                client, model_id, payload, args.max_tokens, args.temperature,
                only_batches=only_batches,
                output_json=args.output_json,
                output_csv=args.output_csv,
            )
            if args.output_raw:
                for i, resp in enumerate(raw_responses, 1):
                    args.output_raw.write(f"=== batch {i} ===\n{resp}\n")
                args.output_raw.close()
                print("Raw output written to", args.output_raw.name, file=sys.stderr)
        else:
            json_raw, csv_section, response = _run_model(
                client, model_id, payload, args.max_tokens, args.temperature,
            )
            if args.output_raw:
                args.output_raw.write(response)
                args.output_raw.close()
                print("Raw output written to", args.output_raw.name, file=sys.stderr)
            if args.output_json:
                _write_json(args.output_json, _try_parse_json(json_raw), is_jsonl=False)
                print("Corrected JSON written to", args.output_json, file=sys.stderr)
            if args.output_csv:
                with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
                    f.write(csv_section)
                    if csv_section and not csv_section.endswith("\n"):
                        f.write("\n")
                print("Corrections CSV written to", args.output_csv, file=sys.stderr)

    except (json.JSONDecodeError, ValueError) as e:
        print("Could not parse model response:", e, file=sys.stderr)
        raise SystemExit(1)
    except Exception as e:
        print("Inference error:", e, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
