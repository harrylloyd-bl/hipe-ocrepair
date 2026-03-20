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
import getpass
import json
import os
import sys
import time

try:
    from dotenv import load_dotenv  # ty:ignore[unresolved-import]
    if "HF_TOKEN" not in os.environ:
        load_dotenv()
except ImportError:
    pass


# Lookup table mapping friendly short names to full Hugging Face model IDs.
# This lets you type e.g. --model mistral3 instead of the full path.
MODEL_REGISTRY = {
    "qwen3.5-397b-a17b": "Qwen/Qwen3.5-397B-A17B",
    "qwen3-235b-a22b-thinking-2507": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "mistral3": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "gemma3": "google/gemma-3-27b-it",
    "deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
}

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

After the CSV:
DISCOVERED_ERRORS
custom_label|count|example|description
(or "none" if no custom types)"""


def get_client():
    """Create and return a Hugging Face InferenceClient, authenticated with HF_TOKEN."""
    from huggingface_hub import InferenceClient  # ty:ignore[unresolved-import]

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
    return InferenceClient(api_key=token.strip())


MAX_RETRIES = 3
RETRY_BACKOFF = [30, 60, 120]  # seconds to wait between retries


def chat_completion(client, model: str, messages: list, **kwargs):
    """Send a chat completion request with automatic retries on timeout."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            return result.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(
                kw in err_str
                for kw in ("timeout", "gateway", "502", "503", "504", "529")
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
    key = name_or_id.strip().lower()            # normalise to lowercase for lookup
    return MODEL_REGISTRY.get(key, name_or_id.strip())  # fallback to original if not found


def parse_response(response: str):
    """
    Split the raw model output into three parts using the section markers
    we asked for in the prompt:
      1. json_raw         — the corrected JSON (as a string, ready to json.loads)
      2. csv_section      — the CSV corrections log (text between CORRECTIONS_CSV and DISCOVERED_ERRORS)
      3. discovered_section — the discovered error types (text after DISCOVERED_ERRORS)
    """
    # --- Step 1: separate the JSON from everything after CORRECTIONS_CSV ---
    parts = response.split("CORRECTIONS_CSV", 1)  # split at most once
    json_raw = parts[0].strip()                    # everything before the marker = JSON
    rest = parts[1].strip() if len(parts) > 1 else ""  # everything after = CSV + DISCOVERED

    # --- Step 2: strip markdown code fences if the model wrapped the JSON in them ---
    if json_raw.startswith("```"):
        lines = json_raw.split("\n")
        if lines[0].startswith("```"):     # remove opening fence line (e.g. ```json)
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":  # remove closing fence line
            lines = lines[:-1]
        json_raw = "\n".join(lines)        # rejoin without the fences

    # --- Step 3: split the remainder into CSV and DISCOVERED_ERRORS sections ---
    csv_section = ""
    discovered_section = ""
    if rest:
        parts2 = rest.split("DISCOVERED_ERRORS", 1)  # split at most once
        csv_section = parts2[0].strip()               # text before marker = CSV rows
        discovered_section = parts2[1].strip() if len(parts2) > 1 else ""  # text after = discovered

    return json_raw, csv_section, discovered_section


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
        "--max-tokens",
        type=int,
        default=16384,
        help="Max tokens for completion (default: 16384)",
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
        """Single API call: send payload, return (json_raw, csv_section, response)."""
        messages = _build_messages(payload)
        response = chat_completion(
            client, model_id, messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        json_raw, csv_section, _ = parse_response(response)
        return json_raw, csv_section, response

    def _make_batches(records, size):
        """Split a list into batches. size=0 means one batch with everything."""
        if size <= 0:
            return [records]
        return [records[i:i + size] for i in range(0, len(records), size)]

    def _run_batched(client, model_id, records, max_tokens, temperature, label=""):
        """Run model on batches of slim payloads; re-inject corrections into
        the original records. Returns (all_results, merged_csv, raw_responses).
        """
        send_records = _slim_extract(records)
        batches = _make_batches(send_records, batch_size)
        n_calls = len(batches)
        print(
            f"  {label}{len(records)} records in {n_calls} batch(es) "
            f"(batch_size={batch_size or 'all'})",
            file=sys.stderr,
        )

        all_corrections = []
        csv_header = None
        csv_data_lines = []
        raw_responses = []

        for batch_idx, batch in enumerate(batches, 1):
            if batch_idx > 1:
                time.sleep(3)
            print(
                f"  Batch {batch_idx}/{n_calls} ({len(batch)} records)...",
                file=sys.stderr,
            )
            json_raw, csv_section, response = _run_model(
                client, model_id, batch, max_tokens, temperature,
            )
            raw_responses.append(response)

            result = json.loads(json_raw)
            if isinstance(result, list):
                all_corrections.extend(result)
            else:
                all_corrections.append(result)

            hdr, rows = _csv_body_lines(csv_section)
            if hdr and csv_header is None:
                csv_header = hdr
            csv_data_lines.extend(rows)

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
                    )
                    _write_json(json_path, all_results, is_jsonl=True)
                    with open(csv_path, "w", encoding="utf-8", newline="") as f:
                        f.write(merged_csv)
                else:
                    json_raw, csv_section, _ = _run_model(
                        client, mid, payload, args.max_tokens, args.temperature,
                    )
                    _write_json(json_path, json.loads(json_raw), is_jsonl=False)
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
    if args.cheapest and ":" not in model_id:
        model_id = f"{model_id}:cheapest"

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
            )
            if args.output_raw:
                for i, resp in enumerate(raw_responses, 1):
                    args.output_raw.write(f"=== batch {i} ===\n{resp}\n")
                args.output_raw.close()
                print("Raw output written to", args.output_raw.name, file=sys.stderr)
            if args.output_json:
                _write_json(args.output_json, all_results, is_jsonl=True)
                print("Corrected JSONL written to", args.output_json, file=sys.stderr)
            if args.output_csv:
                with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
                    f.write(merged_csv)
                print("Corrections CSV written to", args.output_csv, file=sys.stderr)
        else:
            json_raw, csv_section, response = _run_model(
                client, model_id, payload, args.max_tokens, args.temperature,
            )
            if args.output_raw:
                args.output_raw.write(response)
                args.output_raw.close()
                print("Raw output written to", args.output_raw.name, file=sys.stderr)
            if args.output_json:
                _write_json(args.output_json, json.loads(json_raw), is_jsonl=False)
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
