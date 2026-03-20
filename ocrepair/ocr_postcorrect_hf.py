#!/usr/bin/env python3
"""
OCR post-correction via Hugging Face Inference API using the cheapest provider.

Uses the model suffix :cheapest so the HF router selects the lowest-cost provider
(lowest price per output token). Requires HF_TOKEN with Inference Providers permission.

Usage:
  export HF_TOKEN=your_token
  python ocr_postcorrect_hf.py input.json --model qwen3.5-397b-a17b -j corrected.json -c corrections.csv
  python ocr_postcorrect_hf.py input.json --model mistral3 -j out.json -c out.csv
  python ocr_postcorrect_hf.py --list-models
  python ocr_postcorrect_hf.py input.json --run-all --output-dir ./results
"""

import argparse  # for parsing command-line arguments
import json      # for reading/writing JSON files
import os        # for environment variables and file paths
import sys       # for stderr output and exit codes

try:
    from dotenv import load_dotenv  # for environment variables from .env  # ty:ignore[unresolved-import]
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

# The full system prompt sent to the model.
# It instructs the model to:
#   1) correct OCR tokens in the JSON (fill "ocr_postcorrection_output")
#   2) produce a CSV corrections log (after the marker CORRECTIONS_CSV)
#   3) produce a discovered error types section (after the marker DISCOVERED_ERRORS)
SYSTEM_PROMPT = """You are an expert in historical OCR post-correction for newspapers from the 17th to 20th century in English. You will receive a JSON containing OCR tokens from historical newspapers. 
STEP 1 - Your task: 
1. Read the "ocr_hypothesis" field in each token
2. Correct OCR errors and write the corrected text into "ocr_postcorrection_output"
3. Do NOT modify any other field in the JSON
4. Do NOT alter person names or place names
5. Check the "language" field and correct only within that language — do not translate or deviate
6. Pay close attention to accented characters — preserve or restore them correctly
7. Fix four classes of common historical OCR errors:
- Over-segmentation: words being broken across lines to fit within the margins and hyphens being replaced with spaces e.g. [incorrect] "before the follow ing morning" -> [corrected] "before the follow-ing morning"
- Under-segmentation: spaces being removed between words e.g. [incorrect] "Two Closes of Rich oldSwarth LAND" -> [corrected] "Two Closes of Rich old Swarth LAND"
- Misrecognized character: e.g. [incorrect] "aſter" -> [corrected] "after"; [incorrect] "We sailed from Kalaniita Bay, ar.d soon we made the coast" -> [corrected] "We sailed from Kalaniita Bay, and soon we made the coast"
8. If a token is too ambiguous to correct confidently, copy the ocr_hypothesis unchanged into ocr_postcorrection_output
9. Return ONLY the modified JSON. No explanation, no preamble, no markdown fences.

STEP 2 — After the corrected JSON, output a CSV corrections log.
The CSV must have exactly these columns:
token_id,ocr_hypothesis,ocr_postcorrection_output,error_type,confidence,uncertain_chars,notes

Rules for the CSV:
- error_type: use one of these known types when they apply: over_segmentation, misrecognized_char, under_segmentation, wrong_word, uncertain, no_change
- If the error does not fit any known type, invent a descriptive snake_case label and prefix it with "custom:" (e.g. custom:long_s_substitution, custom:ink_bleed_merge, custom:ligature_ct). Be specific — do not use generic custom labels like custom:other.
- You may assign multiple error types to one token by separating them with a pipe character | (e.g. broken_hyphen|spacing or garbled_char|custom:long_s_substitution)
- confidence must be: high, medium, or low
- uncertain_chars: copy the corrected text but wrap every character you are unsure about in angle brackets ⟨⟩. If fully confident, leave empty. Examples: "t⟨h⟩e" means the h is uncertain; "⟨J⟩ustice" means the capital J is uncertain; "" means full confidence.
- notes is a short free-text explanation (no commas inside notes — use semicolons instead)
- Include one row per token, even if no change was made (use no_change)
- Do not wrap the CSV in markdown fences

STEP 3 — After the CSV, output a discovered error types section.
Format it exactly like this:
DISCOVERED_ERRORS
custom_label|count|example|description
(one row per unique custom: type found, with how many tokens used it, a short example, and a plain English description of the error pattern)
Do not wrap in markdown fences.

Output format — follow this structure exactly:
1. The corrected JSON (no markdown fences)
2. A blank line
3. The text: CORRECTIONS_CSV
4. The CSV rows
5. A blank line
6. The text: DISCOVERED_ERRORS
7. The discovered error rows (only if custom: types were used; otherwise write DISCOVERED_ERRORS then "none")"""


def get_client():
    """Create and return a Hugging Face InferenceClient, authenticated with HF_TOKEN."""
    from huggingface_hub import InferenceClient  # HF's Python SDK for inference  # ty:ignore[unresolved-import]

    # Read the token from the environment variable
    token = os.environ.get("HF_TOKEN")
    if not token:
        # Abort early with a helpful message if no token is set
        raise SystemExit(
            "HF_TOKEN is required. Create a token at https://huggingface.co/settings/tokens "
            "with 'Make calls to Inference Providers' and set: export HF_TOKEN=..."
        )
    # Return an authenticated client; all API calls will use this token
    return InferenceClient(api_key=token)


def chat_completion(client, model: str, messages: list, **kwargs):
    """
    Send a chat completion request and return the model's text response.
    `client`   — the InferenceClient from get_client()
    `model`    — full HF model ID, possibly with :cheapest suffix
    `messages` — list of {"role": ..., "content": ...} dicts (system + user)
    `**kwargs` — extra params forwarded to the API (max_tokens, temperature, etc.)
    """
    # Call the HF chat completions endpoint (OpenAI-compatible format)
    result = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    # The response contains a list of choices; we take the first one's text content
    return result.choices[0].message.content


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


def main():
    # ── Argument parser ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="OCR post-correction via Hugging Face (cheapest provider)"
    )
    # Positional: the input JSON file containing OCR tokens
    parser.add_argument(
        "input_json",
        nargs="?",                                     # optional so --list-models works without it
        type=argparse.FileType("r", encoding="utf-8"), # opens the file for reading
        default=None,
        help="Path to JSON file with OCR tokens (must contain ocr_hypothesis per token)",
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
        help="Model: short name (e.g. qwen3.5-397b-a17b, mistral3, gemma3, deepseek-v3.2) or full HF ID. :cheapest appended unless --no-cheapest.",
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
    # --no-cheapest: skip the :cheapest suffix (use HF's default routing = fastest)
    parser.add_argument(
        "--no-cheapest",
        action="store_true",
        help="Do not append :cheapest to model (use default/fastest provider)",
    )
    # --max-tokens: cap on how many tokens the model can generate
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens for completion (default: 8192)",
    )
    # --temperature: controls randomness; low = more deterministic
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

    # ── Load input JSON and build the messages for the model ─────────────
    with args.input_json as f:     # read and parse the input JSON file
        payload = json.load(f)

    # Serialise the JSON payload as a compact string to send as the user message
    user_content = json.dumps(payload, ensure_ascii=False, indent=0)

    # Build the chat messages: system prompt (instructions) + user message (the data)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},  # tells the model what to do
        {"role": "user", "content": user_content},      # the actual OCR tokens to correct
    ]

    # ── --run-all branch: loop through every registered model ────────────
    if args.run_all:
        out_dir = args.output_dir or os.getcwd()  # default output dir = current directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=False)        # create dir if it doesn't exist

        client = get_client()  # one client for all requests (same token)

        for short_name, full_id in MODEL_REGISTRY.items():
            # Append :cheapest to route to the cheapest provider, unless disabled
            mid = f"{full_id}:cheapest" if not args.no_cheapest else full_id

            # Build output file paths: one .json and one .csv per model
            json_path = os.path.join(out_dir, f"hipe-ocrepair-bench_v0.9_{short_name}.json")
            csv_path = os.path.join(out_dir, f"hipe-ocrepair-bench_v0.9_{short_name}.csv")
            print(f"Running {short_name} ({mid}) -> {json_path}, {csv_path}", file=sys.stderr)

            try:
                # Send the request to this model
                response = chat_completion(
                    client, mid, messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                try:
                    # Split the response into JSON, CSV, and discovered sections
                    json_raw, csv_section, _ = parse_response(response)

                    # Write the corrected JSON (parsed then re-serialised for clean formatting)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(json.loads(json_raw), f, ensure_ascii=False, indent=2)

                    # Write the CSV corrections log
                    with open(csv_path, "w", encoding="utf-8", newline="") as f:
                        f.write(csv_section)
                        if csv_section and not csv_section.endswith("\n"):
                            f.write("\n")  # ensure file ends with a newline

                except (json.JSONDecodeError, ValueError) as e:
                    # If the model's JSON was malformed, log but continue with next model
                    print(f"  Could not parse response: {e}", file=sys.stderr)

            except Exception as e:
                # If the API call itself failed, log and continue with next model
                print(f"  Error: {e}", file=sys.stderr)

        return  # done with --run-all

    # ── Single-model branch ──────────────────────────────────────────────

    # Resolve short name to full HF model ID (e.g. "mistral3" -> "mistralai/...")
    model_id = resolve_model(args.model)
    # Append :cheapest so HF routes to the lowest-cost provider, unless disabled
    if not args.no_cheapest and ":" not in model_id:
        model_id = f"{model_id}:cheapest"

    # Create the HF client and send the request
    client = get_client()
    try:
        response = chat_completion(
            client,
            model_id,
            messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except Exception as e:
        print("Inference error:", e, file=sys.stderr)
        raise SystemExit(1)

    # Warn if the user didn't request any output files
    if not any([args.output_json, args.output_csv, args.output_raw]):
        print("No output requested: use -j (JSON), -c (CSV), and/or -o (raw). Example: -j out.json -c corrections.csv", file=sys.stderr)

    # Optionally save the full raw model response (JSON + CSV + DISCOVERED_ERRORS as one text blob)
    if args.output_raw:
        args.output_raw.write(response)  # write entire response as-is
        args.output_raw.close()
        print("Raw output written to", args.output_raw.name, file=sys.stderr)

    # Parse the response and write the structured output files
    try:
        # Split model output into the three sections
        json_raw, csv_section, _ = parse_response(response)

        # -j: write the corrected JSON (same structure as input, with ocr_postcorrection_output filled)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(json.loads(json_raw), f, ensure_ascii=False, indent=2)
            print("Corrected JSON written to", args.output_json, file=sys.stderr)

        # -c: write the corrections CSV log
        if args.output_csv:
            with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
                f.write(csv_section)
                if csv_section and not csv_section.endswith("\n"):
                    f.write("\n")  # ensure file ends with a newline
            print("Corrections CSV written to", args.output_csv, file=sys.stderr)

    except (json.JSONDecodeError, ValueError) as e:
        # If the model response couldn't be parsed, exit with error
        print("Could not parse model response:", e, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    # main()
    print("hello")
