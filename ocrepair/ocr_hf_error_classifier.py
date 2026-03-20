#!/usr/bin/env python3
"""
OCR error classification via Hugging Face Inference API (same model registry as ocr_postcorrect_hf).

Compares ground truth to OCR hypothesis, asks the model to list common error types using
canonical historical-OCR classes plus reusable custom categories. Writes JSON that preserves
the input and adds an OCR_mistake field per relevant item.

Requires HF_TOKEN with Inference Providers permission (same as ocr_postcorrect_hf).

Usage:
  export HF_TOKEN=your_token
  python ocrepair/ocr_hf_error_classifier.py input.json -j out.json --model mistral3
  python ocrepair/ocr_hf_error_classifier.py input.json -j out.json --model qwen3.5-397b-a17b
  python ocrepair/ocr_hf_error_classifier.py --list-models
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

try:
    from dotenv import load_dotenv  # ty:ignore[unresolved-import]

    if "HF_TOKEN" not in os.environ:
        load_dotenv()
except ImportError:
    pass


# Same registry as ocr_postcorrect_hf.py
MODEL_REGISTRY = {
    "qwen3.5-397b-a17b": "Qwen/Qwen3.5-397B-A17B",
    "qwen3-235b-a22b-thinking-2507": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "mistral3": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "gemma3": "google/gemma-3-27b-it",
    "deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
}

SYSTEM_PROMPT = """You are an expert in historical OCR error analysis for newspapers and books from roughly the 17th to 20th century in English, French, and German. You will receive a JSON document. Items may include a ground-truth string and an OCR hypothesis string (field names may vary slightly, e.g. ground_truth / gt / reference and ocr_hypothesis / hypothesis / ocr).

Your task:
1. For every object that has both a ground-truth field and an OCR hypothesis field, compare them and identify which OCR mistakes explain the differences.
2. Compile the most common error *patterns* you observe across the document (not only token-by-token noise).
3. Add a new field named exactly `OCR_mistake` to each such object. Do not remove or rename existing fields.

Use the following **four** canonical classes. Use these **exact** snake_case identifiers in `canonical_classes` when they apply:
- `over_segmentation`: words broken across lines; hyphens or word boundaries replaced with spaces, e.g. [incorrect] "before the follow ing morning" → [correct] "before the follow-ing morning"
- `under_segmentation`: spaces removed between words, e.g. [incorrect] "Two Closes of Rich oldSwarth LAND" → [correct] "Two Closes of Rich old Swarth LAND"
- `misrecognized_char`: wrong glyph or punctuation, e.g. [incorrect] "aſter" → [correct] "after"; [incorrect] "We sailed from Kalaniita Bay, ar.d soon we made the coast" → [correct] "We sailed from Kalaniita Bay, and soon we made the coast"
- `missing_character`: dropped letters or similar, e.g. [incorrect] "er a good deal of argument, the facts were agreed" → [correct] "After a good deal of argument, the facts were agreed"

Also align with the post-correction CSV taxonomy when useful: you may set `extra_labels` to a subset of `wrong_word`, `uncertain`, `no_change` when they describe the situation better than the four classes alone (e.g. `no_change` when hypothesis matches ground truth; `uncertain` when you cannot confidently label).

If an error **cannot** be described by these four classes (and optional extra labels), assign a **new** reusable category in `custom_categories`: use short `snake_case` names (e.g. `repeated_line`, `column_skip`). Before inventing a new label, **check** categories you already introduced in this response and **reuse** the same label when the same pattern appears again.

The value of `OCR_mistake` must be a JSON object with this shape:
{
  "canonical_classes": ["over_segmentation", ...],   // subset of the four above; empty array if none apply
  "extra_labels": ["wrong_word"],                    // optional: any of wrong_word, uncertain, no_change; omit or use [] if none
  "custom_categories": ["my_custom_label", ...],     // additional labels you defined; empty if none
  "brief_notes": "one short sentence; no unescaped newlines"
}

If there is genuinely no difference between ground truth and OCR hypothesis for an item, set:
  "canonical_classes": [],
  "extra_labels": ["no_change"],
  "custom_categories": [],
  "brief_notes": "no_difference"

Also add **one** top-level key to the root JSON object (do not remove existing top-level keys):
  "ocr_error_document_summary": {
    "most_common_patterns": [
      { "category": "over_segmentation", "approx_count_or_rank": "..." , "example_snippet": "..." }
    ],
    "custom_categories_defined": ["label1", "label2"],
    "notes": "short overall comment"
  }

Rules:
- Preserve the exact nesting and order of the input JSON as much as possible; only add `OCR_mistake` on objects that had both GT and hypothesis, and add `ocr_error_document_summary` at the root.
- Do not wrap the final output in markdown code fences.
- Return **only** valid JSON."""


def get_client():
    """Create Hugging Face InferenceClient authenticated with HF_TOKEN."""
    from huggingface_hub import InferenceClient  # ty:ignore[unresolved-import]

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is required. Create a token at https://huggingface.co/settings/tokens "
            "with 'Make calls to Inference Providers' and set: export HF_TOKEN=..."
        )
    return InferenceClient(api_key=token)


def chat_completion(client, model: str, messages: list, **kwargs) -> str:
    result = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    return result.choices[0].message.content


def resolve_model(name_or_id: str) -> str:
    key = name_or_id.strip().lower()
    return MODEL_REGISTRY.get(key, name_or_id.strip())


def extract_json_object(text: str) -> str:
    """Strip optional markdown fences and return the JSON payload string."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def parse_model_json(response: str) -> dict[str, Any]:
    raw = extract_json_object(response)
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify OCR errors vs ground truth via Hugging Face (cheapest provider)"
    )
    parser.add_argument(
        "input_json",
        nargs="?",
        type=argparse.FileType("r", encoding="utf-8"),
        default=None,
        help="JSON with ground truth and ocr_hypothesis fields",
    )
    parser.add_argument(
        "-j",
        "--output-json",
        type=str,
        default=None,
        metavar="FILE",
        help="Write augmented JSON (with OCR_mistake) to FILE",
    )
    parser.add_argument(
        "-o",
        "--output-raw",
        type=argparse.FileType("w", encoding="utf-8"),
        default=None,
        metavar="FILE",
        help="Write full raw model response to FILE (debug)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OCR_HF_CLASSIFIER_MODEL", os.environ.get("OCR_HF_MODEL", "")),
        help="Short name from MODEL_REGISTRY or full HF model ID; :cheapest unless --no-cheapest",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print registered models and exit",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run every model in MODEL_REGISTRY; write <short_name>_classified.json in --output-dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for --run-all outputs (default: cwd)",
    )
    parser.add_argument(
        "--no-cheapest",
        action="store_true",
        help="Do not append :cheapest to the model id",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max completion tokens (default: 8192)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Registered models (same as ocr_postcorrect_hf.py):")
        for short, full in MODEL_REGISTRY.items():
            print(f"  {short}\n    -> {full}")
        return

    if args.input_json is None:
        parser.error("input_json is required (or use --list-models)")

    if not args.model and not args.run_all:
        parser.error(
            "--model is required (or set OCR_HF_CLASSIFIER_MODEL / OCR_HF_MODEL, or use --run-all)"
        )

    with args.input_json as f:
        payload = json.load(f)

    user_content = json.dumps(payload, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if args.run_all:
        out_dir = args.output_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        client = get_client()
        for short_name, full_id in MODEL_REGISTRY.items():
            mid = f"{full_id}:cheapest" if not args.no_cheapest else full_id
            out_path = os.path.join(out_dir, f"{short_name}_classified.json")
            print(f"Running {short_name} ({mid}) -> {out_path}", file=sys.stderr)
            try:
                response = chat_completion(
                    client,
                    mid,
                    messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                data = parse_model_json(response)
                with open(out_path, "w", encoding="utf-8") as out_f:
                    json.dump(data, out_f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
        return

    model_id = resolve_model(args.model)
    if not args.no_cheapest and ":" not in model_id:
        model_id = f"{model_id}:cheapest"

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

    if args.output_raw:
        args.output_raw.write(response)
        args.output_raw.close()
        print("Raw output written to", args.output_raw.name, file=sys.stderr)

    try:
        data = parse_model_json(response)
    except (json.JSONDecodeError, ValueError) as e:
        print("Could not parse model response as JSON:", e, file=sys.stderr)
        print("--- response (first 2000 chars) ---", file=sys.stderr)
        print(response[:2000], file=sys.stderr)
        raise SystemExit(1)

    if not args.output_json:
        json.dump(data, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Written:", args.output_json, file=sys.stderr)


if __name__ == "__main__":
    main()
