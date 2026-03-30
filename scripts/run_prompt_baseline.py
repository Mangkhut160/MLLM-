import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_engine.sft_builder import load_jsonl_records
from src.prompt_baseline.prompt_templates import build_audit_prompt
from src.prompt_baseline.response_parser import parse_audit_response


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_response(args: argparse.Namespace) -> str | None:
    if args.response and args.response_file:
        raise SystemExit("ERROR: provide either --response or --response-file, not both")

    if args.response:
        return args.response

    if args.response_file:
        response_path = _resolve_path(args.response_file)
        if not response_path.is_file():
            raise SystemExit(f"ERROR: response file not found: {response_path}")
        return response_path.read_text(encoding="utf-8")

    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the prompt baseline for one canonical sample."
    )
    parser.add_argument(
        "--samples",
        default="data/canonical/samples.jsonl",
        help="Canonical sample JSONL path.",
    )
    parser.add_argument("--sample-id", required=True, help="Sample id to inspect.")
    parser.add_argument(
        "--response",
        help="Raw model response string to parse as canonical JSON.",
    )
    parser.add_argument(
        "--response-file",
        help="Path to a file containing raw model output to parse.",
    )
    args = parser.parse_args()

    samples_path = _resolve_path(args.samples)
    if not samples_path.is_file():
        raise SystemExit(f"ERROR: samples file not found: {samples_path}")

    samples = load_jsonl_records(samples_path)
    sample = next(
        (candidate for candidate in samples if candidate["sample_id"] == args.sample_id),
        None,
    )
    if sample is None:
        raise SystemExit(f"ERROR: sample_id not found: {args.sample_id}")

    prompt = build_audit_prompt(sample["input"]["form"])
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Brand image: {sample['input']['images']['brand_image']}")
    print(f"Spec image: {sample['input']['images']['spec_image']}")
    print("Prompt:")
    print(prompt)

    response = _load_response(args)
    if response is None:
        print("No response provided. Use --response or --response-file to parse output.")
        print("Expected target:")
        print(json.dumps(sample["output"], ensure_ascii=False, indent=2))
        return 0

    try:
        parsed = parse_audit_response(response)
    except ValueError as exc:
        print(f"ERROR: invalid response: {exc}", file=sys.stderr)
        return 1

    print("Parsed response:")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
