import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_engine.sft_builder import load_jsonl_records
from src.inference.predict import predict_from_response


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


def _find_sample(samples: list[dict[str, object]], sample_id: str) -> dict[str, object]:
    sample = next(
        (candidate for candidate in samples if candidate["sample_id"] == sample_id),
        None,
    )
    if sample is None:
        raise SystemExit(f"ERROR: sample_id not found: {sample_id}")
    return sample


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render an online audit decision for one canonical sample."
    )
    parser.add_argument(
        "--samples",
        default="data/canonical/samples.jsonl",
        help="Canonical sample JSONL path.",
    )
    parser.add_argument("--sample-id", required=True, help="Sample id to inspect.")
    parser.add_argument(
        "--response",
        help="Structured JSON response payload to parse.",
    )
    parser.add_argument(
        "--response-file",
        help="Path to a file containing a structured JSON response payload.",
    )
    args = parser.parse_args()

    samples_path = _resolve_path(args.samples)
    if not samples_path.is_file():
        raise SystemExit(f"ERROR: samples file not found: {samples_path}")

    samples = load_jsonl_records(samples_path)
    sample = _find_sample(samples, args.sample_id)
    raw_response = _load_response(args)

    try:
        prediction = predict_from_response(
            raw_response=raw_response,
            fallback_response=sample["output"],
        )
    except ValueError as exc:
        print(f"ERROR: invalid response: {exc}", file=sys.stderr)
        return 1

    print(f"Sample ID: {sample['sample_id']}")
    print("Input form:")
    print(json.dumps(sample["input"]["form"], ensure_ascii=False, indent=2))
    print("Images:")
    print(json.dumps(sample["input"]["images"], ensure_ascii=False, indent=2))
    print("Structured response:")
    print(
        json.dumps(
            {
                "point_results": prediction["point_results"],
                "reject_tags": prediction["reject_tags"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Decision:")
    print(json.dumps(prediction["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
