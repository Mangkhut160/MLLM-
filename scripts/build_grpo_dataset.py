import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_engine.grpo_builder import build_grpo_records
from src.data_engine.sft_builder import load_jsonl_records, write_jsonl_records


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GRPO cold-start dataset.")
    parser.add_argument(
        "--input",
        default="data/canonical/samples.jsonl",
        help="Canonical or SFT-compatible JSONL input path.",
    )
    parser.add_argument(
        "--output",
        default="data/grpo/cold_start.jsonl",
        help="GRPO cold-start JSONL output path.",
    )
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)

    if not input_path.is_file():
        raise SystemExit(f"ERROR: input file not found: {input_path}")

    samples = load_jsonl_records(input_path)
    records = build_grpo_records(samples)
    write_jsonl_records(records, output_path)

    print(
        "Built GRPO cold-start dataset from {total} samples.".format(
            total=len(records)
        )
    )
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
