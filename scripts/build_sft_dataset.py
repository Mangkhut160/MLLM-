import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_engine.sft_builder import (
    build_sft_records,
    load_jsonl_records,
    split_canonical_samples,
    write_jsonl_records,
)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SFT train/val splits.")
    parser.add_argument(
        "--input",
        default="data/canonical/samples.jsonl",
        help="Canonical JSONL input path.",
    )
    parser.add_argument(
        "--train-output",
        default="data/sft/train_sft.jsonl",
        help="Train SFT JSONL output path.",
    )
    parser.add_argument(
        "--val-output",
        default="data/sft/val_sft.jsonl",
        help="Validation SFT JSONL output path.",
    )
    parser.add_argument(
        "--test-output",
        default="data/canonical/test.jsonl",
        help="Canonical JSONL test split output path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic shuffle seed for train/val/test split.",
    )
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    train_output_path = _resolve_path(args.train_output)
    val_output_path = _resolve_path(args.val_output)
    test_output_path = _resolve_path(args.test_output)

    if not input_path.is_file():
        raise SystemExit(f"ERROR: input file not found: {input_path}")

    canonical_samples = load_jsonl_records(input_path)
    split = split_canonical_samples(canonical_samples, seed=args.seed)

    train_records = build_sft_records(split["train"])
    val_records = build_sft_records(split["val"])

    write_jsonl_records(train_records, train_output_path)
    write_jsonl_records(val_records, val_output_path)
    write_jsonl_records(split["test"], test_output_path)

    print(
        "Built SFT dataset from {total} canonical samples with seed={seed}: "
        "train={train_count}, val={val_count}, test={test_count}.".format(
            total=len(canonical_samples),
            seed=args.seed,
            train_count=len(train_records),
            val_count=len(val_records),
            test_count=len(split["test"]),
        )
    )
    print(f"Train output: {train_output_path}")
    print(f"Val output: {val_output_path}")
    print(f"Test output: {test_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
