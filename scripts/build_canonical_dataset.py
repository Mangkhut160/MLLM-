import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_engine.canonical_builder import build_canonical_records


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _resolve_image_dir(repo_root: Path, preferred: str, fallback: str) -> Path:
    preferred_path = _resolve_path(repo_root, preferred)
    if preferred_path.is_dir():
        return preferred_path

    fallback_path = _resolve_path(repo_root, fallback)
    if fallback_path.is_dir():
        return fallback_path

    return preferred_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical dataset JSONL.")
    parser.add_argument("--csv", default="data.csv", help="Path to the source CSV.")
    parser.add_argument(
        "--brand-dir",
        default="brand_new/brand_new",
        help="Directory containing brand images.",
    )
    parser.add_argument(
        "--spec-dir",
        default="charge_new/charge_new",
        help="Directory containing spec images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for canonical samples.",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    csv_path = _resolve_path(repo_root, args.csv)
    brand_dir = _resolve_image_dir(repo_root, args.brand_dir, "brand_new")
    spec_dir = _resolve_image_dir(repo_root, args.spec_dir, "charge_new")
    output_path = _resolve_path(repo_root, args.output)

    records, stats = build_canonical_records(
        csv_path=csv_path,
        brand_dir=brand_dir,
        spec_dir=spec_dir,
        repo_root=repo_root,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Kept {stats['kept']} rows, dropped {stats['dropped']} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
