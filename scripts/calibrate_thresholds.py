import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_engine.schema import CANONICAL_POINT_KEYS, FINAL_RESULT_PASS, POINT_STATUS_PASS
from src.evaluation.thresholds import calibrate_threshold, calibrate_thresholds


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"ERROR: invalid JSON on line {line_number} in {path}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise SystemExit(
                    f"ERROR: line {line_number} in {path} must decode to an object"
                )
            records.append(record)
    return records


def _build_details(
    records: list[dict[str, object]],
    *,
    min_precision: float,
) -> dict[str, dict[str, float | int]]:
    details: dict[str, dict[str, float | int]] = {}

    for point_key in CANONICAL_POINT_KEYS:
        scored_examples: list[tuple[float, bool]] = []
        for record in records:
            reference = record.get("reference")
            if not isinstance(reference, dict):
                raise SystemExit("ERROR: each record must contain a reference object")
            point_results = reference.get("point_results")
            if not isinstance(point_results, dict):
                raise SystemExit(
                    "ERROR: each reference object must contain point_results"
                )
            point_scores = record.get("point_scores")
            if not isinstance(point_scores, dict):
                raise SystemExit("ERROR: each record must contain point_scores")
            scored_examples.append(
                (
                    float(point_scores[point_key]),
                    point_results[point_key] == POINT_STATUS_PASS,
                )
            )

        details[point_key] = calibrate_threshold(
            scored_examples,
            min_precision=min_precision,
        )

    final_examples: list[tuple[float, bool]] = []
    for record in records:
        reference = record.get("reference")
        if not isinstance(reference, dict):
            raise SystemExit("ERROR: each record must contain a reference object")
        final_examples.append(
            (
                float(record["final_score"]),
                reference["final_result"] == FINAL_RESULT_PASS,
            )
        )

    details["final_result"] = calibrate_threshold(
        final_examples,
        min_precision=min_precision,
    )
    return details


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate auto-pass score thresholds from scored reference JSONL."
    )
    parser.add_argument(
        "--scores",
        required=True,
        help=(
            "JSONL file with reference.point_results, reference.final_result, "
            "point_scores, and final_score."
        ),
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=1.0,
        help="Minimum precision required when selecting each threshold.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the calibration JSON report.",
    )
    args = parser.parse_args()

    scores_path = _resolve_path(args.scores)
    if not scores_path.is_file():
        raise SystemExit(f"ERROR: scores file not found: {scores_path}")

    records = _load_jsonl_records(scores_path)
    threshold_values = calibrate_thresholds(
        records,
        min_precision=args.min_precision,
    )
    details = _build_details(records, min_precision=args.min_precision)
    payload = {
        "thresholds": threshold_values,
        "details": details,
    }

    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        output_path = _resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
