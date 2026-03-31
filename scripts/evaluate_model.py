import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import (
    compute_final_result_metrics,
    compute_point_metrics,
    compute_reject_tag_metrics,
)


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


def _extract_output(record: dict[str, object]) -> dict[str, object]:
    output = record.get("output")
    if isinstance(output, dict):
        return output
    return record


def _align_records(
    predictions: list[dict[str, object]],
    references: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prediction_ids = [record.get("sample_id") for record in predictions]
    reference_ids = [record.get("sample_id") for record in references]

    if all(isinstance(sample_id, str) for sample_id in prediction_ids) and all(
        isinstance(sample_id, str) for sample_id in reference_ids
    ):
        references_by_id = {
            str(record["sample_id"]): _extract_output(record) for record in references
        }
        aligned_predictions: list[dict[str, object]] = []
        aligned_references: list[dict[str, object]] = []

        for record in predictions:
            sample_id = str(record["sample_id"])
            if sample_id not in references_by_id:
                raise SystemExit(
                    f"ERROR: reference sample_id not found for prediction: {sample_id}"
                )
            aligned_predictions.append(_extract_output(record))
            aligned_references.append(references_by_id[sample_id])
        return aligned_predictions, aligned_references

    if len(predictions) != len(references):
        raise SystemExit(
            "ERROR: predictions and references must have the same number of records"
        )

    return (
        [_extract_output(record) for record in predictions],
        [_extract_output(record) for record in references],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate structured battery-audit predictions against references."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="JSONL file of model outputs or records containing an output field.",
    )
    parser.add_argument(
        "--references",
        required=True,
        help="JSONL file of reference outputs or records containing an output field.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the metrics JSON report.",
    )
    args = parser.parse_args()

    predictions_path = _resolve_path(args.predictions)
    references_path = _resolve_path(args.references)
    if not predictions_path.is_file():
        raise SystemExit(f"ERROR: predictions file not found: {predictions_path}")
    if not references_path.is_file():
        raise SystemExit(f"ERROR: references file not found: {references_path}")

    prediction_records = _load_jsonl_records(predictions_path)
    reference_records = _load_jsonl_records(references_path)
    aligned_predictions, aligned_references = _align_records(
        prediction_records,
        reference_records,
    )

    metrics = {
        "point_metrics": compute_point_metrics(aligned_predictions, aligned_references),
        "final_result_metrics": compute_final_result_metrics(
            aligned_predictions,
            aligned_references,
        ),
        "reject_tag_metrics": compute_reject_tag_metrics(
            aligned_predictions,
            aligned_references,
        ),
    }

    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        output_path = _resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
