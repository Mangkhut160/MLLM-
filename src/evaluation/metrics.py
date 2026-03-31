from collections.abc import Mapping, Sequence

from src.data_engine.schema import CANONICAL_POINT_KEYS, FINAL_RESULT_REVIEW, POINT_STATUS_REJECT


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _classification_metrics(
    pairs: Sequence[tuple[str, str]],
    *,
    positive_label: str,
) -> dict[str, float | int]:
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for predicted, reference in pairs:
        predicted_positive = predicted == positive_label
        reference_positive = reference == positive_label

        if predicted_positive and reference_positive:
            true_positive += 1
        elif predicted_positive:
            false_positive += 1
        elif reference_positive:
            false_negative += 1
        else:
            true_negative += 1

    total = len(pairs)
    correct = true_positive + true_negative
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "total": total,
        "correct": correct,
        "accuracy": _safe_divide(correct, total),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _ensure_same_length(
    predictions: Sequence[Mapping[str, object]],
    references: Sequence[Mapping[str, object]],
) -> None:
    if len(predictions) != len(references):
        raise ValueError(
            "Predictions and references must contain the same number of records"
        )


def _point_results(record: Mapping[str, object]) -> Mapping[str, str]:
    point_results = record.get("point_results")
    if not isinstance(point_results, Mapping):
        raise ValueError("Each record must contain a point_results mapping")
    return point_results  # type: ignore[return-value]


def _final_result(record: Mapping[str, object]) -> str:
    final_result = record.get("final_result")
    if not isinstance(final_result, str):
        raise ValueError("Each record must contain a string final_result")
    return final_result


def _reject_tags(record: Mapping[str, object]) -> set[str]:
    reject_tags = record.get("reject_tags")
    if not isinstance(reject_tags, Sequence) or isinstance(reject_tags, (str, bytes)):
        raise ValueError("Each record must contain a reject_tags list")
    if not all(isinstance(tag, str) for tag in reject_tags):
        raise ValueError("reject_tags entries must be strings")
    return set(reject_tags)


def compute_point_metrics(
    predictions: Sequence[Mapping[str, object]],
    references: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    _ensure_same_length(predictions, references)

    overall_pairs: list[tuple[str, str]] = []
    per_point: dict[str, dict[str, float | int]] = {}

    for point_key in CANONICAL_POINT_KEYS:
        point_pairs: list[tuple[str, str]] = []
        for predicted_record, reference_record in zip(predictions, references):
            predicted_point_results = _point_results(predicted_record)
            reference_point_results = _point_results(reference_record)
            point_pairs.append(
                (
                    predicted_point_results[point_key],
                    reference_point_results[point_key],
                )
            )

        per_point[point_key] = _classification_metrics(
            point_pairs,
            positive_label=POINT_STATUS_REJECT,
        )
        overall_pairs.extend(point_pairs)

    overall = _classification_metrics(
        overall_pairs,
        positive_label=POINT_STATUS_REJECT,
    )
    return {
        "overall": overall,
        "per_point": per_point,
    }


def compute_final_result_metrics(
    predictions: Sequence[Mapping[str, object]],
    references: Sequence[Mapping[str, object]],
) -> dict[str, float | int]:
    _ensure_same_length(predictions, references)

    result_pairs = [
        (_final_result(predicted_record), _final_result(reference_record))
        for predicted_record, reference_record in zip(predictions, references)
    ]
    return _classification_metrics(
        result_pairs,
        positive_label=FINAL_RESULT_REVIEW,
    )


def compute_reject_tag_metrics(
    predictions: Sequence[Mapping[str, object]],
    references: Sequence[Mapping[str, object]],
) -> dict[str, float | int]:
    _ensure_same_length(predictions, references)

    exact_match = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for predicted_record, reference_record in zip(predictions, references):
        predicted_tags = _reject_tags(predicted_record)
        reference_tags = _reject_tags(reference_record)

        if predicted_tags == reference_tags:
            exact_match += 1

        true_positive += len(predicted_tags & reference_tags)
        false_positive += len(predicted_tags - reference_tags)
        false_negative += len(reference_tags - predicted_tags)

    total_samples = len(predictions)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "total_samples": total_samples,
        "exact_match": exact_match,
        "exact_match_accuracy": _safe_divide(exact_match, total_samples),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
