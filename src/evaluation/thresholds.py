from collections.abc import Mapping, Sequence

from src.data_engine.schema import (
    CANONICAL_POINT_KEYS,
    FINAL_RESULT_PASS,
    POINT_STATUS_PASS,
)

DEFAULT_POINT_THRESHOLD = 0.5
DEFAULT_FINAL_RESULT_THRESHOLD = 0.5


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _resolve_thresholds(thresholds: Mapping[str, float] | None) -> dict[str, float]:
    resolved = {key: DEFAULT_POINT_THRESHOLD for key in CANONICAL_POINT_KEYS}
    resolved["final_result"] = DEFAULT_FINAL_RESULT_THRESHOLD

    if thresholds:
        for key, value in thresholds.items():
            resolved[key] = float(value)

    return resolved


def should_auto_pass(
    point_predictions: Mapping[str, str],
    point_scores: Mapping[str, float],
    final_score: float,
    thresholds: Mapping[str, float] | None = None,
) -> bool:
    resolved_thresholds = _resolve_thresholds(thresholds)

    for point_key in CANONICAL_POINT_KEYS:
        if point_predictions[point_key] != POINT_STATUS_PASS:
            return False
        if float(point_scores[point_key]) < resolved_thresholds[point_key]:
            return False

    return float(final_score) >= resolved_thresholds["final_result"]


def calibrate_threshold(
    scored_examples: Sequence[tuple[float, bool]],
    *,
    min_precision: float = 1.0,
) -> dict[str, float | int]:
    if not scored_examples:
        raise ValueError("scored_examples must not be empty")
    if min_precision <= 0 or min_precision > 1:
        raise ValueError("min_precision must be in the range (0, 1]")

    max_score = max(score for score, _ in scored_examples)
    candidates = sorted(
        {float(score) for score, _ in scored_examples} | {float(max_score) + 1e-12},
        reverse=True,
    )
    positive_count = sum(1 for _, label in scored_examples if label)
    best_result: dict[str, float | int] | None = None

    for threshold in candidates:
        selected = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for score, label in scored_examples:
            predicted_positive = float(score) >= threshold
            if predicted_positive:
                selected += 1
                if label:
                    true_positive += 1
                else:
                    false_positive += 1
            elif label:
                false_negative += 1

        precision = 1.0 if selected == 0 else _safe_divide(true_positive, selected)
        recall = _safe_divide(true_positive, positive_count)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        result = {
            "threshold": threshold,
            "selected": selected,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        if precision < min_precision:
            continue

        if best_result is None:
            best_result = result
            continue

        if result["recall"] > best_result["recall"]:
            best_result = result
            continue

        if (
            result["recall"] == best_result["recall"]
            and result["threshold"] < best_result["threshold"]
        ):
            best_result = result

    if best_result is None:
        raise RuntimeError("Unable to calibrate a threshold for the provided examples")

    return best_result


def calibrate_thresholds(
    scored_records: Sequence[Mapping[str, object]],
    *,
    min_precision: float = 1.0,
) -> dict[str, float]:
    if not scored_records:
        raise ValueError("scored_records must not be empty")

    thresholds: dict[str, float] = {}

    for point_key in CANONICAL_POINT_KEYS:
        point_examples: list[tuple[float, bool]] = []
        for record in scored_records:
            reference = record["reference"]
            if not isinstance(reference, Mapping):
                raise ValueError("Each scored record must contain a reference mapping")
            point_results = reference["point_results"]
            if not isinstance(point_results, Mapping):
                raise ValueError(
                    "Each scored record reference must contain point_results"
                )
            point_scores = record["point_scores"]
            if not isinstance(point_scores, Mapping):
                raise ValueError("Each scored record must contain point_scores")
            point_examples.append(
                (
                    float(point_scores[point_key]),
                    point_results[point_key] == POINT_STATUS_PASS,
                )
            )

        thresholds[point_key] = float(
            calibrate_threshold(point_examples, min_precision=min_precision)["threshold"]
        )

    final_examples: list[tuple[float, bool]] = []
    for record in scored_records:
        reference = record["reference"]
        if not isinstance(reference, Mapping):
            raise ValueError("Each scored record must contain a reference mapping")
        final_examples.append(
            (
                float(record["final_score"]),
                reference["final_result"] == FINAL_RESULT_PASS,
            )
        )

    thresholds["final_result"] = float(
        calibrate_threshold(final_examples, min_precision=min_precision)["threshold"]
    )
    return thresholds
