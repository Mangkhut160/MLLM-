import pytest

from src.data_engine.schema import (
    CANONICAL_POINT_KEYS,
    FINAL_RESULT_PASS,
    FINAL_RESULT_REVIEW,
    POINT_STATUS_REJECT,
    default_point_results,
)
from src.evaluation.thresholds import (
    calibrate_threshold,
    calibrate_thresholds,
    should_auto_pass,
)


def _thresholds(value: float = 0.98) -> dict[str, float]:
    thresholds = {key: value for key in CANONICAL_POINT_KEYS}
    thresholds["final_result"] = value
    return thresholds


def _scored_record(
    *,
    point_updates: dict[str, str] | None = None,
    point_score_updates: dict[str, float] | None = None,
    final_result: str = FINAL_RESULT_PASS,
    final_score: float = 0.99,
) -> dict[str, object]:
    point_results = default_point_results()
    if point_updates:
        point_results.update(point_updates)

    point_scores = {key: 0.99 for key in CANONICAL_POINT_KEYS}
    if point_score_updates:
        point_scores.update(point_score_updates)

    return {
        "reference": {
            "point_results": point_results,
            "final_result": final_result,
        },
        "point_scores": point_scores,
        "final_score": final_score,
    }


def test_should_auto_pass_requires_all_point_predictions_to_pass():
    point_predictions = default_point_results()
    point_predictions["date_check"] = POINT_STATUS_REJECT

    assert (
        should_auto_pass(
            point_predictions=point_predictions,
            point_scores={key: 0.99 for key in CANONICAL_POINT_KEYS},
            final_score=0.99,
            thresholds=_thresholds(),
        )
        is False
    )


def test_should_auto_pass_requires_each_score_to_clear_its_threshold():
    point_predictions = default_point_results()
    point_scores = {key: 0.99 for key in CANONICAL_POINT_KEYS}
    point_scores["spec_check"] = 0.97

    assert (
        should_auto_pass(
            point_predictions=point_predictions,
            point_scores=point_scores,
            final_score=0.99,
            thresholds=_thresholds(),
        )
        is False
    )

    assert (
        should_auto_pass(
            point_predictions=point_predictions,
            point_scores={key: 0.99 for key in CANONICAL_POINT_KEYS},
            final_score=0.97,
            thresholds=_thresholds(),
        )
        is False
    )


def test_calibrate_threshold_maximizes_recall_under_precision_floor():
    result = calibrate_threshold(
        [
            (0.98, True),
            (0.90, True),
            (0.88, True),
            (0.89, False),
            (0.40, False),
        ],
        min_precision=1.0,
    )

    assert result["threshold"] == pytest.approx(0.90)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["selected"] == 2


def test_calibrate_thresholds_returns_point_and_final_thresholds():
    records = [
        _scored_record(
            point_score_updates={
                "brand_check": 0.98,
                "date_check": 0.96,
                "voltage_capacity_check": 0.97,
                "spec_check": 0.94,
                "image_check": 0.95,
            },
            final_score=0.97,
        ),
        _scored_record(
            point_score_updates={
                "brand_check": 0.90,
                "date_check": 0.95,
                "voltage_capacity_check": 0.96,
                "spec_check": 0.95,
                "image_check": 0.96,
            },
            final_score=0.95,
        ),
        _scored_record(
            point_updates={"brand_check": POINT_STATUS_REJECT},
            point_score_updates={
                "brand_check": 0.89,
                "date_check": 0.94,
                "voltage_capacity_check": 0.95,
                "spec_check": 0.96,
                "image_check": 0.97,
            },
            final_result=FINAL_RESULT_REVIEW,
            final_score=0.94,
        ),
        _scored_record(
            point_score_updates={
                "brand_check": 0.88,
                "date_check": 0.93,
                "voltage_capacity_check": 0.94,
                "spec_check": 0.97,
                "image_check": 0.98,
            },
            final_score=0.93,
        ),
    ]

    thresholds = calibrate_thresholds(records, min_precision=1.0)

    assert set(thresholds) == set(CANONICAL_POINT_KEYS) | {"final_result"}
    assert thresholds["brand_check"] == pytest.approx(0.90)
    assert thresholds["final_result"] == pytest.approx(0.95)
