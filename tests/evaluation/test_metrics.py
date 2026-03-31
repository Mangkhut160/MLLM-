import pytest

from src.data_engine.schema import (
    FINAL_RESULT_PASS,
    FINAL_RESULT_REVIEW,
    POINT_STATUS_REJECT,
    default_point_results,
)
from src.evaluation.metrics import (
    compute_final_result_metrics,
    compute_point_metrics,
    compute_reject_tag_metrics,
)


def _output(
    *,
    point_updates: dict[str, str] | None = None,
    final_result: str = FINAL_RESULT_PASS,
    reject_tags: list[str] | None = None,
) -> dict[str, object]:
    point_results = default_point_results()
    if point_updates:
        point_results.update(point_updates)
    return {
        "point_results": point_results,
        "final_result": final_result,
        "reject_tags": reject_tags or [],
    }


def test_compute_point_metrics_tracks_overall_and_per_point_results():
    references = [
        _output(),
        _output(
            point_updates={
                "date_check": POINT_STATUS_REJECT,
                "image_check": POINT_STATUS_REJECT,
            },
            final_result=FINAL_RESULT_REVIEW,
            reject_tags=["date_mismatch", "image_mismatch"],
        ),
    ]
    predictions = [
        _output(point_updates={"brand_check": POINT_STATUS_REJECT}),
        _output(
            point_updates={"date_check": POINT_STATUS_REJECT},
            final_result=FINAL_RESULT_REVIEW,
            reject_tags=["image_mismatch", "date_mismatch"],
        ),
    ]

    metrics = compute_point_metrics(predictions, references)

    assert metrics["overall"]["total"] == 10
    assert metrics["overall"]["correct"] == 8
    assert metrics["overall"]["accuracy"] == pytest.approx(0.8)
    assert metrics["overall"]["true_positive"] == 1
    assert metrics["overall"]["true_negative"] == 7
    assert metrics["overall"]["false_positive"] == 1
    assert metrics["overall"]["false_negative"] == 1
    assert metrics["overall"]["precision"] == pytest.approx(0.5)
    assert metrics["overall"]["recall"] == pytest.approx(0.5)
    assert metrics["overall"]["f1"] == pytest.approx(0.5)

    assert metrics["per_point"]["brand_check"]["false_positive"] == 1
    assert metrics["per_point"]["date_check"]["true_positive"] == 1
    assert metrics["per_point"]["image_check"]["false_negative"] == 1
    assert metrics["per_point"]["spec_check"]["accuracy"] == pytest.approx(1.0)


def test_compute_final_result_metrics_uses_review_as_positive_class():
    references = [
        _output(final_result=FINAL_RESULT_PASS),
        _output(final_result=FINAL_RESULT_REVIEW),
        _output(final_result=FINAL_RESULT_REVIEW),
    ]
    predictions = [
        _output(final_result=FINAL_RESULT_PASS),
        _output(final_result=FINAL_RESULT_PASS),
        _output(final_result=FINAL_RESULT_REVIEW),
    ]

    metrics = compute_final_result_metrics(predictions, references)

    assert metrics["total"] == 3
    assert metrics["correct"] == 2
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_compute_reject_tag_metrics_is_order_insensitive_per_sample():
    references = [
        _output(reject_tags=["spec_mismatch", "date_mismatch"]),
        _output(reject_tags=[]),
    ]
    predictions = [
        _output(reject_tags=["date_mismatch", "spec_mismatch"]),
        _output(reject_tags=["extra_tag"]),
    ]

    metrics = compute_reject_tag_metrics(predictions, references)

    assert metrics["total_samples"] == 2
    assert metrics["exact_match"] == 1
    assert metrics["exact_match_accuracy"] == pytest.approx(0.5)
    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)
