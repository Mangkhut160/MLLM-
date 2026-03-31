import json

import pytest

from src.data_engine.schema import (
    CANONICAL_POINT_KEYS,
    POINT_STATUS_PASS,
    POINT_STATUS_REJECT,
    default_point_results,
)
from src.inference.decision import build_decision
from src.inference.predict import predict_decision, predict_from_response


def test_build_decision_returns_manual_review_when_trigger_points_exist():
    point_results = default_point_results()
    point_results["date_check"] = POINT_STATUS_REJECT

    decision = build_decision(
        point_results=point_results,
        trigger_points=["date_check"],
    )

    assert decision == {
        "mode": "manual_review",
        "auto_pass_allowed": False,
        "trigger_points": ["date_check"],
    }


def test_build_decision_returns_auto_pass_when_no_trigger_points_exist():
    decision = build_decision(
        point_results=default_point_results(),
        trigger_points=[],
    )

    assert decision == {
        "mode": "auto_pass",
        "auto_pass_allowed": True,
        "trigger_points": [],
    }


def test_build_decision_derives_manual_review_from_point_results():
    point_results = default_point_results()
    point_results["date_check"] = POINT_STATUS_REJECT

    decision = build_decision(
        point_results=point_results,
        trigger_points=[],
    )

    assert decision == {
        "mode": "manual_review",
        "auto_pass_allowed": False,
        "trigger_points": ["date_check"],
    }


def test_predict_decision_rejects_incomplete_point_results():
    with pytest.raises(ValueError, match="point_results missing keys"):
        predict_decision(
            {
                "point_results": {"brand_check": POINT_STATUS_PASS},
                "reject_tags": [],
            }
        )


def test_predict_from_response_returns_manual_review_for_rejected_point():
    point_results = {key: POINT_STATUS_PASS for key in CANONICAL_POINT_KEYS}
    point_results["voltage_capacity_check"] = POINT_STATUS_REJECT
    raw_response = json.dumps(
        {
            "point_results": point_results,
            "final_result": "人工复核",
            "reject_tags": ["电池额定电压/额定容量与填写信息不一致"],
        },
        ensure_ascii=False,
    )

    prediction = predict_from_response(
        raw_response=raw_response,
        fallback_response={
            "point_results": default_point_results(),
            "final_result": "通过",
            "reject_tags": [],
        },
    )

    assert prediction["decision"] == {
        "mode": "manual_review",
        "auto_pass_allowed": False,
        "trigger_points": ["voltage_capacity_check"],
    }
