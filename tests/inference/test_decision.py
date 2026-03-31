from src.data_engine.schema import POINT_STATUS_REJECT, default_point_results
from src.inference.decision import build_decision


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
