import json

from src.data_engine.schema import (
    FINAL_RESULT_PASS,
    FINAL_RESULT_REVIEW,
    POINT_STATUS_REJECT,
    default_point_results,
)
from src.data_engine.sft_builder import build_sft_record
from src.data_engine.grpo_builder import build_grpo_record, derive_short_think


def _sample(sample_id: str) -> dict[str, object]:
    point_results = default_point_results()
    point_results["voltage_capacity_check"] = POINT_STATUS_REJECT
    return {
        "sample_id": sample_id,
        "input": {
            "form": {
                "production_date": "2024-12-20 15:51:43",
                "voltage": "45V",
                "brand": "Tianneng",
                "capacity": "72Ah",
            },
            "images": {
                "brand_image": f"brand_new/brand_new/{sample_id}.png",
                "spec_image": f"charge_new/charge_new/{sample_id}.png",
            },
        },
        "output": {
            "point_results": point_results,
            "final_result": FINAL_RESULT_REVIEW,
            "reject_tags": [
                "voltage label mismatches the submitted spec sheet details"
            ],
        },
    }


def test_derive_short_think_truncates_reject_reason_to_50_chars():
    think = derive_short_think(_sample("460790679"))

    assert think.startswith("Review:")
    assert think.endswith("...")
    assert len(think) == 50


def test_derive_short_think_returns_short_pass_rationale():
    sample = _sample("460790679")
    sample["output"]["point_results"] = default_point_results()
    sample["output"]["final_result"] = FINAL_RESULT_PASS
    sample["output"]["reject_tags"] = []

    assert derive_short_think(sample) == "Pass: all checks consistent"


def test_build_grpo_record_preserves_input_and_messages():
    sample = _sample("460790679")
    think = "Review: voltage mismatch"

    record = build_grpo_record(sample, think)

    assert record == {
        "sample_id": "460790679",
        "input": sample["input"],
        "messages": build_sft_record(sample)["messages"],
        "think": think,
        "answer": json.dumps(sample["output"], ensure_ascii=False),
    }
