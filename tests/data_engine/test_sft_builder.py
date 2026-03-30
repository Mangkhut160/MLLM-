import json

from src.data_engine.schema import (
    FINAL_RESULT_PASS,
    FINAL_RESULT_REVIEW,
    POINT_STATUS_PASS,
    POINT_STATUS_REJECT,
    default_point_results,
)
from src.data_engine.sft_builder import build_sft_record, split_canonical_samples
from src.prompt_baseline.prompt_templates import build_audit_prompt


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
            "reject_tags": ["tag-a"],
        },
    }


def test_build_sft_record_formats_images_then_prompt_and_target():
    sample = _sample("460790679")

    record = build_sft_record(sample)

    assert record["sample_id"] == "460790679"
    assert len(record["messages"]) == 1

    message = record["messages"][0]
    assert message["role"] == "user"
    assert message["content"] == [
        {
            "type": "input_image",
            "image_path": "brand_new/brand_new/460790679.png",
        },
        {
            "type": "input_image",
            "image_path": "charge_new/charge_new/460790679.png",
        },
        {
            "type": "input_text",
            "text": build_audit_prompt(sample["input"]["form"]),
        },
    ]
    assert record["target"] == json.dumps(sample["output"], ensure_ascii=False)


def test_build_audit_prompt_includes_explicit_response_contract():
    prompt = build_audit_prompt(_sample("460790679")["input"]["form"])

    assert "Image 1" in prompt
    assert "brand image" in prompt
    assert "Image 2" in prompt
    assert "spec image" in prompt
    assert "JSON object" in prompt
    assert "point_results" in prompt
    assert "final_result" in prompt
    assert "reject_tags" in prompt
    assert "brand_check" in prompt
    assert "date_check" in prompt
    assert "voltage_capacity_check" in prompt
    assert "spec_check" in prompt
    assert "image_check" in prompt
    assert POINT_STATUS_PASS in prompt
    assert POINT_STATUS_REJECT in prompt
    assert FINAL_RESULT_PASS in prompt
    assert FINAL_RESULT_REVIEW in prompt


def test_split_canonical_samples_is_deterministic_and_exhaustive():
    samples = [_sample(str(index)) for index in range(10)]

    first = split_canonical_samples(samples, seed=7)
    second = split_canonical_samples(samples, seed=7)

    assert first == second
    assert [sample["sample_id"] for sample in first["train"]] == [
        "8",
        "3",
        "1",
        "4",
        "7",
        "0",
        "9",
        "6",
    ]
    assert [sample["sample_id"] for sample in first["val"]] == ["2"]
    assert [sample["sample_id"] for sample in first["test"]] == ["5"]
