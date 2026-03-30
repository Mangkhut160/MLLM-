import json

from src.data_engine.sft_builder import build_sft_record, split_canonical_samples


def _sample(sample_id: str) -> dict[str, object]:
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
            "point_results": {
                "brand_check": "通过",
                "date_check": "通过",
                "voltage_capacity_check": "驳回",
                "spec_check": "通过",
                "image_check": "通过",
            },
            "final_result": "人工复核",
            "reject_tags": ["电池额定电压/额定容量与填写信息不一致"],
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
            "text": (
                "You are auditing a battery listing. Check brand/date/voltage-capacity/"
                "spec/image compliance.\n"
                "Form fields:\n"
                "- brand: Tianneng\n"
                "- production_date: 2024-12-20 15:51:43\n"
                "- voltage: 45V\n"
                "- capacity: 72Ah\n"
                "Return JSON with keys point_results, final_result, reject_tags."
            ),
        },
    ]
    assert record["target"] == json.dumps(sample["output"], ensure_ascii=False)


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
