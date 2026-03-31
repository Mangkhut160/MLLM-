import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from src.data_engine.schema import (
    FINAL_RESULT_REVIEW,
    POINT_STATUS_REJECT,
    default_point_results,
)
from src.data_engine.sft_builder import build_sft_record
from src.prompt_baseline.prompt_templates import build_audit_prompt


def _load_formatter_module():
    try:
        spec = importlib.util.find_spec("src.modeling.sft_formatter")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Expected src.modeling.sft_formatter to exist: {exc}")
    if spec is None:
        pytest.fail("Expected src.modeling.sft_formatter to exist.")
    return importlib.import_module("src.modeling.sft_formatter")


def _load_train_sft_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "train_sft.py"
    if not module_path.is_file():
        pytest.fail(f"Expected training CLI to exist: {module_path}")

    spec = importlib.util.spec_from_file_location("train_sft", module_path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Unable to load training CLI module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(sample_id: str = "460790679") -> dict[str, object]:
    point_results = default_point_results()
    point_results["voltage_capacity_check"] = POINT_STATUS_REJECT
    return {
        "sample_id": sample_id,
        "input": {
            "form": {
                "production_date": "2024-12-20 15:51:43",
                "voltage": "45V",
                "brand": "天能",
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
            "reject_tags": ["规格不符", "需要复核"],
        },
    }


def test_format_training_example_matches_task_four_sft_contract():
    module = _load_formatter_module()
    sample = _sample()

    record = module.format_training_example(sample)

    assert record == build_sft_record(sample)
    assert record["messages"][0]["content"][0] == {
        "type": "input_image",
        "image_path": "brand_new/brand_new/460790679.png",
    }
    assert record["messages"][0]["content"][1] == {
        "type": "input_image",
        "image_path": "charge_new/charge_new/460790679.png",
    }
    assert record["messages"][0]["content"][2] == {
        "type": "input_text",
        "text": build_audit_prompt(sample["input"]["form"]),
    }


def test_format_training_example_serializes_structured_target_with_utf8():
    module = _load_formatter_module()
    sample = _sample()

    record = module.format_training_example(sample)

    assert record["target"] == json.dumps(sample["output"], ensure_ascii=False)
    assert "\\u" not in record["target"]


def test_train_sft_parser_defaults_to_autodl_tmp_paths():
    module = _load_train_sft_module()

    args = module.build_arg_parser().parse_args([])

    assert args.model_path.startswith("/root/autodl-tmp/")
    assert args.train_data.startswith("/root/autodl-tmp/")
    assert args.val_data.startswith("/root/autodl-tmp/")
    assert args.output_dir.startswith("/root/autodl-tmp/")
