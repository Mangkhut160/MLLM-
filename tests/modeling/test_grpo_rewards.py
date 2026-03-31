import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from src.data_engine.schema import FINAL_RESULT_PASS, FINAL_RESULT_REVIEW, default_point_results


def _load_rewards_module():
    try:
        spec = importlib.util.find_spec("src.modeling.grpo_rewards")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Expected src.modeling.grpo_rewards to exist: {exc}")
    if spec is None:
        pytest.fail("Expected src.modeling.grpo_rewards to exist.")
    return importlib.import_module("src.modeling.grpo_rewards")


def _load_train_grpo_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "train_grpo.py"
    if not module_path.is_file():
        pytest.fail(f"Expected GRPO training CLI to exist: {module_path}")

    spec = importlib.util.spec_from_file_location("train_grpo", module_path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Unable to load GRPO training CLI module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _answer(
    *,
    final_result: str = FINAL_RESULT_REVIEW,
    reject_tags: list[str] | None = None,
) -> str:
    point_results = default_point_results()
    payload = {
        "point_results": point_results,
        "final_result": final_result,
        "reject_tags": reject_tags or [],
    }
    return json.dumps(payload, ensure_ascii=False)


def test_reward_format_returns_one_for_canonical_answer_shape():
    module = _load_rewards_module()

    assert module.reward_format(_answer()) == pytest.approx(1.0)


def test_reward_result_matches_expected_final_result():
    module = _load_rewards_module()
    answer = _answer(final_result=FINAL_RESULT_PASS)
    reference = _answer(final_result=FINAL_RESULT_PASS)

    assert module.reward_result(answer, reference) == pytest.approx(1.0)


def test_reward_tag_uses_partial_credit_for_overlap():
    module = _load_rewards_module()
    answer = _answer(reject_tags=["spec mismatch"])
    reference = _answer(reject_tags=["spec mismatch", "image mismatch"])

    assert module.reward_tag(answer, reference) == pytest.approx(0.5)


def test_reward_length_penalizes_overlong_think_strings():
    module = _load_rewards_module()

    assert module.reward_length("short rationale") == pytest.approx(1.0)
    assert module.reward_length("x" * 75) == pytest.approx(-0.5)


def test_train_grpo_parser_defaults_to_autodl_tmp_paths_and_reward_weights():
    module = _load_train_grpo_module()

    args = module.build_arg_parser().parse_args([])

    assert args.model_path == "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"
    assert args.train_data == "/root/autodl-tmp/battery-audit/data/grpo/cold_start.jsonl"
    assert args.output_dir == "/root/autodl-tmp/battery-audit/output/grpo"
    assert args.format_reward_weight == pytest.approx(1.0)
    assert args.result_reward_weight == pytest.approx(1.0)
    assert args.tag_reward_weight == pytest.approx(1.0)
    assert args.length_reward_weight == pytest.approx(1.0)
