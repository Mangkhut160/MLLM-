from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_engine.grpo_builder import MAX_THINK_CHARS
from src.data_engine.sft_builder import load_jsonl_records
from src.modeling.grpo_rewards import reward_format, reward_length

AUTODL_TMP_ROOT = PurePosixPath("/root/autodl-tmp")
DEFAULT_MODEL_PATH = AUTODL_TMP_ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_TRAIN_DATA = AUTODL_TMP_ROOT / "battery-audit" / "data" / "grpo" / "cold_start.jsonl"
DEFAULT_OUTPUT_DIR = AUTODL_TMP_ROOT / "battery-audit" / "output" / "grpo"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GRPO training preflight for structured battery-audit data."
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Base multimodal model path for GRPO fine-tuning.",
    )
    parser.add_argument(
        "--train-data",
        default=str(DEFAULT_TRAIN_DATA),
        help="JSONL GRPO dataset containing sample_id, think, answer, and messages.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for GRPO run metadata and future checkpoints.",
    )
    parser.add_argument(
        "--format-reward-weight",
        type=float,
        default=1.0,
        help="Weight applied to the answer-format reward.",
    )
    parser.add_argument(
        "--result-reward-weight",
        type=float,
        default=1.0,
        help="Weight applied to the final-result reward.",
    )
    parser.add_argument(
        "--tag-reward-weight",
        type=float,
        default=1.0,
        help="Weight applied to the reject-tag reward.",
    )
    parser.add_argument(
        "--length-reward-weight",
        type=float,
        default=1.0,
        help="Weight applied to the think-length reward.",
    )
    parser.add_argument(
        "--max-think-chars",
        type=int,
        default=MAX_THINK_CHARS,
        help="Maximum allowed think length before length reward becomes negative.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=3072,
        help="Maximum prompt length for future GRPO tokenization.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=512,
        help="Maximum completion length for future GRPO rollouts.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of sampled completions per prompt.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device batch size reserved for the future GRPO trainer.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps reserved for the future GRPO trainer.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Optimizer learning rate reserved for the future GRPO trainer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic preflight metadata.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the resolved GRPO configuration without writing files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_training(args)


def run_training(args: argparse.Namespace) -> int:
    train_path = Path(args.train_data).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not train_path.is_file():
        raise SystemExit(f"ERROR: train data file not found: {train_path}")

    records = load_jsonl_records(train_path)
    if not records:
        raise SystemExit(f"ERROR: train data file is empty: {train_path}")

    validated_records = [_validate_grpo_record(record, index, args.max_think_chars) for index, record in enumerate(records)]
    resolved_config = _build_run_config(args, record_count=len(validated_records))

    if args.dry_run:
        print(json.dumps(resolved_config, ensure_ascii=False, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "grpo_run_config.json"
    config_path.write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Validated {len(validated_records)} GRPO records from {train_path} and wrote {config_path}."
    )
    print("GRPO trainer integration is pending; preflight completed successfully.")
    return 0


def _validate_grpo_record(
    record: dict[str, Any],
    index: int,
    max_think_chars: int,
) -> dict[str, Any]:
    if not isinstance(record.get("sample_id"), str) or not record["sample_id"].strip():
        raise SystemExit(f"ERROR: record {index} is missing a non-empty sample_id")
    if not isinstance(record.get("think"), str) or not record["think"].strip():
        raise SystemExit(f"ERROR: record {index} is missing a non-empty think string")
    if reward_length(record["think"], max_chars=max_think_chars) < 0:
        raise SystemExit(
            f"ERROR: record {index} think exceeds max_think_chars={max_think_chars}"
        )
    if reward_format(record.get("answer")) == 0.0:
        raise SystemExit(f"ERROR: record {index} answer does not match the GRPO schema")

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise SystemExit(f"ERROR: record {index} is missing non-empty messages")

    return record


def _build_run_config(args: argparse.Namespace, *, record_count: int) -> dict[str, Any]:
    return {
        "model_path": args.model_path,
        "train_data": args.train_data,
        "output_dir": args.output_dir,
        "record_count": record_count,
        "reward_weights": {
            "format": args.format_reward_weight,
            "result": args.result_reward_weight,
            "tag": args.tag_reward_weight,
            "length": args.length_reward_weight,
        },
        "max_think_chars": args.max_think_chars,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }


if __name__ == "__main__":
    raise SystemExit(main())
