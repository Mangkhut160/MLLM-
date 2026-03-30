from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_engine.sft_builder import load_jsonl_records
from src.modeling.sft_formatter import ensure_training_record, to_qwen_messages

AUTODL_TMP_ROOT = PurePosixPath("/root/autodl-tmp")
DEFAULT_MODEL_PATH = AUTODL_TMP_ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_TRAIN_DATA = AUTODL_TMP_ROOT / "battery-audit" / "data" / "sft" / "train_sft.jsonl"
DEFAULT_VAL_DATA = AUTODL_TMP_ROOT / "battery-audit" / "data" / "sft" / "val_sft.jsonl"
DEFAULT_OUTPUT_DIR = AUTODL_TMP_ROOT / "battery-audit" / "output" / "stage_one_sft"
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the stage-one SFT model for structured battery audits."
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Base multimodal model path.",
    )
    parser.add_argument(
        "--train-data",
        default=str(DEFAULT_TRAIN_DATA),
        help="JSONL train dataset containing canonical or prebuilt SFT records.",
    )
    parser.add_argument(
        "--val-data",
        default=str(DEFAULT_VAL_DATA),
        help="JSONL validation dataset containing canonical or prebuilt SFT records.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for checkpoints and exported adapters.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum token sequence length after prompt-plus-target concatenation.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=1,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Per-device evaluation batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for the learning-rate scheduler.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Step interval for training logs.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="Step interval for validation runs.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Step interval for checkpoint saves.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability.",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=list(DEFAULT_LORA_TARGET_MODULES),
        help="Module names that receive LoRA adapters.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention implementation passed to the base model loader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training.",
    )
    parser.add_argument(
        "--report-to",
        default="none",
        help="Comma-separated reporting integrations, or 'none'.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    parser.add_argument(
        "--bf16",
        dest="bf16",
        action="store_true",
        help="Enable bfloat16 training.",
    )
    parser.add_argument(
        "--no-bf16",
        dest="bf16",
        action="store_false",
        help="Disable bfloat16 training.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing.",
    )
    parser.set_defaults(bf16=True, gradient_checkpointing=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_training(args)


def run_training(args: argparse.Namespace) -> int:
    train_path = Path(args.train_data).expanduser()
    val_path = Path(args.val_data).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not train_path.is_file():
        raise SystemExit(f"ERROR: train data file not found: {train_path}")
    if not val_path.is_file():
        raise SystemExit(f"ERROR: val data file not found: {val_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    deps = _import_training_dependencies()
    train_records = _load_training_records(train_path)
    val_records = _load_training_records(val_path)

    if not train_records:
        raise SystemExit(f"ERROR: train data file is empty: {train_path}")
    if not val_records:
        raise SystemExit(f"ERROR: val data file is empty: {val_path}")

    torch = deps["torch"]
    tokenizer = deps["AutoTokenizer"].from_pretrained(
        args.model_path,
        padding_side="right",
    )
    processor = deps["AutoProcessor"].from_pretrained(
        args.model_path,
        padding_side="right",
    )
    model = deps["Qwen2_5_VLForConditionalGeneration"].from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = deps["LoraConfig"](
        task_type="CAUSAL_LM",
        target_modules=list(args.lora_target_modules),
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = deps["get_peft_model"](model, lora_config)

    train_dataset = deps["Dataset"].from_list(train_records).map(
        lambda example: _tokenize_record(
            example,
            processor=processor,
            tokenizer=tokenizer,
            process_vision_info=deps["process_vision_info"],
            torch_module=torch,
            max_length=args.max_length,
        ),
        batched=False,
        remove_columns=["sample_id", "messages", "target"],
    )
    val_dataset = deps["Dataset"].from_list(val_records).map(
        lambda example: _tokenize_record(
            example,
            processor=processor,
            tokenizer=tokenizer,
            process_vision_info=deps["process_vision_info"],
            torch_module=torch,
            max_length=args.max_length,
        ),
        batched=False,
        remove_columns=["sample_id", "messages", "target"],
    )

    training_args = deps["TrainingArguments"](
        output_dir=str(output_dir),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        remove_unused_columns=False,
        report_to=_parse_report_to(args.report_to),
    )

    trainer = deps["Trainer"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=_VisionSFTCollator(
            torch_module=torch,
            pad_token_id=tokenizer.pad_token_id,
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return 0


def _load_training_records(path: Path) -> list[dict[str, Any]]:
    return [ensure_training_record(record) for record in load_jsonl_records(path)]


def _tokenize_record(
    example: dict[str, Any],
    *,
    processor: Any,
    tokenizer: Any,
    process_vision_info: Any,
    torch_module: Any,
    max_length: int,
) -> dict[str, Any]:
    messages = to_qwen_messages(example)
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    model_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
    )

    target_tokens = tokenizer(example["target"], add_special_tokens=False)
    eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id or pad_token_id.")

    prompt_input_ids = model_inputs["input_ids"][0].to(dtype=torch_module.long)
    prompt_attention_mask = model_inputs["attention_mask"][0].to(dtype=torch_module.long)
    target_input_ids = torch_module.tensor(
        target_tokens["input_ids"] + [eos_token_id],
        dtype=torch_module.long,
    )
    target_attention_mask = torch_module.tensor(
        target_tokens["attention_mask"] + [1],
        dtype=torch_module.long,
    )

    input_ids = torch_module.cat((prompt_input_ids, target_input_ids), dim=0)
    attention_mask = torch_module.cat(
        (prompt_attention_mask, target_attention_mask),
        dim=0,
    )
    labels = torch_module.cat(
        (
            torch_module.full(
                (prompt_input_ids.size(0),),
                -100,
                dtype=torch_module.long,
            ),
            target_input_ids,
        ),
        dim=0,
    )

    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": _squeeze_batch_tensor(model_inputs["pixel_values"]),
        "image_grid_thw": _squeeze_batch_tensor(model_inputs["image_grid_thw"]),
    }


def _squeeze_batch_tensor(value: Any) -> Any:
    if hasattr(value, "dim") and value.dim() > 0 and value.size(0) == 1:
        return value.squeeze(0)
    return value


def _parse_report_to(value: str) -> list[str]:
    if value == "none":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _import_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from qwen_vl_utils import process_vision_info
        from transformers import (
            AutoProcessor,
            AutoTokenizer,
            Qwen2_5_VLForConditionalGeneration,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "ERROR: missing training dependency for scripts/train_sft.py: "
            f"{exc}"
        ) from exc

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "process_vision_info": process_vision_info,
        "AutoProcessor": AutoProcessor,
        "AutoTokenizer": AutoTokenizer,
        "Qwen2_5_VLForConditionalGeneration": Qwen2_5_VLForConditionalGeneration,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


class _VisionSFTCollator:
    def __init__(self, *, torch_module: Any, pad_token_id: int | None) -> None:
        self._torch = torch_module
        self._pad_token_id = 0 if pad_token_id is None else pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        pad_sequence = self._torch.nn.utils.rnn.pad_sequence
        return {
            "input_ids": pad_sequence(
                [feature["input_ids"] for feature in features],
                batch_first=True,
                padding_value=self._pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [feature["attention_mask"] for feature in features],
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                [feature["labels"] for feature in features],
                batch_first=True,
                padding_value=-100,
            ),
            "pixel_values": self._torch.stack(
                [feature["pixel_values"] for feature in features]
            ),
            "image_grid_thw": self._torch.stack(
                [feature["image_grid_thw"] for feature in features]
            ),
        }


if __name__ == "__main__":
    raise SystemExit(main())
