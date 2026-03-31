from __future__ import annotations

from typing import Any

from src.data_engine.sft_builder import build_sft_record


def format_training_example(sample: dict[str, object]) -> dict[str, object]:
    """Return the canonical stage-one SFT record for a canonical sample."""
    return build_sft_record(sample)


def ensure_training_record(sample_or_record: dict[str, Any]) -> dict[str, Any]:
    if "messages" in sample_or_record and "target" in sample_or_record:
        return sample_or_record
    return format_training_example(sample_or_record)


def to_qwen_messages(sample_or_record: dict[str, Any]) -> list[dict[str, Any]]:
    record = ensure_training_record(sample_or_record)
    return [
        {
            "role": message["role"],
            "content": [_to_qwen_content_item(item) for item in message["content"]],
        }
        for message in record["messages"]
    ]


def _to_qwen_content_item(item: dict[str, Any]) -> dict[str, Any]:
    item_type = item["type"]
    if item_type == "input_image":
        return {"type": "image", "image": item["image_path"]}
    if item_type == "input_text":
        return {"type": "text", "text": item["text"]}
    if item_type in {"image", "text"}:
        return item
    raise ValueError(f"Unsupported message content type: {item_type}")
