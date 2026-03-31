import json
from typing import Sequence

from .schema import FINAL_RESULT_PASS
from .sft_builder import build_sft_record

MAX_THINK_CHARS = 50
PASS_THINK = "Pass: all checks consistent"
FALLBACK_REVIEW_THINK = "Review: manual verification needed"


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _truncate_think(value: str) -> str:
    if len(value) <= MAX_THINK_CHARS:
        return value
    return value[: MAX_THINK_CHARS - 3] + "..."


def derive_short_think(sample: dict[str, object]) -> str:
    output = sample["output"]
    reject_tags = output.get("reject_tags") or []
    if reject_tags:
        return _truncate_think(_normalize_text(f"Review: {reject_tags[0]}"))

    if output.get("final_result") == FINAL_RESULT_PASS:
        return PASS_THINK

    return _truncate_think(FALLBACK_REVIEW_THINK)


def build_grpo_record(sample: dict[str, object], think: str) -> dict[str, object]:
    normalized_think = _normalize_text(think)
    if not normalized_think:
        raise ValueError("think must be non-empty")
    if len(normalized_think) > MAX_THINK_CHARS:
        raise ValueError("think must be 50 characters or fewer")

    record: dict[str, object] = {
        "sample_id": sample["sample_id"],
        "think": normalized_think,
        "answer": json.dumps(sample["output"], ensure_ascii=False),
    }

    if "input" in sample:
        record["input"] = sample["input"]

    if "messages" in sample:
        record["messages"] = sample["messages"]
    elif "input" in sample:
        record["messages"] = build_sft_record(sample)["messages"]

    return record


def build_grpo_records(samples: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [
        build_grpo_record(sample, derive_short_think(sample))
        for sample in samples
    ]
