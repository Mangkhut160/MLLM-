from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from src.data_engine.grpo_builder import MAX_THINK_CHARS
from src.data_engine.schema import (
    CANONICAL_POINT_KEYS,
    FINAL_RESULT_PASS,
    FINAL_RESULT_REVIEW,
    POINT_STATUS_PASS,
    POINT_STATUS_REJECT,
)

REQUIRED_TOP_LEVEL_KEYS = ("point_results", "final_result", "reject_tags")
ALLOWED_POINT_STATUSES = {POINT_STATUS_PASS, POINT_STATUS_REJECT}
ALLOWED_FINAL_RESULTS = {FINAL_RESULT_PASS, FINAL_RESULT_REVIEW}


def reward_format(answer: str | Mapping[str, Any]) -> float:
    payload = _parse_answer(answer)
    if payload is None:
        return 0.0
    if tuple(payload.keys()) != REQUIRED_TOP_LEVEL_KEYS:
        return 0.0

    point_results = payload["point_results"]
    if not isinstance(point_results, Mapping):
        return 0.0
    if tuple(point_results.keys()) != tuple(CANONICAL_POINT_KEYS):
        return 0.0
    if any(value not in ALLOWED_POINT_STATUSES for value in point_results.values()):
        return 0.0

    if payload["final_result"] not in ALLOWED_FINAL_RESULTS:
        return 0.0

    reject_tags = payload["reject_tags"]
    if not isinstance(reject_tags, list):
        return 0.0
    if any(not isinstance(tag, str) or not tag.strip() for tag in reject_tags):
        return 0.0

    return 1.0


def reward_result(
    answer: str | Mapping[str, Any],
    reference: str | Mapping[str, Any],
) -> float:
    candidate = _parse_answer(answer)
    target = _parse_answer(reference)
    if candidate is None or target is None:
        return 0.0
    if reward_format(candidate) == 0.0 or reward_format(target) == 0.0:
        return 0.0
    return 1.0 if candidate["final_result"] == target["final_result"] else 0.0


def reward_tag(
    answer: str | Mapping[str, Any],
    reference: str | Mapping[str, Any],
) -> float:
    candidate = _parse_answer(answer)
    target = _parse_answer(reference)
    if candidate is None or target is None:
        return 0.0
    if reward_format(candidate) == 0.0 or reward_format(target) == 0.0:
        return 0.0

    candidate_tags = set(candidate["reject_tags"])
    target_tags = set(target["reject_tags"])
    if not candidate_tags and not target_tags:
        return 1.0

    union = candidate_tags | target_tags
    if not union:
        return 0.0
    return len(candidate_tags & target_tags) / len(union)


def reward_length(think: str, *, max_chars: int = MAX_THINK_CHARS) -> float:
    normalized_think = _normalize_think(think)
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not normalized_think:
        return -1.0

    overflow = len(normalized_think) - max_chars
    if overflow <= 0:
        return 1.0
    return -(overflow / max_chars)


def _normalize_think(think: str) -> str:
    return " ".join(think.split())


def _parse_answer(answer: str | Mapping[str, Any]) -> dict[str, Any] | None:
    if isinstance(answer, Mapping):
        return dict(answer)
    if not isinstance(answer, str):
        return None
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


__all__ = [
    "reward_format",
    "reward_length",
    "reward_result",
    "reward_tag",
]
