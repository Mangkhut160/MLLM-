import json
from json import JSONDecodeError

from src.data_engine.schema import (
    CANONICAL_POINT_KEYS,
    FINAL_RESULT_PASS,
    FINAL_RESULT_REVIEW,
    POINT_STATUS_PASS,
    POINT_STATUS_REJECT,
)

ALLOWED_POINT_STATUSES = {POINT_STATUS_PASS, POINT_STATUS_REJECT}
ALLOWED_FINAL_RESULTS = {FINAL_RESULT_PASS, FINAL_RESULT_REVIEW}


def _strip_code_fence(raw_text: str) -> str:
    stripped = raw_text.strip().lstrip("\ufeff")
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()

    return stripped


def parse_audit_response(raw_text: str) -> dict[str, object]:
    candidate = _strip_code_fence(raw_text)

    try:
        payload = json.loads(candidate)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Response JSON must decode to an object")

    required_keys = ["point_results", "final_result", "reject_tags"]
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")

    point_results = payload["point_results"]
    if not isinstance(point_results, dict):
        raise ValueError("point_results must be an object")

    missing_point_keys = [
        key for key in CANONICAL_POINT_KEYS if key not in point_results
    ]
    if missing_point_keys:
        raise ValueError(
            f"Missing point_results keys: {', '.join(missing_point_keys)}"
        )

    for key, value in point_results.items():
        if value not in ALLOWED_POINT_STATUSES:
            raise ValueError(f"Invalid point_results status for {key}: {value}")

    final_result = payload["final_result"]
    if final_result not in ALLOWED_FINAL_RESULTS:
        raise ValueError(f"Invalid final_result: {final_result}")

    reject_tags = payload["reject_tags"]
    if not isinstance(reject_tags, list):
        raise ValueError("reject_tags must be a list")
    if not all(isinstance(tag, str) for tag in reject_tags):
        raise ValueError("reject_tags must contain only strings")

    return payload
