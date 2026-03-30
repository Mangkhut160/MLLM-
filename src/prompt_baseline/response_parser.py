import json
from json import JSONDecodeError

from src.data_engine.schema import CANONICAL_POINT_KEYS


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

    if not isinstance(payload["final_result"], str):
        raise ValueError("final_result must be a string")

    if not isinstance(payload["reject_tags"], list):
        raise ValueError("reject_tags must be a list")

    return payload
