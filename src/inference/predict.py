import json
from collections.abc import Mapping

from src.data_engine.schema import CANONICAL_POINT_KEYS, POINT_STATUS_PASS
from src.inference.decision import build_decision
from src.prompt_baseline.response_parser import parse_audit_response


def derive_trigger_points(point_results: Mapping[str, str]) -> list[str]:
    return [
        point_key
        for point_key in CANONICAL_POINT_KEYS
        if point_results[point_key] != POINT_STATUS_PASS
    ]


def _normalize_point_results(point_results: Mapping[str, object]) -> dict[str, str]:
    missing_point_keys = [
        point_key for point_key in CANONICAL_POINT_KEYS if point_key not in point_results
    ]
    if missing_point_keys:
        missing = ", ".join(missing_point_keys)
        raise ValueError(f"point_results missing keys: {missing}")

    normalized: dict[str, str] = {}
    for point_key in CANONICAL_POINT_KEYS:
        value = point_results[point_key]
        if not isinstance(value, str):
            raise ValueError(f"point_results[{point_key}] must be a string")
        normalized[point_key] = value
    return normalized


def predict_decision(parsed_response: Mapping[str, object]) -> dict[str, object]:
    point_results = parsed_response["point_results"]
    if not isinstance(point_results, Mapping):
        raise ValueError("parsed_response must contain point_results")
    normalized_point_results = _normalize_point_results(point_results)

    reject_tags = parsed_response["reject_tags"]
    if not isinstance(reject_tags, list):
        raise ValueError("parsed_response must contain reject_tags")
    if any(not isinstance(tag, str) for tag in reject_tags):
        raise ValueError("reject_tags must contain strings only")

    trigger_points = derive_trigger_points(normalized_point_results)
    return {
        "point_results": normalized_point_results,
        "reject_tags": reject_tags,
        "decision": build_decision(normalized_point_results, trigger_points),
    }


def predict_from_response(
    *,
    raw_response: str | None,
    fallback_response: Mapping[str, object],
) -> dict[str, object]:
    if raw_response is None:
        normalized_response = json.dumps(fallback_response, ensure_ascii=False)
    else:
        normalized_response = raw_response

    parsed_response = parse_audit_response(normalized_response)
    return predict_decision(parsed_response)
