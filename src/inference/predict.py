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


def predict_decision(parsed_response: Mapping[str, object]) -> dict[str, object]:
    point_results = parsed_response["point_results"]
    if not isinstance(point_results, Mapping):
        raise ValueError("parsed_response must contain point_results")

    trigger_points = derive_trigger_points(point_results)
    return {
        "point_results": dict(point_results),
        "reject_tags": list(parsed_response["reject_tags"]),
        "decision": build_decision(point_results, trigger_points),
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
