import json

import pytest

from src.data_engine.schema import (
    FINAL_RESULT_PASS,
    POINT_STATUS_REJECT,
    default_point_results,
)
from src.prompt_baseline.response_parser import parse_audit_response


def _valid_payload() -> dict[str, object]:
    point_results = default_point_results()
    point_results["voltage_capacity_check"] = POINT_STATUS_REJECT
    return {
        "point_results": point_results,
        "final_result": FINAL_RESULT_PASS,
        "reject_tags": ["tag-a"],
    }


def test_parse_audit_response_accepts_valid_json():
    raw = json.dumps(_valid_payload(), ensure_ascii=False)

    parsed = parse_audit_response(raw)

    assert parsed["point_results"]["voltage_capacity_check"] == POINT_STATUS_REJECT
    assert parsed["final_result"] == FINAL_RESULT_PASS
    assert parsed["reject_tags"] == ["tag-a"]


def test_parse_audit_response_accepts_json_in_markdown_code_fence():
    raw = f"""```json
{json.dumps(_valid_payload(), ensure_ascii=False)}
```"""

    parsed = parse_audit_response(raw)

    assert parsed["final_result"] == FINAL_RESULT_PASS


def test_parse_audit_response_accepts_utf8_bom_prefixed_json():
    raw = "\ufeff" + json.dumps(_valid_payload(), ensure_ascii=False)

    parsed = parse_audit_response(raw)

    assert parsed["final_result"] == FINAL_RESULT_PASS


def test_parse_audit_response_fails_clearly_on_invalid_json():
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_audit_response("not-json")


def test_parse_audit_response_fails_clearly_on_invalid_schema():
    with pytest.raises(
        ValueError, match="Missing required keys: final_result, reject_tags"
    ):
        parse_audit_response(json.dumps({"point_results": {}}))


def test_parse_audit_response_rejects_invalid_point_status_value():
    payload = _valid_payload()
    payload["point_results"]["brand_check"] = "invalid"

    with pytest.raises(
        ValueError, match="Invalid point_results status for brand_check: invalid"
    ):
        parse_audit_response(json.dumps(payload, ensure_ascii=False))


def test_parse_audit_response_rejects_invalid_final_result():
    payload = _valid_payload()
    payload["final_result"] = "invalid"

    with pytest.raises(ValueError, match="Invalid final_result: invalid"):
        parse_audit_response(json.dumps(payload, ensure_ascii=False))


def test_parse_audit_response_rejects_non_string_reject_tag():
    payload = _valid_payload()
    payload["reject_tags"] = ["tag-a", 123]

    with pytest.raises(ValueError, match="reject_tags must contain only strings"):
        parse_audit_response(json.dumps(payload, ensure_ascii=False))
