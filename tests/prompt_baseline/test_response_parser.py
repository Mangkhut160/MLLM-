import json

import pytest

from src.prompt_baseline.response_parser import parse_audit_response


def test_parse_audit_response_accepts_valid_json():
    raw = json.dumps(
        {
            "point_results": {
                "brand_check": "通过",
                "date_check": "通过",
                "voltage_capacity_check": "驳回",
                "spec_check": "通过",
                "image_check": "通过",
            },
            "final_result": "人工复核",
            "reject_tags": ["电池额定电压/额定容量与填写信息不一致"],
        },
        ensure_ascii=False,
    )

    parsed = parse_audit_response(raw)

    assert parsed["point_results"]["voltage_capacity_check"] == "驳回"
    assert parsed["final_result"] == "人工复核"
    assert parsed["reject_tags"] == ["电池额定电压/额定容量与填写信息不一致"]


def test_parse_audit_response_accepts_json_in_markdown_code_fence():
    raw = """```json
{"point_results":{"brand_check":"通过","date_check":"通过","voltage_capacity_check":"通过","spec_check":"通过","image_check":"通过"},"final_result":"通过","reject_tags":[]}
```"""

    parsed = parse_audit_response(raw)

    assert parsed["final_result"] == "通过"


def test_parse_audit_response_accepts_utf8_bom_prefixed_json():
    raw = (
        "\ufeff"
        '{"point_results":{"brand_check":"通过","date_check":"通过","voltage_capacity_check":"通过","spec_check":"通过","image_check":"通过"},"final_result":"通过","reject_tags":[]}'
    )

    parsed = parse_audit_response(raw)

    assert parsed["final_result"] == "通过"


def test_parse_audit_response_fails_clearly_on_invalid_json():
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_audit_response("not-json")


def test_parse_audit_response_fails_clearly_on_invalid_schema():
    with pytest.raises(ValueError, match="Missing required keys: final_result, reject_tags"):
        parse_audit_response(json.dumps({"point_results": {}}))
