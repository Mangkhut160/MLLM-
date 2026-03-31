from typing import Dict, List

from .schema import (
    FINAL_RESULT_PASS,
    FINAL_RESULT_REVIEW,
    POINT_STATUS_REJECT,
    default_point_results,
)

RAW_LABEL_MAPPING: Dict[str, str] = {
    "使用电池不符合规范": "spec_check",
    "使用电池规格严重超标，不符合国标要求": "spec_check",
    "电池品牌图片与填写品牌信息不一致": "brand_check",
    "电池生产日期与填写信息不一致": "date_check",
    "电池额定电压/额定容量与填写信息不一致": "voltage_capacity_check",
    "非电池相关信息图片": "image_check",
    "非规范电池信息图片": "image_check",
}


def _parse_raw_labels(raw_labels: str) -> List[str]:
    if not raw_labels:
        return []

    parts = [segment.strip() for segment in raw_labels.split("|||")]
    return [segment for segment in parts if segment]


def map_labels(raw_labels: str) -> Dict[str, object]:
    """
    Convert raw inspection labels into the canonical schema.

    Empty or whitespace-only input results in all points passing with no reject tags.
    """
    parsed_labels = _parse_raw_labels(raw_labels)
    point_results = default_point_results()
    reject_tags: List[str] = []

    for label in parsed_labels:
        reject_tags.append(label)
        canonical_key = RAW_LABEL_MAPPING.get(label)
        if canonical_key:
            point_results[canonical_key] = POINT_STATUS_REJECT

    final_result = FINAL_RESULT_REVIEW if parsed_labels else FINAL_RESULT_PASS

    return {
        "point_results": point_results,
        "final_result": final_result,
        "reject_tags": reject_tags,
    }
