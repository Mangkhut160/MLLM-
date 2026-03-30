import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data_engine.label_mapping import map_labels

CANONICAL_POINTS = [
    "brand_check",
    "date_check",
    "voltage_capacity_check",
    "spec_check",
    "image_check",
]


def test_raw_label_maps_to_date_check_rejection():
    raw_label = "电池生产日期与填写信息不一致"
    mapped = map_labels(raw_label)

    assert mapped["final_result"] == "人工复核"
    assert mapped["reject_tags"] == [raw_label]
    assert mapped["point_results"]["date_check"] == "驳回"
    for key in CANONICAL_POINTS:
        if key == "date_check":
            continue
        assert mapped["point_results"][key] == "通过"


def test_empty_label_results_in_all_pass():
    mapped = map_labels("")
    assert mapped["final_result"] == "通过"
    assert mapped["reject_tags"] == []
    for key in CANONICAL_POINTS:
        assert mapped["point_results"][key] == "通过"
