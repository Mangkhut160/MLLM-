from src.data_engine.label_mapping import map_labels
from src.data_engine.schema import FINAL_RESULT_REVIEW, POINT_STATUS_REJECT

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


def test_multi_label_rejection_marks_each_canonical_point():
    brand_label = "电池品牌图片与填写品牌信息不一致"
    date_label = "电池生产日期与填写信息不一致"
    combined = f"{brand_label}|||{date_label}"
    mapped = map_labels(combined)

    assert mapped["final_result"] == FINAL_RESULT_REVIEW
    assert mapped["reject_tags"] == [brand_label, date_label]
    assert mapped["point_results"]["brand_check"] == POINT_STATUS_REJECT
    assert mapped["point_results"]["date_check"] == POINT_STATUS_REJECT


def test_empty_label_results_in_all_pass():
    mapped = map_labels("")
    assert mapped["final_result"] == "通过"
    assert mapped["reject_tags"] == []
    for key in CANONICAL_POINTS:
        assert mapped["point_results"][key] == "通过"
