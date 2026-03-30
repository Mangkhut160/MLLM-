CANONICAL_POINT_KEYS = [
    "brand_check",
    "date_check",
    "voltage_capacity_check",
    "spec_check",
    "image_check",
]

POINT_STATUS_PASS = "通过"
POINT_STATUS_REJECT = "驳回"

FINAL_RESULT_PASS = "通过"
FINAL_RESULT_REVIEW = "人工复核"


def default_point_results():
    """Return the canonical point keys with default pass status."""
    return {key: POINT_STATUS_PASS for key in CANONICAL_POINT_KEYS}
