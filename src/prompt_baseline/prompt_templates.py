from src.data_engine.schema import (
    CANONICAL_POINT_KEYS,
    FINAL_RESULT_PASS,
    FINAL_RESULT_REVIEW,
    POINT_STATUS_PASS,
    POINT_STATUS_REJECT,
)


def build_audit_prompt(form: dict[str, str]) -> str:
    point_keys = ", ".join(CANONICAL_POINT_KEYS)

    return (
        "You are auditing a battery listing. Check brand/date/voltage-capacity/"
        "spec/image compliance.\n"
        "Attached images:\n"
        "- Image 1: brand image\n"
        "- Image 2: spec image\n"
        "Form fields:\n"
        f"- brand: {form['brand']}\n"
        f"- production_date: {form['production_date']}\n"
        f"- voltage: {form['voltage']}\n"
        f"- capacity: {form['capacity']}\n"
        "Return only a JSON object with top-level keys: point_results, "
        "final_result, reject_tags.\n"
        f"The point_results object must include these keys: {point_keys}.\n"
        f"Allowed point_results values: {POINT_STATUS_PASS}, "
        f"{POINT_STATUS_REJECT}.\n"
        f"Allowed final_result values: {FINAL_RESULT_PASS}, "
        f"{FINAL_RESULT_REVIEW}.\n"
        "reject_tags must be a list of strings."
    )
