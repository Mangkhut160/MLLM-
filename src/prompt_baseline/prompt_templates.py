def build_audit_prompt(form: dict[str, str]) -> str:
    return (
        "You are auditing a battery listing. Check brand/date/voltage-capacity/"
        "spec/image compliance.\n"
        "Form fields:\n"
        f"- brand: {form['brand']}\n"
        f"- production_date: {form['production_date']}\n"
        f"- voltage: {form['voltage']}\n"
        f"- capacity: {form['capacity']}\n"
        "Return JSON with keys point_results, final_result, reject_tags."
    )
