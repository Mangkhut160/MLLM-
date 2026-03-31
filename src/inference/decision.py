from collections.abc import Mapping, Sequence


def build_decision(
    point_results: Mapping[str, str],
    trigger_points: Sequence[str],
) -> dict[str, object]:
    trigger_list = list(trigger_points)
    unknown_trigger_points = [
        point_key for point_key in trigger_list if point_key not in point_results
    ]
    if unknown_trigger_points:
        unknown_points = ", ".join(unknown_trigger_points)
        raise ValueError(f"Unknown trigger points: {unknown_points}")

    return {
        "mode": "manual_review" if trigger_list else "auto_pass",
        "auto_pass_allowed": not trigger_list,
        "trigger_points": trigger_list,
    }
