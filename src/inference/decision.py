from collections.abc import Mapping, Sequence

from src.data_engine.schema import CANONICAL_POINT_KEYS, POINT_STATUS_PASS


def build_decision(
    point_results: Mapping[str, str],
    trigger_points: Sequence[str],
) -> dict[str, object]:
    missing_point_keys = [
        point_key for point_key in CANONICAL_POINT_KEYS if point_key not in point_results
    ]
    if missing_point_keys:
        missing_points = ", ".join(missing_point_keys)
        raise ValueError(f"Missing point results for: {missing_points}")

    requested_trigger_points = list(trigger_points)
    unknown_trigger_points = [
        point_key for point_key in requested_trigger_points if point_key not in point_results
    ]
    if unknown_trigger_points:
        unknown_points = ", ".join(unknown_trigger_points)
        raise ValueError(f"Unknown trigger points: {unknown_points}")

    derived_trigger_points = [
        point_key
        for point_key in CANONICAL_POINT_KEYS
        if point_results[point_key] != POINT_STATUS_PASS
    ]
    trigger_list = list(dict.fromkeys(derived_trigger_points + requested_trigger_points))

    return {
        "mode": "manual_review" if trigger_list else "auto_pass",
        "auto_pass_allowed": not trigger_list,
        "trigger_points": trigger_list,
    }
