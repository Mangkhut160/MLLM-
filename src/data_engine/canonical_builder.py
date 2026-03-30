import csv
from pathlib import Path
from typing import Optional

from .label_mapping import map_labels


def parse_person_info(person_info: str) -> dict[str, str]:
    parts = [part.strip() for part in person_info.split("$%$")]
    if len(parts) != 4:
        raise ValueError("person_info must contain exactly 4 fields separated by $%$")

    return {
        "production_date": parts[0],
        "voltage": parts[1],
        "brand": parts[2],
        "capacity": parts[3],
    }


def _resolve_brand_image(brand_dir: Path, sample_id: str) -> Optional[Path]:
    primary = brand_dir / f"{sample_id}.png"
    if primary.is_file():
        return primary

    fallback = brand_dir / f"{sample_id}_brand.png"
    if fallback.is_file():
        return fallback

    return None


def _resolve_spec_image(spec_dir: Path, sample_id: str) -> Optional[Path]:
    candidate = spec_dir / f"{sample_id}.png"
    return candidate if candidate.is_file() else None


def _relative_posix_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_canonical_records(
    *,
    csv_path: Path,
    brand_dir: Path,
    spec_dir: Path,
    repo_root: Path,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    records: list[dict[str, object]] = []
    dropped = 0

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = (row.get("id") or "").strip()
            if not sample_id:
                dropped += 1
                continue

            brand_image = _resolve_brand_image(brand_dir, sample_id)
            spec_image = _resolve_spec_image(spec_dir, sample_id)

            if not brand_image or not spec_image:
                dropped += 1
                continue
            try:
                parsed_person_info = parse_person_info(row.get("person_info", ""))
            except ValueError:
                dropped += 1
                continue

            records.append(
                {
                    "sample_id": sample_id,
                    "input": {
                        "form": parsed_person_info,
                        "images": {
                            "brand_image": _relative_posix_path(
                                brand_image, repo_root
                            ),
                            "spec_image": _relative_posix_path(spec_image, repo_root),
                        },
                    },
                    "output": map_labels(row.get("label", "")),
                }
            )

    return records, {"kept": len(records), "dropped": dropped}
