import csv
import shutil
import tempfile
from pathlib import Path

from src.data_engine.canonical_builder import (
    build_canonical_records,
    parse_person_info,
)
from src.data_engine.label_mapping import map_labels


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "person_info", "label"])
        writer.writeheader()
        writer.writerows(rows)


def test_parse_person_info_extracts_expected_fields():
    raw = "2024-12-20 15:51:43$%$45V$%$天能$%$72Ah"

    assert parse_person_info(raw) == {
        "production_date": "2024-12-20 15:51:43",
        "voltage": "45V",
        "brand": "天能",
        "capacity": "72Ah",
    }


def test_build_canonical_records_associates_images_and_drops_missing():
    repo_root = Path(__file__).resolve().parents[2]
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="canonical-", dir=temp_root))

    try:
        brand_dir = temp_dir / "brand_new" / "brand_new"
        spec_dir = temp_dir / "charge_new" / "charge_new"
        brand_dir.mkdir(parents=True)
        spec_dir.mkdir(parents=True)

        (brand_dir / "123_brand.png").write_bytes(b"brand")
        (spec_dir / "123.png").write_bytes(b"spec")
        (spec_dir / "456.png").write_bytes(b"spec2")

        csv_path = temp_dir / "data.csv"
        raw = "2024-12-20 15:51:43$%$45V$%$天能$%$72Ah"
        _write_csv(
            csv_path,
            [
                {"id": "123", "person_info": raw, "label": "label-a"},
                {"id": "456", "person_info": raw, "label": "label-b"},
            ],
        )

        records, stats = build_canonical_records(
            csv_path=csv_path,
            brand_dir=brand_dir,
            spec_dir=spec_dir,
            repo_root=temp_dir,
        )

        assert stats == {"kept": 1, "dropped": 1}
        assert records == [
            {
                "sample_id": "123",
                "input": {
                    "form": parse_person_info(raw),
                    "images": {
                        "brand_image": "brand_new/brand_new/123_brand.png",
                        "spec_image": "charge_new/charge_new/123.png",
                    },
                },
                "output": map_labels("label-a"),
            }
        ]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
