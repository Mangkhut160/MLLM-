import json
import random
from pathlib import Path
from typing import Iterable, Sequence

from src.prompt_baseline.prompt_templates import build_audit_prompt


def load_jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def write_jsonl_records(records: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_sft_record(sample: dict[str, object]) -> dict[str, object]:
    sample_input = sample["input"]
    images = sample_input["images"]
    form = sample_input["form"]

    return {
        "sample_id": sample["sample_id"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_path": images["brand_image"],
                    },
                    {
                        "type": "input_image",
                        "image_path": images["spec_image"],
                    },
                    {
                        "type": "input_text",
                        "text": build_audit_prompt(form),
                    },
                ],
            }
        ],
        "target": json.dumps(sample["output"], ensure_ascii=False),
    }


def build_sft_records(samples: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [build_sft_record(sample) for sample in samples]


def split_canonical_samples(
    samples: Sequence[dict[str, object]],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[dict[str, object]]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
