import json
import subprocess
import sys
import tempfile
from pathlib import Path

from src.data_engine.schema import FINAL_RESULT_PASS, default_point_results

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sample(sample_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "input": {
            "form": {
                "production_date": "2024-12-20 15:51:43",
                "voltage": "45V",
                "brand": "Tianneng",
                "capacity": "72Ah",
            },
            "images": {
                "brand_image": f"brand_new/brand_new/{sample_id}.png",
                "spec_image": f"charge_new/charge_new/{sample_id}.png",
            },
        },
        "output": {
            "point_results": default_point_results(),
            "final_result": FINAL_RESULT_PASS,
            "reject_tags": [],
        },
    }


def _make_temp_dir() -> Path:
    temp_root = REPO_ROOT / ".pytest_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="prompt-baseline-", dir=temp_root))


def test_run_prompt_baseline_exits_cleanly_on_parse_failure():
    temp_dir = _make_temp_dir()

    try:
        samples_path = temp_dir / "samples.jsonl"
        samples_path.write_text(
            json.dumps(_sample("460790679"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        invalid_response = json.dumps(
            {
                "point_results": {
                    **default_point_results(),
                    "brand_check": "invalid",
                },
                "final_result": FINAL_RESULT_PASS,
                "reject_tags": [],
            },
            ensure_ascii=False,
        )

        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_prompt_baseline.py",
                "--samples",
                str(samples_path),
                "--sample-id",
                "460790679",
                "--response",
                invalid_response,
            ],
            capture_output=True,
            cwd=REPO_ROOT,
            text=True,
            check=False,
        )

        combined_output = result.stdout + result.stderr

        assert result.returncode != 0
        assert "ERROR: invalid response:" in combined_output
        assert "Traceback" not in combined_output
    finally:
        for path in temp_dir.glob("*"):
            path.unlink()
        temp_dir.rmdir()
