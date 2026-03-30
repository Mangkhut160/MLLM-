# Battery Audit Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage battery audit MVP from the current dataset and training skeleton, producing a canonical dataset pipeline, an SFT audit model, evaluation and threshold calibration, then extend it with GRPO-based short-reason optimization.

**Architecture:** Keep `data.csv` plus the two image folders as the immutable source of truth. Build deterministic dataset-generation scripts that emit canonical, SFT, and GRPO formats; train `Qwen2.5-VL` with `LoRA` for structured audit output first, then add `GRPO` for short reasons while keeping online decisions dependent only on structured answers and calibrated thresholds.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, PEFT, Datasets, pytest, Qwen2.5-VL, optional SwanLab

---

## File Structure

### New files to create

- `.gitignore`
- `README.md`
- `data/README.md`
- `src/data_engine/__init__.py`
- `src/data_engine/schema.py`
- `src/data_engine/label_mapping.py`
- `src/data_engine/canonical_builder.py`
- `src/data_engine/sft_builder.py`
- `src/data_engine/grpo_builder.py`
- `src/prompt_baseline/__init__.py`
- `src/prompt_baseline/prompt_templates.py`
- `src/prompt_baseline/response_parser.py`
- `src/modeling/__init__.py`
- `src/modeling/sft_formatter.py`
- `src/modeling/grpo_rewards.py`
- `src/evaluation/__init__.py`
- `src/evaluation/metrics.py`
- `src/evaluation/thresholds.py`
- `src/inference/__init__.py`
- `src/inference/decision.py`
- `src/inference/predict.py`
- `scripts/build_canonical_dataset.py`
- `scripts/build_sft_dataset.py`
- `scripts/build_grpo_dataset.py`
- `scripts/run_prompt_baseline.py`
- `scripts/train_sft.py`
- `scripts/train_grpo.py`
- `scripts/evaluate_model.py`
- `scripts/calibrate_thresholds.py`
- `scripts/demo_inference.py`
- `tests/data_engine/test_label_mapping.py`
- `tests/data_engine/test_canonical_builder.py`
- `tests/data_engine/test_sft_builder.py`
- `tests/data_engine/test_grpo_builder.py`
- `tests/prompt_baseline/test_response_parser.py`
- `tests/modeling/test_sft_formatter.py`
- `tests/modeling/test_grpo_rewards.py`
- `tests/evaluation/test_metrics.py`
- `tests/evaluation/test_thresholds.py`
- `tests/inference/test_decision.py`

### Existing files to modify or relocate

- Move `audit_data_process (1).ipynb` to `legacy/audit_data_process (1).ipynb`
- Move `train_lora.py` to `legacy/train_lora.py`
- Move `run.sh` to `legacy/run.sh`
- Keep `data.csv` in place for the first migration commit, then copy to `data/raw/data.csv`
- Keep `brand_new/` and `charge_new/` in place for the first migration commit, then copy to `data/raw/brand_new/` and `data/raw/charge_new/`

### Existing files to stop depending on

- `config.json`

### Output directories to create during implementation

- `data/raw/`
- `data/canonical/`
- `data/sft/`
- `data/grpo/`
- `outputs/checkpoints/`
- `outputs/eval/`
- `outputs/calibration/`
- `outputs/demo/`
- `legacy/`

## Phase Plan

- Phase 1: Repository hygiene, canonical dataset, prompt baseline, SFT pipeline, evaluation, threshold calibration, demo inference
- Phase 2: Cold-start reason dataset, GRPO rewards, GRPO training, phase comparison report

### Task 1: Clean the Repository and Lock the Project Layout

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `data/README.md`
- Create: `legacy/.gitkeep`
- Modify: repository root layout
- Test: repository cleanliness via `git status --short`

- [ ] **Step 1: Add ignore rules for generated, temporary, and OS noise**

```gitignore
.superpowers/
__MACOSX/
*.pyc
__pycache__/
.pytest_cache/
.DS_Store
._*
outputs/
```

- [ ] **Step 2: Add a minimal root README that explains the source-of-truth files**

```md
# Battery Audit Prototype

Source-of-truth data:
- `data.csv`
- `brand_new/`
- `charge_new/`

Design docs and implementation plans live under `docs/superpowers/`.
```

- [ ] **Step 3: Add `data/README.md` describing the four data layers**

```md
# Data Layout

- `raw/`: immutable source data copied from the original materials
- `canonical/`: normalized JSONL samples
- `sft/`: stage-one supervised training sets
- `grpo/`: stage-two RL training sets
```

- [ ] **Step 4: Remove tracked noise from git without deleting the real dataset**

Run:
```powershell
git rm -r --cached .superpowers
git rm -r --cached brand_new/__MACOSX charge_new/__MACOSX
git rm --cached brand_new/brand_new/.DS_Store
```

Expected: cached temp files removed from the next commit only

- [ ] **Step 5: Verify the worktree only shows intended changes**

Run:
```powershell
git status --short
```

Expected: only `.gitignore`, `README.md`, `data/README.md`, and cleanup changes appear

- [ ] **Step 6: Commit**

```bash
git add .gitignore README.md data/README.md
git commit -m "chore: clean repository and define project layout"
```

### Task 2: Define the Canonical Data Schema and Label Mapping

**Files:**
- Create: `src/data_engine/schema.py`
- Create: `src/data_engine/label_mapping.py`
- Create: `tests/data_engine/test_label_mapping.py`
- Test: `tests/data_engine/test_label_mapping.py`

- [ ] **Step 1: Write failing tests for raw label mapping and empty-label handling**

```python
from src.data_engine.label_mapping import map_labels


def test_map_single_reject_label():
    result = map_labels("电池生产日期与填写信息不一致")
    assert result["point_results"]["date_check"] == "驳回"
    assert result["final_result"] == "人工复核"


def test_map_empty_label_to_all_pass():
    result = map_labels("")
    assert all(v == "通过" for v in result["point_results"].values())
    assert result["final_result"] == "通过"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
pytest tests/data_engine/test_label_mapping.py -v
```

Expected: import or function-not-found failure

- [ ] **Step 3: Implement the canonical schema and label mapping**

```python
POINT_KEYS = [
    "brand_check",
    "date_check",
    "voltage_capacity_check",
    "spec_check",
    "image_check",
]


def make_all_pass():
    return {key: "通过" for key in POINT_KEYS}
```

```python
RAW_TO_POINT = {
    "使用电池不符合规范": "spec_check",
    "使用电池规格严重超标，不符合国标要求": "spec_check",
    "电池品牌图片与填写品牌信息不一致": "brand_check",
    "电池生产日期与填写信息不一致": "date_check",
    "电池额定电压/额定容量与填写信息不一致": "voltage_capacity_check",
    "非电池相关信息图片": "image_check",
    "非规范电池信息图片": "image_check",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
pytest tests/data_engine/test_label_mapping.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_engine/schema.py src/data_engine/label_mapping.py tests/data_engine/test_label_mapping.py
git commit -m "feat: add canonical label mapping for battery audit"
```

### Task 3: Build the Canonical Dataset from `data.csv` and Image Folders

**Files:**
- Create: `src/data_engine/canonical_builder.py`
- Create: `scripts/build_canonical_dataset.py`
- Create: `tests/data_engine/test_canonical_builder.py`
- Test: `tests/data_engine/test_canonical_builder.py`

- [ ] **Step 1: Write failing tests for `person_info` parsing and image association**

```python
from src.data_engine.canonical_builder import parse_person_info


def test_parse_person_info():
    parsed = parse_person_info("2024-12-20 15:51:43$%$45V$%$天能$%$72Ah")
    assert parsed["production_date"] == "2024-12-20 15:51:43"
    assert parsed["voltage"] == "45V"
    assert parsed["brand"] == "天能"
    assert parsed["capacity"] == "72Ah"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
pytest tests/data_engine/test_canonical_builder.py -v
```

Expected: import or attribute failure

- [ ] **Step 3: Implement the canonical builder and CLI**

```python
def parse_person_info(raw: str) -> dict:
    production_date, voltage, brand, capacity = raw.split("$%$")
    return {
        "production_date": production_date,
        "voltage": voltage,
        "brand": brand,
        "capacity": capacity,
    }
```

```python
sample = {
    "sample_id": row["id"],
    "input": {"form": parse_person_info(row["person_info"]), "images": image_paths},
    "output": map_labels(row.get("label", "")),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
pytest tests/data_engine/test_canonical_builder.py -v
```

Expected: PASS

- [ ] **Step 5: Generate canonical data on the real repository**

Run:
```powershell
python scripts/build_canonical_dataset.py --csv data.csv --brand-dir brand_new/brand_new --spec-dir charge_new/charge_new --output data/canonical/samples.jsonl
```

Expected: `data/canonical/samples.jsonl` created plus a short summary of kept / dropped samples

- [ ] **Step 6: Commit**

```bash
git add src/data_engine/canonical_builder.py scripts/build_canonical_dataset.py tests/data_engine/test_canonical_builder.py data/canonical/samples.jsonl
git commit -m "feat: build canonical battery audit dataset"
```

### Task 4: Create SFT Training Sets and the Prompt Baseline

**Files:**
- Create: `src/data_engine/sft_builder.py`
- Create: `src/prompt_baseline/prompt_templates.py`
- Create: `src/prompt_baseline/response_parser.py`
- Create: `scripts/build_sft_dataset.py`
- Create: `scripts/run_prompt_baseline.py`
- Create: `tests/data_engine/test_sft_builder.py`
- Create: `tests/prompt_baseline/test_response_parser.py`
- Test: `tests/data_engine/test_sft_builder.py`
- Test: `tests/prompt_baseline/test_response_parser.py`

- [ ] **Step 1: Write failing tests for SFT sample formatting and prompt-output parsing**

```python
from src.data_engine.sft_builder import build_sft_record


def test_build_sft_record_contains_two_images_and_json_answer():
    record = build_sft_record(sample)
    assert len(record["messages"][0]["content"]) == 3
    assert "brand_check" in record["target"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
pytest tests/data_engine/test_sft_builder.py tests/prompt_baseline/test_response_parser.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement prompt templates, parsers, and SFT data builder**

```python
AUDIT_PROMPT = """请根据用户填写信息与两张图片，对品牌、日期、电压容量、规格合规、图片规范进行审核，并输出结构化 JSON。"""
```

```python
def build_sft_record(sample: dict) -> dict:
    return {
        "messages": [...],
        "target": json.dumps(sample["output"], ensure_ascii=False),
    }
```

- [ ] **Step 4: Build real SFT data**

Run:
```powershell
python scripts/build_sft_dataset.py --input data/canonical/samples.jsonl --train-out data/sft/train_sft.jsonl --val-out data/sft/val_sft.jsonl --test-out data/canonical/test.jsonl
```

Expected: three JSONL outputs and a split summary

- [ ] **Step 5: Run tests to verify they pass**

Run:
```powershell
pytest tests/data_engine/test_sft_builder.py tests/prompt_baseline/test_response_parser.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/data_engine/sft_builder.py src/prompt_baseline/prompt_templates.py src/prompt_baseline/response_parser.py scripts/build_sft_dataset.py scripts/run_prompt_baseline.py tests/data_engine/test_sft_builder.py tests/prompt_baseline/test_response_parser.py data/sft/train_sft.jsonl data/sft/val_sft.jsonl
git commit -m "feat: add prompt baseline and sft dataset generation"
```

### Task 5: Implement Stage-One SFT Training for Structured Audit Output

**Files:**
- Create: `src/modeling/sft_formatter.py`
- Create: `scripts/train_sft.py`
- Create: `tests/modeling/test_sft_formatter.py`
- Modify: training references from legacy `train_lora.py`
- Test: `tests/modeling/test_sft_formatter.py`

- [ ] **Step 1: Write failing tests for the message formatter and target labels**

```python
from src.modeling.sft_formatter import format_training_example


def test_format_training_example_returns_two_image_message():
    payload = format_training_example(sample)
    assert payload["messages"][0]["content"][0]["type"] == "image"
    assert payload["messages"][0]["content"][1]["type"] == "image"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
pytest tests/modeling/test_sft_formatter.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement the formatter and SFT training CLI**

```python
def format_training_example(sample: dict) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["input"]["images"]["brand_image"]},
                    {"type": "image", "image": sample["input"]["images"]["spec_image"]},
                    {"type": "text", "text": AUDIT_PROMPT_WITH_FORM_FIELDS},
                ],
            }
        ],
        "target": json.dumps(sample["output"], ensure_ascii=False),
    }
```

- [ ] **Step 4: Run unit test to verify it passes**

Run:
```powershell
pytest tests/modeling/test_sft_formatter.py -v
```

Expected: PASS

- [ ] **Step 5: Smoke-test the training entrypoint**

Run:
```powershell
python scripts/train_sft.py --help
```

Expected: usage output with dataset and model arguments

- [ ] **Step 6: Commit**

```bash
git add src/modeling/sft_formatter.py scripts/train_sft.py tests/modeling/test_sft_formatter.py
git commit -m "feat: add stage-one sft training entrypoint"
```

### Task 6: Add Evaluation Metrics and Threshold Calibration

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `src/evaluation/thresholds.py`
- Create: `scripts/evaluate_model.py`
- Create: `scripts/calibrate_thresholds.py`
- Create: `tests/evaluation/test_metrics.py`
- Create: `tests/evaluation/test_thresholds.py`
- Test: `tests/evaluation/test_metrics.py`
- Test: `tests/evaluation/test_thresholds.py`

- [ ] **Step 1: Write failing tests for point metrics and auto-pass threshold logic**

```python
from src.evaluation.thresholds import should_auto_pass


def test_should_auto_pass_requires_all_key_points():
    assert not should_auto_pass(
        point_predictions={"brand_check": "通过", "date_check": "驳回"},
        point_scores={"brand_check": 0.99, "date_check": 0.99},
        final_score=0.99,
        thresholds={"brand_check": 0.98, "date_check": 0.98, "final_result": 0.98},
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
pytest tests/evaluation/test_metrics.py tests/evaluation/test_thresholds.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement evaluation and calibration modules**

```python
def should_auto_pass(point_predictions, point_scores, final_score, thresholds):
    for key, label in point_predictions.items():
        if label != "通过":
            return False
        if point_scores[key] < thresholds[key]:
            return False
    return final_score >= thresholds["final_result"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```powershell
pytest tests/evaluation/test_metrics.py tests/evaluation/test_thresholds.py -v
```

Expected: PASS

- [ ] **Step 5: Smoke-test the CLI**

Run:
```powershell
python scripts/evaluate_model.py --help
python scripts/calibrate_thresholds.py --help
```

Expected: both commands print usage without crashing

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/metrics.py src/evaluation/thresholds.py scripts/evaluate_model.py scripts/calibrate_thresholds.py tests/evaluation/test_metrics.py tests/evaluation/test_thresholds.py
git commit -m "feat: add evaluation metrics and threshold calibration"
```

### Task 7: Add Online Decision Logic and Demo Inference

**Files:**
- Create: `src/inference/decision.py`
- Create: `src/inference/predict.py`
- Create: `scripts/demo_inference.py`
- Create: `tests/inference/test_decision.py`
- Test: `tests/inference/test_decision.py`

- [ ] **Step 1: Write failing tests for final decision modes**

```python
from src.inference.decision import build_decision


def test_build_decision_returns_manual_review_when_any_point_fails():
    decision = build_decision(
        point_results={"brand_check": "通过", "date_check": "驳回"},
        trigger_points=["date_check"],
    )
    assert decision["mode"] == "manual_review"
    assert decision["auto_pass_allowed"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
pytest tests/inference/test_decision.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement the decision builder and demo CLI**

```python
def build_decision(point_results: dict, trigger_points: list[str]) -> dict:
    return {
        "mode": "auto_pass" if not trigger_points else "manual_review",
        "auto_pass_allowed": not trigger_points,
        "trigger_points": trigger_points,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
pytest tests/inference/test_decision.py -v
```

Expected: PASS

- [ ] **Step 5: Smoke-test the demo entrypoint**

Run:
```powershell
python scripts/demo_inference.py --help
```

Expected: usage output only

- [ ] **Step 6: Commit**

```bash
git add src/inference/decision.py src/inference/predict.py scripts/demo_inference.py tests/inference/test_decision.py
git commit -m "feat: add inference decision layer and demo cli"
```

### Task 8: Build Cold-Start `think + answer` Data for GRPO

**Files:**
- Create: `src/data_engine/grpo_builder.py`
- Create: `scripts/build_grpo_dataset.py`
- Create: `tests/data_engine/test_grpo_builder.py`
- Test: `tests/data_engine/test_grpo_builder.py`

- [ ] **Step 1: Write failing tests for short-reason formatting**

```python
from src.data_engine.grpo_builder import build_grpo_record


def test_build_grpo_record_contains_short_think_and_structured_answer():
    record = build_grpo_record(sample, "参数图容量不符，需人工复核。")
    assert record["think"].startswith("参数图")
    assert len(record["think"]) <= 50
    assert "final_result" in record["answer"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
pytest tests/data_engine/test_grpo_builder.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement the GRPO data builder**

```python
def build_grpo_record(sample: dict, think: str) -> dict:
    assert len(think) <= 50
    return {
        "messages": sample["messages"],
        "think": think,
        "answer": json.dumps(sample["output"], ensure_ascii=False),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```powershell
pytest tests/data_engine/test_grpo_builder.py -v
```

Expected: PASS

- [ ] **Step 5: Generate cold-start GRPO data**

Run:
```powershell
python scripts/build_grpo_dataset.py --input data/canonical/samples.jsonl --output data/grpo/cold_start.jsonl
```

Expected: `data/grpo/cold_start.jsonl` created with short reasons and structured answers

- [ ] **Step 6: Commit**

```bash
git add src/data_engine/grpo_builder.py scripts/build_grpo_dataset.py tests/data_engine/test_grpo_builder.py data/grpo/cold_start.jsonl
git commit -m "feat: add grpo cold-start dataset builder"
```

### Task 9: Implement GRPO Rewards and Training Entry Point

**Files:**
- Create: `src/modeling/grpo_rewards.py`
- Create: `scripts/train_grpo.py`
- Create: `tests/modeling/test_grpo_rewards.py`
- Test: `tests/modeling/test_grpo_rewards.py`

- [ ] **Step 1: Write failing tests for format, result, tag, and length rewards**

```python
from src.modeling.grpo_rewards import reward_length


def test_reward_length_penalizes_overlong_think():
    assert reward_length("这是一条明显超过五十字的审核理由这是一条明显超过五十字的审核理由这是一条明显超过五十字的审核理由") < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
pytest tests/modeling/test_grpo_rewards.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement the reward helpers and training CLI**

```python
def reward_length(think: str) -> float:
    if len(think) == 0:
        return -1.0
    if len(think) <= 50:
        return 1.0
    return -1.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```powershell
pytest tests/modeling/test_grpo_rewards.py -v
```

Expected: PASS

- [ ] **Step 5: Smoke-test the GRPO entrypoint**

Run:
```powershell
python scripts/train_grpo.py --help
```

Expected: usage output only

- [ ] **Step 6: Commit**

```bash
git add src/modeling/grpo_rewards.py scripts/train_grpo.py tests/modeling/test_grpo_rewards.py
git commit -m "feat: add grpo rewards and training entrypoint"
```

### Task 10: Document Results and Phase Comparison

**Files:**
- Modify: `README.md`
- Create: `docs/project/phase-1-report.md`
- Create: `docs/project/phase-2-report.md`
- Create: `docs/project/example-cases.md`
- Test: report generation via evaluation scripts

- [ ] **Step 1: Add execution instructions and dataset flow to `README.md`**

```md
## Pipeline
1. Build canonical data
2. Build SFT data
3. Train stage-one SFT model
4. Evaluate and calibrate thresholds
5. Build GRPO data
6. Train stage-two GRPO model
```

- [ ] **Step 2: Add a phase-one report template**

```md
# Phase 1 Report

- Dataset summary
- SFT configuration
- Point-level metrics
- Threshold calibration result
- Demo cases
```

- [ ] **Step 3: Add a phase-two comparison report template**

```md
# Phase 2 Report

- Cold-start data summary
- GRPO reward configuration
- Stage-1 vs Stage-2 metric comparison
- Reason quality examples
```

- [ ] **Step 4: Run the core reporting commands once**

Run:
```powershell
python scripts/evaluate_model.py --help
python scripts/calibrate_thresholds.py --help
python scripts/demo_inference.py --help
```

Expected: all three commands print usage successfully

- [ ] **Step 5: Commit**

```bash
git add README.md docs/project/phase-1-report.md docs/project/phase-2-report.md docs/project/example-cases.md
git commit -m "docs: add implementation reports and execution guide"
```
