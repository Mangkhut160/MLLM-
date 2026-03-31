# Battery Audit Prototype

## Source of Truth

- `data.csv`
- `brand_new/`
- `charge_new/`

Design documents and implementation plans live under `docs/superpowers/`.
Project reports and reporting templates live under `docs/project/`.

## Data Layers

- `data/canonical/`: normalized JSONL records built from the CSV plus paired images
- `data/sft/`: stage-one supervised fine-tuning datasets
- `data/grpo/`: stage-two GRPO cold-start datasets
- `outputs/`: suggested local destination for checkpoints, metrics, and calibration artifacts

## Pipeline

1. Build canonical data
2. Build SFT data
3. Train stage-one SFT model
4. Evaluate and calibrate thresholds
5. Build GRPO data
6. Train stage-two GRPO model

## Dataset Flow

`data.csv` + `brand_new/` + `charge_new/`
-> `data/canonical/samples.jsonl`
-> `data/sft/train_sft.jsonl` + `data/sft/val_sft.jsonl` + `data/canonical/test.jsonl`
-> stage-one predictions, metrics, and threshold-calibration inputs under `outputs/`
-> `data/grpo/cold_start.jsonl`
-> stage-two GRPO run metadata and future checkpoints

## Execution Guide

### 1. Build canonical data

```bash
python scripts/build_canonical_dataset.py \
  --csv data.csv \
  --brand-dir brand_new/brand_new \
  --spec-dir charge_new/charge_new \
  --output data/canonical/samples.jsonl
```

This step normalizes the raw CSV rows, resolves paired image paths, and writes canonical JSONL samples for all downstream tasks.

### 2. Build SFT data

```bash
python scripts/build_sft_dataset.py \
  --input data/canonical/samples.jsonl \
  --train-output data/sft/train_sft.jsonl \
  --val-output data/sft/val_sft.jsonl \
  --test-output data/canonical/test.jsonl
```

This step creates deterministic train and validation SFT records plus a held-out canonical test split.

### 3. Train stage-one SFT model

```bash
python scripts/train_sft.py \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --train-data data/sft/train_sft.jsonl \
  --val-data data/sft/val_sft.jsonl \
  --output-dir outputs/checkpoints/stage_one_sft
```

Override `--model-path` with your local base model. The repository-relative dataset paths above avoid the AutoDL defaults baked into the CLI.

### 4. Evaluate and calibrate thresholds

Evaluate structured predictions against the held-out references:

```bash
python scripts/evaluate_model.py \
  --predictions outputs/eval/stage_one_predictions.jsonl \
  --references data/canonical/test.jsonl \
  --output outputs/eval/stage_one_metrics.json
```

Calibrate thresholds from a scored JSONL file containing `reference`, `point_scores`, and `final_score` fields:

```bash
python scripts/calibrate_thresholds.py \
  --scores outputs/eval/stage_one_scores.jsonl \
  --output outputs/calibration/stage_one_thresholds.json
```

### 5. Build GRPO data

```bash
python scripts/build_grpo_dataset.py \
  --input data/canonical/samples.jsonl \
  --output data/grpo/cold_start.jsonl
```

This produces short-reason cold-start records for the stage-two GRPO phase.

### 6. Train stage-two GRPO model

```bash
python scripts/train_grpo.py \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --train-data data/grpo/cold_start.jsonl \
  --output-dir outputs/checkpoints/stage_two_grpo
```

At the current implementation stage, `scripts/train_grpo.py` validates the GRPO dataset and writes run metadata for the phase-two configuration.

## Reporting and Case Review

- Use `docs/project/phase-1-report.md` for the stage-one implementation report.
- Use `docs/project/phase-2-report.md` for the stage-one versus stage-two comparison report.
- Use `docs/project/example-cases.md` to curate representative examples and record per-case observations.
- Use `docs/project/autodl-runbook.md` for the AutoDL environment setup, `/root/autodl-tmp` layout, and stage-one/stage-two run commands.

## Demo Inference

Inspect one canonical sample with either a provided response payload or the stored reference output:

```bash
python scripts/demo_inference.py \
  --samples data/canonical/samples.jsonl \
  --sample-id <sample_id>
```
