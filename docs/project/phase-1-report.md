# Phase 1 Report

Use this template for the stage-one SFT implementation report.

## Metadata

- Report date:
- Author:
- Commit SHA:
- Base model:
- Stage-one checkpoint:
- Canonical dataset path:
- SFT train path:
- SFT validation path:
- Test reference path:
- Metrics report path:
- Threshold calibration path:

## Scope

Summarize what phase one covers, which artifacts were evaluated, and what "done" means for the stage-one release candidate.

## Dataset Summary

| Item | Value | Notes |
| --- | --- | --- |
| Raw CSV rows |  |  |
| Canonical samples kept |  |  |
| Canonical samples dropped |  |  |
| Train split size |  |  |
| Validation split size |  |  |
| Test split size |  |  |

## SFT Configuration

| Setting | Value | Notes |
| --- | --- | --- |
| Model path |  |  |
| LoRA target modules |  |  |
| LoRA rank |  |  |
| LoRA alpha |  |  |
| LoRA dropout |  |  |
| Max length |  |  |
| Train batch size |  |  |
| Eval batch size |  |  |
| Gradient accumulation steps |  |  |
| Learning rate |  |  |
| Epochs |  |  |
| Seed |  |  |

## Evaluation Results

### Point-Level Metrics

| Metric | brand_check | date_check | voltage_capacity_check | spec_check | image_check |
| --- | --- | --- | --- | --- | --- |
| Accuracy |  |  |  |  |  |
| Precision |  |  |  |  |  |
| Recall |  |  |  |  |  |
| F1 |  |  |  |  |  |

### Final Result Metrics

| Metric | Value | Notes |
| --- | --- | --- |
| Accuracy |  |  |
| Precision |  |  |
| Recall |  |  |
| F1 |  |  |

### Reject-Tag Metrics

| Metric | Value | Notes |
| --- | --- | --- |
| Exact match |  |  |
| Micro precision |  |  |
| Micro recall |  |  |
| Micro F1 |  |  |

## Threshold Calibration

| Key | Threshold | Precision | Recall | Selected Positives | Notes |
| --- | --- | --- | --- | --- | --- |
| brand_check |  |  |  |  |  |
| date_check |  |  |  |  |  |
| voltage_capacity_check |  |  |  |  |  |
| spec_check |  |  |  |  |  |
| image_check |  |  |  |  |  |
| final_result |  |  |  |  |  |

## Example Cases

Reference `docs/project/example-cases.md` and summarize the representative cases used in the review.

| Sample ID | Category | Expected Outcome | Observed Outcome | Decision | Notes |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

## Operational Notes

- Auto-pass policy summary:
- Manual-review trigger summary:
- Known error modes:
- Data quality issues:

## Risks and Follow-Ups

- Risk 1:
- Risk 2:
- Next action:

## Reproduction Commands

```bash
python scripts/build_canonical_dataset.py --output data/canonical/samples.jsonl
python scripts/build_sft_dataset.py
python scripts/train_sft.py --model-path /path/to/model --train-data data/sft/train_sft.jsonl --val-data data/sft/val_sft.jsonl --output-dir outputs/checkpoints/stage_one_sft
python scripts/evaluate_model.py --predictions outputs/eval/stage_one_predictions.jsonl --references data/canonical/test.jsonl --output outputs/eval/stage_one_metrics.json
python scripts/calibrate_thresholds.py --scores outputs/eval/stage_one_scores.jsonl --output outputs/calibration/stage_one_thresholds.json
```
