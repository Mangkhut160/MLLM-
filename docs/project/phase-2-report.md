# Phase 2 Report

Use this template for the stage-two GRPO comparison report.

## Metadata

- Report date:
- Author:
- Commit SHA:
- Stage-one checkpoint:
- Stage-two run config:
- Phase-one metrics report:
- Phase-two metrics report:
- Threshold file(s):

## Comparison Goal

State what phase two is intended to improve over phase one, how success is measured, and whether the comparison is checkpoint-to-checkpoint or preflight-to-baseline.

## Cold-Start GRPO Data Summary

| Item | Value | Notes |
| --- | --- | --- |
| Input source path |  |  |
| Cold-start output path |  |  |
| Total records |  |  |
| Average think length |  |  |
| Max think length |  |  |
| Known exclusions |  |  |

## GRPO Configuration

| Setting | Value | Notes |
| --- | --- | --- |
| Model path |  |  |
| Output dir |  |  |
| Format reward weight |  |  |
| Result reward weight |  |  |
| Tag reward weight |  |  |
| Length reward weight |  |  |
| Max think chars |  |  |
| Max prompt length |  |  |
| Max completion length |  |  |
| Num generations |  |  |
| Learning rate |  |  |
| Seed |  |  |

## Metric Comparison

### Point-Level Metrics

| Metric | Stage 1 | Stage 2 | Delta | Notes |
| --- | --- | --- | --- | --- |
| brand_check F1 |  |  |  |  |
| date_check F1 |  |  |  |  |
| voltage_capacity_check F1 |  |  |  |  |
| spec_check F1 |  |  |  |  |
| image_check F1 |  |  |  |  |

### Final Result Metrics

| Metric | Stage 1 | Stage 2 | Delta | Notes |
| --- | --- | --- | --- | --- |
| Accuracy |  |  |  |  |
| Precision |  |  |  |  |
| Recall |  |  |  |  |
| F1 |  |  |  |  |

### Reject-Tag Metrics

| Metric | Stage 1 | Stage 2 | Delta | Notes |
| --- | --- | --- | --- | --- |
| Exact match |  |  |  |  |
| Micro precision |  |  |  |  |
| Micro recall |  |  |  |  |
| Micro F1 |  |  |  |  |

## Reason Quality Review

| Sample ID | Category | Stage 1 Reasoning | Stage 2 Reasoning | Better Stage | Notes |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

## Example Case Comparison

Reference `docs/project/example-cases.md` and record the cases that best show gains, regressions, or no material change.

| Sample ID | Category | Expected Outcome | Stage 1 Outcome | Stage 2 Outcome | Comparison |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

## Operational Impact

- Auto-pass rate change:
- Manual-review rate change:
- Calibration impact:
- Latency or cost impact:

## Risks and Recommendation

- Primary regression risk:
- Primary quality gain:
- Recommendation:
- Required next validation:

## Reproduction Commands

```bash
python scripts/build_grpo_dataset.py --input data/canonical/samples.jsonl --output data/grpo/cold_start.jsonl
python scripts/train_grpo.py --model-path /path/to/model --train-data data/grpo/cold_start.jsonl --output-dir outputs/checkpoints/stage_two_grpo
python scripts/evaluate_model.py --predictions outputs/eval/stage_two_predictions.jsonl --references data/canonical/test.jsonl --output outputs/eval/stage_two_metrics.json
```
