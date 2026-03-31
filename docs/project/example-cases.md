# Example Cases

Use this document to curate representative examples for phase-one reporting and phase-two comparison work.

## Recommended Categories

| Category | Why It Matters | Typical Expected Outcome |
| --- | --- | --- |
| Clean pass | Verifies the happy path and auto-pass behavior. | All points pass and the final result passes. |
| Brand mismatch | Confirms brand-image versus form consistency checks. | `brand_check` fails and manual review is triggered. |
| Date mismatch | Confirms production-date extraction and date validation. | `date_check` fails and manual review is triggered. |
| Voltage or capacity mismatch | Confirms numeric/spec parsing against the form. | `voltage_capacity_check` fails. |
| Spec non-compliance | Covers batteries that should fail the spec rule itself. | `spec_check` fails. |
| Image content issue | Covers non-battery or non-compliant images. | `image_check` fails. |
| Multiple simultaneous issues | Tests multi-tag output and decision aggregation. | More than one point fails. |
| Borderline or ambiguous sample | Captures uncertain cases where thresholds or policy matter most. | Expected outcome should be called out explicitly. |
| Formatting or parsing robustness | Ensures structured outputs still parse cleanly. | Decision remains stable after parsing. |
| Reason-quality showcase | Useful for stage-one versus stage-two qualitative comparisons. | Same decision, better explanation quality. |

## How to Record a Case

1. Pick one sample per row in the summary table below.
2. Keep the raw `sample_id` and the canonical file path so the case can be reproduced.
3. Record the ground-truth `point_results`, `reject_tags`, and `final_result` before comparing any model output.
4. Save the exact stage-one and stage-two structured outputs used in the review.
5. Note whether the case is a gain, regression, or neutral comparison.
6. When thresholds affect the final online decision, record the threshold file version used in the review.

## Summary Table

| Sample ID | Category | Split | Expected Outcome | Stage 1 Outcome | Stage 2 Outcome | Notes |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

## Detailed Case Template

### Case: `<sample_id>`

- Category:
- Dataset split:
- Canonical record path:
- Source image paths:
- Input form summary:
- Ground-truth point results:
- Ground-truth reject tags:
- Ground-truth final result:
- Stage 1 structured response:
- Stage 1 decision:
- Stage 2 structured response:
- Stage 2 decision:
- Threshold file used:
- Why this case matters:
- Reviewer notes:

## Review Tips

- Prefer cases that cover each audit point at least once.
- Keep a small set of high-signal examples instead of a long unsorted dump.
- Use the same cases in both reports so stage-one and stage-two comparisons stay consistent.
- If stage two is not available yet, fill the phase-one fields now and leave the comparison fields blank until the next run.
