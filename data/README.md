# Data Layers

- `raw/`: Immutable source data copied directly from the original materials.
- `canonical/`: Normalized JSONL samples derived from `raw/` for downstream processing.
- `sft/`: Stage-one supervised fine-tuning datasets.
- `grpo/`: Stage-two reinforcement-learning (RL) training sets.
