#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python scripts/train_sft.py \
  --model-path "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct" \
  --train-data "/root/autodl-tmp/battery-audit/data/sft/train_sft.jsonl" \
  --val-data "/root/autodl-tmp/battery-audit/data/sft/val_sft.jsonl" \
  --output-dir "/root/autodl-tmp/battery-audit/output/stage_one_sft" \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --num-train-epochs 1 \
  --eval-steps 200 \
  --logging-steps 10 \
  --save-steps 200 \
  --save-total-limit 2 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05
