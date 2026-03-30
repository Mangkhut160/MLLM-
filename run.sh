#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python train_lora.py \
  --base_image_path "/your/image/path" \
  --output_dir "/your/output/json/dir" \
  --model_path "/your/model/path" \
  --train_json "/your/output/json/dir/train.json" \
  --val_json "/your/output/json/dir/val.json" \
  --batch_size 4 \
  --eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --eval_steps 375 \
  --logging_steps 50 \
  --save_steps 375 \
  --save_total_limit 1 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --output_model_dir "/your/output/model/dir" 