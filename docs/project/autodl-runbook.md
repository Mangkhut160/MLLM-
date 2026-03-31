# AutoDL 运行清单

## 目标

这份清单用于在 AutoDL 环境中完成以下事情：

1. 准备 `/root/autodl-tmp` 目录结构
2. 放置或检查 `Qwen2.5-VL-7B-Instruct` 模型
3. 准备 Stage-1 / Stage-2 所需数据
4. 运行 Stage-1 SFT
5. 运行 Stage-2 GRPO 预检

当前仓库中的训练脚本默认路径已经固定为：

- 模型目录：`/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct`
- Stage-1 训练数据：`/root/autodl-tmp/battery-audit/data/sft/train_sft.jsonl`
- Stage-1 验证数据：`/root/autodl-tmp/battery-audit/data/sft/val_sft.jsonl`
- Stage-1 输出目录：`/root/autodl-tmp/battery-audit/output/stage_one_sft`
- Stage-2 训练数据：`/root/autodl-tmp/battery-audit/data/grpo/cold_start.jsonl`
- Stage-2 输出目录：`/root/autodl-tmp/battery-audit/output/grpo`

## 1. 准备目录

```bash
mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/battery-audit/data/sft
mkdir -p /root/autodl-tmp/battery-audit/data/grpo
mkdir -p /root/autodl-tmp/battery-audit/output/stage_one_sft
mkdir -p /root/autodl-tmp/battery-audit/output/grpo
```

## 2. 准备模型

确保 `Qwen2.5-VL-7B-Instruct` 最终位于：

```bash
/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct
```

如果你通过 Hugging Face 下载模型，最终也要整理到上面的目录。

可先快速检查目录：

```bash
ls -la /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct
```

## 3. 准备数据

如果仓库已经在 AutoDL 中，可以直接把当前仓库内生成好的 JSONL 文件复制过去：

```bash
cp data/sft/train_sft.jsonl /root/autodl-tmp/battery-audit/data/sft/train_sft.jsonl
cp data/sft/val_sft.jsonl /root/autodl-tmp/battery-audit/data/sft/val_sft.jsonl
cp data/grpo/cold_start.jsonl /root/autodl-tmp/battery-audit/data/grpo/cold_start.jsonl
cp data/canonical/test.jsonl /root/autodl-tmp/battery-audit/data/test.jsonl
```

也可以不复制，直接在命令行里覆盖脚本默认参数，指向仓库内的路径。

## 4. 安装并验证依赖

先在 AutoDL 环境里安装项目依赖。当前脚本至少依赖：

- `torch`
- `transformers`
- `datasets`
- `peft`
- `qwen-vl-utils`

示例：

```bash
pip install torch transformers datasets peft accelerate sentencepiece qwen-vl-utils
```

安装后先做 import 检查：

```bash
python - <<'PY'
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
print("imports ok")
PY
```

如果这里失败，优先解决依赖版本问题，再继续训练。

## 5. Stage-1 SFT 训练前检查

先确认数据存在：

```bash
ls -lh /root/autodl-tmp/battery-audit/data/sft/train_sft.jsonl
ls -lh /root/autodl-tmp/battery-audit/data/sft/val_sft.jsonl
```

再确认训练脚本帮助信息正常：

```bash
python scripts/train_sft.py --help
```

## 6. Stage-1 SFT 训练命令

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_sft.py \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
  --train-data /root/autodl-tmp/battery-audit/data/sft/train_sft.jsonl \
  --val-data /root/autodl-tmp/battery-audit/data/sft/val_sft.jsonl \
  --output-dir /root/autodl-tmp/battery-audit/output/stage_one_sft \
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
```

如果你的显存更小，可以进一步降低：

- `--train-batch-size`
- `--eval-batch-size`
- 增大 `--gradient-accumulation-steps`

## 7. Stage-2 GRPO 预检

先确认 cold-start 数据存在：

```bash
ls -lh /root/autodl-tmp/battery-audit/data/grpo/cold_start.jsonl
```

再检查脚本帮助：

```bash
python scripts/train_grpo.py --help
```

当前 `train_grpo.py` 已经能做参数与数据预检，但还不是完整 RL trainer。可以先跑：

```bash
python scripts/train_grpo.py \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
  --train-data /root/autodl-tmp/battery-audit/data/grpo/cold_start.jsonl \
  --output-dir /root/autodl-tmp/battery-audit/output/grpo \
  --dry-run
```

这一步的目的，是确认：

- 模型路径
- 冷启动数据路径
- 奖励权重参数
- 输出目录

都已经对齐。

## 8. 常见问题

### 1. `Qwen2_5_VLForConditionalGeneration` 导入失败

说明当前 `transformers` 版本不支持该模型类。先升级或调整到支持该导入的版本，再继续。

### 2. 找不到 `/root/autodl-tmp/...` 数据

说明你还没把 JSONL 文件复制到脚本默认位置。要么复制到默认目录，要么显式传参覆盖 `--train-data` / `--val-data` / `--output-dir`。

### 3. 显存不够

优先调整：

- `--train-batch-size`
- `--eval-batch-size`
- `--gradient-accumulation-steps`
- GPU 数量

### 4. `flash_attention_2` 报错

可以在训练命令里临时覆盖：

```bash
--attn-implementation eager
```

## 9. 推荐顺序

实际执行建议按这个顺序：

1. 安装依赖并通过 import 检查
2. 准备 `/root/autodl-tmp` 数据和模型
3. 运行 `python scripts/train_sft.py --help`
4. 运行 Stage-1 SFT
5. 检查 Stage-1 输出目录
6. 运行 `python scripts/train_grpo.py --dry-run`
7. 再考虑补完整的 Stage-2 RL trainer
