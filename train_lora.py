import json
import os
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from datasets import load_dataset
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from swanlab.integration.huggingface import SwanLabCallback
import swanlab
import argparse


# ========== argparse 参数解析 ==========
parser = argparse.ArgumentParser()
parser.add_argument('--base_image_path', type=str, default="/root/autodl-tmp/qwen/dataset/images")
parser.add_argument('--output_dir', type=str, default="/root/autodl-tmp/qwen/output/output_dataset")
parser.add_argument('--model_path', type=str, default="/root/autodl-tmp/qwen/qwen2.5-vl-7b")
parser.add_argument('--train_json', type=str, default="/root/autodl-tmp/qwen/output/output_dataset/train.json")
parser.add_argument('--val_json', type=str, default="/root/autodl-tmp/qwen/output/output_dataset/val.json")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--eval_batch_size', type=int, default=4)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--num_train_epochs', type=int, default=1)
parser.add_argument('--eval_steps', type=int, default=375)
parser.add_argument('--logging_steps', type=int, default=50)
parser.add_argument('--save_steps', type=int, default=375)
parser.add_argument('--save_total_limit', type=int, default=1)
parser.add_argument('--lora_r', type=int, default=8)
parser.add_argument('--lora_alpha', type=int, default=16)
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--output_model_dir', type=str, default="/root/autodl-tmp/qwen/output/Qwen2.5-VL-LoRA")
args = parser.parse_args()

# ========== 用参数替换原有硬编码 ===========
base_image_path = args.base_image_path
output_dir = args.output_dir


# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

def clean_caption(text):
    """移除词性标注并重组句子"""
    words = []
    for word in text.strip().split():
        if ":" in word:  # 处理带标注的词
            words.append(word.split(":", 1)[0])
        else:            # 处理无标注的词（兼容旧格式）
            words.append(word)
    return "".join(words)  # 中文无需空格

def process_split(split_name):
    """处理单个数据集"""
    input_file = f"{split_name}.txt"
    output_file = os.path.join(output_dir, f"{split_name}.json")  # 修改输出文件名
    
    conversations = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 计算前1/5数据量
    split_lines = len(lines)
    # split_lines = len(lines)
    if split_lines == 0:
        split_lines = 1  # 保证至少处理1条数据
    
    print(f"\n{split_name.upper()} 总样本量：{len(lines)}，本次处理前 {split_lines} 条")
    
    for idx, line in tqdm(enumerate(lines[:split_lines]), total=split_lines, desc=f"处理 {split_name}"):
        # 分割图片ID和描述
        try:
            img_part, caption = line.strip().split(" ", 1)
        except:
            print(f"跳过无效行：{line}")
            continue
        
        # 提取图片ID（去除#zhm#0后缀）
        image_id = img_part.split("#")[0]
        
        # 生成图片绝对路径
        image_path = os.path.abspath(
            os.path.join(base_image_path, f"{image_id}.jpg")
        )
        
        # 清洗描述文本
        clean_text = clean_caption(caption)
        
        # 构建对话数据
        conversations.append({
            "id": f"{split_name}_{idx+1}",
            "conversations": [
                {
                    "from": "user",
                    "value": f"<|vision_start|>{image_path}<|vision_end|>"
                },
                {
                    "from": "assistant",
                    "value": clean_text
                }
            ]
        })
    
    # 保存JSON文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    print(f"✅ {split_name.upper()} 文件已生成：{output_file}")

# 处理所有数据集分割
for split in ["train", "val", "test"]:
    process_split(split)

print("\n数据集准备完成！")


def process_func(example):
    """
    预处理输入数据
    """
    MAX_LENGTH = 8192
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    # 构造多模态对话
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"{file_path}", "resized_height": 256, "resized_width": 256},
                {"type": "text", "text": "你是一位擅长图像描述的专家，请你用简洁的中文描述图片内容，使用'动词+形容词+名词'的结构，例如:一位穿蓝色衬衫的金发女士正在招手叫出租车"},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # 构造目标输出
    response = tokenizer(f"{output_content}", add_special_tokens=False)
    input_ids = inputs["input_ids"][0].tolist() + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0].tolist() + response["attention_mask"] + [1]
    labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]
    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    # 确保图像数据维度正确
    pixel_values = inputs["pixel_values"]
    if isinstance(pixel_values, torch.Tensor):
        if len(pixel_values.shape) == 4:  # [batch_size, channels, height, width]
            pixel_values = pixel_values.squeeze(0)  # 移除batch维度
    else:
        pixel_values = torch.tensor(pixel_values)
        if len(pixel_values.shape) == 4:
            pixel_values = pixel_values.squeeze(0)
    
    # 确保image_grid_thw维度正确
    image_grid_thw = inputs["image_grid_thw"]
    if isinstance(image_grid_thw, torch.Tensor):
        if len(image_grid_thw.shape) > 1:
            image_grid_thw = image_grid_thw.squeeze(0)
    else:
        image_grid_thw = torch.tensor(image_grid_thw)
        if len(image_grid_thw.shape) > 1:
            image_grid_thw = image_grid_thw.squeeze(0)
    
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw
    }

 
# 加载 Qwen2.5-VL-7B-Instruct 模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

# 加载 tokenizer 和 processor，确保左侧填充
tokenizer = AutoTokenizer.from_pretrained(
    args.model_path,
    padding_side='left',
)
processor = AutoProcessor.from_pretrained(
    args.model_path,
    tokenizer=tokenizer,
    tokenizer_padding_side='left',
)
 
# 允许梯度更新
model.enable_input_require_grads()

train_ds = Dataset.from_json(args.train_json)
val_ds = Dataset.from_json(args.val_json)


# 应用预处理函数
train_dataset = train_ds.map(process_func, batched=False)
val_dataset = val_ds.map(process_func, batched=False)


 
 # lora配置
config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    init_lora_weights="gaussian"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将 LoRA 应用于模型
peft_model = get_peft_model(model, config)
peft_model = peft_model.to(device)

def freeze_vision(model):
    """冻结所有视觉相关参数"""
    for name, param in model.named_parameters():
        if "visual.merger" in name:# 特征融合层保持可训练
            param.requires_grad = True
        elif "visual" in name:
            param.requires_grad = False
freeze_vision(peft_model)



swanlab_callback = SwanLabCallback(
    project="Qwen2.5-fintune",
    experiment_name="Qwen2.5-3B-VL(all data)",
    description="使用通义千问Qwen2.5-VL模型内容审核微调。",
    config={
        "model": args.model_path,
        "peft_config": {
            "lora_rank": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        },
        # 训练参数
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.num_train_epochs
    },
     dashboard={
        "layout": [
            # 第一行显示训练指标
            [
                {"name": "loss", "type": "line", "goal": "minimize"},
                {"name": "rougeL", "type": "line", "goal": "maximize"}
            ],
            # 第二行显示验证指标
            [
                {"name": "bleu4", "type": "bar", "goal": "maximize"},
                {"name": "cider", "type": "line", "goal": "maximize"}
            ]
        ] 
            
    }
)

# 首先安装必要的评估库
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
import numpy as np

args_trainer = TrainingArguments(
    output_dir=args.output_model_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    evaluation_strategy="steps",
    eval_steps=args.eval_steps,
    logging_steps=args.logging_steps,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    optim="adamw_torch",
    bf16=True,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
)


# 开始训练
    
trainer = Trainer(
    model=peft_model,
    args=args_trainer,
    train_dataset=train_dataset,  
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)
 
trainer.train()