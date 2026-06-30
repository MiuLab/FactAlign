#!/usr/bin/env python3
"""
Full-weight SFT of Gemma-2B on 4× V100 32GB with DeepSpeed ZeRO-2.

Run with:
    torchrun --nproc_per_node=4 scripts/fine_tune.py
"""

#%% ==============================================================================
# 0. Load python modules
# ==============================================================================

import gc
import os
import pathlib

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

# Experiment tracking
import wandb
import mlflow
import mlflow.data
from mlflow.data.huggingface_dataset import from_huggingface  # mlflow >= 2.9

load_dotenv()

#%% ==============================================================================
# 1. Model and tokenizer
# ==============================================================================

MODEL_ID = "google/gemma-2b"
OUTPUT_DIR = "results/models/google-gemma-2b/fine-tuned/v2/"
FINAL_PATH = os.path.join(OUTPUT_DIR, "final_merged")
DEEPSPEED_CONFIG = "configs/ds_config_zero2.json"
TOKENIZER_ID = "google/gemma-2b-it"   # same vocab as gemma-2b, ships with chat template

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
tokenizer.pad_token = "<pad>"      # Gemma has this token; avoids eos/pad aliasing
tokenizer.padding_side = 'right' # todo

# Prompt-masked loss: only assistant tokens contribute to cross-entropy.
# Use token IDs for the response template to avoid string-matching ambiguity.
response_template_ids = tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,      # use fp32 for model weights but  fp16 for optimizer, the setup is more stable.
    use_cache=False,                # Required with gradient checkpointing
)

#%% ==============================================================================
# 2. Dataset
# ==============================================================================

def format_to_chat(example: dict) -> dict:
    messages = []
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        if not messages and role == "assistant":        # skip leading assistant turns
            continue
        if messages and messages[-1]["role"] == role:   # merge consecutive same-role turns
            messages[-1]["content"] += "\n\n" + content
        else:
            messages.append({"role": role, "content": content})
    return {
        "text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    }


dataset = load_dataset("HuggingFaceH4/deita-10k-v0-sft", split="train_sft")
dataset = dataset.map(format_to_chat, remove_columns=dataset.column_names)
dataset = dataset.shuffle(seed=42)

#%% ==============================================================================
# 3. Trainer
# ==============================================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    packing=False,
    max_seq_length=2048,
    dataset_text_field="text",
    learning_rate=2.0e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.0,
    fp16=True,                       # V100 has fp16 tensor cores; NO bf16 on Volta
    seed=42,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    save_only_model=False,
    report_to=["mlflow", "wandb"],
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    deepspeed=DEEPSPEED_CONFIG,      # DeepSpeed manages distributed comms; DDP flags not needed
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

#%% ==============================================================================
# 4. Train
# ==============================================================================
def get_latest_checkpoint(checkpoint_dir: str) -> str | None:
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    return os.path.join(checkpoint_dir, sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1])

latest_checkpoint = get_latest_checkpoint(OUTPUT_DIR)

if int(os.environ.get("LOCAL_RANK", 0)) == 0 and latest_checkpoint:
    print(f"Resuming from checkpoint: {latest_checkpoint}")

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "factalign-fine-tuning"),
        name=os.environ.get("WANDB_RUN_NAME", None),
        config={
            "model_id": MODEL_ID,
            "dataset_source": "HuggingFaceH4/deita-10k-v0-sft",
            "dataset_split": "train_sft",
            "dataset_size": len(dataset),
            "dataset_seed": 42,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2.0e-5,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "max_seq_length": 2048,
            "fp16": True,
            "deepspeed": DEEPSPEED_CONFIG,
        },
    )
    mlflow.start_run()
    mlflow.log_params({
        "dataset_source": "HuggingFaceH4/deita-10k-v0-sft",
        "dataset_split": "train_sft",
        "dataset_size": len(dataset),
        "dataset_seed": 42,
        "model_id": MODEL_ID,
    })

# ALL ranks train
trainer.train(resume_from_checkpoint=latest_checkpoint)

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print("Training complete!")
    mlflow.end_run()
    wandb.finish()

gc.collect()
torch.cuda.empty_cache()

# ALL ranks save (ZeRO collective)
trainer.save_model(FINAL_PATH)
if trainer.is_world_process_zero():
    tokenizer.save_pretrained(FINAL_PATH)
    print(f"Model saved at: {FINAL_PATH}")
