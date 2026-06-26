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
from trl import SFTConfig, SFTTrainer

load_dotenv()

#%% ==============================================================================
# 1. Model and tokenizer
# ==============================================================================

MODEL_ID = "google/gemma-2b"
OUTPUT_DIR = "results/models/gemma-2b/fine-tuned/"
FINAL_PATH = os.path.join(OUTPUT_DIR, "final_merged")
DEEPSPEED_CONFIG = "configs/ds_config_zero2.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.chat_template is None:
    tokenizer.chat_template = AutoTokenizer.from_pretrained("google/gemma-2b-it").chat_template

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,      # use fp32 for model weights but  fp16 for optimizer, the setup is more stable.
    attn_implementation="sdpa",
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
dataset = dataset.shuffle(seed=42).select(range(2000))

#%% ==============================================================================
# 3. Trainer
# ==============================================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,   # effective batch = 4 GPUs × 2 × 2 = 16
    packing=True,
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
    report_to="none",
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
if latest_checkpoint:
    print(f"Resuming from checkpoint: {latest_checkpoint}")

trainer.train(resume_from_checkpoint=latest_checkpoint)
print("Training complete!")

#%% ==============================================================================
# 5. Save
# ==============================================================================

gc.collect()
torch.cuda.empty_cache()

# trainer.save_model() coordinates across all ZeRO stages correctly
trainer.save_model(FINAL_PATH)
if trainer.is_world_process_zero():
    tokenizer.save_pretrained(FINAL_PATH)
    print(f"Model saved at: {FINAL_PATH}")
