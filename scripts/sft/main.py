# ==============================================================================
# %%
# 1. Load model and tokenizer
# ==============================================================================
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

model_id = "google/gemma-2b"
output_dir = "./results/models/gemma-2b/sft/v1"
final_path = os.path.join(output_dir, "final_merged")

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2", # H100 loves Flash Attention 2
    use_cache=False,
)

if tokenizer.chat_template is None:
    tokenizer.chat_template = AutoTokenizer.from_pretrained(
        "google/gemma-2b-it"
    ).chat_template

tokenizer.save_pretrained(final_path)

# ==============================================================================
# %%
# 2. Load Dataset
# ==============================================================================

import pandas as pd
from datasets import load_dataset

# dataset = load_dataset("hkust-nlp/deita-10k-v0", split="train")
dataset = load_dataset("HuggingFaceH4/deita-10k-v0-sft", split="train")

filtered_dataset = dataset.filter(lambda x: pd.DataFrame(x['conversations'])['from'].iloc[0] == "human")

def format_sharegpt_to_gemma(example):
    raw_messages = example["conversations"]
    cleaned_messages = []
    
    for msg in raw_messages:
        # Standardize roles if necessary (e.g., from ShareGPT 'human'/'gpt' to 'user'/'assistant')
        role = msg["from"]
        if role == "human": role = "user"
        if role == "gpt": role = "assistant"
        
        content = msg["value"]

        # 1. Skip leading assistant messages (conversations must start with a user)
        if not cleaned_messages and role == "assistant":
            continue
            
        # 2. Merge consecutive messages from the same role
        if cleaned_messages and cleaned_messages[-1]["role"] == role:
            cleaned_messages[-1]["content"] += "\n\n" + content
        else:
            # 3. Add alternating message
            cleaned_messages.append({"role": role, "content": content})

    # Optional: If the last message is from the user and you only want complete pairs, 
    # you might want to drop it. Usually, it's fine for fine-tuning to leave it.
    
    # Now apply the chat template to the safely cleaned messages
    formatted_string = tokenizer.apply_chat_template(
        cleaned_messages, 
        tokenize=False,
        add_generation_prompt=False,
    )
    
    return {"text": formatted_string}

# Apply the map function
formatted_dataset = filtered_dataset.map(format_sharegpt_to_gemma, num_proc=4)
formatted_dataset_text = formatted_dataset.select_columns(["text"])
sampled_formatted_dataset_text = formatted_dataset_text.shuffle(seed=42).select(range(500))

# ==============================================================================
# %%
# 3. Initialize Trainer
# ==============================================================================

from trl import SFTTrainer, SFTConfig


training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=8, # Increased for H100 throughput
    gradient_accumulation_steps=64,
    packing=True,
    max_seq_length=2048,
    dataset_text_field="text",
    learning_rate=2.0e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.0,
    optim="adamw_torch",        # Native torch optimizer is faster on H100
    bf16=True,                  # Native H100 format
    tf32=True,
    seed=42,
    logging_steps=10,
    save_strategy="steps",      # Save checkpoints at intervals
    save_steps=100,             # Save checkpoint every 100 steps
    save_total_limit=3,         # Keep only last 3 checkpoints to manage disk space
    save_only_model=False,
    report_to="none",
    gradient_checkpointing=True,
    dataset_kwargs={
        "add_special_tokens": False,   # template already added <bos>
        "append_concat_token": False,  # don't auto-append EOS between packed seqs
    },
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=sampled_formatted_dataset_text,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True
)


# %%
# ==============================================================================
# 4. Start Supervised Fine-Tuning
# ==============================================================================

# Check for latest checkpoint to resume training
import os
from pathlib import Path

def get_latest_checkpoint(checkpoint_dir):
    """Get the latest checkpoint directory if it exists."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None

    # Sort by step number
    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
    return os.path.join(checkpoint_dir, latest)

latest_checkpoint = get_latest_checkpoint(output_dir)
if latest_checkpoint:
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("Starting training from scratch")
    trainer.train()

print("Training complete!")

# %%
# ==============================================================================
# 5. Save Final Model Locally
# ==============================================================================

import gc, torch

gc.collect()
torch.cuda.empty_cache()

print("Saving final model...")

# Save the final full model
trainer.model.save_pretrained(final_path, safe_serialization=True)
tokenizer.save_pretrained(final_path)

# Push the full model to Hub
# trainer.model.push_to_hub('Factiverse/gemma-2b-it-sft-merged')

print(f"Success! Model saved at: {final_path}")

