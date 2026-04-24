# %%
# 2.1 Load model from huggingface
# ===================================================

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

model_id = "google/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Base models do not have a chat template by default, so we apply the standard Gemma template
tokenizer.chat_template = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}"
    "{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}"
    "{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}"
)

tokenizer.pad_token = tokenizer.eos_token

# from transformers import BitsAndBytesConfig

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,  # nested quantization
# )

# On H100, we skip quantization for better performance and easier merging
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation="flash_attention_2", # H100 loves Flash Attention 2
)

# %%
# 2.2 Test model
# ===================================================

# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# inputs = tokenizer.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	tokenize=True,
# 	return_dict=True,
# 	return_tensors="pt",
# ).to(model.device)

# outputs = model.generate(**inputs, max_new_tokens=40)
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


# %% 
# 3. Load Dataset
# ===================================================

import pandas as pd
from datasets import load_dataset

dataset = load_dataset("hkust-nlp/deita-10k-v0", split="train")

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
        add_generation_prompt=False
    )
    
    return {"text": formatted_string}

# Apply the map function
formatted_dataset = filtered_dataset.map(format_sharegpt_to_gemma, num_proc=4)
formatted_dataset_text = formatted_dataset.select_columns(["text"])
sampled_formatted_dataset_text = formatted_dataset_text.shuffle(seed=42).select(range(50))


# %%
# 4. Initialize Trainer
# ===================================================

from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# peft_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
# )

peft_config = LoraConfig(
    r=32, # Increased rank since you have the memory
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

output_dir = "./results/models/gemma-2b/sft/v1"

training_args = SFTConfig(
    output_dir=output_dir,
    # num_train_epochs=1,
    per_device_train_batch_size=4, # Increased for H100 throughput
    gradient_accumulation_steps=8,
    packing=True,
    gradient_checkpointing=True,
    max_seq_length=2048,
    dataset_text_field="text",
    optim="adamw_torch",        # Native torch optimizer is faster on H100
    bf16=True,                  # Native H100 format
    logging_steps=10,
    save_strategy="steps",      # Save checkpoints at intervals
    save_steps=100,             # Save checkpoint every 100 steps
    save_total_limit=3,         # Keep only last 3 checkpoints to manage disk space
    save_only_model=True,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset_text,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_args,
)

# %%
# 5. Start Supervised Fine-Tuning
# ===================================================

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
# 6. Merge Weights & Save Locally
# ===================================================
# Since we didn't use 4-bit, we can merge IMMEDIATELY.
import gc, torch

gc.collect()
torch.cuda.empty_cache()

print("Merging weights...")

merged_model = trainer.model.merge_and_unload()

# Save the final full model
final_path = os.path.join(output_dir, "final_merged")
merged_model.save_pretrained(final_path, safe_serialization=True)
tokenizer.save_pretrained(final_path)

# Push the full model (not just adapters) to Hub
# merged_model.push_to_hub('Factiverse/gemma-2b-it-sft-merged')

print(f"Success! Merged model saved at: {final_path}")
