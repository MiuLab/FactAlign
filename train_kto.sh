#!/bin/bash

export WANDB_PROJECT=FactAlign
export WANDB_LOG_MODEL=false
export WANDB_DISABLED=false
export HF_HOME=$(pwd)/cache
export HF_TOKEN_PATH=${HOME}/.huggingface/token

NUM_GPUS=2

# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file=configs/multi_gpu.yaml --num_processes ${NUM_GPUS} train_kto.py \
#     --model_name_or_path=Columbia-NLP/gemma-2b-zephyr-sft \
#     --per_device_train_batch_size 1 \
#     --num_train_epochs 1.0 \
#     --learning_rate 1e-5 \
#     --lr_scheduler_type=linear \
#     --gradient_accumulation_steps 8 \
#     --gradient_checkpointing True \
#     --optim adafactor \
#     --logging_steps 10 \
#     --eval_steps 500 \
#     --output_dir=kto-aligned-model \
#     --warmup_ratio 0.1 \
#     --report_to wandb \
#     --fp16 \
#     --logging_first_step


ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes ${NUM_GPUS} \
    run_kto.py configs/kto_deepspeed.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes ${NUM_GPUS} \
    run_kto.py configs/kto_deepspeed_2.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes ${NUM_GPUS} \
    run_kto.py configs/kto_deepspeed_3.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes ${NUM_GPUS} \
    run_kto.py configs/kto_deepspeed_4.yaml
