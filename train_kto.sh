#!/bin/bash

export WANDB_PROJECT=factalign-kto
export WANDB_RUN_NAME=gemma-2b-v2-it_1
export WANDB_LOG_MODEL=false
export WANDB_DISABLED=false
export WANDB_API_KEY="..."
export HF_HOME=${HOME}/.cache/huggingface
export HF_TOKEN_PATH=${HOME}/.cache/huggingface/token

NUM_GPUS=4
CUDA_VISIBLE_DEVICES="1,2,6,7"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes ${NUM_GPUS} \
    run_kto.py configs/kto_gemmma_2b.yaml

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/deepspeed_zero4.yaml --num_processes ${NUM_GPUS} \
#     run_kto.py configs/kto_gemma_2b_zephyr_sft_deepspeed.yaml