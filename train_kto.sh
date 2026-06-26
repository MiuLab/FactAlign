#!/bin/bash

export WANDB_PROJECT=FactAlign
export WANDB_LOG_MODEL=false
export WANDB_DISABLED=true
export HF_HOME=${HOME}/.cache/huggingface
export HF_TOKEN_PATH=${HOME}/.cache/huggingface/token

NUM_GPUS=1

ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes ${NUM_GPUS} \
    run_kto.py configs/kto_gemma_2b_zephyr_sft.yaml

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/deepspeed_zero4.yaml --num_processes ${NUM_GPUS} \
#     run_kto.py configs/kto_gemma_2b_zephyr_sft_deepspeed.yaml