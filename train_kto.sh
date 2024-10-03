#!/bin/bash

export WANDB_PROJECT=FactAlign
export WANDB_LOG_MODEL=false
export WANDB_DISABLED=false
export HF_HOME=$(pwd)/cache
export HF_TOKEN_PATH=${HOME}/.huggingface/token

NUM_GPUS=2

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/deepspeed_zero2.yaml --num_processes ${NUM_GPUS} \
    run_kto.py configs/kto_deepspeed.yaml