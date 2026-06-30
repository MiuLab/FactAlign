#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=300G
#SBATCH --nodelist=gorina9
#SBATCH --job-name=fine-tune-with-flash
#SBATCH --output=fine-tune-with-flash.out

set -euo pipefail

# uv init 

# CUDA 12.1 toolkit (matches torch 2.2.0 +cu121) so flash-attn can compile its kernels during `uv sync`.
# `uenv` sources /xu/bin/xenv, which isn't `set -u`-clean (unbound var `hr`), so
# relax nounset just around it.
set +u
uenv cuda-12.1.0
set -u
# export CUDA_HOME=/usr/local/_cuda
# export PATH="$CUDA_HOME/bin:$PATH"
echo $CUDA_VISIBLE_DEVICES
nvcc --version
uv sync
# Launch one process per allocated GPU. `gpu` auto-detects the count from
# CUDA_VISIBLE_DEVICES (set by SLURM --gres), so this stays correct regardless
# of how many GPUs the job requests — no more "duplicate GPU" NCCL crashes.
uv run torchrun --nproc_per_node=gpu scripts/fine_tune.py