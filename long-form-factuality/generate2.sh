#!/bin/bash

# Generate params
# EXP_DIR can be a local path or a username on Hugging Face
# EXP_DIR="../fact-align/experiments/Phi-3-mini-4k-instruct"
EXP_DIR="chaoweihuang"
# PREFIX="phi3mini-"
PREFIX="llama_3_8b-"
# SUFFIX="-object_114"
SUFFIX="-object_seen_57"
# DATASET="longfact/object_sampled_114"
DATASET="longfact/object_sampled_seen_57"
TEMPERATURE=0.5

MODELS=(
    "llama3-kto-mix-14k-kto"
    "llama3-kto-mix-14k-lf-response-llama3-f1_100_0.8-fg0.8-kto"
    "llama3-kto-mix-14k-lf-response-llama3-iter2-f1_100_0.8-fg0.5-fgudw4.0-kto-fg"
)

# SAFE params
# MODEL_SHORT="gpt_35_turbo_0125"
MODEL_SHORT="gpt_4o_mini"


# Generate responses
for MODEL in "${MODELS[@]}"; do
    # for TARGETPROB in 0.7 0.8 0.9; do
    #     python3 generate.py \
    #         ${DATASET} \
    #         ${EXP_DIR}/${MODEL} \
    #         results/${PREFIX}${MODEL}${SUFFIX}-prob_${TARGETPROB}.json \
    #         --backend vllm \
    #         --temperature ${TEMPERATURE} \
    #         --target_prob ${TARGETPROB}
    # done
    python3 generate.py \
        ${DATASET} \
        ${EXP_DIR}/${MODEL} \
        results/${PREFIX}${MODEL}${SUFFIX}.json \
        --backend vllm \
        --temperature ${TEMPERATURE}
done

# Eval by SAFE
for MODEL in "${MODELS[@]}"; do
    # for TARGETPROB in 0.7 0.8 0.9; do
    #     python -m eval.run_eval \
    #         --result_path results/${PREFIX}${MODEL}${SUFFIX}-prob_${TARGETPROB}.json \
    #         --parallelize \
    #         --n_shards 1 \
    #         --shard_idx 0 \
    #         --save_path results/${PREFIX}${MODEL}${SUFFIX}-prob_${TARGETPROB}-SAFE.json \
    #         --model_short ${MODEL_SHORT}
    # done
    python -m eval.run_eval \
        --result_path results/${PREFIX}${MODEL}${SUFFIX}.json \
        --parallelize \
        --n_shards 1 \
        --shard_idx 0 \
        --save_path results/${PREFIX}${MODEL}${SUFFIX}-SAFE-gpt4omini.json \
        --model_short ${MODEL_SHORT}
done