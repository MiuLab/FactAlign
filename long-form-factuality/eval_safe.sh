#!/bin/bash

# MODEL_SHORT="gpt_35_turbo_0125"
MODEL_SHORT="gpt_4o_mini"

FILES=(
    "results/gemma2bsft-object_114.json"
    "results/gemma2bsft-kto-mix-14k-lf-response-f1_0.75-object_38.json"
    "results/gemma2bsft-kto-mix-14k-lf-response-f1_0.75-iter2-object_38.json"
    "results/gemma2bsft-kto-mix-14k-lf-response-f1_0.75-fg-calibration_r12pct-object_38.json"
    # "results/phi3mini-object_114.json"
    # "results/phi3mini-kto-mix-14k-lf-response-phi3-f1_100_0.7-fg1.0-kto-object_114.json"
    # "results/phi3mini-kto-mix-14k-lf-response-phi3-f1_100_0.7-fg1.0-kto-fg-object_114.json"
    # "results/llama_3_8b-object_114.json"
    # "results/llama_3_8b-kto-mix-14k-lf-response-llama3-f1_100_0.8-fg0.8-kto-object-114.json"
    # "results/llama_3_8b-llama3-kto-mix-14k-lf-response-llama3-f1_100_0.8-fg0.5-fgudw4.0-kto-fg-fb0.5-object_114.json"
)

for FILE in "${FILES[@]}"; do
    python -m eval.run_eval \
        --result_path $FILE \
        --parallelize \
        --n_shards 1 \
        --shard_idx 0 \
        --save_path "${FILE%.json}-SAFE-gpt4omini.json" \
        --model_short $MODEL_SHORT
done