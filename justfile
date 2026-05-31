generate_responses:
    PYTHONPATH=. python long-form-factuality/generate.py \
        "long-form-factuality/longfact/longfact-objects_gpt4_01-12-2024_noduplicates" \
        "google/gemma-2b-it" \
        "results/google/gemma-2b-it_objects_all.json" \
        --backend vllm \
        --temperature 0.5

sft:
    PYTHONPATH=. python scripts/sft/main.py

kto:
    bash train_kto.sh


vllm-gemma-4-31B-it:
    vllm serve google/gemma-4-31B-it \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.85 \
        --default-chat-template-kwargs '{"enable_thinking": false}'