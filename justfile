# generate_responses:
#     PYTHONPATH=. python long-form-factuality/generate.py \
#         "long-form-factuality/longfact/longfact-objects_gpt4_01-12-2024_noduplicates" \
#         "google/gemma-2b-it" \
#         "results/google/gemma-2b-it_objects_all.json" \
#         --backend vllm \
#         --temperature 0.5

sft:
    PYTHONPATH=. python scripts/sft/main.py

kto:
    bash train_kto.sh

vllm_gemma:
    vllm serve google/gemma-2b-it --gpu-memory-utilization 0.85

vllm_llama:
    vllm serve "meta-llama/Llama-3.1-8B-Instruct" --gpu-memory-utilization 0.85

slurm-fine-tune:
    cd fine-tuning && sbatch slurm/slurm_fine_tune.sh