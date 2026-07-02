# generate_responses:
#     PYTHONPATH=. python long-form-factuality/generate.py \
#         "long-form-factuality/longfact/longfact-objects_gpt4_01-12-2024_noduplicates" \
#         "google/gemma-2b-it" \
#         "results/google/gemma-2b-it_objects_all.json" \
#         --backend vllm \
#         --temperature 0.5


gen_prompts:
    PYTHONPATH=/home/pratuat/repositories/FactAlign/long-form-factuality python -m data_creation.pipeline

gen_response:
    PYTHONPATH=/home/pratuat/repositories/FactAlign/long-form-factuality python -m main.pipeline

run_eval:
    PYTHONPATH=/home/pratuat/repositories/FactAlign/long-form-factuality python -m eval.run_eval \
        --result_path='results/evals/gemma-2b-sft-v2-2026-07-02-04-38-09-v15.json' \
        --eval_side2=False \
        --parallelize=True

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


long-form-factuality/results/evals/gemma-2b-sft-v2-2026-07-02-04-38-09-v15.json


PYTHONPATH=. python -m eval.run_eval \
    --result_path='results/evals/gemma-2b-sft-v2-2026-07-02-04-38-09-v15.json' \
    --eval_side2=False \
    --parallelize=True

PYTHONPATH=. python -m eval.run_eval \
    --result_path='results/evals/gemma-2b-kto-v2-it_1-2026-07-02-05-01-38-v15.json' \
    --eval_side2=False \
    --parallelize=True