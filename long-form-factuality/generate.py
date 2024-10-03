import argparse
import glob
import json

import openai

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from common import shared_config


calibration_prompt_template = " Your response should have a factual precision of {prob}% or higher."


# Check for repetitions in the responses
# by checking the n-gram repetitions
def check_repetitions(response, n=10, threshold=5):
    ngram_counts = {}
    tokens = response.split()
    for i in range(len(tokens) - n):
        ngram = tuple(tokens[i:i+n])
        if ngram in ngram_counts:
            ngram_counts[ngram] += 1
        else:
            ngram_counts[ngram] = 1
    
    repetitives = []
    for ngram, count in ngram_counts.items():
        if count > threshold:
            repetitives.append((ngram, count))
    
    return repetitives


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Path to dataset.')
    parser.add_argument('model', type=str, help='Model name.')
    parser.add_argument('output', type=str, help='Output file.')
    parser.add_argument('--target_prob', type=float, default=None, help='Target probability.')
    parser.add_argument('--backend', type=str, default="vllm", choices=["vllm", "openai"])
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    args = parser.parse_args()

    prompts = []
    for f in glob.glob(f"{args.dataset}/*.jsonl"):
        for line in open(f):
            prompt = json.loads(line)["prompt"]
            if shared_config.prompt_postamble not in prompt:
                prompt += " " + shared_config.prompt_postamble
                prompt = prompt.strip()
            
            if args.target_prob is not None:
                prompt += calibration_prompt_template.format(prob=args.target_prob * 100)

            prompts.append(prompt)

    per_prompt_results = []
    if args.backend == "vllm":
        model = LLM(
            args.model,
            dtype="float16",
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=0.95,
            enforce_eager=True,
            download_dir="./cache"
        )

        sampling_params = SamplingParams(
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n=args.num_samples
        )
        print(sampling_params)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        chat_prompts = [
            tokenizer.apply_chat_template(
                [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for prompt in prompts
        ]

        results = model.generate(chat_prompts, sampling_params)

        for prompt, result in zip(prompts, results):
            per_prompt_results.append({
                "prompt": prompt,
                "correct_answers": [],
                "incorrect_answers": [],
                "side1_response": "[PLACEHOLDER RESPONSE]",
                "side2_response": result.outputs[0].text
            })
    elif args.backend == "openai":
        openai.api_key = shared_config.openai_api_key
        for prompt in prompts:
            result = openai.ChatCompletion.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                n=args.num_samples,
            )

            per_prompt_results.append({
                "prompt": prompt,
                "correct_answers": [],
                "incorrect_answers": [],
                "side1_response": "[PLACEHOLDER RESPONSE]",
                "side2_response": result.choices[0].message.content
            })

    results = {
        "add_universal_postamble": "True",
        "max_num_examples": "-1",
        "num_sentences": "1",
        "parallelize": "True",
        "responder_model": f"{args.backend}:{args.model}",
        "responder_model_short": args.model,
        "response_length_postamble": "Respond in exactly 1 sentence.",
        "save_results": "True",
        "shared_config": "<module 'common.shared_config' from '/work/cwhuang0921/long-form-factuality/common/shared_config.py'>",
        "show_responder_prompts": "True",
        "show_responder_responses": "True",
        "shuffle_data": "False",
        "side_1": "placeholder",
        "side_2": "vanilla_prompting",
        "task": args.dataset,
        "task_short": args.dataset,
        "use_length_ablation": "False",
        "per_prompt_data": per_prompt_results
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)

    # Check for repetitions in the responses
    repetition_count = 0
    for result in per_prompt_results:
        repetitions = check_repetitions(result["side2_response"], n=10, threshold=5)
        if repetitions:
            repetition_count += 1
            print(f"Prompt: {result['prompt']}")
            print(f"Response: {result['side2_response']}")

    print(f"Repetitions: {repetition_count} / {len(per_prompt_results)}")