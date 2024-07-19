#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from alignment.data import is_openai_format
from peft import PeftConfig, PeftModel
from trl import setup_chat_format

from kto_trainer_fg import FGKTOConfig, FGKTOTrainer


logger = logging.getLogger(__name__)


def apply_chat_template(
    example,
    tokenizer,
    auto_insert_empty_system_msg: bool = True,
):
    if all(k in example.keys() for k in ("prompt", "completion", "label")):
        if not is_openai_format(example["prompt"]) or not is_openai_format(example["completion"]):
            raise ValueError(
                f"Could not format example as dialogue for `kto` task! Require OpenAI format for all messages"
            )
        example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False)
        example["completion"] = tokenizer.apply_chat_template(example["completion"], tokenize=False)
    else:
        raise ValueError(
            f"Could not format example as dialogue for `kto` task! Requires the keys `[prompt, completion, label]`"
            f" but found {list(example.keys())} instead."
        )

    return example


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, FGKTOConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["prompt", "completion", "completion_sentences", "label", "sentence_label"],
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    ################
    # Load tokenizer
    ################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################################################
    # Load model (required here to setup the chat format)
    #####################################################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    if is_adapter_model(model_args.model_name_or_path, model_args.model_revision):
        logger.info(f"Loading adapter for {model_args.model_name_or_path=}")

        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )

    # For ChatML we need to add special tokens and resize the embedding layer
    if "<|im_start|>" in tokenizer.chat_template:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # For Phi-3 tokenizer, we need to remove the eos_token from the chat template
    if tokenizer.chat_template == "{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}":
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        desc="Formatting prompt-completion pairs with prompt template",
        load_from_cache_file=False,
    )

    ##########################
    # Decontaminate benchmarks
    ##########################
    # num_raw_train_samples = len(raw_datasets["train"])
    # raw_datasets = raw_datasets.filter(
    #     decontaminate_humaneval,
    #     fn_kwargs={"text_column": "completion"},
    #     batched=True,
    #     batch_size=10_000,
    #     num_proc=1,
    #     desc="Decontaminating HumanEval samples",
    # )
    # num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
    # logger.info(
    #     f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
    # )

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(raw_datasets["train"])), 3):
    #     logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
    #     logger.info(
    #         f"Completion sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['completion']}"
    #     )

    ref_model = None
    if not model_args.use_peft:
        logger.info(f"Loading reference model for {model_args.model_name_or_path=}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )

    ########################
    # Instantiate KTOTrainer
    ########################
    trainer = FGKTOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"] if "test" in raw_datasets else None,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args) if model_args.use_peft else None,
    )

    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval and "test" in raw_datasets:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
