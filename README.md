FactAlign: Long-form Factuality Alignment of Large Language Models
===

<p align="center">
📃 <a href="https://arxiv.org/abs/2410.01691" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/collections/chaoweihuang/factalign-66fe175ff44983580bff96e0" target="_blank">Models & Datasets</a>
</p>


This repository contains the code, models, and data for our paper **"FactAlign: Long-form Factuality Alignment of Large Language Models"** accepted at **EMNLP 2024 Findings**.
Please cite the following reference if you use the code or models.

```
@inproceedings{huang2024infactalign,
      title={{FactAlign}: Long-form Factuality Alignment of Large Language Models}, 
      author={Chao-Wei Huang and Yun-Nung Chen},
      year={2024},
      booktitle={Findings of the Association for Computational Linguistics: EMNLP 2024 (Findings of EMNLP 2024)}
}
```

![image](https://github.com/user-attachments/assets/98d05042-e684-44c1-b0ad-8ea5ef0f53d6)


## Overview
**FactAlign** is a alignment framework designed to enhance the factuality of LLMs' long-form responses. FactAlign leverages recent advances in automatic factuality assessment to guide the alignment process. Additionally, we introduce **fKTO**, a fine-grained, sentence-level alignment algorithm that extends the Kahneman-Tversky Optimization (KTO) alignment method. 

FactAlign significantly improves the factual accuracy of LLM responses on benchmarks such as [LongFact](https://arxiv.org/abs/2403.18802) and [FactScore](https://aclanthology.org/2023.emnlp-main.741/).


## Install Dependencies

Make a new Python 3.9+ environment using `virtualenv` or `conda`.

```bash
conda create -n fact-align python=3.10
conda activate fact-align
# Install python dependencies. We specify the versions in the requirements.txt file, but newer versions should work generally okay.
pip install -r requirements.txt
```

We also use the [`alignment-handbook`](https://github.com/huggingface/alignment-handbook/tree/main) package for the alignment algorithms. Install it using the following command:

```bash
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .
```

Note that we used [this commit](https://github.com/huggingface/alignment-handbook/commit/a9cf95a) of the `alignment-handbook` package. Newer versions should generally work.


## Data

The datasets we generated for training FactAlign, including the long-form responses and the corresponding factuality assessments, are available in our [Huggingface collection](https://huggingface.co/collections/chaoweihuang/factalign-66fe175ff44983580bff96e0").

In order to generate the datasets, we use the adapted version of [Search-Augmented Factuality Evaluator (SAFE) from Google Deepmind](https://github.com/google-deepmind/long-form-factuality).

Please navigate to the `long-form-factuality` directory and refer to the [README](long-form-factuality/README.md) for more details on how to generate the datasets.


## Training

### Configuration
First, modify the configuration files in the `configs` directory to make sure it fits your local machine.

We used DeepSpeed Zero2 to train `gemma-2b` and `Phi3-Mini` models on 2xV100 32GB, and DeepSpeed Zero3 for training the `LLaMA-3-8B` models on 4xA100 40GB. Please modify the `deepspeed_config_file` path in the `configs/deepspeed_zero*.yaml` files to fit your local machine.

The `configs/kto_*deepspeed.yaml` files are the configurations for training the FactAlign model. You can adjust the hyperparameters in these files.

### Training the FactAlign Model
To train the FactAlign model, run the following command:

```bash
bash train_kto.sh
```

The trained model will be saved in the `output_dir` specified in the configuration file.


## Evaluation
We used the [LongFact](https://arxiv.org/abs/2403.18802) and [FactScore](https://aclanthology.org/2023.emnlp-main.741/) benchmarks to evaluate the performance of FactAlign.

FactAlign significantly improves the factual accuracy of LLM responses on these benchmarks.
![image](https://github.com/user-attachments/assets/07e241fd-a222-485f-b2b2-75b55283b103) 

### LongFact
For LongFact, we used the adapted SAFE evaluation script. Please refer to the [README](long-form-factuality/README.md) for more details.

### FactScore
For FactScore, we used the forked version of the official FactScore evaluation script, which supports up-to-date OpenAI API. Please refer to [their repository](https://github.com/wj210/factscore) for more details.