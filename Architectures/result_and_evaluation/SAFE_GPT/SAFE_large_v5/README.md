---
base_model: trainer/configs/large_config.json
tags:
- safe
- datamol-io
- molecule-design
- smiles
- generated_from_trainer
datasets:
- sagawa/ZINC-canonicalized
model-index:
- name: SAFE_large_v5
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](None)
# SAFE_large_v5

This model is a fine-tuned version of [trainer/configs/large_config.json](https://huggingface.co/trainer/configs/large_config.json) on the sagawa/ZINC-canonicalized dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-06
- train_batch_size: 32
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 10000
- training_steps: 50000

### Training results



### Framework versions

- Transformers 4.42.3
- Pytorch 2.3.1+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1
