---
base_model: ../trainer/configs/small_config.json
tags:
- safe
- datamol-io
- molecule-design
- smiles
- generated_from_trainer
datasets:
- ../../Datasets/MOSES/datasets
model-index:
- name: SAFE_small
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# SAFE_small

This model is a fine-tuned version of [../trainer/configs/small_config.json](https://huggingface.co/../trainer/configs/small_config.json) on the ../../Datasets/MOSES/datasets dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0005
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 128
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10.0

### Training results



### Framework versions

- Transformers 4.40.2
- Pytorch 2.5.0.dev20240621+cu121
- Datasets 2.19.1
- Tokenizers 0.19.1
