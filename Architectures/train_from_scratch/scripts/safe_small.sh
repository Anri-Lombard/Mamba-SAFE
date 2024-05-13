#!/bin/bash

config_path="../SAFE_GPT/trainer/configs/small_config.json"
# model_path="../SAFE_GPT/trainer/model.py"
tokenizer_path="../SAFE_GPT/tokenizer.json"
dataset_path="katielink/moses"
output_dir="../trained/SAFE-small"

safe-train --config $config_path \
    --tokenizer $tokenizer_path \
    --dataset $dataset_path \
    --num_labels 9 \
    --torch_compile True \
    --optim "adamw_torch" \
    --learning_rate 5e-4 \
    --prop_loss_coeff 1e-3 \
    --gradient_accumulation_steps 1 \
    --output_dir $output_dir \
    --max_steps 5