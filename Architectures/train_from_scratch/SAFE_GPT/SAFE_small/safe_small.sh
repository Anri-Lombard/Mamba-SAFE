#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:a100-3g-20gb:1
#SBATCH --time=04:00:00
#SBATCH --job-name="SAFE_small"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

# Set CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(ncvd)

# Load necessary modules
module load python/miniconda3-py310

# Activate virtual environment
source activate architecture_venv

config_path="../trainer/configs/small_config.json"
tokenizer_path="../tokenizer.json"
dataset_path="../../Datasets/MOSES/datasets"
output_dir="./trained/SAFE_small"

safe-train --config $config_path \
  --tokenizer $tokenizer_path \
  --dataset $dataset_path \
  --text_column "SAFE" \
  --torch_compile True \
  --optim "adamw_torch" \
  --learning_rate 5e-4 \
  --prop_loss_coeff 1e-3 \
  --gradient_accumulation_steps 1 \
  --output_dir $output_dir \
  --num_labels 9 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --lr_scheduler_type "linear" \
  --warmup_steps 500 \
  --logging_steps 100 \
  --evaluation_strategy "steps" \
  --eval_steps 500 \
  --save_steps 500 \
  --load_best_model_at_end True \
  --metric_for_best_model "eval_loss" \
  --greater_is_better False

# Deactivate virtual environment
conda deactivate