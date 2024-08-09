#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:ampere80:1
#SBATCH --time=48:00:00
#SBATCH --job-name="SAFE_20M_validated_v2"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

# export WANDB_MODE="disabled"
# export WANDB_MODE="offline"
export WANDB_API_KEY="d68cdf1a94da49b43fbfb7fd90246c39d7c34237"
# Set the wandb cache and directory paths
wandb_cache_dir="/scratch/lmbanr001/wandb_cache"
wandb_dir="/scratch/lmbanr001/wandb"

# Check if the wandb cache directory exists
if [ -d "$wandb_cache_dir" ]; then
  # If it exists, remove all its contents
  rm -rf "$wandb_cache_dir"/*
else
  # If it doesn't exist, create the directory
  mkdir -p "$wandb_cache_dir"
fi

# Check if the wandb directory exists
if [ -d "$wandb_dir" ]; then
  # If it exists, remove all its contents
  rm -rf "$wandb_dir"/*
else
  # If it doesn't exist, create the directory
  mkdir -p "$wandb_dir"
fi

# Set the wandb environment variables
export WANDB_CACHE_DIR=$wandb_cache_dir
export WANDB_DIR=$wandb_dir

# Set the TMPDIR environment variable
export TMPDIR="/scratch/lmbanr001/tmp_safe_small"

# Check if the TMPDIR directory exists
if [ -d "$TMPDIR" ]; then
  # If it exists, remove all its contents
  rm -rf "$TMPDIR"/*
else
  # If it doesn't exist, create the directory
  mkdir -p "$TMPDIR"
fi

# Load necessary modules
module load python/miniconda3-py310

# Activate virtual environment
source activate architecture_venv

config_path="../trainer/configs/small_config.json"
tokenizer_path="../tokenizer.json"
dataset_path="/scratch/lmbanr001/Datasets/MOSES/safe_datasets"
# dataset_path="sagawa/ZINC-canonicalized"
output_dir="/scratch/lmbanr001/SAFE_20M_validated_v2"
# checkpoint_path="/scratch/lmbanr001/SAFE_20M_validated/checkpoint-213000"
# config_path="/scratch/lmbanr001/SAFE_20M_validated/checkpoint-213000/config.json"

mkdir -p $output_dir
mkdir -p $wandb_cache_dir

safe-train --config $config_path \
  --tokenizer $tokenizer_path \
  --dataset $dataset_path \
  --text_column "SAFE" \
  --optim "adamw_torch" \
  --learning_rate 5e-4 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --report_to "wandb" \
  --warmup_steps 20000 \
  --logging_first_step True \
  --logging_steps 100 \
  --eval_accumulation_steps 1000 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --eval_strategy "steps" \
  --wandb_project "SAFE_small" \
  --num_train_epochs 10 \
  --save_total_limit 1 \
  --output_dir $output_dir \
  --overwrite_output_dir True \
  --do_train True \
  --do_eval True \
  --save_safetensors True \
  --gradient_checkpointing True \
  --num_train_epochs 10 \
  --prediction_loss_only True \
  --max_grad_norm 1.0 \
  --weight_decay 0.1 \
  --include_descriptors False

# Deactivate virtual environment
conda deactivate
