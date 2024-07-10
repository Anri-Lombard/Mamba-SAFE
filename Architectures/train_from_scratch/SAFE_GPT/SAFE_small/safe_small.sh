#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=48:00:00
#SBATCH --job-name="SAFE_small"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

# export WANDB_MODE="disabled"
export WANDB_MODE="offline"
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
dataset_path="../../Datasets/MOSES/datasets"
output_dir="/scratch/lmbanr001/SAFE_small"

mkdir -p $output_dir
mkdir -p $wandb_cache_dir

safe-train --config $config_path \
  --tokenizer $tokenizer_path \
  --dataset $dataset_path \
  --text_column "SMILES" \
  --is_tokenized False \
  --streaming True \
  --optim "adamw_torch" \
  --learning_rate 5e-4 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --warmup_steps 2000 \
  --eval_steps 500 \
  --save_steps 500 \
  --num_train_epochs 10 \
  --save_total_limit 1 \
  --prop_loss_coeff 1e-3 \
  --output_dir $output_dir \
  --overwrite_output_dir True \
  --do_train True \
  --do_eval True \
  --save_safetensors True \
  --gradient_checkpointing True \
  --eval_accumulation_steps 100 \
  --max_steps 125_000

# Deactivate virtual environment
conda deactivate
