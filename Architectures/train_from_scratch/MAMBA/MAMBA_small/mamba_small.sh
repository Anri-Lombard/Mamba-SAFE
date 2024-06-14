#!/bin/bash

#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=1:00:00
#SBATCH --job-name="MAMBA_small"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(ncvd)

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
export TMPDIR="/scratch/lmbanr001/tmp"

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

config_path="small_config.json"
dataset_path="../../Datasets/MOSES/datasets"
output_dir="/scratch/lmbanr001/MAMBA_small"

mkdir -p $output_dir
mkdir -p $wandb_cache_dir

python3 train_mamba_small.py --dataset_path $dataset_path \
    --output_dir $output_dir \
    --config_path $config_path \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --num_train_epochs 3 \
    --warmup_steps 1000 \
    --num_train_steps 10000 \
    --save_steps 500 \
    --max_checkpoints 2

# Deactivate virtual environment
conda deactivate