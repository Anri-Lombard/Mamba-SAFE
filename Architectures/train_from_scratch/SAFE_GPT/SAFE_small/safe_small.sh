#!/bin/bash

#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-3g-20gb:1
#SBATCH --time=48:00:00
#SBATCH --job-name="SAFE_small"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(ncvd)

# Set your wandb API key
export WANDB_API_KEY="d68cdf1a94da49b43fbfb7fd90246c39d7c34237"
export WANDB_MODE="offline"
wandb_cache_dir="/scratch/lmbanr001/wandb_cache"
wandb_dir="/scratch/lmbanr001/wandb"
export WANDB_CACHE_DIR=$wandb_cache_dir # Just in case
export WANDB_DIR=$wandb_dir
mkdir -p $wandb_cache_dir
mkdir -p $wandb_dir

# Set the TMPDIR environment variable
export TMPDIR="/scratch/lmbanr001/tmp"
mkdir -p $TMPDIR

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
  --text_column "SAFE" \
  --torch_compile True \
  --optim "adamw_torch" \
  --learning_rate 5e-4 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --eval_steps 500 \
  --save_steps 500 \
  --save_total_limit 2 \
  --prop_loss_coeff 1e-3 \
  --output_dir $output_dir \
  --overwrite_output_dir True \
  --do_train True

# Remove the tmp directory for the next run
rm -rf $TMPDIR

# Deactivate virtual environment
conda deactivate