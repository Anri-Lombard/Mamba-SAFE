#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=48:00:00
#SBATCH --job-name="MAMBA_small"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

# Set up wandb
export WANDB_MODE="offline"
export WANDB_API_KEY="d68cdf1a94da49b43fbfb7fd90246c39d7c34237"
wandb_cache_dir="/scratch/lmbanr001/wandb_cache"
wandb_dir="/scratch/lmbanr001/wandb"

# Clean up and create directories
for dir in "$wandb_cache_dir" "$wandb_dir"; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"/*
    else
        mkdir -p "$dir"
    fi
done

export WANDB_CACHE_DIR=$wandb_cache_dir
export WANDB_DIR=$wandb_dir

# Set up TMPDIR
export TMPDIR="/scratch/lmbanr001/tmp_mamba"
if [ -d "$TMPDIR" ]; then
    rm -rf "$TMPDIR"/*
else
    mkdir -p "$TMPDIR"
fi

# Load necessary modules
module load python/miniconda3-py310 compilers/gcc11.2

# Activate virtual environment
source activate architecture_venv

# Set up paths
config_path="mamba_config.json"
tokenizer_path="tokenizer.json"
dataset_path="../../Datasets/MOSES/datasets"
output_dir="/scratch/lmbanr001/MAMBA_small"

mkdir -p $output_dir

# Run the training script using cli.py
python3 trainer/cli.py \
    --config $config_path \
    --tokenizer $tokenizer_path \
    --dataset $dataset_path \
    --text_column "SMILES" \
    --is_tokenized False \
    --streaming True \
    --model_type "mamba" \
    --optim "adamw_torch" \
    --learning_rate 5e-4 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 2000 \
    --eval_steps 500 \
    --save_steps 500 \
    --num_train_epochs 10 \
    --save_total_limit 2 \
    --prop_loss_coeff 1e-3 \
    --output_dir $output_dir \
    --overwrite_output_dir True \
    --do_train True \
    --save_safetensors True \
    --gradient_checkpointing True \
    --eval_accumulation_steps 100 \
    --max_steps 30_000_000

# Deactivate virtual environment
conda deactivate
