#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:ampere80:1
#SBATCH --time=48:00:00
#SBATCH --job-name="MAMBA_large"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

# Set up wandb
export WANDB_MODE="offline"
export WANDB_API_KEY="d68cdf1a94da49b43fbfb7fd90246c39d7c34237"
wandb_cache_dir="/scratch/lmbanr001/wandb_cache"
wandb_dir="/scratch/lmbanr001/wandb"

rm -rf "$HOME/.wandb"

# Clean up and create directories
for dir in "$wandb_cache_dir" "$wandb_dir"; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
    else
        mkdir -p "$dir"
    fi
done

export WANDB_CACHE_DIR=$wandb_cache_dir
export WANDB_DIR=$wandb_dir

# Set up TMPDIR
export TMPDIR="/scratch/lmbanr001/tmp_mamba_large"
if [ -d "$TMPDIR" ]; then
    rm -rf "$TMPDIR"
else
    mkdir -p "$TMPDIR"
fi

export WANDB_TMPDIR=$TMPDIR

# Set up the PYTHONPATH
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Load necessary modules
module load python/miniconda3-py310 compilers/gcc11.2

# Activate virtual environment
source activate architecture_venv

pip3 install --force-reinstall --no-deps wandb==0.16.6 protobuf==4.25.3

# Set up paths
config_path="mamba_config.json"
tokenizer_path="tokenizer.json"
dataset_path="anrilombard/safe-gpt-small"
output_dir="/scratch/lmbanr001/MAMBA_large"

mkdir -p $output_dir

# Run the training script using cli.py
python3 trainer/cli.py \
    --config $config_path \
    --tokenizer $tokenizer_path \
    --dataset $dataset_path \
    --text_column "input" \
    --is_tokenized False \
    --streaming True \
    --model_type "mamba" \
    --optim "adamw_torch" \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 5000 \
    --eval_steps 500 \
    --save_steps 500 \
    --num_train_epochs 10 \
    --save_total_limit 1 \
    --prop_loss_coeff 1e-2 \
    --output_dir $output_dir \
    --overwrite_output_dir True \
    --do_train True \
    --push_to_hub True \
    --hub_token "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" \
    --hub_model_id "anrilombard/mamba-medium" \
    --save_safetensors True \
    --gradient_checkpointing True \
    --eval_accumulation_steps 100 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --max_steps 50_000 \

# Deactivate virtual environment
conda deactivate
