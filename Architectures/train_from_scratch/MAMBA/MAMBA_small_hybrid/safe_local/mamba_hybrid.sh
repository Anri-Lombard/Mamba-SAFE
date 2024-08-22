#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:ampere80:1
#SBATCH --time=48:00:00
#SBATCH --job-name="HYBRID_20M_dropout_little"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

# Set up wandb
# export WANDB_MODE="offline"
export WANDB_API_KEY="d68cdf1a94da49b43fbfb7fd90246c39d7c34237"
wandb_cache_dir="/scratch/lmbanr001/wandb_cache_hybrid"
wandb_dir="/scratch/lmbanr001/wandb_hybrid"

rm -rf "$HOME/.wandb"

# Clean up and create directories
for dir in "$wandb_cache_dir" "$wandb_dir"; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
    fi
    mkdir -p "$dir"
done

export WANDB_CACHE_DIR=$wandb_cache_dir
export WANDB_DIR=$wandb_dir

# Set up TMPDIR
export TMPDIR="/scratch/lmbanr001/tmp_hybrid"
if [ -d "$TMPDIR" ]; then
    rm -rf "$TMPDIR"
else
    mkdir -p "$TMPDIR"
fi

export WANDB_TMPDIR=$TMPDIR

# Set up the PYTHONPATH
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# export PYTHONPATH="$SCRIPT_DIR/..:$PYTHONPATH"
export PYTHONPATH="/home/lmbanr001/hons2024/MAMBA/MAMBA_small_hybrid:$PYTHONPATH"

echo "Current working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# Load necessary modules
module load python/miniconda3-py310 compilers/gcc11.2

# Activate virtual environment
source activate architecture_venv

# Set up paths
config_path="mamba_config.json"
tokenizer_path="tokenizer.json"
# config_path="/scratch/lmbanr001/MAMBA_small_dropout/checkpoint-110000/config.json"
# TODO: continue from last checkpoint when crash happens
# model_path="/scratch/lmbanr001/MAMBA_small_dropout/checkpoint-110000"
dataset_path="/scratch/lmbanr001/Datasets/MOSES/safe_datasets"
# output_dir="/scratch/lmbanr001/MAMBA_small_dropout_continued"
# TODO: change this before next run to prevent deletion!
output_dir="/scratch/lmbanr001/HYBRID_20M_dropout_little"


# Run the training script using cli.py
# TODO: add model path next time
python3 trainer/cli.py \
    --config_path $config_path \
    --tokenizer_path $tokenizer_path \
    --dataset_path $dataset_path \
    --text_column "SAFE" \
    --optim "adamw_torch" \
    --report_to "wandb" \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --learning_rate 5e-4 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 20000 \
    --logging_first_step True \
    --save_steps 2000 \
    --eval_steps 2000 \
    --eval_accumulation_steps 1000 \
    --eval_strategy "steps" \
    --wandb_project "MAMBA_hybrid" \
    --logging_steps 100 \
    --num_train_epochs 10 \
    --save_total_limit 1 \
    --output_dir $output_dir \
    --overwrite_output_dir True \
    --do_train True \
    --do_eval True \
    --save_safetensors True \
    --gradient_checkpointing True \
    --max_grad_norm 1.0 \
    --weight_decay 0.1

# Deactivate virtual environment
conda deactivate
