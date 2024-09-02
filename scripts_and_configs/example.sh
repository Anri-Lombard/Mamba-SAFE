#!/bin/bash
#SBATCH --account=...
#SBATCH --partition=...
#SBATCH --nodes=...
#SBATCH --ntasks=...
#SBATCH --gres=...
#SBATCH --time=...
#SBATCH --job-name=...
#SBATCH --mail-user=...
#SBATCH --mail-type=...

# Set up wandb
export WANDB_API_KEY="..."
wandb_cache_dir="..."
wandb_dir="..."

export WANDB__SERVICE_WAIT=300

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
export TMPDIR="..."
if [ -d "$TMPDIR" ]; then
    rm -rf "$TMPDIR"
else
    mkdir -p "$TMPDIR"
fi

export WANDB_TMPDIR=$TMPDIR

# Set up the PYTHONPATH
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# make sure this script is in the same directory as mamba_safe and replace
# ... with your path to the mamba_safe directory
export PYTHONPATH="...:$PYTHONPATH"

echo "Current working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"


# Set up paths
config_path="example_config.json"
tokenizer_path="tokenizer.json"
dataset_path=".../safe_zinc_dataset"
output_dir="..."

# Run the training script using cli.py
# TODO: add model path next time
python3 trainer/cli.py \
    --resume_from_checkpoint $checkpoint_path \
    --config_path $config_path \
    --tokenizer_path $tokenizer_path \
    --dataset_path $dataset_path \
    --text_column "safe" \
    --optim "adamw_torch" \
    --report_to "wandb" \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 100 \
    --per_device_eval_batch_size 100 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 10_000 \
    --logging_first_step True \
    --save_steps 10_000 \
    --eval_steps 10_000 \
    --eval_accumulation_steps 1000 \
    --eval_strategy "steps" \
    --wandb_project "MAMBA_large" \
    --logging_steps 100 \
    --save_total_limit 1 \
    --output_dir $output_dir \
    --overwrite_output_dir True \
    --do_train True \
    --do_eval True \
    --save_safetensors True \
    --gradient_checkpointing True \
    --max_grad_norm 1.0 \
    --weight_decay 0.1 \
    --max_steps 250_000

# Deactivate virtual environment
conda deactivate
