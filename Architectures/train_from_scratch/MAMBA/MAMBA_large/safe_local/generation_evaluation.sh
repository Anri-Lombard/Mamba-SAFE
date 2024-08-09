#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:ampere80:1
#SBATCH --time=48:00:00
#SBATCH --job-name="MAMBA_large_evaluate"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

# Set up the PYTHONPATH
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Load necessary modules
module load python/miniconda3-py310 compilers/gcc11.2

# Activate virtual environment
source activate architecture_venv

model_dir="/scratch/lmbanr001/MAMBA_small_v9/checkpoint-5000"
tokenizer_path="/scratch/lmbanr001/MAMBA_small_v9/tokenizer.json"
output_dir="/scratch/lmbanr001/MAMBA_large_v9/evaluation"

export TOKENIZERS_PARALLELISM="false"

# python3 evaluate_mamba.py --model_dir $model_dir \
#     --tokenizer_path $tokenizer_path \
#     --num_samples 1000 \
#     --max_length 30 \
#     --output_dir $output_dir
python3 evaluate_mamba_small.py --model_dir $model_dir \
    --tokenizer_path $tokenizer_path \
    --num_samples 100 \
    --max_length 40

conda deactivate