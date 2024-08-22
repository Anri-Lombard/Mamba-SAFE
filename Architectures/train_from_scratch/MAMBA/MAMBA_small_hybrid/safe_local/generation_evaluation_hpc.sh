#!/bin/bash
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:ampere80:1
#SBATCH --time=48:00:00
#SBATCH --job-name="HYBRID_generate_10k_samples"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

# Set up the PYTHONPATH
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Load necessary modules
module load python/miniconda3-py310 compilers/gcc11.2

# Activate virtual environment
source activate architecture_venv

model_dir="/scratch/lmbanr001/HYBRID_20M_dropout_little"
tokenizer_path="tokenizer.json"
output_file="/scratch/lmbanr001/Results/HYBRID_20M_dropout_little_10000_samples.txt"

export TOKENIZERS_PARALLELISM="false"

python3 simplified_molecule_generator.py --model_dir $model_dir \
    --tokenizer_path $tokenizer_path \
    --num_samples 100 \
    --max_length 100 \
    --top_k 50 \
    --top_p 0.95 \
    --temperature 1.0 \
    --n_trials 100 \
    --max_retries 10 \
    --output_file $output_file

conda deactivate