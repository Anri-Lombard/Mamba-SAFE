#!/bin/sh
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:a100-3g-20gb:1
#SBATCH --time=04:00:00
#SBATCH --job-name="ConvertSMILEStoSAFE"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

# Set CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(ncvd)

# Load necessary modules
module load python/miniconda3-py310

# Activate virtual environment
source activate architecture_venv

# Run script
python3 convert_smiles_to_safe.py

# Deactivate virtual environment
conda deactivate