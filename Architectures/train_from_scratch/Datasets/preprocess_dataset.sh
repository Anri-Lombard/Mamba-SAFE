#!/bin/sh
#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:a100-3g-20gb:1
#SBATCH --time=04:00:00
#SBATCH --job-name="ConvertSMILEStoSAFE"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

# Load necessary modules
module load python/miniconda3-py310

# Activate virtual environment
source activate architecture_venv

# Run script
python3 Datasets/convert_smiles_to_selfies.py

# Deactivate virtual environment
conda deactivate