#!/bin/sh
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --job-name="ConvertSMILEStoSAFE_large_dataset"
#SBATCH --mail-user=lmbanr001@myuct.ac.za
#SBATCH --mail-type=ALL

# Load necessary modules
module load python/miniconda3-py310

# Activate virtual environment
source activate architecture_venv

# Run script
python3 convert_smiles_to_safe_zinc.py

# Deactivate virtual environment
conda deactivate