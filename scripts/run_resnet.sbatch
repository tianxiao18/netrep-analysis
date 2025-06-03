#!/bin/bash
#SBATCH --job-name=resnet-train
#SBATCH --partition=eval
#SBATCH -C h200
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=experiments/exp_resnet_sigma_0/%x-%j.out

# Load modules (if needed)
# module load python/3.10.13  # Or whatever you use

# Activate virtualenv
source ~/netrep-analysis/venv/bin/activate

# Run your script
python scripts/resnet.py --sigma=0
