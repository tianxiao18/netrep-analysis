#!/bin/bash
#SBATCH --job-name=resnet-train
#SBATCH --partition=eval
#SBATCH -C h200
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/tmp/%x-%j.out

EXPERIMENT_NAME="fixed_blur"
SIGMA=8

LOGDIR="experiments/${EXPERIMENT_NAME}/exp_resnet_sigma_${SIGMA}"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

# Activate virtualenv
source ~/netrep-analysis/venv/bin/activate

# Run your script
python scripts/resnet.py --experiment=${EXPERIMENT_NAME} --sigma=${SIGMA}
