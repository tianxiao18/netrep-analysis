#!/bin/bash
#SBATCH --job-name=resnet-train
#SBATCH --partition=eval
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=36:00:00
#SBATCH --output=/tmp/%x-%j.out
                  
LOGDIR="/mnt/home/the10/ceph/results/netrep/experiments/aug_combined/weak_random_blur_cutout/exp_resnet_temp_4.0_cutout_patch_size_120.0"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

# copy the files
# bash scripts/download_imagenet.sh

# Activate virtualenv
source ~/netrep-analysis/venv/bin/activate

# Run your script
python scripts/resnet_aug_combined.py --augments weak_random_blur=4.0 cutout=120.0
