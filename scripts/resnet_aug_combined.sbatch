#!/bin/bash
#SBATCH --job-name=resnet-train
#SBATCH --partition=eval
#SBATCH --nodelist=workergpu[313-318]
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=/tmp/%x-%j.out
                  
LOGDIR="/mnt/home/the10/ceph/results/netrep/experiments/aug_combined/weak_random_blur_cutout/exp_resnet_temp_2.0_cutout_patch_size_80.0"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

# copy the files
# bash scripts/download_imagenet.sh

# Activate virtualenv
source ~/netrep-analysis/venv/bin/activate

# Run your script
python scripts/resnet_aug_combined.py --augments weak_random_blur=2.0 cutout=80.0
