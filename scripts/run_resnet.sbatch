#!/bin/bash
#SBATCH --job-name=resnet-train
#SBATCH --partition=eval
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=36:00:00
#SBATCH --output=/tmp/%x-%j.out

AUG_TYPE="weak_random_blur"
AUG_PARAM=1.0

LOGDIR="experiments/${AUG_TYPE}/exp_resnet_temp_${AUG_PARAM}"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

# copy the files
bash scripts/download_imagenet.sh

# Activate virtualenv
source ~/netrep-analysis/venv/bin/activate

# Run your script
python scripts/resnet.py --aug_type=${AUG_TYPE} --aug_param=${AUG_PARAM} --data_path=/tmp/imagenet
