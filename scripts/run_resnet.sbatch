#!/bin/bash
#SBATCH --job-name=resnet-train
#SBATCH --partition=eval
#SBATCH --nodelist=workergpu[160-170]
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=36:00:00
#SBATCH --output=/tmp/%x-%j.out

AUG_TYPE="clean"
AUG_PARAM="0"
MODEL="resnet"

declare -A param_map=(["fixed_blur"]="sigma",["weak_random_blur"]="temp",["sp_noise"]="sp_prob", ["cutout"]="cutout_patch_size",["clean"]="")
                  
LOGDIR="/mnt/home/the10/ceph/results/netrep/experiments/${AUG_TYPE}/exp_${MODEL}_${param_map[$AUG_TYPE]}_${AUG_PARAM}"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

# copy the files
bash scripts/download_imagenet.sh

# Activate virtualenv
source ~/netrep-analysis/venv/bin/activate

# Run your script
python scripts/resnet.py --aug_type=${AUG_TYPE} --model=${MODEL} --data_path=/tmp/imagenet
