#!/bin/bash
#SBATCH --job-name=resnet-train
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/tmp/%x-%j.out

AUG_TYPE="cutout"
AUG_PARAM="8.0"
OP_PARAM="0.2"
MODEL="resnet"
# AUG_TYPE="clean"
# AUG_PARAM="0"
# MODEL="wide_resnet"

declare -A param_map=(["fixed_blur"]="sigma" ["weak_random_blur"]="temp" ["sp_noise"]="sp_prob" ["cutout"]="cutout_patch_size" ["clean"]="")
                  
LOGDIR="/mnt/home/the10/ceph/results/netrep/experiments/${AUG_TYPE}/exp_${MODEL}_${param_map[$AUG_TYPE]}_${AUG_PARAM}_${OP_PARAM}"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

# copy the files
bash scripts/download_imagenet.sh

# Activate virtualenv
source ~/netrep-analysis/venv/bin/activate

# Run your script
# python scripts/resnet.py --aug_type=${AUG_TYPE} --aug_param=16.0 --op_param=0.5 --model=${MODEL} --dataset=cifar10 --momentum=0.9 --weight_decay=5e-4 --warmup_epochs=0 --epochs=200 --data_path=/mnt/home/the10/ceph/dataset/cifar10 --seed=44
# python scripts/resnet.py --aug_type=${AUG_TYPE} --aug_param=28.0 --op_param=1.0 --model=${MODEL} --dataset=cifar10 --base_lr=3e-4 --weight_decay=0.05 --warmup_epochs=0 --epochs=50 --data_path=/mnt/home/the10/ceph/dataset/cifar10 --checkpoint=/mnt/home/the10/ceph/results/netrep/experiments/clean/exp_wide_resnet_clean_0.0/checkpoints/final_model.pth --seed=42

for AUG_PARAM in 80.0; do
    for OP_PARAM in 0.2 0.3 0.5 0.8; do
        echo "Running with AUG_PARAM=${AUG_PARAM} OP_PARAM=${OP_PARAM}"
        python scripts/resnet.py \
            --aug_type=${AUG_TYPE} \
            --aug_param=${AUG_PARAM} \
            --op_param=${OP_PARAM} \
            --model=${MODEL} \
            --dataset=imagenet \
            --base_lr=5e-5 \
            --weight_decay=0.01 \
            --warmup_epochs=0 \
            --epochs=20 \
            --data_path=/tmp/imagenet \
            --checkpoint=/mnt/home/the10/ceph/results/netrep/experiments/clean/exp_resnet_clean_0.0/checkpoints/final_model.pth\
            --seed=42
    done
done