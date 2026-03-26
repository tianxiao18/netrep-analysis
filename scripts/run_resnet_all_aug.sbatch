#!/bin/bash
#SBATCH --job-name=resnet-train
#SBATCH --partition=gpu
#SBATCH --nodelist=workergpu[017-072]
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=/tmp/%x-%j.out

# AUG_TYPE="cutout_crop"
AUG_TYPE="cutout"
AUG_PARAM="12.0"
OP_PARAM="0.1"
MODEL="resnet18"
# AUG_TYPE="clean"
# AUG_PARAM="0"
# MODEL="wide_resnet"

declare -A param_map=(["fixed_blur"]="sigma" ["weak_random_blur"]="temp" ["sp_noise"]="sp_prob" ["cutout"]="cutout_patch_size" ["cutout"]="cutout_patch_size" ["clean"]="")
                  
LOGDIR="/mnt/home/the10/ceph/results/netrep/experiments/${AUG_TYPE}/exp_${MODEL}_${param_map[$AUG_TYPE]}_${AUG_PARAM}_${OP_PARAM}"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

# copy the files
# bash scripts/download_imagenet.sh

# Activate virtualenv
source ~/netrep-analysis/venv/bin/activate

# train from scratch
# python scripts/resnet.py \
#     --aug_type=clean \
#     --aug_param=0.0 \
#     --model=resnet18 \
#     --dataset=cifar10 \
#     --base_lr=0.03 \
#     --weight_decay=5e-4 \
#     --warmup_epochs=0 \
#     --momentum=0.9 \
#     --epochs=100 \
#     --data_path=/mnt/home/the10/ceph/dataset/cifar10 \
#     --seed=42

# for AUG_PARAM in 4.0 12.0 18.0 24.0; do
#     python scripts/resnet.py --aug_type=rotate --data_path=/mnt/home/the10/ceph/dataset/cifar10 --aug_param=${AUG_PARAM} --model=resnet18 --dataset=cifar10 --base_lr=0.03 --weight_decay=5e-4 --momentum=0.9 --warmup_epochs=0 --epochs=100
# done
# for AUG_PARAM in 4.0 12.0 18.0 24.0; do
#     python scripts/resnet.py --aug_type=sheer --data_path=/mnt/home/the10/ceph/dataset/cifar10 --aug_param=${AUG_PARAM} --model=resnet18 --dataset=cifar10 --base_lr=0.03 --weight_decay=5e-4 --momentum=0.9 --warmup_epochs=0 --epochs=100
# done
# for AUG_PARAM in 0.1 0.2 0.3 0.4; do
#     python scripts/resnet.py --aug_type=jitter --data_path=/mnt/home/the10/ceph/dataset/cifar10 --aug_param=${AUG_PARAM} --model=resnet18 --dataset=cifar10 --base_lr=0.03 --weight_decay=5e-4 --momentum=0.9 --warmup_epochs=0 --epochs=100
# done
# for AUG_PARAM in 0.03 0.05 0.08 0.1; do
#     python scripts/resnet.py --aug_type=grayscale --data_path=/mnt/home/the10/ceph/dataset/cifar10 --aug_param=${AUG_PARAM} --model=resnet18 --dataset=cifar10 --base_lr=0.03 --weight_decay=5e-4 --momentum=0.9 --warmup_epochs=0 --epochs=100
# done
# for AUG_PARAM in 0.01 0.02 0.03 0.04; do
#     python scripts/resnet.py --aug_type=gaussian_noise --data_path=/mnt/home/the10/ceph/dataset/cifar10 --aug_param=${AUG_PARAM} --model=resnet18 --dataset=cifar10 --base_lr=0.03 --weight_decay=5e-4 --momentum=0.9 --warmup_epochs=0 --epochs=100
# done
# for AUG_PARAM in 0.05 0.1 0.15 0.2; do
#     python scripts/resnet.py --aug_type=sp_noise --data_path=/mnt/home/the10/ceph/dataset/cifar10 --aug_param=${AUG_PARAM} --model=resnet18 --dataset=cifar10 --base_lr=0.03 --weight_decay=5e-4 --momentum=0.9 --warmup_epochs=0 --epochs=100
# done
for AUG_PARAM in 4.0 5.0 6.0 7.0; do
    python scripts/resnet.py --aug_type=cutout --data_path=/mnt/home/the10/ceph/dataset/cifar10 --aug_param=${AUG_PARAM} --model=resnet18 --dataset=cifar10 --base_lr=0.03 --weight_decay=5e-4 --momentum=0.9 --warmup_epochs=0 --epochs=100
done
# for AUG_PARAM in 0.6 0.7 0.8 0.9; do
#     python scripts/resnet.py --aug_type=crop --data_path=/mnt/home/the10/ceph/dataset/cifar10 --aug_param=${AUG_PARAM} --model=resnet18 --dataset=cifar10 --base_lr=0.03 --weight_decay=5e-4 --momentum=0.9 --warmup_epochs=0 --epochs=100
# done