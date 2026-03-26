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
MODEL="resnet"
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

# Run your script
# python scripts/resnet.py --aug_type=${AUG_TYPE} --aug_param=16.0 --op_param=0.5 --model=${MODEL} --dataset=cifar10 --momentum=0.9 --weight_decay=5e-4 --warmup_epochs=0 --epochs=200 --data_path=/mnt/home/the10/ceph/dataset/cifar10 --seed=44
# python scripts/resnet.py --aug_type=${AUG_TYPE} --aug_param=28.0 --op_param=1.0 --model=${MODEL} --dataset=cifar10 --base_lr=3e-4 --weight_decay=0.05 --warmup_epochs=0 --epochs=50 --data_path=/mnt/home/the10/ceph/dataset/cifar10 --checkpoint=/mnt/home/the10/ceph/results/netrep/experiments/clean/exp_wide_resnet_clean_0.0/checkpoints/final_model.pth --seed=42

# here seed42 = full parameter grid pretrained with 42, seed43 = MVP for analysis, seed44=pretrained with 44, seed45=pretrained with 43
# pretraining + fine tuning
# for AUG_PARAM in 4.0 8.0 10.0 12.0 16.0 20.0 24.0; do
#     for OP_PARAM in 0.05 0.1 0.2 0.3 0.5 0.8 1.0; do
#         echo "Running with AUG_PARAM=${AUG_PARAM} OP_PARAM=${OP_PARAM}"
#         python scripts/resnet.py \
#             --aug_type=${AUG_TYPE} \
#             --aug_param=${AUG_PARAM} \
#             --op_param=${OP_PARAM} \
#             --model=${MODEL} \
#             --dataset=cifar10 \
#             --epochs=50 \
#             --base_lr=0.01 \
#             --weight_decay=1e-4 \
#             --warmup_epochs=0 \
#             --momentum=0.9 \
#             --epochs=50 \
#             --data_path=/mnt/home/the10/ceph/dataset/cifar10 \
#             --seed=42 \
#             --pretrained
#     done
# done

python scripts/resnet.py \
    --aug_type=${AUG_TYPE} \
    --aug_param=0.0 \
    --op_param=0.0 \
    --model=${MODEL} \
    --dataset=cifar10 \
    --base_lr=0.01 \
    --weight_decay=1e-4 \
    --warmup_epochs=0 \
    --momentum=0.9 \
    --epochs=20 \
    --data_path=/mnt/home/the10/ceph/dataset/cifar10 \
    --seed=42 \
    --pretrained

for AUG_PARAM in 4.0 8.0 10.0 12.0 16.0 20.0 24.0; do
    for OP_PARAM in 0.05 0.1 0.2 0.3 0.5 0.8 1.0; do
        echo "Running with AUG_PARAM=${AUG_PARAM} OP_PARAM=${OP_PARAM}"
        python scripts/resnet.py \
            --aug_type=${AUG_TYPE} \
            --aug_param=${AUG_PARAM} \
            --op_param=${OP_PARAM} \
            --model=${MODEL} \
            --dataset=cifar10 \
            --base_lr=0.01 \
            --weight_decay=1e-4 \
            --warmup_epochs=0 \
            --momentum=0.9 \
            --epochs=20 \
            --data_path=/mnt/home/the10/ceph/dataset/cifar10 \
            --seed=42 \
            --pretrained
    done
done