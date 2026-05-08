#!/bin/bash
#SBATCH --job-name=vit-imagenet
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=150:00:00
#SBATCH --output=/tmp/%x-%j.out

AUG_TYPE="cutout"
AUG_PARAM="8.0"
OP_PARAM="0.2"
MODEL="vit"
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

# Cosine+warmup (per-epoch): avoids OneCycle mid-run LR peak that matches val “crater then recover”.
python scripts/resnet.py \
    --aug_type=${AUG_TYPE} \
    --aug_param=0.0 \
    --op_param=0.0 \
    --model=${MODEL} \
    --dataset=imagenet \
    --base_lr=5e-4 \
    --weight_decay=1e-2 \
    --warmup_epochs=0 \
    --lr_schedule=cosine \
    --epochs=20 \
    --data_path=/tmp/imagenet \
    --pretrained \
    --label_smoothing=0.1 \
    --seed=44

for AUG_PARAM in 24.0 40.0 56.0 72.0 88.0 104.0 120.0; do
    for OP_PARAM in 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
        echo "Running with AUG_PARAM=${AUG_PARAM} OP_PARAM=${OP_PARAM}"
        python scripts/resnet.py \
            --aug_type=${AUG_TYPE} \
            --aug_param=${AUG_PARAM} \
            --op_param=${OP_PARAM} \
            --model=${MODEL} \
            --dataset=imagenet \
            --base_lr=5e-4 \
            --weight_decay=1e-2 \
            --warmup_epochs=0 \
            --lr_schedule=cosine \
            --epochs=20 \
            --data_path=/tmp/imagenet \
            --pretrained \
            --label_smoothing=0.1 \
            --seed=44
    done
done