#!/bin/bash
#SBATCH --job-name=resnet-train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/tmp/%x-%j.out

AUG_TYPE="weak_random_blur"
AUG_PARAM=1.0

LOGDIR="/mnt/home/the10/ceph/results/netrep/experiments/clean"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

# Activate virtualenv
source ~/netrep-analysis/venv/bin/activate

# Run your script
# for i in {0..90..10}; do
#     echo "Processing epoch$i"
#     python scripts/extract_rep.py --experiment=weak_random_blur/exp_resnet_sigma_1 --model=epoch$i
# done
python scripts/extract_rep.py --experiment=clean
python scripts/extract_rep.py --experiment=weak_random_blur/exp_resnet_temp_1.0
python scripts/extract_rep.py --experiment=weak_random_blur/exp_resnet_temp_2.0
python scripts/extract_rep.py --experiment=weak_random_blur/exp_resnet_temp_3.0
python scripts/extract_rep.py --experiment=weak_random_blur/exp_resnet_temp_4.0
python scripts/extract_rep.py --experiment=weak_random_blur/exp_resnet_temp_0.2
python scripts/extract_rep.py --experiment=weak_random_blur/exp_resnet_temp_0.4
python scripts/extract_rep.py --experiment=weak_random_blur/exp_resnet_temp_0.6
python scripts/extract_rep.py --experiment=weak_random_blur/exp_resnet_temp_0.8