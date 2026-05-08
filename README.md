# netrep-analysis

How does **data augmentation** affect the **geometry** of learned representations beyond better accuracy? This repo implements analyses in **shape space**, comparing layers with **Riemannian shape distance** (invariant to scaling, translation, rotation, and reflection). Stronger augmentation yields cleaner trajectories in that space; different augmentations move reps along different directions; geometry along trajectories relates to **ensembling** gains, with similar patterns across architectures and seeds.

**Pipeline:** `resnet.py` (train) → `extract_rep.py` (layer activity + distance matrices) → `repr_analysis.py` (plots / summaries).

## Environment

From the repo root:

```bash
python -m venv venv
source venv/bin/activate   # or your equivalent
pip install -r requirements.txt
```

Then run the Python entrypoints from the **`scripts/`** directory (they resolve the project root automatically):

```bash
cd scripts
```

## 1. Train (`resnet.py`)

Example (CIFAR-10, DenseNet, cutout):

```bash
python resnet.py \
  --aug_type=cutout \
  --data_path=/mnt/home/the10/ceph/dataset/cifar10 \
  --aug_param=12 \
  --op_param=0.2 \
  --model=densenet \
  --dataset=cifar10 \
  --base_lr=0.01 \
  --weight_decay=0.01 \
  --momentum=0.9 \
  --warmup_epochs=0 \
  --epochs=50
```

More options: `python resnet.py --help`. Checkpoints and logging paths are set inside the script (including Weights & Biases).

## 2. Extract representations (`extract_rep.py`)

Example **grid over all experiments** matching trained runs (here ResNet-18, cutout family):

```bash
python extract_rep.py \
  --experiment=all \
  --model_name=resnet18 \
  --data_path=/mnt/home/the10/ceph/dataset/cifar10 \
  --dataset=cifar10 \
  --aug_type=cutout
```

More options: `python extract_rep.py --help`.

## 3. Analyze representations (`repr_analysis.py`)

Uses outputs from extraction (same experiment/model/aug wiring):

```bash
python repr_analysis.py \
  --experiment=all \
  --model_name=resnet18 \
  --data_path=/mnt/home/the10/ceph/dataset/cifar10 \
  --aug_type=cutout
```

More options: `python repr_analysis.py --help`.

**Order:** train → `extract_rep.py` → `repr_analysis.py`.
