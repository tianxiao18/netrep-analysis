import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from netrep.metrics import LinearMetric
from netrep.activity_extractor import LayerActivityExtractor
from netrep.dualPCA import DualPCA
from netrep.visualize import visualize_coordinates_all, visualize_coordinates, visualize_distance, visualize_layer_aligned, visualize_distance_matrices
from argparse import ArgumentParser
from netrep.models import get_model
from netrep.optimize import evaluate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import itertools
from netrep.metrics import LinearMetric
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from tqdm import tqdm
from adjustText import adjust_text
import re
import random
import gc
import math

def parse_args():
    parser = ArgumentParser(description="PyTorch Resnet Trainer")
    parser.add_argument(
        "--test_size",
        type=int,
        default=10000,
        help="Number of test images to extract representation"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/home/the10/ceph/dataset/cifar10"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="clean"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=24
    )

    parser.add_argument(
        "--model",
        type=str,
        default="final_model"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10"
    )

    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True
    )

    parser.add_argument(
        "--edit",
        type=bool,
        default=False
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )


    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    set_seed(42)

    base_path = '/mnt/home/the10/ceph/results/netrep'

    pretrain_str = "/pretrained" if args.pretrained else ""
    pretrain_str = f"{pretrain_str}_seed{args.seed}" if args.seed else pretrain_str
    experiment_ls = [f"cutout{pretrain_str}/exp_wide_resnet_cutout_patch_size_{c}_{s}" for c in [12.0, 16.0, 20.0, 24.0] for s in [0.2, 0.3, 0.5, 0.8, 1.0]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2023, 0.1994, 0.2010]),
        ])

    val_dataset = datasets.CIFAR10(root=args.data_path, train=False, transform=eval_tf, download=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    accuracies = []

    for exp in experiment_ls:
        checkpoint_path = f'{base_path}/experiments/{exp}/checkpoints/final_model.pth'
        print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model = get_model(device, "wide_resnet")
        model.load_state_dict(checkpoint)

        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        # val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        val_acc = 100.0 * correct / total
        c, s = exp.split('_')[-2:]

        print(f"Test Accuracy for c={c}, s={s}: {val_acc:.2f}%")
        accuracies.append((c, s, val_acc))

    plt.figure(figsize=(6, 5))
    x, y, acc = zip(*accuracies)
    sc = plt.scatter(y, x, c=acc, cmap='viridis', s=60, edgecolor='k')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()
    plt.colorbar(sc, label='Accuracy')
    plt.xlabel("sigma")
    plt.ylabel("cutout size")
    plt.title("Scatter plot colored by accuracy")
    plt.savefig('accuracy.png')


if __name__ == "__main__":
    main()