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
from netrep.visualize import visualize_coordinates_all, visualize_coordinates, visualize_distance, visualize_layer_aligned
from argparse import ArgumentParser

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
        default="/mnt/home/the10/ceph/dataset/imagenet/val"
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
        default="best_model"
    )

    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def exp_to_name(exp_ls):
    name_ls = []
    param_ls = []

    symbol_map = {'temp': 'T', 'cutout_patch_size': 'C'}

    for exp in exp_ls:        
        if exp != 'clean':
            words = exp.split('/')[-1].split('_')
            symbol = symbol_map['_'.join(words[2:-1])]
            param = words[-1]
        else:
            param = '0'
            symbol = 'UNK'
        param_ls.append(param)
        name_ls.append(f'{symbol}={param}')
    
    for i, exp in enumerate(exp_ls):
        if exp == 'clean':
            next_symbol = next((entry.split('=')[0] for entry in name_ls[1:] if entry.split('=')[0] != 'UNK'),None)
            name_ls[i] = f'{next_symbol}=0'

    return name_ls, param_ls

def visualize_distance_all(distmat, layer_names, network_names, output_path):
    distmat = np.array(distmat)
    num_layers = len(layer_names)

    # Get vmin and vmax from off-diagonal entries only
    mask = np.eye(distmat.shape[0], dtype=bool)
    off_diag_vals = distmat[~mask]

    distmat = np.delete(distmat, 2*np.arange(num_layers), axis=0)
    distmat = np.delete(distmat, 2*np.arange(num_layers), axis=1)

    vmin = off_diag_vals.min()
    vmax = off_diag_vals.max()
    row, col = np.unravel_index(np.argmax(distmat), distmat.shape)
    print(row, col)

    fig, axes = plt.subplots(4, 4, figsize=(4 * 4, 4 * 4))

    # Flatten axes to index easily even if there's only one row
    axes = axes.flatten()

    for l in range(num_layers):
        indices = np.arange(l, distmat.shape[0], num_layers)
        selected_distmat = distmat[indices[:, None], indices]

        ax = axes[l]
        sns.heatmap(
            selected_distmat,
            cmap="viridis",
            square=True,
            xticklabels=network_names,
            yticklabels=network_names,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar=False
        )
        ax.set_title(layer_names[l])
        ax.set_xlabel("Network")
        ax.set_ylabel("Network")

    plt.tight_layout()
    plt.savefig(f'{output_path}/dist.png', dpi=250)
    np.save(f'{output_path}/distance_matrix.npy', distmat)
    print(f"Saving to {output_path}/dist.png")
    plt.close()


def main():
    args = parse_args()
    set_seed(42)

    base_path = '/mnt/home/the10/ceph/results/netrep'
    result_path = f'{base_path}/experiments/{args.experiment}/results'
    output_path = f'{result_path}/analysis'

    os.makedirs(output_path, exist_ok=True)
    experiment_ls = ['clean', 
                     'cutout/exp_resnet_cutout_patch_size_60.0', 'cutout/exp_resnet_cutout_patch_size_100.0', 'cutout/exp_resnet_cutout_patch_size_120.0'     
        # 'weak_random_blur/exp_resnet_temp_0.2', 'weak_random_blur/exp_resnet_temp_0.4' ,'weak_random_blur/exp_resnet_temp_0.6', 'weak_random_blur/exp_resnet_temp_0.8',
        ]

    distmat = np.load(f"{result_path}/distance_matrix.npy")
    network_indices = [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 4.0, 4.1, 4.2]
    network_names = [f'module.layer{i}' for i in network_indices]

    l = len(network_indices)
    name_ls = exp_to_name(experiment_ls)[0]
    # indices = np.arange(1, len(distmat), l)
    # distmat = distmat[indices[:, None], indices]
    # print(distmat.shape, indices)
    visualize_distance_all(distmat, network_names, name_ls, output_path)

    embedding = MDS(n_components=200, metric= True, eps = 0.00001, normalized_stress='auto', dissimilarity='precomputed', random_state=42)
    Z = embedding.fit_transform(np.abs(np.real(distmat)))
    print(embedding.stress_)

    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(Z)
    print(coordinates.shape, Z.shape)

    visualize_coordinates_all(coordinates, experiment_ls, output_path, name_ls)

if __name__ == "__main__":
    main()