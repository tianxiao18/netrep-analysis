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


def compute_distances(X):
    # first precompute X X^T
    n = X[0].shape[0]
    G = np.zeros((len(X), n, n))

    for i in range(len(X)):
        x = X[i].reshape(n, -1)
        G[i] = x @ x.T

    distmat = np.zeros((len(X), len(X)))
    total_pairs = len(X) * (len(X) - 1) // 2

    # then compute euclidean distance X X^T - Y Y^T
    for i, j in tqdm(itertools.combinations(range(len(X)), 2), total=total_pairs, desc="Computing distances"):
        distmat[i, j] = np.linalg.norm(G[i] -G[j], ord='fro')

    distmat += distmat.T
    return distmat

def compute_distance_from_G(G):
    distmat = np.zeros((len(G), len(G)))
    total_pairs = len(G) * (len(G) - 1) // 2

    # then compute euclidean distance X X^T - Y Y^T
    for i, j in tqdm(itertools.combinations(range(len(G)), 2), total=total_pairs, desc="Computing distances"):
        distmat[i, j] = np.linalg.norm(G[i]/G[i].trace()-G[j]/G[j].trace(), ord='fro')

    distmat += distmat.T
    return distmat

def visualize_distance_matrices(dist_matrix_path, sample_sizes, seeds, i=0, j=1):
    all_distances = np.load(dist_matrix_path)
    n_sample_sizes, n_seeds, n_networks, _ = all_distances.shape
    print(all_distances.shape)

    n_plots = n_networks * n_networks
    n_cols = n_networks // 4
    n_rows = math.ceil(n_networks / n_cols)
    cmap = cm.get_cmap('viridis', n_networks)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    for i in range(n_networks):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        for j in range(n_networks):
            means = []
            stds = []
            for s in range(n_sample_sizes):
                vals = all_distances[s, :, i, j]  # all seeds
                means.append(np.mean(vals))
                stds.append(np.std(vals))

            label = f'j={j}' if i != j else f'j={j} (self)'
            ax.errorbar(sample_sizes, means, yerr=stds, label=label, capsize=3, fmt='-o', markersize=4, color=cmap(j))

        ax.set_xscale('log')
        ax.set_xlim(min(sample_sizes) * 0.9, max(sample_sizes) * 1.1)
        ax.set_title(f'Distances from i={i}')
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('Distance')

    plt.tight_layout()
    plt.savefig('d_vs_n.png')

    fig, axes = plt.subplots(n_seeds, n_sample_sizes, figsize=(4 * n_sample_sizes, 4 * n_seeds))

    for r_idx in range(n_seeds):
        for s_idx in range(n_sample_sizes):
            ax = axes[r_idx, s_idx] if n_seeds > 1 else axes[s_idx]
            sns.heatmap(all_distances[s_idx, r_idx], ax=ax, cmap="viridis", square=True, cbar=False)
            ax.set_title(f"N={sample_sizes[s_idx]}, Seed={seeds[r_idx]}")
            ax.set_xlabel("Net")
            ax.set_ylabel("Net")

    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
    plt.savefig('dist_pca_heatmaps.png')

def extract_activity_all(experiment_ls, base_path, args):

    activities = LayerActivityExtractor(checkpoint_path=f'{base_path}/experiments/{experiment_ls[0]}/checkpoints/best_model.pth',
            image_folder=args.data_path,batch_size=args.batch_size, num_workers=args.num_workers,test_size=args.test_size
        ).get_activities()
    network_names = activities.keys()
    X = list(activities.values())
    n, l, n_e = len(X[0]), len(network_names), len(experiment_ls)
    
    G_all = np.zeros((l*n_e, n, n), dtype=np.float32)
    network_names_all = []

    for j, exp in enumerate(experiment_ls):
        print(f"Loading network representation from {exp}...")

        activities = LayerActivityExtractor(checkpoint_path=f'{base_path}/experiments/{exp}/checkpoints/best_model.pth',
            image_folder=args.data_path,batch_size=args.batch_size, num_workers=args.num_workers,test_size=args.test_size
        ).get_activities()

        network_names = activities.keys()
        X = list(activities.values())
        for i in range(len(X)):
            x = X[i].reshape(len(X[i]), -1)
            G = x @ x.T
            G_all[j*l+i] = G.astype(np.float32)
        network_names_all.extend(network_names)

    return G_all, network_names_all

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def exp_to_name(exp_ls, symbol):
    name_ls = []
    param_ls = []
    for exp in exp_ls:
        param = exp.split('_')[-1] if exp != 'clean' else '0'
        param_ls.append(param)
        name_ls.append(f'{symbol}={param}')
    return name_ls, param_ls

def main():
    args = parse_args()
    set_seed(42)

    base_path = '/mnt/home/the10/ceph/results/netrep'
    output_path = f'{base_path}/experiments/{args.experiment}'
    result_path = f'{output_path}/results'
    result_path = f'{result_path}/{args.model}' if args.model != 'best_model' else result_path

    os.makedirs(result_path, exist_ok=True)
    experiment_ls = ['clean', 'weak_random_blur/exp_resnet_temp_0.2', 'weak_random_blur/exp_resnet_temp_0.4' ,'weak_random_blur/exp_resnet_temp_0.6', 'weak_random_blur/exp_resnet_temp_0.8',
        'weak_random_blur/exp_resnet_temp_1.0', 'weak_random_blur/exp_resnet_temp_2.0', 'weak_random_blur/exp_resnet_temp_3.0', 'weak_random_blur/exp_resnet_temp_4.0']

    if not os.path.isfile(f"{result_path}/euclidean_distance_matrix.npy"):
        print("Computing distance matrix...")

        # extract activity for each layer in the network
        if 'all' not in args.experiment:
            extractor = LayerActivityExtractor(
                checkpoint_path=f'{output_path}/checkpoints/{args.model}.pth',
                image_folder=args.data_path,
                batch_size=args.batch_size, 
                num_workers=args.num_workers,
                test_size=args.test_size
            )
                
            print("Extracting network representation...")
            activities = extractor.get_activities()
            X = list(activities.values())
            network_names = activities.keys()

        # extract activity for all networks
        else:
            G, network_names = extract_activity_all(experiment_ls, base_path, args)

        distmat = compute_distance_from_G(G)
    else:
        print("Loading precomputed distance matrix...")
        activities = LayerActivityExtractor(checkpoint_path=f'{base_path}/experiments/{experiment_ls[0]}/checkpoints/best_model.pth',
            image_folder=args.data_path,batch_size=args.batch_size, num_workers=args.num_workers,test_size=args.test_size
        ).get_activities()
        network_names = list(activities.keys()) * len(experiment_ls)
        distmat = np.load(f"{result_path}/euclidean_distance_matrix.npy")
        
    
    # TODO: temporary cut off first experiments for display, fix later
    # if 'all' in args.experiment:
    #     n_layers = len(network_names) // len(experiment_ls)
    #     distmat = distmat[n_layers:, n_layers:]
    #     network_names = network_names[n_layers:]
    #     experiment_ls = experiment_ls[1:]

    l = len(network_names) // len(experiment_ls)
    expanded_experiment_ls = [exp for exp in experiment_ls for _ in range(l)]
    name_ls, _ = exp_to_name(expanded_experiment_ls, symbol='T')
    _, param_ls = exp_to_name(experiment_ls, symbol='T')
    visualize_distance(distmat, name_ls, result_path)

    embedding = MDS(n_components=200, metric= True, eps = 0.00001, normalized_stress='auto', dissimilarity='precomputed', random_state=42)
    Z = embedding.fit_transform(np.abs(np.real(distmat)))
    print(embedding.stress_)

    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(Z)
    if 'all' in args.experiment:
        print(result_path)
        visualize_coordinates_all(coordinates, network_names, result_path, param_ls)
        visualize_layer_aligned(coordinates, network_names, result_path, param_ls, mds_dim=0)
        visualize_layer_aligned(coordinates, network_names, result_path, param_ls, mds_dim=1)
    else:
        visualize_coordinates(coordinates, network_names, result_path)

    
def visualize_distance(distmat, network_names, output_path):
    distmat = np.array(distmat)
    mask = np.eye(distmat.shape[0], dtype=bool)

    # Get vmin and vmax from off-diagonal entries only
    off_diag_vals = distmat[~mask]
    vmin = off_diag_vals.min()
    vmax = off_diag_vals.max()

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        distmat,
        cmap="viridis",
        square=True,
        xticklabels=dedup_labels_centered(network_names),
        yticklabels=dedup_labels_centered(network_names),
        vmin=vmin,
        vmax=vmax
    )

    plt.title("Representation Distance Matrix")
    plt.xlabel("Network Index")
    plt.ylabel("Network Index")
    plt.tight_layout()
    plt.savefig(f'{output_path}/euclidean_dist.png', dpi=250)
    np.save(f'{output_path}/euclidean_distance_matrix.npy', distmat)
    print(f"Saving to {output_path}/euclidean_dist.png")
    plt.close()

def dedup_labels_centered(labels):
    from collections import defaultdict

    label_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_indices[label].append(i)

    new_labels = [""] * len(labels)
    for label, indices in label_to_indices.items():
        mid = indices[len(indices) // 2]  # center index
        new_labels[mid] = label
    print(new_labels, label_to_indices)
    return new_labels

def visualize_coordinates(coords, network_names, output_path):
    plt.figure(figsize=(10, 8))

    plt.scatter(coords[:, 0], coords[:, 1], s=150, c=range(len(coords)))
    plt.plot(coords[:, 0], coords[:, 1], c='black')
    for i, name in enumerate(network_names):
        name = '.'.join(name.split('.')[1:])
        plt.text(coords[i, 0], coords[i, 1]+ 0.03, name, fontsize=10, ha='center', va='center')

    plt.title(f"MDS Embedding")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.colorbar()
    plt.savefig(f'{output_path}/euclidean_mds_embedding.png')
    print(f"Saving to {output_path}/euclidean_mds_embedding.png")
    plt.close()

def visualize_coordinates_all(coords, network_names, output_path, experiments):
    plt.figure(figsize=(10, 8))
    n_networks = len(network_names) // len(experiments)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'h', '*', 'p', 'x']
    texts = []
    print(experiments)

    for e in range(len(experiments)):
        selected_coords = coords[e*n_networks: (e+1)*n_networks]
        selected_network_names = network_names[e*n_networks: (e+1)*n_networks]

        plt.scatter(selected_coords[:, 0], selected_coords[:, 1], s=150, c=range(len(selected_coords)), marker=markers[e])
        plt.plot(selected_coords[:, 0], selected_coords[:, 1], color='gray', alpha=0.5, linestyle='--')

        for i, name in enumerate(selected_network_names):
            name = '.'.join(name.split('.')[1:])
            text = plt.text(selected_coords[i, 0], selected_coords[i, 1]+ 0.03, name, fontsize=8, ha='center', va='center')
            texts.append(text)

    for i in range(len(experiments)):
        plt.scatter([], [], marker=markers[i], color='gray', label=f'T={experiments[i]}')
    
    adjust_text(texts)
    plt.title(f"MDS Embedding")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.colorbar()
    plt.legend()
    plt.savefig(f'{output_path}/euclidean_mds_embedding.png')
    print(f"Saving to {output_path}/euclidean_mds_embedding.png")
    plt.close()

def visualize_layer_aligned(coords, network_names, output_path, experiments, mds_dim=1):
    plt.figure(figsize=(15, 4))
    n_networks = len(network_names) // len(experiments)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'h', '*', 'p', 'x']
    texts = []

    # Extract layer index for x-axis
    def extract_layer_id(name):
        match = re.search(r'layer(\d+\.\d+)', name)
        return float(match.group(1)) if match else -1

    layer_ids = [extract_layer_id(name) for name in network_names]

    for e in range(len(experiments)):
        start = e * n_networks
        end = (e + 1) * n_networks
        selected_coords = coords[start:end]
        selected_layer_ids = layer_ids[start:end]
        selected_network_names = network_names[start:end]

        plt.scatter(selected_layer_ids, selected_coords[:, mds_dim], s=150, c=range(len(selected_coords)), marker=markers[e], label=f'T={experiments[e]}')

        for i in range(n_networks):
            x = selected_layer_ids[i]
            y = selected_coords[i, mds_dim]
            label = '.'.join(selected_network_names[i].split('.')[1:])
            text = plt.text(x, y, label, fontsize=8, ha='center', va='center')
            texts.append(text)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.xlabel("Layer Index")
    plt.ylabel("MDS Dimension")
    plt.title(f"Layer-Aligned MDS Dim {mds_dim}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/euclidean_layer_aligned_mds{mds_dim}.png')
    print(f"Saving to {output_path}/euclidean_layer_aligned_mds{mds_dim}.png")
    plt.close()

if __name__ == "__main__":
    main()