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
import seaborn as sns
from tqdm import tqdm
from adjustText import adjust_text
import re

def parse_args():
    parser = ArgumentParser(description="PyTorch Resnet Trainer")
    parser.add_argument(
        "--test_size",
        type=int,
        default=1000,
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

    return parser.parse_args()

def split_activity(activities, save_path=None, n_components=1000):
    X_train, X_test = [], []

    if save_path and os.path.isfile(save_path):
        print(f"Loading PCA-reduced activities from {save_path}")
        pca_activities = np.load(save_path)

    else:
        print("Computing PCA-reduced activities...")
        pca_activities = {}

        for key in activities:
            X = activities[key]
            pca = PCA(n_components=n_components)
            
            X_pca = pca.fit_transform(X.reshape(X.shape[0], -1))
            print(f"{key} variance explained: {np.sum(pca.explained_variance_ratio_)}")
            pca_activities[key] = X_pca
            
        if save_path:
            np.savez_compressed(save_path, **pca_activities)
            print(f"Saved PCA activities to {save_path}")

    for key in pca_activities:
        X_pca = pca_activities[key]
        X1_train, X1_test = train_test_split(X_pca, test_size=0.2, random_state=42)
        X_train.append(X1_train)
        X_test.append(X1_test)

    return X_train, X_test


def compute_distances(X_train, X_test):
    distmat = np.zeros((len(X_train), len(X_train)))
    total_pairs = len(X_train) * (len(X_train) - 1) // 2

    # use procruste's metrics
    for i, j in tqdm(itertools.combinations(range(len(X_train)), 2), total=total_pairs, desc="Computing distances"):
        metric = LinearMetric(alpha=1.0, center_columns=True, score_method='angular')
        metric.fit(X_train[i], X_train[j])
        dist = metric.score(X_test[i], X_test[j])
        distmat[i, j] = dist

    distmat += distmat.T
    return distmat

def extract_activity_all(experiment_ls, base_path):
    X_train_all, X_test_all, network_names_all = [], [], []

    for exp in experiment_ls:
        output_path = f'{base_path}/experiments/{exp}'
        embed_path = f'{output_path}/results/activities.npz'
        save_path = f'{output_path}/results/activities_pca.npz'

        print(f"Loading network representation from {exp}...")
        activities = np.load(embed_path)
        
        X_train, X_test = split_activity(activities, save_path, n_components=1000)
        X_train_all.extend(X_train)
        X_test_all.extend(X_test)
        network_names_all.extend(activities.keys())

    return X_train_all, X_test_all, network_names_all

def main():
    args = parse_args()

    base_path = '/mnt/home/the10/netrep-analysis'
    output_path = f'{base_path}/experiments/{args.experiment}'
    result_path = f'{output_path}/results'
    embed_path = f'{result_path}/activities.npz'

    os.makedirs(result_path, exist_ok=True)
    experiment_ls = ['clean', 'weak_random_blur/exp_resnet_sigma_1', 'weak_random_blur/exp_resnet_sigma_2',
                     'weak_random_blur/exp_resnet_sigma_3', 'weak_random_blur/exp_resnet_sigma_4']

    # extract activity for each layer in the network
    if args.experiment != 'all':
        extractor = LayerActivityExtractor(
            checkpoint_path=f'{output_path}/checkpoints/best_model.pth',
            image_folder=args.data_path,
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            test_size=args.test_size
        )
            
        if not os.path.isfile(embed_path):
            print("Extracting network representation...")
            activities = extractor.get_activities()
            np.savez_compressed(embed_path, **activities)
        else:
            print("Loading network representation...")
            activities = np.load(embed_path)

        X_train, X_test = split_activity(activities, save_path=f'{result_path}/activities_pca.npz', n_components=1000)
        network_names = activities.keys()
    # extract activity for all networks
    else:
        X_train, X_test, network_names = extract_activity_all(experiment_ls, base_path)

    # compute procruste's distance between pair of networks
    if not os.path.isfile(f"{result_path}/distance_matrix.npy"):
        distmat = compute_distances(X_train, X_test)
    else:
        distmat = np.load(f"{result_path}/distance_matrix.npy")
    
    # TODO: temporary cut off last experiments for display, fix later
    if args.experiment == 'all':
        n_layers = len(network_names) // len(experiment_ls)
        distmat = distmat[:(len(experiment_ls)-1)*n_layers, :(len(experiment_ls)-1)*n_layers]
        network_names = network_names[:(len(experiment_ls)-1)*n_layers]

    visualize_distance(distmat, network_names, result_path)

    embedding = MDS(n_components=200, metric= True, eps = 0.00001, normalized_stress='auto', dissimilarity='precomputed', random_state=42)
    Z = embedding.fit_transform(np.abs(np.real(distmat)))
    print(embedding.stress_)

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(Z)
    if args.experiment == 'all':
        visualize_coordinates_all(coordinates, network_names, result_path, experiment_ls[:-1])
        visualize_layer_aligned(coordinates, network_names, result_path, experiment_ls[:-1], mds_dim=0)
        visualize_layer_aligned(coordinates, network_names, result_path, experiment_ls[:-1], mds_dim=1)
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
        xticklabels=network_names,
        yticklabels=network_names,
        vmin=vmin,
        vmax=vmax
    )

    plt.title("Representation Distance Matrix")
    plt.xlabel("Network Index")
    plt.ylabel("Network Index")
    plt.tight_layout()
    plt.savefig(f'{output_path}/dist.png')
    np.save(f'{output_path}/distance_matrix.npy', distmat)
    print(f"Saving to {output_path}/dist.png")
    plt.close()


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
    plt.savefig(f'{output_path}/mds_embedding.png')
    print(f"Saving to {output_path}/mds_embedding.png")
    plt.close()

def visualize_coordinates_all(coords, network_names, output_path, experiments):
    plt.figure(figsize=(10, 8))
    n_networks = len(network_names) // len(experiments)
    markers = ['o', 's', '^', 'D']
    texts = []

    for e in range(len(experiments)):
        selected_coords = coords[e*n_networks: (e+1)*n_networks]
        selected_network_names = network_names[e*n_networks: (e+1)*n_networks]

        plt.scatter(selected_coords[:, 0], selected_coords[:, 1], s=150, c=range(len(selected_coords)), marker=markers[e])
        plt.plot(selected_coords[:, 0], selected_coords[:, 1], color='gray', alpha=0.5, linestyle='--')

        for i, name in enumerate(selected_network_names):
            name = '.'.join(name.split('.')[1:])
            text = plt.text(selected_coords[i, 0], selected_coords[i, 1]+ 0.03, name, fontsize=8, ha='center', va='center')
            texts.append(text)

    for net_i in range(4):
        plt.scatter([], [], marker=markers[net_i], color='gray', label=f'T={net_i}')
    
    adjust_text(texts)
    plt.title(f"MDS Embedding")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.colorbar()
    plt.legend()
    plt.savefig(f'{output_path}/mds_embedding.png')
    print(f"Saving to {output_path}/mds_embedding.png")
    plt.close()

def visualize_layer_aligned(coords, network_names, output_path, experiments, mds_dim=1):
    plt.figure(figsize=(15, 4))
    n_networks = len(network_names) // len(experiments)
    markers = ['o', 's', '^', 'D']
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

        plt.scatter(selected_layer_ids, selected_coords[:, mds_dim], s=150, c=range(len(selected_coords)), marker=markers[e], label=f'T={e}')

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
    plt.savefig(f'{output_path}/layer_aligned_mds{mds_dim}.png')
    print(f"Saving to {output_path}/layer_aligned_mds{mds_dim}.png")
    plt.close()

if __name__ == "__main__":
    main()