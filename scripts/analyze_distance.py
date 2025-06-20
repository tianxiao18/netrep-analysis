import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from adjustText import adjust_text
import re

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
    plt.savefig(f'dist.png')
    print(f"Saving to dist.png")
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
        plt.scatter([], [], marker=markers[net_i], color='gray', label=f'{experiments[net_i][-7:]}')
    
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
    plt.figure(figsize=(12, 6))
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

        plt.scatter(selected_layer_ids, selected_coords[:, mds_dim], s=150, c=range(len(selected_coords)), marker=markers[e], label=f'{experiments[e][-7:]}')

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
    


def main():
    dist_path = '/mnt/home/the10/netrep-analysis/experiments/all/results/distance_matrix.npy'
    dist = np.load(dist_path)
    n_exp = 5
    n_layers = len(dist)//n_exp
    experiment_ls = ['clean', 'weak_random_blur/exp_resnet_sigma_1', 'weak_random_blur/exp_resnet_sigma_2',
                     'weak_random_blur/exp_resnet_sigma_3', 'weak_random_blur/exp_resnet_sigma_4']

    distmat = dist
    network_names = ['module.layer1.0', 'module.layer1.1', 'module.layer1.2', 'module.layer2.0', 'module.layer2.1', 'module.layer2.2', 'module.layer2.3', 'module.layer3.0', 'module.layer3.1', 'module.layer3.2', 'module.layer3.3', 'module.layer3.4', 'module.layer3.5', 'module.layer4.0', 'module.layer4.1', 'module.layer4.2']
    network_names = network_names + network_names + network_names + network_names

    visualize_distance(distmat, network_names, '.')

    embedding = MDS(n_components=200, metric= True, eps = 0.00001, normalized_stress='auto', dissimilarity='precomputed', random_state=42)
    Z = embedding.fit_transform(np.abs(np.real(distmat)))
    print(embedding.stress_)

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(Z)
    print(coordinates.shape)
    visualize_coordinates_all(coordinates, network_names, '.', experiment_ls[:-1])
    visualize_layer_aligned(coordinates, network_names, '.', experiment_ls[:-1], mds_dim=0)
    visualize_layer_aligned(coordinates, network_names, '.', experiment_ls[:-1], mds_dim=1)

if __name__ == "__main__":
    main()