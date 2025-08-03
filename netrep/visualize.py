import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import seaborn as sns
from tqdm import tqdm
from adjustText import adjust_text
import re
import random
import gc
import math

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
    plt.savefig(f'{output_path}/dist.png', dpi=250)
    np.save(f'{output_path}/distance_matrix.npy', distmat)
    print(f"Saving to {output_path}/dist.png")
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
    plt.savefig(f'{output_path}/mds_embedding.png')
    print(f"Saving to {output_path}/mds_embedding.png")
    plt.close()

def visualize_coordinates_all(coords, network_names, output_path, experiments):
    plt.figure(figsize=(10, 8))
    n_networks = len(network_names) // len(experiments)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'h', '*', 'p', 'x']
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

    for i in range(len(experiments)):
        plt.scatter([], [], marker=markers[i], color='gray', label=experiments[i])
    
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

        plt.scatter(selected_layer_ids, selected_coords[:, mds_dim], s=150, c=range(len(selected_coords)), marker=markers[e], label=experiments[e])

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

