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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

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

    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True
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

def exp_to_name(exp_ls):
    name_ls = []
    param_ls = []

    symbol_map = {'temp': 'T', 'cutout_patch_size': 'C', 'sigma': 0.3}

    for exp in exp_ls:
        if 'wide_resnet' and 'cutout_patch_size' in exp:
            words = exp.split('/')[-1].split('_')
            param_ls.append(words[-2])
            name_ls.append(f'C={words[-2]}, s={words[-1]}')
            continue
        if len(exp) > 60:
            symbol = 'combined'
            param = '0'
        elif 'clean' not in exp:                
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
    vmax, vmin = 0, 100
    for l in range(num_layers):
        indices = np.arange(l, distmat.shape[0], num_layers)
        selected_distmat = distmat[indices[:, None], indices]
        masked_selected_distmat = selected_distmat[~np.eye(len(selected_distmat), dtype=bool)]

        vmax = max(np.max(masked_selected_distmat), vmax)
        vmin = min(np.min(masked_selected_distmat), vmin)

    row, col = 4, len(network_names)//4
    fig, axes = plt.subplots(row, col, figsize=(col*5, row*4), gridspec_kw={'wspace': 0, 'hspace': 0})
    axes = axes.flatten()
    print(layer_names)

    sns.set(style='white')
    plt.rcParams.update({'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 10,
                        'ytick.labelsize': 10, 'axes.titlesize': 16})

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
            vmax=vmax+0.1,
            ax=ax,
            cbar=True,
            cbar_kws={"shrink": 0.5}
        )
        title = '.'.join(layer_names[l].split('.')[1:]) if '.' in layer_names[l] else layer_names[l]
        ax.set_title(title)
        ax.set_xlabel("Network")
        ax.set_ylabel("Network")

        if l % col > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        if l < (row - 1) * col:
            ax.set_xlabel('')
            ax.set_xticklabels([])

    plt.tight_layout()
    plt.savefig(f'{output_path}/dist.png')
    np.save(f'{output_path}/distance_matrix.npy', distmat)
    print(f"Saving to {output_path}/dist.png")
    plt.close()

def visualize_distance_histogram(distmat, layer_names, network_names, output_path):
    num_layers = len(layer_names)

    T_mask = np.array(['T' in n for n in network_names], dtype=bool)
    C_mask = np.array(['C' in n for n in network_names], dtype=bool)
    T_mask[0] = False
    
    distmat_T = distmat[np.repeat(T_mask, num_layers)].flatten()
    distmat_C = distmat[np.repeat(C_mask, num_layers)].flatten()
    print(distmat_T.shape, distmat_C.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    axes[0].hist(distmat_T, bins=20, color='cornflowerblue', edgecolor='black')
    axes[0].set_title('Blur')
    axes[0].set_xlabel('Distance')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(distmat_C, bins=20, color='lightcoral', edgecolor='black')
    axes[1].set_title('Cutout')
    axes[1].set_xlabel('Distance')

    plt.tight_layout()
    plt.savefig(f'{output_path}/hist.png', dpi=300)
    print(f"Saving to {output_path}/hist.png")
    plt.close()

def visualize_coordinates_2d(distmat, layer_names, network_names, output_path):
    num_layers = len(layer_names)
    rows, cols = 4, len(network_names)//4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for ax in axes[num_layers:]:
        ax.axis('off')

    coord_list = []
    for l in range(num_layers):
        indices = np.arange(l, distmat.shape[0], num_layers)
        selected_distmat = distmat[indices[:, None], indices]

        embedding = MDS(
            n_components=200,
            metric=True,
            eps=1e-5,
            normalized_stress='auto',
            dissimilarity='precomputed',
            random_state=42
        )
        Z = embedding.fit_transform(np.abs(np.real(selected_distmat)))
        print(f"Layer {layer_names[l]} stress: {embedding.stress_:.4f}")

        pca = PCA(n_components=2, random_state=42)
        coordinates = pca.fit_transform(Z)
        coord_list.append(coordinates)
    
    all_coords = np.vstack(coord_list)
    x_min, x_max = np.min(all_coords[:, 0]-0.05), np.max(all_coords[:, 0]+0.05)
    y_min, y_max = np.min(all_coords[:, 1]-0.05), np.max(all_coords[:, 1]+0.05)

    network_names = [name.split(',') for name in network_names]

    x = [float(c.split("=")[1]) for c, _ in network_names]
    y = [float(s.split("=")[1]) for _, s in network_names]

    C_vals = sorted(set(x))
    s_vals = sorted(set(y))

    C_to_idx = {c: i for i, c in enumerate(C_vals)}
    s_to_idx = {s: i for i, s in enumerate(s_vals)}

    C_colors = cm.Greens(np.linspace(0.3, 1.0, len(C_vals)))
    s_shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'h', '*', 'p', 'x'][:len(s_vals)]

    x = [float(c.split("=")[1]) for c, _ in network_names]
    y = [float(s.split("=")[1]) for _, s in network_names]

    C_colors = cm.Greens(np.linspace(0.3, 1, len(C_vals)))

    for l in range(num_layers):
        coordinates = coord_list[l]
        ax = axes[l]

        # scatter points
        for i, (c_val, s_val) in enumerate(zip(x, y)):
            color = C_colors[C_to_idx[c_val]]
            marker = s_shapes[s_to_idx[s_val]]
            ax.scatter(coordinates[i, 0], coordinates[i, 1], marker=marker, color=color, s=150)
        
        # plot grids
        for c_val in C_vals:
            idx = [i for i, xx in enumerate(x) if xx == c_val]
            pts = coordinates[idx]
            ax.plot(pts[:,0], pts[:,1], linestyle=":", color="gray", zorder=0)

        for s_val in s_vals:
            idx = [i for i, yy in enumerate(y) if yy == s_val]
            pts = coordinates[idx]
            ax.plot(pts[:,0], pts[:,1], linestyle=":", color="gray", zorder=0)       

        title = '.'.join(layer_names[l].split('.')[1:]) if '.' in layer_names[l] else layer_names[l]
        ax.set_title(title)
        ax.set_xlabel("MDS Dimension 1")
        ax.set_ylabel("MDS Dimension 2")

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            
    ax.legend(fontsize=8, labelspacing=0.8, frameon=False)

    handles = (
        [mpatches.Patch(color=C_colors[i], label=f"C={val}") for i, val in enumerate(C_vals)] +
        [mlines.Line2D([], [], color="black", marker=s_shapes[i],
                    ls="None", ms=10, label=f"s={val}") for i, val in enumerate(s_vals)]
    )
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 0.5), loc="center left")
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f'{output_path}/mds_embedding.png')
    print(f"Saved to {output_path}/mds_embedding.png")
    plt.close()

def visualize_coordinates_all(distmat, layer_names, network_names, output_path):
    num_layers = len(layer_names)
    rows, cols = 4, len(network_names)//4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for ax in axes[num_layers:]:
        ax.axis('off')

    all_coords = []

    coord_list = []
    for l in range(num_layers):
        indices = np.arange(l, distmat.shape[0], num_layers)
        selected_distmat = distmat[indices[:, None], indices]

        embedding = MDS(
            n_components=200,
            metric=True,
            eps=1e-5,
            normalized_stress='auto',
            dissimilarity='precomputed',
            random_state=42
        )
        Z = embedding.fit_transform(np.abs(np.real(selected_distmat)))
        print(f"Layer {layer_names[l]} stress: {embedding.stress_:.4f}")

        pca = PCA(n_components=2, random_state=42)
        coordinates = pca.fit_transform(Z)
        coord_list.append(coordinates)
        all_coords.append(coordinates)

    all_coords = np.vstack(all_coords)
    x_min, x_max = np.min(all_coords[:, 0]-0.05), np.max(all_coords[:, 0]+0.05)
    y_min, y_max = np.min(all_coords[:, 1]-0.05), np.max(all_coords[:, 1]+0.05)

    num_Ts = sum(['T' in n for n in network_names])
    num_Cs = sum(['C' in n for n in network_names])
    selected_Ts = num_Ts-1 if num_Ts > 1 else num_Ts
    T_colors = cm.Purples(np.linspace(0.3, 1, selected_Ts))
    C_colors = cm.Greens(np.linspace(0.3, 1, num_Cs))

    for l in range(num_layers):
        coordinates = coord_list[l]
        ax = axes[l]
        t_count, c_count = 0, 0
        for i, name in enumerate(network_names):
            if '=0' in name:
                marker, color = 'o', (0.6, 0.6, 0.6, 1.0)
            elif name.startswith('T='):
                marker = 'o'
                color = T_colors[t_count]
                t_count += 1
            else:
                marker = 's'
                color = C_colors[c_count]
                c_count += 1
            ax.scatter(coordinates[i, 0], coordinates[i, 1], marker=marker, color=color, label=network_names[i], s=150)
        
        # ax.plot(coordinates[:num_Ts, 0], coordinates[:num_Ts, 1], color='gray', alpha=0.5, linestyle='--')
        # C_indices = np.r_[0, np.arange(num_Ts, num_Ts+num_Cs)]
        # ax.plot(coordinates[C_indices, 0], coordinates[C_indices, 1], color='gray', alpha=0.5, linestyle='--')
        ax.plot(coordinates[:, 0], coordinates[:, 1], color='gray', alpha=0.5, linestyle='--')

        ax.set_title('.'.join(layer_names[l].split('.')[1:]))
        ax.set_xlabel("MDS Dimension 1")
        ax.set_ylabel("MDS Dimension 2")
        ax.legend(fontsize=8, labelspacing=0.8, frameon=False)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{output_path}/mds_embedding.png')
    print(f"Saved to {output_path}/mds_embedding.png")
    plt.close()

def compute_angles(a, b, c):
    return np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

def compute_all_angles(distmat):
    n = distmat.shape[0]
    angles = np.zeros((n, n, n))  # angle at i between j and k

    for i in tqdm(range(n)):
        for j in range(n):
            for k in range(n):
                if i == j or i == k or j == k:
                    continue  # degenerate triangle
                angles[i, j, k] = compute_angles(distmat[i, j], distmat[i, k], distmat[j, k])

    return angles

def visualize_angles(angles, layer_names, network_names, output_path, symbol="T"):
    T_indices = np.where(np.char.find(network_names, symbol) >= 0)[0]
    angle_lists = []
    num_layers = len(layer_names)

    # Get angles vmin and vmax from off-diagonal entries only
    vmax, vmin = 0, 100
    for l in range(num_layers):
        indices = np.arange(l, angles.shape[0], num_layers)
        selected_angles = angles[np.ix_(indices, indices, indices)]
        nonzero_angles = selected_angles[selected_angles != 0]

        vmax = max(np.max(nonzero_angles), vmax)
        vmin = min(np.min(nonzero_angles), vmin)

    fig, axes = plt.subplots(4, 5, figsize=(20, 20), gridspec_kw={'wspace': 0, 'hspace': 0})
    axes = axes.flatten()

    for l in range(num_layers):
        indices = np.arange(l, angles.shape[0], num_layers)
        selected_angles = angles[np.ix_(indices, indices, indices)]

        ax = axes[l]
        sns.heatmap(
            selected_angles[0],
            cmap="viridis",
            square=True,
            xticklabels=network_names,
            yticklabels=network_names,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar=True,
            cbar_kws={"shrink": 0.5}
        )
        ax.set_title('.'.join(layer_names[l].split('.')[1:]))
        ax.set_xlabel("Network")
        ax.set_ylabel("Network")

        if l % 4 > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        if l // 4 < 3:
            ax.set_xlabel('')
            ax.set_xticklabels([])

        angle_list = []
        for i in range(1, len(T_indices) - 1):
            a = selected_angles[T_indices[i - 1], T_indices[i], T_indices[i + 1]]
            angle_list.append(np.degrees(a))
        angle_lists.append(angle_list)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/angle_matrix.png', dpi=300)
    print(f"Saved to {output_path}/angle_matrix.png")
    plt.close()

    x = np.arange(num_layers)
    plt.figure(figsize=(10, 5))
    for l in range(num_layers):
        y_vals = angle_lists[l]
        x_vals = [x[l]] * len(y_vals)
        plt.scatter(x_vals, y_vals, alpha=0.7)

    plt.xticks(x, [name.replace('module.', '') for name in layer_names], rotation=45, ha='right')
    plt.ylabel("Angle (degree)")
    plt.xlabel("Layer")
    plt.title(f"Mean Angle Between Consecutive {symbol}")
    plt.tight_layout()
    plt.savefig(f"{output_path}/angles_{symbol}.png", dpi=300)
    plt.close()
    print(f"Saved to {output_path}/angles_{symbol}.png")
    

def compare_angles(angles, layer_names, network_names, output_path):
    angles[0]
    num_layers = len(layer_names)

    T_mask = np.array(['T' in n for n in network_names], dtype=bool)
    C_mask = np.array(['C' in n for n in network_names], dtype=bool)
    T_mask[0] = False
    T_mean, T_std, C_mean, C_std, cross_mean, cross_std = [], [], [], [], [], []
    
    for l in range(num_layers):
        indices = np.arange(l, angles.shape[0], num_layers)
        selected_angles = angles[np.ix_(indices, indices, indices)][0]

        T_vals = np.degrees(selected_angles[np.ix_(T_mask, T_mask)])
        C_vals = np.degrees(selected_angles[np.ix_(C_mask, C_mask)])
        X_vals = np.degrees(selected_angles[np.ix_(T_mask, C_mask)])
        
        T_nonzero = T_vals[T_vals != 0]
        C_nonzero = C_vals[C_vals != 0]
        X_nonzero = X_vals[X_vals != 0]

        T_mean.append(np.mean(T_nonzero))
        T_std.append(np.std(T_nonzero))

        C_mean.append(np.mean(C_nonzero))
        C_std.append(np.std(C_nonzero))

        cross_mean.append(np.mean(X_nonzero))
        cross_std.append(np.std(X_nonzero))
    
    x = np.arange(num_layers)
    plt.figure(figsize=(12, 5))

    # T vs cross
    plt.subplot(1, 2, 1)
    plt.errorbar(x - 0.05, T_mean, yerr=T_std, fmt='-o', label='T-T')
    plt.errorbar(x + 0.05, cross_mean, yerr=cross_std, fmt='-o', label='T-C')
    plt.title("Angles: T vs Cross")
    plt.xlabel("Layer")
    plt.ylabel("Angle (degrees)")
    plt.xticks(x, [l.replace("module.", "") for l in layer_names], rotation=45, ha='right')
    plt.legend()

    # C vs cross
    plt.subplot(1, 2, 2)
    plt.errorbar(x - 0.05, C_mean, yerr=C_std, fmt='-o', label='C-C')
    plt.errorbar(x + 0.05, cross_mean, yerr=cross_std, fmt='-o', label='C-T')
    plt.title("Angles: C vs Cross")
    plt.xlabel("Layer")
    plt.ylabel("Angle (degrees)")
    plt.xticks(x, [l.replace("module.", "") for l in layer_names], rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}/angle_compare_TC.png", dpi=300)
    plt.close()
    print(f"Saved to {output_path}/angle_compare_TC.png")

def main():
    args = parse_args()
    set_seed(42)

    base_path = '/mnt/home/the10/ceph/results/netrep'
    output_path = f'{base_path}/experiments/{args.experiment}'
    result_path = f'{output_path}/results_seed{args.seed}' if args.seed else f'{output_path}/results'
    output_path = f'{result_path}/analysis'

    os.makedirs(output_path, exist_ok=True)
    pretrain_str = "/pretrained" if args.pretrained else ""
    pretrain_str = f"{pretrain_str}_seed{args.seed}" if args.seed else pretrain_str
    experiment_ls = [f"cutout{pretrain_str}/exp_wide_resnet_cutout_patch_size_{c}_{s}" for c in [12.0, 16.0, 20.0, 24.0] for s in [0.2, 0.3, 0.5, 0.8, 1.0]]
    # experiment_ls += [f"cutout_seed43/exp_wide_resnet_cutout_patch_size_16.0_0.5"]
    # experiment_ls += [f"cutout_seed44/exp_wide_resnet_cutout_patch_size_16.0_0.5"]
    # experiment_ls.remove("cutout/pretrained/exp_wide_resnet_cutout_patch_size_20.0_0.2")

    print(f"Loading distmat from {result_path}/distance_matrix.npy")
    distmat = np.load(f"{result_path}/distance_matrix.npy")
    # network_indices = [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 4.0, 4.1, 4.2]
    network_indices = [3.5, 4.2]
    network_names = [f'module.layer{i}' for i in network_indices] + ['avgpool', 'fc']

    l = len(network_names)
    name_ls = exp_to_name(experiment_ls)[0]

    # TODO: look into cutout patch size 80, blur temp 1.0, remove it for now
    print(len(distmat), l)
    # distmat = np.delete(distmat, np.arange(20*l, 25*l), axis=0)
    # distmat = np.delete(distmat, np.arange(20*l, 25*l), axis=1)
    # distmat = np.delete(distmat, np.arange(10*l, 11*l), axis=0)
    # distmat = np.delete(distmat, np.arange(10*l, 11*l), axis=1)

    print(distmat.shape)
    visualize_distance_all(distmat, network_names, name_ls, output_path)
    visualize_coordinates_2d(distmat, network_names, name_ls, output_path)
    # visualize_distance_histogram(distmat, network_names, name_ls, output_path)

    # angles = compute_all_angles(distmat)
    # compare_angles(angles, network_names, name_ls, output_path)
    # visualize_angles(angles, network_names, name_ls, output_path, symbol='T')
    # name_ls[0] = 'C=0'
    # visualize_angles(angles, network_names, name_ls, output_path, symbol='C')
    # compare_angles(angles, network_names, name_ls, output_path)

if __name__ == "__main__":
    main()