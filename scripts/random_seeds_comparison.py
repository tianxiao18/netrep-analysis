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

from sklearn.model_selection import train_test_split
import itertools
from netrep.metrics import LinearMetric
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
        type=str
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
        type=str
    )

    parser.add_argument(
        "--pretrained",
        action="store_true"
    )

    return parser.parse_args()

def split_activity(activities, save_path=None, n_components=1000, test_size=0.2):
    X_train, X_test = [], []

    print("Computing PCA-reduced activities...")
    pca_activities = {}

    for key in activities.keys():
        X = activities[key]
        if isinstance(X, str):
            X = np.load(X, mmap_mode="r")
        print(key, X.shape)
        X = X.astype(np.float32).reshape(X.shape[0], -1)
        k = min(n_components, X.shape[0], X.shape[1])

        if k < X.shape[1]:
            pca = DualPCA(n_components=k)
            X = pca.fit_transform(X)
            print(f"{key} variance explained in {k} dim: {np.sum(pca.explained_variance_ratio_)}")
        
        pca_activities[key] = X
        
    if save_path:
        np.savez_compressed(save_path, **pca_activities)
        print(f"Saved PCA activities to {save_path}")

    for key in pca_activities:
        X_pca = pca_activities[key]
        if test_size > 0:
            X1_train, X1_test = train_test_split(X_pca, test_size=test_size, random_state=42)
        else:
            X1_train, X1_test = X_pca, []
        X_train.append(X1_train)
        X_test.append(X1_test)

    return X_train, X_test


def compute_layer_distmat(X_train, X_test, indices):
    m = len(indices)
    D = np.zeros((m, m), dtype=float)
    total_pairs = m * (m - 1) // 2

    for a, b in tqdm(itertools.combinations(range(m), 2),
                     total=total_pairs, desc="Computing layer distances"):
        i, j = indices[a], indices[b]

        metric = LinearMetric(alpha=1.0, center_columns=True, score_method="angular")
        metric.fit(X_train[i], X_train[j])
        D[a, b] = metric.score(X_test[i], X_test[j])

    D += D.T
    return D

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

def extract_activity_all(experiment_ls, base_path, args):
    X_train_all, X_test_all, layer_names_all = [], [], []

    # extract network names, assuming all experiments has the same network names
    print(f"Loading from {base_path}/experiments/{experiment_ls[0]}/checkpoints/final_model.pth")
    print(base_path, experiment_ls[0])
    layer_names = LayerActivityExtractor(checkpoint_path=f'{base_path}/experiments/{experiment_ls[0]}/checkpoints/final_model.pth',
        image_folder=args.data_path,batch_size=args.batch_size, num_workers=args.num_workers,test_size=args.test_size, 
        model_name=args.model_name, dataset=args.dataset
    ).get_layer_names()

    for exp in experiment_ls:
        output_path = f'{base_path}/experiments/{exp}'
        os.makedirs(f'{output_path}/results', exist_ok=True)
        save_path = f'{output_path}/results/activities_pca.npz'

        if save_path and os.path.isfile(save_path):
            print(f"Loading PCA-reduced activities from {save_path}")

            with np.load(save_path, allow_pickle=False) as data:
                pca_activities = {k: data[k] for k in data.files}
                X_train = [pca_activities[name].reshape(pca_activities[name].shape[0], -1) for name in layer_names]
                X_test = [pca_activities[name].reshape(pca_activities[name].shape[0], -1) for name in layer_names]
        else:
            print(f"Extracting PCA-reduced activities to {save_path}")
            extractor = LayerActivityExtractor(
                checkpoint_path=f'{output_path}/checkpoints/{args.model}.pth',
                image_folder=args.data_path,
                batch_size=args.batch_size, 
                num_workers=args.num_workers,
                test_size=args.test_size,
                model_name=args.model_name,
                dataset=args.dataset
            )
            layer_names = extractor.get_layer_names()
            activities = extractor.get_activities()
            X_train, X_test = split_activity(activities, save_path=save_path, n_components=1000, test_size=0)

        X_train_all.extend(X_train)
        X_test_all.extend(X_test)
        layer_names_all.extend(layer_names)

    return X_train_all, X_test_all, layer_names_all


def visualize_distance_all(distmat, layer_names, network_names, output_path):
    distmat = np.array(distmat)
    num_layers = len(layer_names)

    # Get vmin and vmax from off-diagonal entries only
    vmax, vmin = 0, 100
    for l in range(num_layers):
        selected_distmat = distmat[l]
        masked_selected_distmat = selected_distmat[~np.eye(len(selected_distmat), dtype=bool)]

        vmax = max(np.max(masked_selected_distmat), vmax)
        vmin = min(np.min(masked_selected_distmat), vmin)

    row, col = 1, 4
    fig, axes = plt.subplots(row, col, figsize=(col*5, row*4), gridspec_kw={'wspace': 0, 'hspace': 0})
    axes = axes.flatten()
    print(layer_names)

    sns.set(style='white')
    plt.rcParams.update({'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 10,
                        'ytick.labelsize': 10, 'axes.titlesize': 16})

    for l in range(num_layers):
        selected_distmat = distmat[l]

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

def visualize_coordinates_2d(coord_list, layer_names, network_names, output_path):
    num_layers = len(layer_names)
    rows, cols = 1, 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for ax in axes[num_layers:]:
        ax.axis('off')

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

    color_values = np.linspace(0.3, 1.0, len(C_vals))
    C_colors = [cm.Purples(color_values), cm.Blues(color_values), cm.Greens(color_values), cm.Oranges(color_values)]
    base_shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'h', '*', 'p', 'x']
    s_shapes = list(itertools.islice(itertools.cycle(base_shapes), len(s_vals)))

    x = [float(c.split("=")[1]) for c, _ in network_names]
    y = [float(s.split("=")[1]) for _, s in network_names]

    cs_to_idx = {(c_val, s_val): i for i, (c_val, s_val) in enumerate(zip(x, y))}
    all_coords = all_coords.reshape(3, len(coord_list), -1, 2) # 3 seeds, 4 layers, 49 augs, 2 coordinates
    print(all_coords.shape, coord_list[0].shape)

    for s in range(3):
        for l in range(num_layers):
            coordinates = all_coords[s][l]
            ax = axes[l]

            # scatter points
            for i, (c_val, s_val) in enumerate(zip(x, y)):
                color = C_colors[l][C_to_idx[c_val]]
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

            special_path = [(0.0, 0.0),(12.0, 0.2),(12.0, 0.3),(12.0, 0.5),(12.0, 1.0)]

            special_indices = [
                cs_to_idx[(c_val, s_val)]
                for (c_val, s_val) in special_path
                if (c_val, s_val) in cs_to_idx
            ]

            if len(special_indices) >= 2:
                pts = coordinates[special_indices]
                for i in range(1, len(pts)):
                    ax.plot([pts[0,0],pts[i,0]], [pts[0,1],pts[i,1]], linestyle=":", color="gray", zorder=0)
            
    ax.legend(fontsize=8, labelspacing=0.8, frameon=False)

    handles = (
        [mpatches.Patch(color=C_colors[-1][i], label=f"C={val}") for i, val in enumerate(C_vals)] +
        [mlines.Line2D([], [], color="black", marker=s_shapes[i],
                    ls="None", ms=10, label=f"s={val}") for i, val in enumerate(s_vals)]
    )
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 0.5), loc="center left")

    plt.tight_layout()
    plt.savefig(f'{output_path}/mds_embedding.png')
    print(f"Saved to {output_path}/mds_embedding.png")
    plt.close()


def main():
    args = parse_args()
    set_seed(42)
    seeds = [42, 43, 44]

    base_path = '/mnt/home/the10/ceph/results/netrep'
    output_path = f'{base_path}/experiments/{args.experiment}'
    print(args.dataset, args.model_name)

    all_Xs, all_network_names = [], []

    for seed in seeds:
        result_path = f'{output_path}/results_seed{seed}' if seed != 42 else f'{output_path}/results'
        result_path = f'{result_path}_pretrained' if args.pretrained else result_path
        result_path = f'{result_path}/{args.model}' if args.model != 'final_model' else result_path
        os.makedirs(result_path, exist_ok=True)
        print(result_path)

        pretrain_str = "/pretrained" if args.pretrained else ""
        pretrain_str = f"{pretrain_str}_seed{seed}" if seed != 42 else pretrain_str
        experiment_ls = [f"cutout{pretrain_str}/exp_{args.model_name}_cutout_patch_size_{c}_{s}" for c in [4.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0] for s in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]]

        X_train, _, network_names = extract_activity_all(experiment_ls, base_path, args)
        all_Xs.extend(X_train)
        all_network_names.extend(network_names)

    # compute procruste's distance between pair of networks
    num_layers = len(set(all_network_names))
    print(len(all_network_names), len(all_Xs))

    dist_list = []
    coord_list = []
    os.makedirs(f'{output_path}/results_seed_combined', exist_ok=True)
    network_indices = [3.1, 4.1]
    layer_names = [f'module.layer{i}' for i in network_indices] + ['avgpool', 'fc']
    network_names = exp_to_name(experiment_ls)[0]

    for l in range(num_layers):
        indices = np.arange(l, len(all_Xs), num_layers)
        print(len(all_Xs), indices)

        distmat_path = f'{output_path}/results_seed_combined/distance_matrix_layer{l}.npy'
        if not os.path.isfile(distmat_path):
            selected_distmat = compute_layer_distmat(all_Xs, all_Xs, indices)
            np.save(distmat_path, selected_distmat)
        else:
            selected_distmat = np.load(distmat_path)
        dist_list.append(selected_distmat)
        print(selected_distmat.shape)

        embedding = MDS(
            n_components=550,
            metric=True,
            eps=1e-5,
            normalized_stress="auto",
            dissimilarity="precomputed",
            random_state=42,
        )
        Z = embedding.fit_transform(np.abs(np.real(selected_distmat)))
        print(f"Layer {all_network_names[l]} stress: {embedding.stress_:.4f}")

        coordinates = PCA(n_components=2, random_state=42).fit_transform(Z)
        coord_list.append(coordinates)
        
    visualize_distance_all(dist_list, layer_names, ['']*len(selected_distmat), f'{output_path}/results_seed_combined')
    visualize_coordinates_2d(coord_list, layer_names, network_names, f'{output_path}/results_seed_combined')

if __name__ == "__main__":
    main()