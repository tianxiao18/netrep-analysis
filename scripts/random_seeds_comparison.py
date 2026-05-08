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
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from tqdm import tqdm
from adjustText import adjust_text
import re
import random
import gc
import math

# Cutout patch × σ grid; flat order is ``for c in CUTOUT_PATCH_GRID: for s in CUTOUT_SIGMA_GRID``.
CUTOUT_PATCH_GRID = [4.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0]
CUTOUT_SIGMA_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]


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

    parser.add_argument(
        "--aug_type",
        type=str,
        default="gaussian_noise"
    )

    parser.add_argument(
        "--load_pca",
        action="store_true"
    )

    parser.add_argument(
        "--single_param",
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
        if 'wide_resnet' in exp or 'cutout_patch_size' in exp:
            words = exp.split('/')[-1].split('_')
            param_ls.append(words[-2])
            name_ls.append(f'C={words[-2]}, s={words[-1]}')
            continue
        if 'cutout_patch_size' in exp:
            words = exp.split('/')[-1].split('_')
            param_ls.append(words[-2])
            name_ls.append(f'C={words[-2]}')
            continue
        if 'rot_deg' in exp and '0.0' not in exp:
            words = exp.split('/')[-1].split('_')
            param_ls.append(words[-3])
            name_ls.append(f'r={words[-2]}, s={words[-1]}')
            continue
        if len(exp.split('/')[-2].split('_')) == 1 or 'weak_random_blur' in exp or 'sp_noise' in exp or 'noise_std' in exp:
            words = exp.split('/')[-1].split('_')
            param_ls.append(words[-2])
            if words[-3] in symbol_map:
                 name_ls.append(f'{symbol_map[words[-3]]}={words[-2]}')
            elif 'clean' in words:
                name_ls.append('clean=0.0')
            else:
                aug_name = '_'.join(words[-4:-2])
                name_ls.append(f'{aug_name}={words[-2]}')
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

        if save_path and os.path.isfile(save_path) and args.load_pca:
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

    # Color scale from off-diagonals only (diagonal is ~0 self-distance and would crush contrast)
    # vmin, vmax = np.inf, -np.inf
    # for l in range(num_layers):
    #     selected_distmat = distmat[l]
    #     n = len(selected_distmat)
    #     off_diagonal = selected_distmat[~np.eye(n, dtype=bool)]

    #     vmax = max(float(np.max(off_diagonal)), vmax)
    #     vmin = min(float(np.min(off_diagonal)), vmin)
    # vmin, vmax = 0.35, 0.85
    row, col = 1, 4
    fig, axes = plt.subplots(row, col, figsize=(col*5, row*4), gridspec_kw={'wspace': 0, 'hspace': 0})
    axes = axes.flatten()
    print(layer_names)

    sns.set(style='white')
    plt.rcParams.update({'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 10,
                        'ytick.labelsize': 10, 'axes.titlesize': 16})

    for l in range(num_layers):
        selected_distmat = distmat[l]
        vmin, vmax = np.inf, -np.inf
        off_diagonal = selected_distmat[~np.eye(len(selected_distmat), dtype=bool)]

        vmax = max(np.max(off_diagonal), vmax)
        vmin = min(np.min(off_diagonal), vmin)

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
    plt.savefig(f'{output_path}/dist.png', dpi=200)
    np.save(f'{output_path}/distance_matrix.npy', distmat)
    print(f"Saving to {output_path}/dist.png")
    plt.close()

def visualize_coordinates_2d(coord_list, layer_names, network_names, output_path, single_param=False):
    """
    Visualizes 2D coordinates for network embeddings.

    If single_param=True, assumes network_names is like ['noise_std=0.01', ...] and only colors by this value.

    Otherwise, expects network_names as pairs: e.g.,
        ['C=0.01,s=0.2', ...] or ['C=0.01,s=0.2', ...]
    """
    num_layers = len(layer_names)
    rows, cols = 1, 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
    axes = axes.flatten()

    for ax in axes[num_layers:]:
        ax.axis('off')

    all_coords = np.vstack(coord_list)
    x_min, x_max = np.min(all_coords[:, 0]-0.05), np.max(all_coords[:, 0]+0.05)
    y_min, y_max = np.min(all_coords[:, 1]-0.05), np.max(all_coords[:, 1]+0.05)

    if single_param:
        # Single hyperparameter mode
        c_vals = [float(name.split("=")[1]) for name in network_names]
        C_vals = sorted(set(c_vals))
        C_to_idx = {c: i for i, c in enumerate(C_vals)}

        color_values = np.linspace(0.3, 1.0, len(C_vals))
        C_colors = cm.Purples(color_values)
        marker = 'o'

        all_coords = all_coords.reshape(3, len(coord_list), -1, 2) # (n_seed, n_layers, n_aug, 2)
        print(all_coords.shape, coord_list[0].shape)

        for s in range(3):
            for l in range(num_layers):
                coordinates = all_coords[s][l]
                ax = axes[l]
                for i, c_val in enumerate(c_vals):
                    color = C_colors[C_to_idx[c_val]]
                    ax.scatter(coordinates[i, 0], coordinates[i, 1], marker=marker, color=color, s=150)

                # Optional grid lines by c
                print(coordinates, c_vals)
                # Plot dotted lines between consecutive c values in their original order
                for i in range(len(coordinates) - 1):
                    ax.plot(
                        [coordinates[i, 0], coordinates[i + 1, 0]],
                        [coordinates[i, 1], coordinates[i + 1, 1]],
                        linestyle=":", color="gray", zorder=0
                    )

                title = '.'.join(layer_names[l].split('.')[1:]) if '.' in layer_names[l] else layer_names[l]
                ax.set_title(title)
                ax.set_xlabel("PC Dimension 1")
                ax.set_ylabel("PC Dimension 2")
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)

        # Color legend only
        handles = [
            mpatches.Patch(color=C_colors[i], label=f"{network_names[j]}")
            for i, j in enumerate([C_to_idx[c] for c in C_vals])
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8, labelspacing=0.8, frameon=False)
 

    else:
        # Original c and s parsing
        network_names_split = [name.split(',') for name in network_names]
        x = [float(c.split("=")[1]) for c, _ in network_names_split]
        y = [float(s.split("=")[1]) for _, s in network_names_split]
        C_vals = sorted(set(x))
        s_vals = sorted(set(y))

        C_to_idx = {c: i for i, c in enumerate(C_vals)}
        s_to_idx = {s: i for i, s in enumerate(s_vals)}

        color_values = np.linspace(0.3, 1.0, len(C_vals))
        C_colors = [cm.Purples(color_values), cm.Blues(color_values), cm.Oranges(color_values), cm.Greens(color_values)]
        base_shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'h', '*', 'p', 'x']
        s_shapes = list(itertools.islice(itertools.cycle(base_shapes), len(s_vals)))

        cs_to_idx = {(c_val, s_val): i for i, (c_val, s_val) in enumerate(zip(x, y))}
        all_coords = all_coords.reshape(3, len(coord_list), -1, 2)
        print(all_coords.shape, coord_list[0].shape)

        for s in range(3):
            for l in range(num_layers):
                coordinates = all_coords[s][l]
                ax = axes[l]
                for i, (c_val, s_val) in enumerate(zip(x, y)):
                    color = C_colors[l][C_to_idx[c_val]]
                    marker = s_shapes[s_to_idx[s_val]]
                    ax.scatter(coordinates[i, 0], coordinates[i, 1], marker=marker, color=color, s=150)

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
                ax.set_xlabel("PC Dimension 1")
                ax.set_ylabel("PC Dimension 2")

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

        handles = (
            [mpatches.Patch(color=C_colors[-1][i], label=f"C={val}") for i, val in enumerate(C_vals)] +
            [mlines.Line2D([], [], color="black", marker=s_shapes[i],
                        ls="None", ms=10, label=f"s={val}") for i, val in enumerate(s_vals)]
        )
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8, labelspacing=0.8, frameon=False)

    plt.tight_layout()
    plt.savefig(f'{output_path}/mds_embedding.png', dpi=200)
    print(f"Saved to {output_path}/mds_embedding.png")
    plt.close()


def variability_seeds_vs_augs(dist_list, layer_names, n_seeds=3, n_augs=49):
    """
    Compares within-seed variability (same aug config, different seeds) vs.
    within-aug variability (same seed, different aug configs) using CV = std/mean.

    dist_list : list of (n_seeds*n_augs, n_seeds*n_augs) distance matrices, one per layer.
                Rows/cols ordered as [seed0_aug0 ... seed0_aug(n_augs-1), seed1_aug0 ...].
    layer_names: list of display names, one per layer.

    Returns a dict  { layer_name: { cv_seed, cv_aug, mean_seed, std_seed, mean_aug, std_aug } }
    """
    results = {}

    for D, lname in zip(dist_list, layer_names):
        D = np.array(D)

        # Within-seed: same aug config j, all pairs of seeds
        seed_dists = []
        for j in range(n_augs):
            for s1 in range(n_seeds):
                for s2 in range(s1 + 1, n_seeds):
                    seed_dists.append(D[s1 * n_augs + j, s2 * n_augs + j])
        seed_dists = np.array(seed_dists)

        # Within-aug: same seed, all off-diagonal aug pairs
        aug_dists = []
        for s in range(n_seeds):
            block = D[s * n_augs:(s + 1) * n_augs, s * n_augs:(s + 1) * n_augs]
            aug_dists.append(block[~np.eye(n_augs, dtype=bool)])
        aug_dists = np.concatenate(aug_dists)

        cv_seed = seed_dists.std() / seed_dists.mean()
        cv_aug  = aug_dists.std()  / aug_dists.mean()

        results[lname] = dict(
            cv_seed=cv_seed,   mean_seed=seed_dists.mean(), std_seed=seed_dists.std(),
            cv_aug=cv_aug,     mean_aug=aug_dists.mean(),   std_aug=aug_dists.std(),
        )
        print(f"[{lname}]")
        print(f"  Seeds  (same aug, diff seed) — mean={seed_dists.mean():.4f}, std={seed_dists.std():.4f}, CV={cv_seed:.4f}")
        print(f"  Augs   (same seed, diff aug) — mean={aug_dists.mean():.4f}, std={aug_dists.std():.4f}, CV={cv_aug:.4f}")

    return results


def plot_variability_comparison(results, output_path):
    layer_names = list(results.keys())
    cv_seed = [results[l]["cv_seed"] for l in layer_names]
    cv_aug  = [results[l]["cv_aug"]  for l in layer_names]

    mean_seed = [results[l]["mean_seed"] for l in layer_names]
    std_seed  = [results[l]["std_seed"] for l in layer_names]
    mean_aug  = [results[l]["mean_aug"]  for l in layer_names]
    std_aug   = [results[l]["std_aug"]  for l in layer_names]

    x = np.arange(len(layer_names))
    w = 0.35

    # --- CV (Coefficient of Variation) plot ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, cv_seed, w, label="Seeds (same aug, diff seed)", color="steelblue")
    ax.bar(x + w / 2, cv_aug,  w, label="Augs  (same seed, diff aug)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.set_ylabel("Coefficient of Variation (std / mean)")
    ax.set_title("Distance variability: seeds vs. augmentations")
    ax.legend()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_path}/seed_vs_aug_variability.png", dpi=200)
    plt.close()
    print(f"Saved to {output_path}/seed_vs_aug_variability.png")

    # --- Mean with error bars (std) plot ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, mean_seed, w, yerr=std_seed, label="Seeds (same aug, diff seed)", color="steelblue", capsize=4)
    ax.bar(x + w / 2, mean_aug,  w, yerr=std_aug,  label="Augs  (same seed, diff aug)", color="coral", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.set_ylabel("Distance (mean ± std)")
    ax.set_title("Mean distance variability: seeds vs. augmentations")
    ax.legend()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_path}/seed_vs_aug_mean_std.png", dpi=200)
    plt.close()
    print(f"Saved to {output_path}/seed_vs_aug_mean_std.png")


def pairwise_d_seed_d_aug_from_ref_varied(D, n_seeds, stride, ref_flat, var_flat):
    """D_seed = mean of cross-seed at ref and at var; D_aug = mean of ref→var within each seed."""
    n_pairs = n_seeds * (n_seeds - 1) // 2
    d_seed = np.empty(n_pairs, dtype=np.float32)
    d_aug = np.empty(n_pairs, dtype=np.float32)
    pair_idx = 0
    for s0 in range(n_seeds):
        for s1 in range(s0 + 1, n_seeds):
            i00 = stride * s0 + ref_flat
            i0v = stride * s0 + var_flat
            i10 = stride * s1 + ref_flat
            i1v = stride * s1 + var_flat
            D1 = D[i00, i0v]
            D2 = D[i10, i1v]
            D3 = D[i00, i10]
            D4 = D[i0v, i1v]
            d_seed[pair_idx] = 0.5 * (float(D3) + float(D4))
            d_aug[pair_idx] = 0.5 * (float(D1) + float(D2))
            pair_idx += 1
    return d_seed, d_aug


def collect_d_seed_d_aug_per_layer(dist_list, num_layers, n_seeds, n_augs):
    """Per layer, D_seed / D_aug for a single aug axis: index 0 = clean ref, 1.. = aug levels."""
    n_pairs = n_seeds * (n_seeds - 1) // 2
    stride = n_augs
    all_D_seed_lists = []
    all_D_aug_lists = []

    for l in range(num_layers):
        D = dist_list[l]
        D_seed_lists = np.zeros((n_pairs, n_augs - 1), dtype=np.float32)
        D_aug_lists = np.zeros((n_pairs, n_augs - 1), dtype=np.float32)
        for x in range(1, n_augs):
            ds, da = pairwise_d_seed_d_aug_from_ref_varied(D, n_seeds, stride, 0, x)
            D_seed_lists[:, x - 1] = ds
            D_aug_lists[:, x - 1] = da
        all_D_seed_lists.append(D_seed_lists)
        all_D_aug_lists.append(D_aug_lists)

    return all_D_seed_lists, all_D_aug_lists, n_pairs


def collect_d_seed_d_aug_per_layer_two_param(
    dist_list, num_layers, n_seeds, n_primary, n_secondary, *, hold_secondary: bool
):
    """
    2D aug grid with flat index ``p * n_secondary + s`` matching
    ``for p in range(n_primary): for s in range(n_secondary):``.

    If ``hold_secondary``: fix secondary index ``s`` (e.g. σ), ref at smallest primary ``p=0``,
    sweep primary ``p = 1..n_primary-1``.

    If ``not hold_secondary``: fix primary ``p``, ref at ``s=0``, sweep ``s = 1..n_secondary-1``.

    Returns per layer arrays ``(n_hold, n_pairs, n_sweep)``.
    """
    stride = n_primary * n_secondary
    n_pairs = n_seeds * (n_seeds - 1) // 2
    if hold_secondary:
        n_hold = n_secondary
        n_sweep = n_primary - 1
    else:
        n_hold = n_primary
        n_sweep = n_secondary - 1

    all_D_seed = []
    all_D_aug = []
    for l in range(num_layers):
        D = dist_list[l]
        S = np.zeros((n_hold, n_pairs, n_sweep), dtype=np.float32)
        A = np.zeros((n_hold, n_pairs, n_sweep), dtype=np.float32)
        for hi in range(n_hold):
            if hold_secondary:
                ref_flat = hi
                for si, xp in enumerate(range(1, n_primary)):
                    var_flat = xp * n_secondary + hi
                    ds, da = pairwise_d_seed_d_aug_from_ref_varied(
                        D, n_seeds, stride, ref_flat, var_flat
                    )
                    S[hi, :, si] = ds
                    A[hi, :, si] = da
            else:
                ref_flat = hi * n_secondary + 0
                for si, xs in enumerate(range(1, n_secondary)):
                    var_flat = hi * n_secondary + xs
                    ds, da = pairwise_d_seed_d_aug_from_ref_varied(
                        D, n_seeds, stride, ref_flat, var_flat
                    )
                    S[hi, :, si] = ds
                    A[hi, :, si] = da
        all_D_seed.append(S)
        all_D_aug.append(A)
    return all_D_seed, all_D_aug, n_pairs


def plot_d_seed_d_aug_vs_aug(
    all_D_seed_lists,
    all_D_aug_lists,
    aug_param_plot,
    layer_names,
    param_name,
    plot_save_path,
    n_pairs,
    num_layers,
    *,
    title_suffix: str = "",
    dpi: float | None = None,
    bbox_inches: str | None = None,
):
    """Mean ± std over seed pairs; vertical dashed line at first aug where mean D_aug > mean D_seed."""

    def _first_x_where_d_aug_exceeds_d_seed(x_vals, mean_aug, mean_seed):
        x_vals = np.asarray(x_vals, dtype=float)
        diff = np.asarray(mean_aug, dtype=float) - np.asarray(mean_seed, dtype=float)
        pos = np.where(diff > 0)[0]
        if len(pos) == 0:
            return None
        return float(x_vals[int(pos[0])])

    fig, axs = plt.subplots(1, num_layers, figsize=(6.5 * num_layers, 4))
    if num_layers == 1:
        axs = [axs]

    for l in range(num_layers):
        ax = axs[l]
        D_seed_lists = all_D_seed_lists[l]
        D_aug_lists = all_D_aug_lists[l]
        mean_seed = D_seed_lists.mean(axis=0)
        mean_aug = D_aug_lists.mean(axis=0)
        if n_pairs > 1:
            std_seed = D_seed_lists.std(axis=0, ddof=1)
            std_aug = D_aug_lists.std(axis=0, ddof=1)
        else:
            std_seed = np.zeros_like(mean_seed)
            std_aug = np.zeros_like(mean_aug)

        ax.errorbar(
            aug_param_plot,
            mean_seed,
            yerr=std_seed,
            marker="o",
            linestyle="-",
            capsize=3,
            label=r"$D_{\mathrm{seed}}$",
        )
        ax.errorbar(
            aug_param_plot,
            mean_aug,
            yerr=std_aug,
            marker="s",
            linestyle="-",
            capsize=3,
            label=r"$D_{\mathrm{aug}}$",
        )
        xc = _first_x_where_d_aug_exceeds_d_seed(aug_param_plot, mean_aug, mean_seed)
        if xc is not None:
            ax.axvline(
                xc,
                linestyle="--",
                color="0.35",
                linewidth=1.2,
                zorder=2,
                label=r"$D_{\mathrm{aug}} > D_{\mathrm{seed}}$",
            )
        ax.set_title(
            f"Layer {layer_names[l]}: D_seed & D_aug vs. {param_name}{title_suffix}"
        )
        ax.set_xlabel(param_name)
        ax.set_ylabel("Distance")
        ax.legend(fontsize=8)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    _save_kw = {}
    if bbox_inches is not None:
        _save_kw["bbox_inches"] = bbox_inches
    plt.savefig(plot_save_path, **_save_kw, dpi=200)
    plt.close()


def main():
    args = parse_args()
    set_seed(42)
    seeds = [42, 43, 44]

    base_path = '/mnt/home/the10/ceph/results/netrep'
    output_path = f'{base_path}/experiments/{args.experiment}'
    print(args.dataset, args.model_name, args.aug_type)

    param_dict = {"cutout": "cutout_patch_size", "gaussian_noise": "noise_std"}

    all_Xs, all_network_names = [], []

    if args.aug_type == "gaussian_noise":
        aug_param = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]
    elif args.aug_type == "cutout":
        aug_param = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0]


    for seed in seeds:
        result_path = f'{output_path}/results_seed{seed}' if seed != 42 else f'{output_path}/results'
        result_path = f'{result_path}_pretrained' if args.pretrained else result_path
        result_path = f'{result_path}/{args.model}' if args.model != 'final_model' else result_path
        os.makedirs(result_path, exist_ok=True)
        print(result_path)

        pretrain_str = "/pretrained" if args.pretrained else ""
        pretrain_str = f"{pretrain_str}_seed{seed}" if seed != 42 else pretrain_str

        seed_str = f"_seed{seed}" if seed != 42 else ""
        if args.single_param:
            experiment_ls = [f"clean{seed_str}/exp_{args.model_name}_clean_0.0_0.0"]
            experiment_ls += [f"{args.aug_type}{pretrain_str}/exp_{args.model_name}_{param_dict[args.aug_type]}_{s}_0.0" for s in aug_param]
        else:
            experiment_ls = [
                f"cutout{pretrain_str}/exp_{args.model_name}_cutout_patch_size_{c}_{s}"
                for c in CUTOUT_PATCH_GRID
                for s in CUTOUT_SIGMA_GRID
            ]
        
        X_train, _, network_names = extract_activity_all(experiment_ls, base_path, args)
        all_Xs.extend(X_train)
        all_network_names.extend(network_names)

    # compute procruste's distance between pair of networks
    num_layers = len(set(all_network_names)) if len(all_network_names) > 0 else 4
    n_networks = len(all_Xs) if len(all_Xs) > 0 else num_layers*len(seeds)*len(experiment_ls)

    dist_list = []
    coord_list = []
    os.makedirs(f'{output_path}/results_seed_combined', exist_ok=True)
    layer_names = [f'stage {i}' for i in range(1, 5)]
    network_names = exp_to_name(experiment_ls)[0]
    os.makedirs(f'{output_path}/results_seed_combined_{args.model_name}', exist_ok=True)

    for l in range(num_layers):
        indices = np.arange(l, len(all_Xs), num_layers)
        print(len(all_Xs), indices)
    
        os.makedirs(f'{output_path}/results_seed_combined_{args.model_name}/{args.aug_type}', exist_ok=True)
        distmat_path = f'{output_path}/results_seed_combined_{args.model_name}/{args.aug_type}/distance_matrix_layer{l}.npy'
        
        if args.load_pca and os.path.isfile(distmat_path):
            selected_distmat = np.load(distmat_path)
        else:
            selected_distmat = compute_layer_distmat(all_Xs, all_Xs, indices)
            np.save(distmat_path, selected_distmat)

        dist_list.append(selected_distmat)
        print(selected_distmat.shape)

        embedding = MDS(
            n_components=250, #450
            metric=True,
            n_init=20,
            max_iter=5000,
            eps=1e-6,
            normalized_stress="auto",
            dissimilarity="precomputed",
            random_state=42,
        )
        Z = embedding.fit_transform(np.abs(np.real(selected_distmat)))
        print(f"Layer {all_network_names[l]} stress: {embedding.stress_:.4f}")

        coordinates = PCA(n_components=2, random_state=42).fit_transform(Z)
        coord_list.append(coordinates)
    
    visualize_distance_all(dist_list, layer_names, ['']*len(selected_distmat), f'{output_path}/results_seed_combined_{args.model_name}/{args.aug_type}')
    visualize_coordinates_2d(coord_list, layer_names, network_names, f'{output_path}/results_seed_combined_{args.model_name}/{args.aug_type}', single_param=args.single_param)

    results = variability_seeds_vs_augs(dist_list, layer_names, n_seeds=len(seeds), n_augs=len(experiment_ls))
    plot_variability_comparison(results, f'{output_path}/results_seed_combined_{args.model_name}/{args.aug_type}')

    n_seeds = len(seeds)
    if args.single_param:
        aug_param_list = [0.0] + aug_param
        n_augs = len(aug_param_list)
        print(n_seeds, n_augs)
        all_D_seed_lists, all_D_aug_lists, n_pairs = collect_d_seed_d_aug_per_layer(
            dist_list, num_layers, n_seeds, n_augs
        )
        plot_d_seed_d_aug_vs_aug(
            all_D_seed_lists,
            all_D_aug_lists,
            aug_param_list[1:],
            layer_names,
            param_dict[args.aug_type],
            f"{output_path}/results_seed_combined_{args.model_name}/{args.aug_type}/D_vs_sigma_layers.png",
            n_pairs,
            num_layers,
        )
    else:
        n_pri, n_sec = len(CUTOUT_PATCH_GRID), len(CUTOUT_SIGMA_GRID)
        print(n_seeds, n_pri * n_sec, "two-param grid (largest cutout vs σ)")
        all_D_seed, all_D_aug, n_pairs = collect_d_seed_d_aug_per_layer_two_param(
            dist_list,
            num_layers,
            n_seeds,
            n_pri,
            n_sec,
            hold_secondary=False,
        )
        patch_max_idx = n_pri - 4
        patch_max = float(CUTOUT_PATCH_GRID[patch_max_idx])
        all_D_seed_lists = [all_D_seed[l][patch_max_idx] for l in range(num_layers)]
        all_D_aug_lists = [all_D_aug[l][patch_max_idx] for l in range(num_layers)]
        sigma_vals = np.asarray(CUTOUT_SIGMA_GRID[1:], dtype=float)
        plot_d_seed_d_aug_vs_aug(
            all_D_seed_lists,
            all_D_aug_lists,
            sigma_vals,
            layer_names,
            "sigma",
            f"{output_path}/results_seed_combined_{args.model_name}/{args.aug_type}/D_seed_aug_max_cutout_vs_sigma.png",
            n_pairs,
            num_layers,
            title_suffix=f" (cutout_patch_size={patch_max:g})",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()