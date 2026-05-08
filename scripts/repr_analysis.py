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
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tqdm import tqdm
from adjustText import adjust_text
import re
import random
import gc
import math
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection

_MDS_AXIS_LABELSIZE = 15
_MDS_TICK_LABELSIZE = 11

# Distance-matrix heatmaps: prominent axis titles; tick size tuned to limit overlap when many nets
_DIST_AXIS_LABELSIZE = 14
_DIST_TICK_LABELSIZE = 10

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
        default="best_model"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18"
    )

    parser.add_argument(
        "--pretrained",
        action="store_true"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--aug_type",
        type=str,
        default="cutout"
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="procrustes"
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


def modality_order_from_experiment_ls(exp_ls):
    """First path segment of each experiment (after clean), unique in first-seen order.

    Matches ``get_experiments_name_two_aug``: index ``i`` corresponds to aug slot ``idx_c``.
    """
    out = []
    for exp in exp_ls[1:]:
        prim = exp.split("/")[0]
        if prim not in out:
            out.append(prim)
    return out


def _compact_distance_ticklabels(names):
    """If every name is key=value with the same key, return short labels (values only).

    Returns (labels_for_plot, axis_label). axis_label is a readable key for the axis when
    compacting; otherwise None to keep the generic "Network" label.
    """
    names = [str(n) for n in names]
    if not names:
        return names, None
    keys, vals = [], []
    for n in names:
        if "=" not in n:
            return names, None
        k, v = n.split("=", 1)
        keys.append(k.strip())
        vals.append(v.strip())
    if len(set(keys)) != 1:
        return names, None
    key = keys[0]
    axis_label = key.replace("_", " ")
    return vals, axis_label


def visualize_distance_all(distmat, layer_names, network_names, output_path, metric="procrustes"):
    distmat = np.array(distmat)
    num_layers = len(layer_names)

    # Get vmin and vmax from off-diagonal entries only
    # vmax, vmin = 0, 100
    # for l in range(num_layers):
    #     indices = np.arange(l, distmat.shape[0], num_layers)
    #     selected_distmat = distmat[indices[:, None], indices]
    #     masked_selected_distmat = selected_distmat[~np.eye(len(selected_distmat), dtype=bool)]

    #     vmax = max(np.max(masked_selected_distmat), vmax)
    #     vmin = min(np.min(masked_selected_distmat), vmin)

    row, col = 1, len(layer_names)//1
    fig, axes = plt.subplots(row, col, figsize=(col * 5, row * 4.7), gridspec_kw={'wspace': 0.06, 'hspace': 0})
    axes = axes.flatten()
    print(layer_names)

    tick_names, axis_label_hint = _compact_distance_ticklabels(network_names)

    sns.set(style='white')

    n_tick = len(tick_names)
    max_tw = max((len(str(t)) for t in tick_names), default=0)
    # More grids → tilt labels earlier; shallow rotation when borderline keeps labels readable vs overlap
    if max_tw <= 5 and n_tick <= 14:
        rot_x, ha_x = 0, "center"
    elif max_tw <= 10 and n_tick <= 22:
        rot_x, ha_x = 38, "right"
    else:
        rot_x, ha_x = 52, "right"

    for l in range(num_layers):
        indices = np.arange(l, distmat.shape[0], num_layers)
        selected_distmat = distmat[indices[:, None], indices]

        ax = axes[l]
        sns.heatmap(
            selected_distmat,
            cmap="viridis",
            square=True,
            xticklabels=tick_names,
            yticklabels=tick_names,
            # vmin=vmin,
            # vmax=vmax+0.1,
            vmin=selected_distmat[np.triu_indices(selected_distmat.shape[0], k=1)].min(),
            vmax=selected_distmat[np.triu_indices(selected_distmat.shape[0], k=1)].max(),
            ax=ax,
            cbar=True,
            cbar_kws={"shrink": 0.5}
        )
        title = '.'.join(layer_names[l].split('.')[1:]) if '.' in layer_names[l] else layer_names[l]
        ax.set_title(title, fontsize=13)

        plt.setp(
            ax.get_xticklabels(),
            rotation=rot_x,
            ha=ha_x,
            fontsize=_DIST_TICK_LABELSIZE,
        )
        plt.setp(ax.get_yticklabels(), fontsize=_DIST_TICK_LABELSIZE)

        axis_lbl = axis_label_hint if axis_label_hint is not None else "Network"
        x_labelpad = 10 if rot_x else 7
        y_labelpad = 7
        ax.set_xlabel(axis_lbl, fontsize=_DIST_AXIS_LABELSIZE, labelpad=x_labelpad)
        ax.set_ylabel(axis_lbl, fontsize=_DIST_AXIS_LABELSIZE, labelpad=y_labelpad)
        if l % col > 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        if l < (row - 1) * col:
            ax.set_xlabel("")
            ax.set_xticklabels([])

    plt.tight_layout(pad=0.6)
    metric_str = "" if metric == "procrustes" else f"{metric}_"
    plt.savefig(
        f'{output_path}/{metric_str}dist.png',
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.18,
    )
    np.save(f'{output_path}/{metric_str}distance_matrix.npy', distmat)
    print(f"Saving to {output_path}/{metric_str}dist.png")
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

def visualize_coordinates_2d(distmat, layer_names, network_names, output_path, metric="procrustes"):
    import matplotlib as mpl
    num_layers = len(layer_names)
    rows, cols = 1, 4
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

    # `network_names` can be either:
    # - two-parameter labels like "C=12.0, s=0.2" (some cutout-style sweeps), or
    # - single-parameter labels like "T=4.0" / "clean=0.0" (most other sweeps).
    # Make parsing robust so visualization doesn't crash on single-parameter runs.
    parsed_pairs = []
    for name in network_names:
        parts = [p.strip() for p in str(name).split(',') if p.strip()]
        if len(parts) >= 2:
            x_part, y_part = parts[0], parts[1]
        elif len(parts) == 1:
            x_part, y_part = parts[0], "s=0.0"
        else:
            x_part, y_part = "C=0.0", "s=0.0"
        parsed_pairs.append((x_part, y_part))

    def _val_after_equals(s):
        s = str(s).strip()
        return float(s.split("=", 1)[1]) if "=" in s else float(s)

    x = [_val_after_equals(x_part) for x_part, _ in parsed_pairs]
    y = [_val_after_equals(y_part) for _, y_part in parsed_pairs]

    C_vals = sorted(set(x))
    s_vals = sorted(set(y))

    C_to_idx = {c: i for i, c in enumerate(C_vals)}
    s_to_idx = {s: i for i, s in enumerate(s_vals)}

    color_values = np.linspace(0.3, 1.0, len(C_vals))
    color_maps = [cm.Purples, cm.Blues, cm.Oranges, cm.Greens]
    C_colors = [cmap(color_values) for cmap in color_maps]
    base_shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'h', '*', 'p', 'x']
    s_shapes = list(itertools.islice(itertools.cycle(base_shapes), len(s_vals)))

    cs_to_idx = {(c_val, s_val): i for i, (c_val, s_val) in enumerate(zip(x, y))}

    # Normalize the color value for the colorbar (use 0...1 scale for colormap)
    norm = mpl.colors.Normalize(vmin=min(C_vals), vmax=max(C_vals))

    for l in range(num_layers):
        coordinates = coord_list[l]
        ax = axes[l]

        # scatter points
        sc_objs = []
        for i, (c_val, s_val) in enumerate(zip(x, y)):
            # Determine color from normalized C value
            color = C_colors[l][C_to_idx[c_val]]
            marker = s_shapes[s_to_idx[s_val]]
            sc = ax.scatter(coordinates[i, 0], coordinates[i, 1], marker=marker, color=color, s=150,
                            edgecolors='none', cmap=None, 
                            )
            sc_objs.append(sc)
        
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
        ax.set_xlabel("PC 1", fontsize=_MDS_AXIS_LABELSIZE)
        ax.set_ylabel("PC 2", fontsize=_MDS_AXIS_LABELSIZE)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=_MDS_TICK_LABELSIZE)

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

        # Autoscale uses marker centers only; large s=150 disks extend past limits and clip.
        ax.margins(0.14)
            
    # Colorbar with correct colormap for each subplot
    for l, ax in enumerate(axes[:num_layers]):
        cmap = color_maps[l % len(color_maps)]
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('C value', fontsize=_MDS_AXIS_LABELSIZE)
        cbar.ax.tick_params(labelsize=_MDS_TICK_LABELSIZE)

    # Add legend for marker shape
    # from matplotlib.lines import Line2D
    # marker_handles = [
    #     Line2D([], [], linestyle='None', marker=s_shapes[i], color='black', label=f's={val}', markersize=10)
    #     for i, val in enumerate(s_vals)
    # ]
    # # Put the marker legend on the last axis, right side
    # axes[-1].legend(
    #     handles=marker_handles,
    #     bbox_to_anchor=(1.05, 0.75), loc="center left",
    #     frameon=False, fontsize=8, title="Shape: s"
    # )

    plt.tight_layout()
    metric_str = "" if metric == "procrustes" else f"{metric}_"
    plt.savefig(f'{output_path}/{metric_str}mds_embedding.png', dpi=200)
    print(f"Saved to {output_path}/{metric_str}mds_embedding.png")
    plt.close()

def visualize_coordinates_2d_all_augmentation(distmat, layer_names, network_names, output_path):
    num_layers = len(layer_names)
    rows, cols = 1, 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
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
    # x_min, x_max = np.median(all_coords[:, 0])-0.05, np.median(all_coords[:, 0])+0.05
    # y_min, y_max = np.median(all_coords[:, 1])-0.05, np.median(all_coords[:, 1])+0.05

    network_names = [name.split('=') for name in network_names]

    color_values = np.linspace(0.3, 1.0, 4)
    C_colors = [cm.Blues(color_values), cm.Purples(color_values), cm.Greens(color_values), cm.Oranges(color_values)]
    base_shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'h', '*', 'p', 'x']
    s_shapes = list(itertools.islice(itertools.cycle(base_shapes), 8))
    print("Network names: ", network_names)

    aug_types = np.array([aug_type for aug_type, _ in network_names[1:]])
    n_augs = len(np.unique(aug_types))

    for l in range(num_layers):
        coordinates = coord_list[l]
        ax = axes[l]
        legend_handles, legend_labels = [], []

        # scatter points
        ax.scatter(coordinates[0, 0], coordinates[0, 1], marker='o', color='black', s=150)

        for i in np.arange(n_augs):
            color = C_colors[l]
            marker = s_shapes[i]
            idx = np.arange(i*4+1, i*4+5)
            ax.scatter(coordinates[idx, 0], coordinates[idx, 1], marker=marker, color=color, s=150)
            
            ax.plot([coordinates[0, 0], coordinates[idx[0], 0]],[coordinates[0, 1], coordinates[idx[0], 1]],
                linestyle=":", color="gray", zorder=0)
            ax.plot(coordinates[idx, 0], coordinates[idx, 1], linestyle=":", color="gray", zorder=0)

            handle = mlines.Line2D([], [], linestyle="None",marker=marker, markersize=10,color='black')
            legend_handles.append(handle)
            legend_labels.append(aug_types[i*4])

        title = '.'.join(layer_names[l].split('.')[1:]) if '.' in layer_names[l] else layer_names[l]
        ax.set_title(title)
        ax.set_xlabel("PC 1", fontsize=_MDS_AXIS_LABELSIZE)
        ax.set_ylabel("PC 2", fontsize=_MDS_AXIS_LABELSIZE)
        # ax.set_xlim(-0.1, 0.1)
        # ax.set_ylim(-0.1, 0.1)
        ax.legend(legend_handles, legend_labels, fontsize=9, frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=_MDS_TICK_LABELSIZE)
        ax.margins(0.14)

    plt.tight_layout()
    plt.savefig(f'{output_path}/mds_embedding.png', dpi=200)
    print(f"Saved to {output_path}/mds_embedding.png")
    plt.close()


def _primary_aug_rgb(name):
    k = name.strip().lower().replace("-", "_")
    k = {"shear": "sheer"}.get(k, k)
    grp = (
        frozenset({"rotate", "sheer", "shear"}),
        frozenset({"jitter", "grayscale"}),
        frozenset({"gaussian_noise", "sp_noise"}),
        frozenset({"cutout", "crop"}),
    )
    cols = ("#377eb8", "#4daf4a", "#984ea3", "#ff7f00")
    return next((cols[i] for i, g in enumerate(grp) if k in g), "#7f7f7f")


def visualize_coordinates_2d_two_augmentation(distmat, layer_names, network_names, output_path, experiment_ls):
    """Rows = primary aug folder; cols = stage. Marker discriminates paired aug; color encodes paired family (four colors)."""
    nn = len(network_names)
    mods = modality_order_from_experiment_ls(experiment_ls)
    n = len(mods)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    expected = len(pairs) + 1
    if nn != expected:
        raise ValueError(f"two-aug: expected {expected} nets, got {nn} (n={n})")

    sh = ["o", "s", "^", "D", "v", "<", ">", "p", "*"]
    L = len(layer_names)
    fig, axes = plt.subplots(n, L, figsize=(4.2 * L, 2.4 * n), squeeze=False, layout="constrained")

    coord_list = []
    for l in range(L):
        idx = np.arange(l, distmat.shape[0], L)
        sub = distmat[idx[:, None], idx]
        mds = MDS(n_components=200, metric=True, eps=1e-5,
                  normalized_stress="auto", dissimilarity="precomputed", random_state=42)
        Z = mds.fit_transform(np.abs(np.real(sub)))
        print(f"Layer {layer_names[l]} stress: {mds.stress_:.4f}")
        coord_list.append(PCA(n_components=2, random_state=42).fit_transform(Z))

    def group_inds(ai):
        o = sorted([(j, p) for p, (ci, j) in enumerate(pairs) if ci == ai])
        return [1 + p for _, p in o]

    groups = [group_inds(ai) for ai in range(n)]

    pair_handles = [
        mlines.Line2D([], [], color=_primary_aug_rgb(mods[j]), marker=sh[j % 9],
                      linestyle="None", markersize=9, label=mods[j])
        for j in range(n)
    ]
    clean_handle = mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=9, label="clean")

    for row in range(n):
        for l in range(L):
            ax = axes[row, l]
            XY = coord_list[l]
            ax.scatter(XY[0, 0], XY[0, 1], c="black", s=120, marker="o", zorder=5, edgecolors="none")
            for net_i in groups[row]:
                _, jb = pairs[net_i - 1]
                ax.scatter(XY[net_i, 0], XY[net_i, 1], marker=sh[jb % 9],
                           color=_primary_aug_rgb(mods[jb]), s=110, zorder=3, edgecolors="none")
            t = layer_names[l]
            if row == 0:
                ax.set_title(".".join(t.split(".")[1:]) if "." in t else t)
            if l == 0:
                ax.annotate(
                    mods[row], xy=(0, 0.5), xycoords="axes fraction", xytext=(-42, 0),
                    textcoords="offset points", va="center", ha="right", fontsize=9, fontweight="bold",
                    color="0.2",
                )
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(axis="both", which="major", labelsize=_MDS_TICK_LABELSIZE)

    fig.supxlabel("PC 1", fontsize=_MDS_AXIS_LABELSIZE)
    fig.supylabel("PC 2", fontsize=_MDS_AXIS_LABELSIZE)
    leg_all = [clean_handle] + pair_handles
    fig.legend(
        handles=leg_all,
        labels=[h.get_label() for h in leg_all],
        title="paired aug — shape + one of four family colors",
        loc="outside lower center",
        ncol=min(14, len(leg_all)),
        frameon=False,
        fontsize=7,
    )

    out = f"{output_path}/mds_embedding_two_aug.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out}")

def visualize_coordinates_3d_plotly(distmat, layer_names, network_names, output_path, metric="procrustes"):
    """
    Interactive 3D version of your function using Plotly subplots.
    Saves:  <output_path>/mds_embedding_3d.html
    """

    os.makedirs(output_path, exist_ok=True)

    num_layers = len(layer_names)

    # --- subplot grid (similar spirit to your matplotlib layout) ---
    rows = 1
    cols = int(np.ceil(num_layers / rows)) if num_layers > 0 else 1

    # nicer titles (same logic you had)
    subplot_titles = [
        ('.'.join(n.split('.')[1:]) if '.' in n else n)
        for n in layer_names
    ]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=subplot_titles
    )

    # --- compute per-layer 3D coords (MDS->PCA(3)) ---
    coord_list = []
    for l in range(num_layers):
        indices = np.arange(l, distmat.shape[0], num_layers)
        selected_distmat = distmat[indices[:, None], indices]

        embedding = MDS(
            n_components=200,
            metric=True,
            eps=1e-5,
            normalized_stress="auto",
            dissimilarity="precomputed",
            random_state=42
        )
        Z = embedding.fit_transform(np.abs(np.real(selected_distmat)))
        print(f"Layer {layer_names[l]} stress: {embedding.stress_:.4f}")

        pca = PCA(n_components=3, random_state=42)
        coordinates = pca.fit_transform(Z)
        coord_list.append(coordinates)

    # --- global axis ranges so all subplots are comparable ---
    all_coords = np.vstack(coord_list) if coord_list else np.zeros((0, 3))
    pad = 0.05
    x_rng = [float(all_coords[:, 0].min() - pad), float(all_coords[:, 0].max() + pad)] if len(all_coords) else [-1, 1]
    y_rng = [float(all_coords[:, 1].min() - pad), float(all_coords[:, 1].max() + pad)] if len(all_coords) else [-1, 1]
    z_rng = [float(all_coords[:, 2].min() - pad), float(all_coords[:, 2].max() + pad)] if len(all_coords) else [-1, 1]

    # --- parse network_names like "C=...,s=..." (same as you do) ---
    network_names_split = [name.split(",") for name in network_names]
    x = [float(c.split("=")[1]) for c, _ in network_names_split]  # C values
    y = [float(s.split("=")[1]) for _, s in network_names_split]  # s values

    C_vals = sorted(set(x))
    s_vals = sorted(set(y))
    C_to_idx = {c: i for i, c in enumerate(C_vals)}
    s_to_idx = {s: i for i, s in enumerate(s_vals)}

    # Colors from matplotlib Greens -> plotly rgba strings
    C_colors_rgba = cm.Greens(np.linspace(0.3, 1.0, len(C_vals)))  # Nx4 floats
    C_colors = [
        f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.3f})"
        for (r, g, b, a) in C_colors_rgba
    ]

    # Plotly marker symbols (a small set; extend if you have more s_vals)
    plotly_symbols = ['circle', 'square', 'diamond', 'circle', 'square', 'diamond', 'circle', 'square', 'diamond']
    s_symbols = plotly_symbols[:len(s_vals)]

    cs_to_idx = {(c_val, s_val): i for i, (c_val, s_val) in enumerate(zip(x, y))}

    # special path (your updated one)
    special_path = [
        (0.0, 0.0),
        (4.0, 0.05),
        (4.0, 0.1),
        (4.0, 0.2),
        (4.0, 0.3),
        (4.0, 0.5),
        (4.0, 0.8),
        (4.0, 1.0),
    ]
    special_indices = [
        cs_to_idx[(c_val, s_val)]
        for (c_val, s_val) in special_path
        if (c_val, s_val) in cs_to_idx
    ]

    # --- add traces per layer/subplot ---
    for l in range(num_layers):
        r = (l // cols) + 1
        c = (l % cols) + 1
        coordinates = coord_list[l]

        # points grouped by (C,s) to preserve BOTH discrete color and discrete symbol
        for i, (c_val, s_val) in enumerate(zip(x, y)):
            color = C_colors[C_to_idx[c_val]]
            symbol = s_symbols[s_to_idx[s_val]]

            fig.add_trace(
                go.Scatter3d(
                    x=[coordinates[i, 0]],
                    y=[coordinates[i, 1]],
                    z=[coordinates[i, 2]],
                    mode="markers",
                    marker=dict(size=6, color=color, symbol=symbol),
                    text=[f"C={c_val}, s={s_val}"],
                    hoverinfo="text",
                    showlegend=False
                ),
                row=r,
                col=c
            )

        # grid lines for fixed C
        for c_val in C_vals:
            idx = [i for i, xx in enumerate(x) if xx == c_val]
            pts = coordinates[idx]
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode="lines",
                    line=dict(color="gray", dash="dot", width=2),
                    hoverinfo="skip",
                    showlegend=False
                ),
                row=r, col=c
            )

        # grid lines for fixed s
        for s_val in s_vals:
            idx = [i for i, yy in enumerate(y) if yy == s_val]
            pts = coordinates[idx]
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode="lines",
                    line=dict(color="gray", dash="dot", width=2),
                    hoverinfo="skip",
                    showlegend=False
                ),
                row=r, col=c
            )

        # special star connections from first special point to others (if present)
        if len(special_indices) >= 2:
            pts = coordinates[special_indices]
            for i in range(1, len(pts)):
                fig.add_trace(
                    go.Scatter3d(
                        x=[pts[0, 0], pts[i, 0]],
                        y=[pts[0, 1], pts[i, 1]],
                        z=[pts[0, 2], pts[i, 2]],
                        mode="lines",
                        line=dict(color="gray", dash="dot", width=2),
                        hoverinfo="skip",
                        showlegend=False
                    ),
                    row=r, col=c
                )

        # per-subplot axis labels/ranges
        fig.update_scenes(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            row=r,
            col=c
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=5, b=10)
        )

    # --- build a single legend (global) using "dummy" traces ---
    # C legend entries (color)
    for c_val in C_vals:
        fig.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=8, color=C_colors[C_to_idx[c_val]], symbol="circle"),
                name=f"C={c_val}",
                showlegend=True
            )
        )
    # s legend entries (symbol)
    for s_val in s_vals:
        fig.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=8, color="black", symbol=s_symbols[s_to_idx[s_val]]),
                name=f"s={s_val}",
                showlegend=True
            )
        )

    fig.update_layout(
        title="3D MDS Embeddings (Interactive)",
        height=rows * 360,
        width=cols * 340,
        legend=dict(itemsizing="constant"),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    metric_str = "" if metric == "procrustes" else f"{metric}_"
    out_html = os.path.join(output_path, f"{metric_str}mds_embedding_3d.html")
    fig.write_html(out_html)
    print(f"Saved interactive plot to {out_html}")
    
def visualize_coordinates_3d(distmat, layer_names, network_names, output_path, metric="procrustes"):
    num_layers = len(layer_names)
    rows, cols = 4, len(network_names)//4
    fig = plt.figure(figsize=(cols * 6, rows * 5))

    axes = []
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        axes.append(ax)

    for ax in axes[num_layers:]:
        ax.set_axis_off()

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

        pca = PCA(n_components=3, random_state=42)
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
    base_shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'h', '*', 'p', 'x']
    s_shapes = list(itertools.islice(itertools.cycle(base_shapes), len(s_vals)))

    x = [float(c.split("=")[1]) for c, _ in network_names]
    y = [float(s.split("=")[1]) for _, s in network_names]

    cs_to_idx = {(c_val, s_val): i for i, (c_val, s_val) in enumerate(zip(x, y))}

    for l in range(num_layers):
        coordinates = coord_list[l]
        ax = axes[l]

        # scatter points
        for i, (c_val, s_val) in enumerate(zip(x, y)):
            color = C_colors[C_to_idx[c_val]]
            marker = s_shapes[s_to_idx[s_val]]
            ax.scatter(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], marker=marker, color=color, s=150)
        
        # plot grids
        for c_val in C_vals:
            idx = [i for i, xx in enumerate(x) if xx == c_val]
            pts = coordinates[idx]
            ax.plot(pts[:,0], pts[:,1], pts[:,2], linestyle=":", color="gray", zorder=0)

        for s_val in s_vals:
            idx = [i for i, yy in enumerate(y) if yy == s_val]
            pts = coordinates[idx]
            ax.plot(pts[:,0], pts[:,1], pts[:,2], linestyle=":", color="gray", zorder=0)       

        title = '.'.join(layer_names[l].split('.')[1:]) if '.' in layer_names[l] else layer_names[l]
        ax.set_title(title)
        ax.set_xlabel("MDS PC 1", fontsize=_MDS_AXIS_LABELSIZE)
        ax.set_ylabel("MDS PC 2", fontsize=_MDS_AXIS_LABELSIZE)
        ax.set_zlabel("MDS PC 3", fontsize=_MDS_AXIS_LABELSIZE)
        ax.tick_params(axis="x", labelsize=_MDS_TICK_LABELSIZE)
        ax.tick_params(axis="y", labelsize=_MDS_TICK_LABELSIZE)
        ax.tick_params(axis="z", labelsize=_MDS_TICK_LABELSIZE)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # special_path = [(0.0, 0.0),(12.0, 0.2),(12.0, 0.3),(12.0, 0.5),(12.0, 1.0)]
        special_path = [(0.0, 0.0),(4.0, 0.05),(4.0, 0.1),(4.0, 0.2),(4.0, 0.3),(4.0, 0.5),(4.0, 0.8),(4.0, 1.0)]

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
        [mpatches.Patch(color=C_colors[i], label=f"C={val}") for i, val in enumerate(C_vals)] +
        [mlines.Line2D([], [], color="black", marker=s_shapes[i],
                    ls="None", ms=10, label=f"s={val}") for i, val in enumerate(s_vals)]
    )
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 0.5), loc="center left")
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f'{output_path}/mds_embedding_3d.png')
    print(f"Saved to {output_path}/mds_embedding_3d.png")
    plt.close()


def visualize_gaussian_curvature(distmat, layer_names, network_names, output_path):
    num_layers = len(layer_names)
    rows, cols = 1, len(layer_names)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), constrained_layout=True)
    fig.set_constrained_layout_pads(wspace=0.02)
    axes = axes.flatten()

    for ax in axes[num_layers:]:
        ax.axis('off')

    coord_list = []
    layer_distmats = []
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
        layer_distmats.append(selected_distmat)

    all_coords = np.vstack(coord_list)
    network_names = [name.split(',') for name in network_names]

    x = [float(c.split("=")[1]) for c, _ in network_names]
    y = [float(s.split("=")[1]) for _, s in network_names]

    C_vals = sorted(set(x))
    s_vals = sorted(set(y))
    cs_to_idx = {(c_val, s_val): i for i, (c_val, s_val) in enumerate(zip(x, y))}

    for l in range(num_layers):
        coordinates = coord_list[l]
        dist_layer = layer_distmats[l]
        ax = axes[l]

        polys = []
        K_faces = []

        for ci in range(len(C_vals) - 1):
            for si in range(len(s_vals) - 1):
                c0, c1 = C_vals[ci], C_vals[ci+1]
                s0, s1 = s_vals[si], s_vals[si+1]

                i00 = cs_to_idx[(c0, s0)]  # (c,   s)
                i01 = cs_to_idx[(c0, s1)]  # (c,   s+ds)
                i10 = cs_to_idx[(c1, s0)]  # (c+dc,s)
                i11 = cs_to_idx[(c1, s1)]  # (c+dc,s+ds)

                # --- T1: (c, s), (c, s+ds), (c+dc, s) ---
                K1 = estimate_curvature(i00, i01, i10, dist_layer)
                poly1 = coordinates[[i00, i01, i10], :]
                polys.append(poly1)
                K_faces.append(K1)

                # --- T2: (c+dc, s), (c, s+ds), (c+dc, s+ds) ---
                K2 = estimate_curvature(i10, i01, i11, dist_layer)
                poly2 = coordinates[[i10, i01, i11], :]
                polys.append(poly2)
                K_faces.append(K2)
                
        K_faces = np.array(K_faces)
        if len(K_faces) > 0:
            norm = plt.Normalize(vmin=-100, vmax=100)
            colors = cm.coolwarm(norm(K_faces))

            mesh = PolyCollection(polys, facecolors=colors,
                                edgecolors='none', alpha=0.7)
            ax.add_collection(mesh)

        ax.scatter(coordinates[:, 0], coordinates[:, 1], color='black', s=150)
       
        title = '.'.join(layer_names[l].split('.')[1:]) if '.' in layer_names[l] else layer_names[l]
        ax.set_title(title)
        ax.set_xlabel("PC 1", fontsize=_MDS_AXIS_LABELSIZE)
        ax.set_ylabel("PC 2", fontsize=_MDS_AXIS_LABELSIZE)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=_MDS_TICK_LABELSIZE)

    plt.tight_layout()
    sm = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location='right', shrink=0.8, pad=0.01)
    cbar.set_label("Gaussian Curvature", fontsize=_MDS_AXIS_LABELSIZE)
    cbar.ax.tick_params(labelsize=_MDS_TICK_LABELSIZE)
    plt.savefig(f'{output_path}/gaussian_curvature.png')
    print(f"Saved to {output_path}/gaussian_curvature.png")
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
        ax.set_xlabel("PC 1", fontsize=_MDS_AXIS_LABELSIZE)
        ax.set_ylabel("PC 2", fontsize=_MDS_AXIS_LABELSIZE)
        ax.legend(fontsize=8, labelspacing=0.8, frameon=False)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=_MDS_TICK_LABELSIZE)

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

def visualize_diff_norms(diff_norms, n_layers, data_path, layer_idx=2, save_dir='.'):
    # diff_norms is big matrix of size (n_aug1*n_aug2*n_layers, n_aug1*n_aug2*n_layers)
    print("Processing diff norms...")
    idx = np.arange(layer_idx, len(diff_norms), n_layers)
    selected_norms = diff_norms[np.ix_(idx, idx)]
    
    n_augs = np.sqrt(len(diff_norms)/n_layers).astype(int) 
    c0, s0 = 0, 0
    cN, sN = 1, 1

    def aug_index(c, s):
        return c * n_augs + s 
    
    anchor   = aug_index(c0, s0)
    s_targets = aug_index(c0, sN)  # (c0, sN)
    c_targets = aug_index(cN, s0)  # (cN, s0)

    # distances: shape (n_augs-1, n_images)
    dist_c0s0_to_c0sn = selected_norms[anchor, s_targets, :]
    dist_c0s0_to_cns0 = selected_norms[anchor, c_targets, :]

    blocks = [dist_c0s0_to_c0sn, dist_c0s0_to_cns0]
    row_names = ["(c0,s0) → (c0,sn)", "(c0,s0) → (cn,s0)"]

    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(nrows, ncols)

    cifar_index = np.zeros((ncols, blocks[0].shape[-1]), dtype=int)

    for c in range(ncols):
        order = np.argsort(blocks[c])  # consistent ordering per row
        cifar_index[c] = order
        axes[0, c].scatter(np.arange(len(order)), blocks[c][order], s=10)
        axes[0, c].set_title(row_names[c])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/diff_norm_anchor_sorted.png")
    plt.close(fig)
    
    eval_tf = transforms.ToTensor()
    dataset = datasets.CIFAR10(root=data_path, train=False, transform=eval_tf, download=False)

    def _show_smallest_and_largest(order_indices, out_name, title,
                                            nrows=4, ncols=5, gap = 1):
        order_indices = np.asarray(order_indices).reshape(-1)

        n_show = nrows * ncols  # 20
        smallest = order_indices[:n_show]
        largest  = order_indices[-n_show:][::-1]

        fig, axes = plt.subplots(nrows, 2 * ncols + gap,
                                figsize=(3 * (2 * ncols + gap), 3 * nrows))
        axes = np.asarray(axes).reshape(nrows, 2 * ncols + gap)

        # turn off the gap column
        gap_start = ncols
        for r in range(nrows):
            axes[r, gap_start].axis("off")

        def _fill_block(idxs, col_offset, block_title):
            axes[0, col_offset].set_title(block_title, fontsize=14, loc="left", y=1.15)

            for k, idx in enumerate(idxs):
                r = k // ncols
                c = col_offset + (k % ncols)
                img, label = dataset[int(idx)]
                axes[r, c].imshow(np.transpose(img.numpy(), (1, 2, 0)))
                axes[r, c].set_title(f"idx={int(idx)}\nlabel={label}", fontsize=9)
                axes[r, c].axis("off")

        # left block: cols [0..4], right block: start after gap => cols [6..10]
        _fill_block(smallest, col_offset=0, block_title="Smallest moved")
        _fill_block(largest, col_offset=ncols + gap, block_title="Largest moved")

        fig.suptitle(title, y=0.995)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, out_name), dpi=200)
        plt.close(fig)

    _show_smallest_and_largest(
        cifar_index[0],
        out_name="c0s0_to_c0sN_smallest_vs_largest.png",
        title=f"(c0,s0)→(c0,sN) | layer={layer_idx}",
        nrows=4, ncols=5
    )

    _show_smallest_and_largest(
        cifar_index[1],
        out_name="c0s0_to_cNs0_smallest_vs_largest.png",
        title=f"(c0,s0)→(cN,s0) | layer={layer_idx}",
        nrows=4, ncols=5
    )

def estimate_curvature(i, j, k, distmat_layer):
    a = distmat_layer[j, k]
    b = distmat_layer[i, k]
    c = distmat_layer[i, j]

    angles_c = compute_angles(c, a, b)
    angles_b = compute_angles(b, a, c)
    angles_a = compute_angles(a, c, b)

    s = (a+b+c)/2
    area = math.sqrt(s*(s-a)*(s-b)*(s-c))
    curvature = (angles_a + angles_b + angles_c - np.pi)/area
    return curvature

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
            selected_angles[7],
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

def get_experiments_name_two_aug(model_name, param_dict, aug1, aug2, aug1_params, aug2_params, seed=42):
    seed_str = f"_seed{seed}" if seed else ""
    experiment_ls = [f"clean{seed_str}/exp_{model_name}_{param_dict['clean']}_0.0_0.0"]

    for idx_c, c in enumerate(aug1_params):
        for idx_s, s in enumerate(aug2_params):
            if idx_c != idx_s:
                experiment_ls.append(f"{aug1[idx_c]}/exp_{model_name}_{param_dict[aug1[idx_c]]}_{c}_{aug2[idx_s]}_{param_dict[aug2[idx_s]]}{s}_0.0")
    return experiment_ls

def compare_angles(angles, layer_names, network_names, output_path):
    num_layers = len(layer_names)

    n_aug1 = np.unique([n.split(',')[0] for n in network_names]).shape[0]
    n_aug2 = np.unique([n.split(',')[1] for n in network_names]).shape[0]

    T_mean, T_std, C_mean, C_std, cross_mean, cross_std = [], [], [], [], [], []
    
    for l in range(num_layers):
        indices = np.arange(l, angles.shape[0], num_layers)
        selected_angles = angles[np.ix_(indices, indices, indices)]

        # select the angle anchor point
        N = len(selected_angles)
        grid = np.arange(N).reshape(n_aug1, n_aug2)

        a1 = np.arange(n_aug1)
        a2 = np.arange(1, n_aug2-1)
        A1, A2 = np.meshgrid(a1, a2, indexing='ij')
        
        anchors = grid[A1, A2].ravel()
        left  = grid[A1, A2-1].ravel()
        right = grid[A1, A2+1].ravel()

        T_vals = np.degrees(selected_angles[anchors, left, right])
        T_vals_2d = T_vals.reshape(len(a1), len(a2))
        plt.imshow(T_vals_2d, origin="lower", aspect="auto")
        plt.colorbar(label="s-values (deg)")
        plt.savefig(f"sigma_{layer_names[l]}.png")
        plt.close()

        a1 = np.arange(1, n_aug1-1)
        a2 = np.arange(n_aug2)
        A1, A2 = np.meshgrid(a1, a2, indexing='ij')

        anchors = grid[A1, A2].ravel()
        left  = grid[A1-1, A2].ravel()
        right = grid[A1+1, A2].ravel()

        C_vals = np.degrees(selected_angles[anchors, left, right])
        C_vals_2d = C_vals.reshape(len(a1), len(a2))
        plt.imshow(C_vals_2d, origin="lower", aspect="auto")
        plt.colorbar(label="c-values (deg)")
        plt.savefig(f"cutout_{layer_names[l]}.png")
        plt.close()

        a1 = np.arange(n_aug1-1)
        a2 = np.arange(n_aug2-1)
        A1, A2 = np.meshgrid(a1, a2, indexing='ij')

        anchors = grid[A1, A2].ravel()
        left  = grid[A1, A2+1].ravel()
        right = grid[A1+1, A2].ravel()

        X_vals = np.degrees(selected_angles[anchors, left, right])
        X_vals_2d = X_vals.reshape(len(a1), len(a2))
        plt.imshow(X_vals_2d, origin="lower", aspect="auto")
        plt.colorbar(label="across-values (deg)")
        plt.savefig(f"across_{layer_names[l]}.png")
        plt.close()

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

def get_experiments_name_all_aug(model_name, param_dict):
    experiment_ls = [f"clean/exp_{model_name}_{param_dict['clean']}_0.0_0.0"]
    experiment_ls += [f"rotate/exp_{model_name}_{param_dict['rotate']}_{c}_0.0" for c in [4.0, 12.0, 18.0, 24.0]]
    experiment_ls += [f"sheer/exp_{model_name}_{param_dict['sheer']}_{c}_0.0" for c in [4.0, 12.0, 18.0, 24.0]]
    experiment_ls += [f"jitter/exp_{model_name}_{param_dict['jitter']}_{c}_0.0" for c in [0.1, 0.2, 0.3, 0.4]]
    experiment_ls += [f"grayscale/exp_{model_name}_{param_dict['grayscale']}_{c}_0.0" for c in [0.1, 0.2, 0.3, 0.4]]
    experiment_ls += [f"gaussian_noise/exp_{model_name}_{param_dict['gaussian_noise']}_{c}_0.0" for c in [0.01, 0.02, 0.03, 0.04]]
    experiment_ls += [f"sp_noise/exp_{model_name}_{param_dict['sp_noise']}_{c}_0.0" for c in [0.05, 0.1, 0.15, 0.2]]
    experiment_ls += [f"cutout/exp_{model_name}_{param_dict['cutout']}_{c}_0.0" for c in [2.0, 4.0, 6.0, 8.0]]
    experiment_ls += [f"crop/exp_{model_name}_{param_dict['crop']}_{c}_0.0" for c in [0.6, 0.7, 0.8, 0.9]]
    return experiment_ls

def main():
    args = parse_args()
    set_seed(42)

    base_path = '/mnt/home/the10/ceph/results/netrep'
    output_path = f'{base_path}/experiments/{args.experiment}/{args.aug_type}'
    result_path = f'{output_path}/results_seed{args.seed}' if args.seed else f'{output_path}/results'
    result_path = f'{result_path}_pretrained' if args.pretrained else result_path
    result_path = f'{result_path}_{args.model_name}' if args.model_name != 'resnet18' else result_path
    output_path = f'{result_path}/analysis'

    os.makedirs(output_path, exist_ok=True)
    pretrain_str = "/pretrained" if args.pretrained else ""
    pretrain_str = f"{pretrain_str}_seed{args.seed}" if args.seed else pretrain_str

    param_dict = {"fixed_blur": "sigma", "weak_random_blur": "temp", 
                  "sp_noise": "sp_prob", "cutout": "cutout_patch_size", "clean": "clean",
                  "cutout_jitter": "cutout_patch_size", "cutout_crop": "cutout_patch_size", "rot_sheer": "rot_deg",
                  "rotate": "rot_deg", "sheer": "shear_deg","gaussian_noise": "noise_std",
                  "jitter": "cj_strength", "grayscale": "gray_prob",
                  "cutout_only": "cutout_patch_size", "crop": "extra_crop_scale"}

    # aug1_param = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]
    # aug2_param = [0.0]
    aug1_param = [4.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0]
    aug2_param = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    # experiment_ls = [f"{args.aug_type}{pretrain_str}/exp_{args.model_name}_{param_dict[args.aug_type]}_{c}_{s}" for c in aug1_param for s in aug2_param]
    if args.aug_type == "combined":
        experiment_ls = get_experiments_name_all_aug(args.model_name, param_dict)
    elif args.aug_type == "two":
        aug_param = [24.0, 24.0, 0.4, 0.1, 0.04, 0.2, 8.0, 0.9]
        aug_types = ["rotate", "sheer", "jitter", "grayscale", "gaussian_noise", "sp_noise", "cutout", "crop"]
        experiment_ls = get_experiments_name_two_aug(args.model_name, param_dict, aug_types, aug_types, aug_param, aug_param, args.seed)
    else:
        experiment_ls = [f"{args.aug_type}{pretrain_str}/exp_{args.model_name}_{param_dict[args.aug_type]}_{c}_{s}" for c in aug1_param for s in aug2_param]

    if args.metric == "cka":
        distmat = np.load(f"{result_path}/cka_distance_matrix.npy")
    elif args.metric == "procrustes":
        distmat = np.load(f"{result_path}/distance_matrix.npy")
    elif args.metric == "unaligned":
        distmat = np.load(f"{result_path}/unaligned_distance_matrix.npy")
    else:
        raise ValueError(f"Invalid metric: {args.metric}")
    # network_indices = [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 4.0, 4.1, 4.2]
    network_indices = [2.1, 3.1, 4.1]
    if args.model_name == "resnet18":
        network_names = [f'module.layer{i}' for i in network_indices] + ['avgpool']
    else:
        network_names = [f'stage {i}' for i in range(1, 5)]

    # diff_norms = np.load(f"{result_path}/diff_norms.npy")
    # visualize_diff_norms(diff_norms, len(network_names), args.data_path, save_dir=output_path)

    l = len(network_names)
    name_ls = exp_to_name(experiment_ls)[0]

    # TODO: look into cutout patch size 80, blur temp 1.0, remove it for now
    print(len(distmat), l)
    # n, g = len(distmat), 4*l
    # distmat = distmat[l:, l:]
    # distmat = np.delete(distmat, np.arange(n - 2*g, n - g), axis=0)
    # distmat = np.delete(distmat, np.arange(n - 2*g, n - g), axis=1)

    # blocks = np.array([0, 1, 7, 8, 14, 15, 21, 22])
    # block_size = 4
    # idx = np.concatenate([
    #     np.arange(b * block_size, (b + 1) * block_size)
    #     for b in blocks
    # ])
    # distmat = np.delete(np.delete(distmat, idx, axis=0), idx, axis=1)

    print(name_ls)
    visualize_distance_all(distmat, network_names, name_ls, output_path, metric=args.metric)

    if args.aug_type == "combined":
        visualize_coordinates_2d_all_augmentation(distmat, network_names, name_ls, output_path)
    elif args.aug_type == "two":
        visualize_coordinates_2d_two_augmentation(
            distmat, network_names, name_ls, output_path, experiment_ls
        )
    else:
        visualize_coordinates_2d(distmat, network_names, name_ls, output_path, metric=args.metric)
        visualize_coordinates_3d_plotly(distmat, network_names, name_ls, output_path, metric=args.metric)
        # visualize_distance_histogram(distmat, network_names, name_ls, output_path)
        # visualize_gaussian_curvature(distmat, network_names, name_ls, output_path)
    
        if 'clean' in experiment_ls[0]:
            distmat = distmat[l:, l:]
            name_ls = name_ls[1:]
            print(distmat.shape)
        
        angles = compute_all_angles(distmat)
        print(angles.shape)

if __name__ == "__main__":
    main()