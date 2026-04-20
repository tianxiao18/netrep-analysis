import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from netrep.dualPCA import DualPCA

def center_scale(X):
    Xc = X - np.mean(X, axis=0, keepdims=True)
    return Xc / np.linalg.norm(Xc)

def spherical_interp(X, Y, t):
    """
    Spherical interpolation between to vectors `X` and `Y` on the
    unit sphere. The value of `t` is between zero and one.
    Returns `X` when `t = 0` and `Y` when `t = 1`.
    """

    # Compute the angle between X and Y. Clipping is done
    # for numerical safety here.
    c = np.sum(X * Y)
    theta = np.arccos(np.clip(c, -1.0, 1.0))

    # Prevent divide by zero error
    if theta < 1e-8:
        return X

    # This is the formula for interpolating along the sphere.
    # See: https://en.wikipedia.org/wiki/Slerp
    a = np.sin((1 - t) * theta) / np.sin(theta)
    b = np.sin(t * theta) / np.sin(theta)
    result = a * X + b * Y

    # Call to center_scale shouldn't be necessary, but is done
    # here for numerical stability.
    return center_scale(result)

def shape_interp(X, Y, t):
    """
    Aligns Y to X by best orthogonal transformation, then calls
    spherical interpolation.
    """
    U, _, Vt = np.linalg.svd(Y.T @ X)
    Y_aligned = Y @ U @ Vt
    return spherical_interp(X, Y_aligned, t)


def sq_proc(p, q):
    """
    Square of the procrustes shape distance.

    Args:
        p: (M, N) array of centered and rescaled neural responses.
        q: (M, N) array of centered and rescaled neural responses.
    """
    singular_values = np.linalg.svd(q.T @ p, compute_uv=False)
    c = np.sum(singular_values)
    return np.arccos(np.clip(c, -1.0, 1.0))#2 * (1 - np.sum(singular_values))

def shape_align(X, Y):
    """
    Aligns Y to X by best orthogonal transformation, then returns aligned Y.
    """
    U, _, Vt = np.linalg.svd(Y.T @ X)
    Y_aligned = Y @ U @ Vt
    return Y_aligned

def landmark_vel(A, B):
    """Compute landmark velocity field on the geodesic from A to B."""
    B_aligned = shape_align(A, B)
    rho = sq_proc(A, B_aligned)
    xdiff = B_aligned - A
    v = (rho/np.sin(rho)) * (B_aligned - np.cos(rho) * A)
    return v, xdiff


def geodesic_angle(v1, v2):
    """Compute the angle between two landmark velocity fields."""
    c = np.sum(v1 * v2)
    return np.arccos(np.clip(c, -1.0, 1.0)/(np.linalg.norm(v1) * np.linalg.norm(v2)))


def plot_angle_matrices(angle_matrices, aug_names, save_path):
    n_augs = len(aug_names)

    nonzero_vals = np.concatenate(
        [m[np.nonzero(m)] for m in angle_matrices]
    )
    vmin = nonzero_vals.min() - 1
    vmax = nonzero_vals.max()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i, (ax, angle_matrix) in enumerate(zip(axes, angle_matrices)):
        im = ax.imshow(
            angle_matrix,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(f"Matrix {i+1}")
        ax.set_xticks(np.arange(n_augs))
        ax.set_yticks(np.arange(n_augs))
        ax.set_xticklabels(aug_names, rotation=45, ha="right")
        ax.set_yticklabels(aug_names)

    # shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label("Mean Geodesic Angle (degrees)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

def reduce_activity(activities, n_dim=1000):
    if np.array(activities).shape[-1] < n_dim:
        return activities
    
    reduced_activity = np.zeros((len(activities), len(activities[0]), n_dim))

    for i, activity in enumerate(activities):
        pca = DualPCA(n_components=n_dim)
        X = pca.fit_transform(activity)
        print(f"variance explained in {n_dim} dim: {np.sum(pca.explained_variance_ratio_)}")
        reduced_activity[i] = X
    
    return reduced_activity

if __name__ == "__main__":
    # layers = ['layer2.1', 'layer3.1', 'layer4.1', 'avgpool']
    layers = ['layer2.1']
    aug_names = ["rotate", "sheer", "jitter", "grayscale", "gaussian_noise", "sp_noise", "cutout", "crop"]
    angle_matrices = np.zeros((len(layers), len(aug_names), len(aug_names)))
    n_augs = len(aug_names)
    result_path = "/mnt/home/the10/netrep-analysis/results"

    for k, layer in enumerate(layers):
        print("reading", layer)
        in_npy  = f"/mnt/home/the10/ceph/results/netrep/results_aggregated/{layer}.npy"
        out_npy = f"{result_path}/{layer}_angle_matrix.npy"

        if os.path.isfile(out_npy):
            angle_matrices = np.load(out_npy)
            angle_matrix = angle_matrices[-1]
            continue 

        raw = np.load(in_npy, mmap_mode="r")
        raw_data = np.empty_like(raw)

        chunk = 1
        for i in tqdm(range(0, raw.shape[0], chunk), desc="Loading into RAM"):
            raw_data[i:i+chunk] = raw[i:i+chunk]

        shape0 = center_scale(raw_data[0])

        rotate = center_scale(raw_data[1:5])
        sheer = center_scale(raw_data[5:9])
        jitter = center_scale(raw_data[9:13])
        grayscale = center_scale(raw_data[13:17])
        gaussian_noise = center_scale(raw_data[17:21])
        sp_noise = center_scale(raw_data[21:25])
        cutout = center_scale(raw_data[25:29])
        crop = center_scale(raw_data[29:33])
        
        shape0 = reduce_activity([shape0])[0]

        # compute all pairwise angles between augmentations
        aug_list = [rotate, sheer, jitter, grayscale, gaussian_noise, sp_noise, cutout, crop]
        new_aug_list = []
        n_augs = len(aug_list)

        for aug_activity in tqdm(aug_list, desc="Reducing activity"):
            new_aug_list.append(reduce_activity(aug_activity))

        angle_matrix = np.zeros((n_augs, n_augs))
        for p in range(4):
            for i in tqdm(range(n_augs)):
                for j in range(n_augs):
                    if i < j:
                        v1, xdiff1 = landmark_vel(shape0, new_aug_list[i][p])
                        v2, xdiff2 = landmark_vel(shape0, new_aug_list[j][p])
                        angle = geodesic_angle(v1, v2)
                        angle_matrix[i, j] = angle_matrix[i,j] + angle  #angle * 180 / np.pi

        angle_matrix = angle_matrix + angle_matrix.T
        angle_matrix = angle_matrix / 4  # average over 4 levels per aug
        angle_matrix = (angle_matrix * 180) / np.pi

        angle_matrices[k] = angle_matrix

    np.save(f"{result_path}/{layer}_angle_matrix.npy", angle_matrices)

    cmin = np.min(angle_matrix[np.nonzero(angle_matrix)])
    cmax = np.max(angle_matrix)

    plt.imshow(angle_matrix, cmap='viridis')
    plt.colorbar(label='Mean Geodesic Angle (degrees)')
    plt.clim(cmin-1, cmax)
    plt.xticks(ticks=np.arange(n_augs), labels=aug_names, rotation=45)
    plt.yticks(ticks=np.arange(n_augs), labels=aug_names)
    plt.tight_layout()
    plt.savefig(f'{result_path}/{layer}_geodesic_angles.png')

    # plot_angle_matrices(
    #     angle_matrices=angle_matrices,
    #     aug_names=aug_names,
    #     save_path="geodesic_angles_1x4.png"
    # )