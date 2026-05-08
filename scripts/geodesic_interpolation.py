import itertools

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Riemannian geometry on shape space
# ---------------------------------------------------------------------------

def center_scale(X):
    Xc = X - np.mean(X, axis=0, keepdims=True)
    return Xc / np.linalg.norm(Xc)


def angular_proc_dist(p, q):
    """Riemannian (angular Procrustes) distance between two shapes."""
    singular_values = np.linalg.svd(q.T @ p, compute_uv=False)
    return np.arccos(np.clip(np.sum(singular_values), -1.0, 1.0))


def proc_dist(X, Y):
    """Euclidean Procrustes distance between two shapes."""
    singular_values = np.linalg.svd(X.T @ Y, compute_uv=False)
    return np.sqrt(2 * (1 - np.sum(singular_values)))


def spherical_interp(X, Y, t):
    """
    Spherical (slerp) interpolation between X and Y on the unit sphere.
    Returns X at t=0 and Y at t=1.
    """
    c = np.sum(X * Y)
    theta = np.arccos(np.clip(c, -1.0, 1.0))
    if theta < 1e-8:
        return X
    a = np.sin((1 - t) * theta) / np.sin(theta)
    b = np.sin(t * theta) / np.sin(theta)
    return center_scale(a * X + b * Y)


def shape_interp(X, Y, t):
    """Align Y to X by optimal orthogonal transformation, then slerp."""
    U, _, Vt = np.linalg.svd(Y.T @ X)
    Y_aligned = Y @ U @ Vt
    return spherical_interp(X, Y_aligned, t)


def exp_map(X, V):
    """
    Exponential map on shape space at base X with tangent vector V.

    Args:
        X: (M, N) centered unit-norm representation.
        V: (M, N) tangent vector orthogonal to X.
    Returns:
        (M, N) centered unit-norm representation.
    """
    norm_v = np.linalg.norm(V)
    return center_scale(np.cos(norm_v) * X + np.sin(norm_v) * (V / norm_v))


def log_map(X, Y):
    """
    Logarithmic map on shape space: tangent vector at X pointing toward Y.

    Args:
        X: (M, N) centered unit-norm representation.
        Y: (M, N) centered unit-norm representation.
    Returns:
        V: (M, N) tangent vector at X.
    """
    U, _, Vt = np.linalg.svd(Y.T @ X)
    Y_aligned = Y @ U @ Vt
    inner = np.clip(np.sum(X * Y_aligned), -1.0, 1.0)
    theta = np.arccos(inner)
    if theta < 1e-6:
        return np.zeros_like(X)
    V = Y_aligned - inner * X
    return V * (theta / np.linalg.norm(V))


def procrustes_align_to_reference(Y, X_ref):
    """
    Optimal orthogonal alignment of Y to X_ref in column space (same rotation
    as inside `log_map` / `shape_interp`).

    Args:
        Y: (M, N) typically `center_scale`'d.
        X_ref: (M, N) fixed reference, same preprocessing as Y.

    Returns:
        Y_aligned: (M, N) = Y @ U @ Vt from SVD(Y.T @ X_ref).
    """
    U, _, Vt = np.linalg.svd(Y.T @ X_ref)
    return Y @ U @ Vt


def predict_composition(X0, Xa, Xb):
    """
    Predict the representation of the composed augmentation (a ∘ b) at X0
    by adding tangent vectors in the log-map and mapping back.
    """
    vA = log_map(X0, Xa)
    vB = log_map(X0, Xb)
    return exp_map(X0, vA + vB)


def geodesic_angle(v1, v2):
    """Angle (radians) between two tangent vectors."""
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-12:
        return 0.0
    return np.arccos(np.clip(np.sum(v1 * v2) / denom, -1.0, 1.0))

# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------

def _compute_pairwise_distances(shapes):
    n = len(shapes)
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        D[i, j] = angular_proc_dist(shapes[i], shapes[j])
    D += D.T
    return D


def _plot_triple_panel(D, labels, save_path):
    """
    Plot a 3-panel figure: (1) distance matrix, (2) 2-D MDS+PCA, (3) 3-D MDS+PCA.
    """
    embedding = MDS(
        n_components=min(200, len(labels) - 1),
        metric=True,
        eps=1e-5,
        normalized_stress="auto",
        dissimilarity="precomputed",
        random_state=42,
    )
    Z = embedding.fit_transform(np.abs(np.real(D)))
    print(f"MDS stress: {embedding.stress_:.4f}")

    coords_2d = PCA(n_components=2, random_state=42).fit_transform(Z)
    coords_3d = PCA(n_components=3, random_state=42).fit_transform(Z)

    fig = plt.figure(figsize=(15, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    mask = ~np.eye(len(D), dtype=bool)
    im = ax1.imshow(D, cmap="viridis", vmin=np.min(D[mask]), vmax=np.max(D[mask]))
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_title("Distances")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(coords_2d[:, 0], coords_2d[:, 1])
    for k, name in enumerate(labels):
        ax2.text(coords_2d[k, 0], coords_2d[k, 1], name, fontsize=8)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("2D PCA of MDS embedding")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2])
    for k, name in enumerate(labels):
        ax3.text(coords_3d[k, 0], coords_3d[k, 1], coords_3d[k, 2], name, fontsize=8)
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_zlabel("PC3")
    ax3.set_title("3D PCA of MDS embedding")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Geodesic interpolation analysis
# ---------------------------------------------------------------------------

def run_geodesic_interpolation(individual_data, save_prefix="geodesic_interpolation"):
    """
    Check whether shape1 lies on the geodesic between shape0 and shape2.

    Plots the angular Procrustes distance from shape1 to each interpolated
    point along the geodesic shape0 → shape2, then a triple-panel MDS/PCA
    summary of the full pairwise distance matrix.
    """
    shape0 = center_scale(individual_data[0])
    shape1 = center_scale(individual_data[1])
    shape2 = center_scale(individual_data[2])

    ts = np.linspace(0, 1)
    interp_shapes = [shape_interp(shape0, shape2, t) for t in ts]
    losses = np.array([angular_proc_dist(shape1, s) for s in tqdm(interp_shapes, desc="Interpolating")])

    fig, ax = plt.subplots()
    ax.plot(ts, losses)
    ax.plot([ts[0], ts[-1]], [losses[0], losses[-1]], "o")
    ax.axhline(0, dashes=[2, 2])
    ax.set_xlabel("Interpolation parameter t")
    ax.set_ylabel("d(shape1, interp(shape0, shape2, t))")
    ax.set_title("Geodesic interpolation: shape0 → shape2 vs. shape1")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png")
    plt.close()
    print(f"Saved {save_prefix}.png")

    all_shapes = [shape1] + interp_shapes
    labels = ["shape1"] + [f"t={t:.2f}" for t in ts]
    D = _compute_pairwise_distances(all_shapes)
    _plot_triple_panel(D, labels, f"{save_prefix}_triple_panel.png")


# ---------------------------------------------------------------------------
# Geodesic angle analysis for compositionality
# ---------------------------------------------------------------------------

def _visualize_composition_errors(distances, n_aug, augmentation_types, save_path):
    """Heatmap of normalised composition prediction error for each aug pair."""
    mat = np.full((n_aug, n_aug), np.nan)
    it = iter(distances)
    for i in range(n_aug):
        for j in range(n_aug):
            if i != j:
                mat[i, j] = next(it)

    fig, ax = plt.subplots()
    im = ax.imshow(mat, cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set(
        xlabel="Augmentation j",
        ylabel="Augmentation i",
        title="Normalised composition prediction error d(X_ab, X̂_ab) / d(X0, X_ab)",
        xticks=range(n_aug),
        yticks=range(n_aug),
    )
    ax.set_xticklabels(augmentation_types, rotation=45, ha="right")
    ax.set_yticklabels(augmentation_types)
    for i in range(n_aug):
        for j in range(n_aug):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", va="center", ha="center", color="w", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def _make_angle_matrix(values, n_aug):
    """Unpack a flat list of (i≠j) values into an (n_aug, n_aug) matrix."""
    mat = np.full((n_aug, n_aug), np.nan)
    it = iter(values)
    for i in range(n_aug):
        for j in range(n_aug):
            if i != j:
                mat[i, j] = next(it)
    return mat


def visualize_geodesic_angles(
    angles_a_b, angles_a_ab, angles_b_ab,
    n_aug, augmentation_types, save_path,
):
    """
    3-panel heatmap of geodesic angles (degrees) between tangent vectors at X0:
      (1) angle( v(X0→Xa), v(X0→Xb) )
      (2) angle( v(X0→Xa), v(X0→Xab) )
      (3) angle( v(X0→Xb), v(X0→Xab) )

    Each panel uses the same (aug_i rows, aug_j cols) layout as the error heatmap.
    """
    panels = {
        "∠(X0→Xa, X0→Xb)":  _make_angle_matrix(angles_a_b,  n_aug),
        "∠(X0→Xa, X0→Xab)": _make_angle_matrix(angles_a_ab, n_aug),
        "∠(X0→Xb, X0→Xab)": _make_angle_matrix(angles_b_ab, n_aug),
    }

    # shared colour scale in degrees
    all_vals = np.concatenate([m[~np.isnan(m)] for m in panels.values()])
    vmin, vmax = np.degrees(all_vals.min()), np.degrees(all_vals.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (title, mat) in zip(axes, panels.items()):
        mat_deg = np.degrees(mat)
        im = ax.imshow(mat_deg, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Aug j")
        ax.set_ylabel("Aug i")
        ax.set_xticks(range(n_aug))
        ax.set_xticklabels(augmentation_types, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_aug))
        ax.set_yticklabels(augmentation_types, fontsize=8)
        for i in range(n_aug):
            for j in range(n_aug):
                if not np.isnan(mat_deg[i, j]):
                    ax.text(j, i, f"{mat_deg[i, j]:.0f}°", va="center", ha="center",
                            color="w", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="degrees")

    fig.suptitle("Geodesic angles between directions from X0", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def run_composition_analysis(individual_data, combined_data, save_prefix="composition"):
    """
    For each ordered pair of augmentation types, predict the composed
    representation via tangent-vector addition and measure the error relative
    to the ground-truth composed model.

    individual_data: array indexed [aug_index, stimuli, neurons]
    combined_data:   array indexed [pair_index, stimuli, neurons] where
                     pair_index follows the ordering of all (i≠j) pairs.
    """
    augmentation_types = ["rotate", "sheer", "jitter", "grayscale", "gaussian_noise", "sp_noise", "crop"]
    aug_indices = [4, 8, 12, 16, 20, 24, 32]  # last-magnitude index for each aug (skipping cutout)

    X0 = center_scale(individual_data[0])
    augmented = individual_data[aug_indices]
    n_aug = len(aug_indices)

    pair_indices = [(i, j) for i in range(n_aug) for j in range(n_aug) if i != j]
    distances = []
    angles_a_b, angles_a_ab, angles_b_ab = [], [], []

    for k, (i, j) in tqdm(enumerate(pair_indices), total=len(pair_indices), desc="Composition pairs"):
        Xa  = center_scale(augmented[i])
        Xb  = center_scale(augmented[j])
        X_ab = center_scale(combined_data[k])
        X_ab_pred = predict_composition(X0, Xa, Xb)

        dist = angular_proc_dist(X_ab, X_ab_pred)
        original_dist = angular_proc_dist(X0, X_ab)
        distances.append(dist / original_dist)

        va  = log_map(X0, Xa)
        vb  = log_map(X0, Xb)
        vab = log_map(X0, X_ab)
        angles_a_b.append(geodesic_angle(va, vb))
        angles_a_ab.append(geodesic_angle(va, vab))
        angles_b_ab.append(geodesic_angle(vb, vab))

    _visualize_composition_errors(distances, n_aug, augmentation_types, f"{save_prefix}_error_heatmap.png")
    visualize_geodesic_angles(
        angles_a_b, angles_a_ab, angles_b_ab,
        n_aug, augmentation_types, f"{save_prefix}_geodesic_angles.png",
    )

    best_k = int(np.argmin(distances))
    best_i, best_j = pair_indices[best_k]
    print(
        f"Best-predicted pair: ({augmentation_types[best_i]}, {augmentation_types[best_j]}), "
        f"normalised error = {distances[best_k]:.4f}"
    )

    Xa = center_scale(augmented[best_i])
    Xb = center_scale(augmented[best_j])
    X_ab = center_scale(combined_data[best_k])
    X_ab_pred = predict_composition(X0, Xa, Xb)

    key_shapes = [X0, Xa, Xb, X_ab, X_ab_pred]
    key_labels = ["X0", "Xa", "Xb", "X_ab", "X̂_ab"]
    D = _compute_pairwise_distances(key_shapes)
    _plot_triple_panel(D, key_labels, f"{save_prefix}_triple_panel.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    individual_data = np.array(
        np.load("/mnt/home/the10/ceph/results/netrep/results_aggregated/avgpool.npz")["arr_0"].astype("float64")
    )
    combined_data = np.array(
        np.load("/mnt/home/the10/ceph/results/netrep/results_aggregated/two_avgpool.npz")["arr_0"].astype("float64")
    )
    print(f"individual_data: {individual_data.shape}, combined_data: {combined_data.shape}")

    run_geodesic_interpolation(individual_data)
    run_composition_analysis(individual_data, combined_data)

if __name__ == "__main__":
    main()
