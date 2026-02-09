import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

raw_data = np.array(np.load("/mnt/home/the10/ceph/results/netrep/results_aggregated/avgpool.npz")['arr_0'].astype('float64'))
aug1 = (4, 8, 12, 16, 20, 24)
# aug2 = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0)
aug2 = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
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
    Rimannian shape distance.

    Args:
        p: (M, N) array of centered and rescaled neural responses.
        q: (M, N) array of centered and rescaled neural responses.
    """
    singular_values = np.linalg.svd(q.T @ p, compute_uv=False)
    return 2 * (1 - np.sum(singular_values))
    # return np.arccos(np.sum(singular_values))

def exp_map(X, V):
    """
    Exponential map on shape space at base X with tangent V.
    This is simply the exponential map on the sphere - the output
    is implicitly defined up to an orthogonal transformation.

    Args:
        X: (M, N) array of centered and unit-norm neural responses.
        V: (M, N) array that is orthogonal to p
    
    Returns:
        Y: (M, N) array of centered and unit-norm neural responses.
    """
    norm_v = np.linalg.norm(V)
    return center_scale(
        np.cos(norm_v) * X + np.sin(norm_v) * (V / norm_v)
    )

def log_map(X, Y):
    """
    Logarithmic map on shape space at base X with respect to Y.
    This is simply the logarithmic map on the sphere after we have
    aligned Y to X by a procrustes transformation.

    Args:
        X: (M, N) array of centered and unit-norm neural responses.
        Y: (M, N) array of centered and unit-norm neural responses.
    
    Returns:
        V: (M, N) array in tangent space at p.
    """
    U, _, Vt = np.linalg.svd(Y.T @ X)
    Y_aligned = Y @ U @ Vt
    inner = np.clip(np.sum(X * Y_aligned), -1.0, 1.0)
    theta = np.arccos(inner)
    if theta < 1e-6:
        return np.zeros_like(X)
    V = Y_aligned - inner * X
    return V * (theta / np.linalg.norm(V))

def proc_dist(X, Y):
    """
    Square of the procrustes shape distance.

    Args:
        X: (M, N) array of centered and rescaled neural responses.
        Y: (M, N) array of centered and rescaled neural responses.
    """
    singular_values = np.linalg.svd(X.T @ Y, compute_uv=False)
    return np.sqrt(2 * (1 - np.sum(singular_values)))


data = np.zeros_like(raw_data)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        data[i, j] = center_scale(raw_data[i, j])

pred_dists = np.zeros((len(aug1) - 1, len(aug2) - 1))

# for i in range(len(aug1) - 1):
#     for j in tqdm(range(len(aug2) - 1)):
#         A = data[i, j]
#         B = data[i, j + 1]
#         C = data[i + 1, j]
#         D = data[i + 1, j + 1]

#         vB = log_map(A, B)
#         vC = log_map(A, C)

#         pred_dists[i, j] = proc_dist(exp_map(A, vB + vC), D) / proc_dist(A, D)
# print(pred_dists)
n_aug1, n_aug2, _, _ = raw_data.shape
fig, ax = plt.subplots(n_aug1, n_aug2-2, figsize=(4*(n_aug2-2), 3*n_aug1))

for i in range(n_aug1):
    for j in range(1, n_aug2-1):
        print(i, j)

        shape0 = center_scale(raw_data[i, j-1])
        shape1 = center_scale(raw_data[i, j])
        shape2 = center_scale(raw_data[i, j+1])

        ts = np.linspace(0, 1)
        losses = np.array([
            sq_proc(shape1, shape_interp(shape0, shape2, a)) for a in tqdm(ts)
        ])
        jj = j-1

        ax[i][jj].plot(ts, losses)
        ax[i][jj].plot([ts[0], ts[-1]], [losses[0], losses[-1]], 'o')
        ax[i][jj].axhline(0, dashes=[2, 2])
        ax[i][jj].set_ylabel(f"d{i, j}", fontsize=8)
        # ax[i][jj].set_xlabel("Interpolated position")
        ax[i][jj].set_title(f"{j-1} → {j+1}", fontsize=8)
        ax[i][jj].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        # ax[i][jj].set_xticklabels([f'[{i, j-1}]', '', '', '', f'[{i, j+1}]]'])
plt.tight_layout()
plt.savefig('geodesics.png')