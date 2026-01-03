import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

raw_data = np.array(np.load("/mnt/home/the10/ceph/results/netrep/results_aggregated/avgpool_origin.npz")['arr_0'].astype('float64'))
aug1 = (4, 8, 12, 16, 20, 24)
aug2 = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0)

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
    return 2 * (1 - np.sum(singular_values))

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