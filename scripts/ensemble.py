import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    raw_data = np.array(np.load("/mnt/home/the10/ceph/results/netrep/results_aggregated/fc_origin.npz")['arr_0'].astype('float64'))
    labels = np.load("cifar10_labels.npy")

    aug1 = (4, 8, 12, 16, 20, 24)
    aug2 = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0)
    n_aug1, n_aug2, n, d = raw_data.shape

    acc_grid = np.empty((n_aug1+1, n_aug2+1), dtype=float)
    for i in range(n_aug1):
        for j in range(n_aug2):
            preds = np.argmax(raw_data[i, j], axis=1)
            acc_grid[i, j] = (preds == labels).mean()

    # Row-wise averag
    row_mean_logits = raw_data.mean(axis=1)
    row_preds = np.argmax(row_mean_logits, axis=2)
    acc_grid[:-1, -1] = (row_preds == labels[None, :]).mean(axis=1)

    # Column-wise average
    col_mean_logits = raw_data.mean(axis=0)
    col_preds = np.argmax(col_mean_logits, axis=2)
    acc_grid[-1, :-1] = (col_preds == labels[None, :]).mean(axis=1)

    # Average everything
    all_mean_logits = raw_data.mean(axis=(0, 1))
    all_preds = np.argmax(all_mean_logits, axis=1)
    acc_grid[-1, -1] = (all_preds == labels).mean()

    plt.figure()
    plt.imshow(acc_grid, origin="upper", aspect="auto")
    plt.colorbar(label="Accuracy")
    plt.xlabel("Gaussian noise sigma")
    plt.ylabel("Cutout patch size")
    plt.title(f"Accuracy Heatmap")
    plt.xticks(ticks=np.arange(n_aug2 + 1),labels=list(aug2) + ["avg"])
    plt.yticks(ticks=np.arange(n_aug1 + 1),labels=list(aug1) + ["avg"])

    for i in range(n_aug1 + 1):
        for j in range(n_aug2 + 1):
            plt.text(
                j, i, f"{acc_grid[i, j]:.3f}",
                ha="center", va="center", fontsize=8
            )
    plt.savefig("ensemble_accuracy.png")

