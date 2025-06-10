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
from argparse import ArgumentParser

from sklearn.model_selection import train_test_split
import itertools
from netrep.metrics import LinearMetric
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = ArgumentParser(description="PyTorch Resnet Trainer")
    parser.add_argument(
        "--test_size",
        type=int,
        default=1000,
        help="Number of test images to extract representation"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/gpuxl/scc/AI_DATASETS/ImageNet/2012/imagenet/val"
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
        default=64
    )

    return parser.parse_args()

def main():
    args = parse_args()

    base_path = '/mnt/home/the10/netrep-analysis'
    output_path = f'{base_path}/experiments/{args.experiment}'
    embed_path = f'{output_path}/results/activities.npz'

    extractor = LayerActivityExtractor(
        checkpoint_path=f'{output_path}/checkpoints/best_model.pth',
        image_folder='/mnt/gpuxl/scc/AI_DATASETS/ImageNet/2012/imagenet/val',
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        test_size=args.test_size
    )

    if not os.path.isfile(embed_path):
        print("Extracting network representation...")
        activities = extractor.get_activities()
        print("Saving extracted representation...")
        np.savez_compressed(embed_path, **activities)
    else:
        print("Loading network representation...")
        activities = np.load(embed_path)

    X_train, X_test = [], []

    for key in list(activities.keys()):
        X = activities[key]
        pca = PCA(n_components=1000)
        X_pca = pca.fit_transform(X.reshape(X.shape[0], -1))
        print("Variance explained: ", np.sum(pca.explained_variance_ratio_))
        X1_train, X1_test = train_test_split(X_pca, test_size=0.2, random_state=42)
        # X_train.append(X1_train.transpose(0, 2, 3, 1).reshape(-1, X1_train.shape[1]))
        # X_test.append(X1_test.transpose(0, 2, 3, 1).reshape(-1, X1_test.shape[1]))
        X_train.append(X1_train)
        X_test.append(X1_test)

    distmat = np.zeros((len(X_train), len(X_train)))

    # use procruste's metrics
    for i, j in itertools.combinations(range(len(X_train)), 2):
        metric = LinearMetric(alpha=1.0, center_columns=True, score_method='angular')
        print(X_train[i].shape, X_train[j].shape)
        metric.fit(X_train[i], X_train[j])
        dist = metric.score(X_test[i], X_test[j])
        distmat[i, j] = dist

    distmat += distmat.T
    visualize_distance(distmat, activities.keys())
    print(distmat)

    embedding = MDS(n_components=200, metric= True, eps = 0.00001, normalized_stress='auto', dissimilarity='precomputed')
    Z = embedding.fit_transform(np.abs(np.real(distmat)))
    print(embedding.stress_)

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(Z)
    visualize_coordinates(coordinates, activities.keys())

    
def visualize_distance(distmat, network_names):
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
        xticklabels=network_names,
        yticklabels=network_names,
        vmin=vmin,
        vmax=vmax
    )

    plt.title("Representation Distance Matrix")
    plt.xlabel("Network Index")
    plt.ylabel("Network Index")
    plt.tight_layout()
    plt.savefig('dist.png')
    plt.close()


def visualize_coordinates(coords, network_names):
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=150, c=range(len(coords)))
    for i, name in enumerate(network_names):
        name = '.'.join(name.split('.')[1:])
        plt.text(coords[i, 0], coords[i, 1]+ 0.03, name, fontsize=10, ha='center', va='center')
    plt.title(f"MDS Embedding")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.colorbar()
    plt.savefig('mds_embedding.png')

if __name__ == "__main__":
    main()