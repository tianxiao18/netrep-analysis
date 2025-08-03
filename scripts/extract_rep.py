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

    return parser.parse_args()

def split_activity(activities, save_path=None, n_components=1000, test_size=0.2):
    X_train, X_test = [], []

    if save_path and os.path.isfile(save_path):
        print(f"Loading PCA-reduced activities from {save_path}")
        pca_activities = np.load(save_path)

    else:
        print("Computing PCA-reduced activities...")
        pca_activities = {}

        for key in activities:
            X = activities[key]
            n_components = min(len(X), n_components)

            pca = DualPCA(n_components=n_components)
            X_pca = pca.fit_transform(X.reshape(X.shape[0], -1))

            print(f"{key} variance explained in {n_components} dim: {np.sum(pca.explained_variance_ratio_)}")
            pca_activities[key] = X_pca
            
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


def compute_distances(X_train, X_test):
    distmat = np.zeros((len(X_train), len(X_train)))
    total_pairs = len(X_train) * (len(X_train) - 1) // 2

    # use procruste's metrics
    for i, j in tqdm(itertools.combinations(range(len(X_train)), 2), total=total_pairs, desc="Computing distances"):
        metric = LinearMetric(alpha=1.0, center_columns=True, score_method='angular')
        metric.fit(X_train[i], X_train[j])
        dist = metric.score(X_test[i], X_test[j])
        distmat[i, j] = dist

    distmat += distmat.T
    return distmat

def distance_vs_sample_size(args, output_path):
    sample_sizes = [100, 200, 500, 1000, 5000, 10000]
    random_seeds = [42, 43, 44, 45, 46]
    all_distances = []

    for test_size in sample_sizes:
        distances = []

        for seed in random_seeds:
            print(f"Computing Procruste's distance of test size {test_size} at seed {seed}")
            set_seed(seed)
            extractor = LayerActivityExtractor(
                    checkpoint_path=f'{output_path}/checkpoints/{args.model}.pth',
                    image_folder=args.data_path,
                    batch_size=args.batch_size, 
                    num_workers=args.num_workers,
                    test_size=test_size,
                    seed=seed
            )

            activities = extractor.get_activities()
            X_train, _ = split_activity(activities, test_size=0)
            # network_names = activities.keys()

            distmat = compute_distances(X_train, X_train)
            print(distmat)
            # visualize_distance(distmat, network_names, result_path)
            del activities, X_train
            gc.collect() 
            distances.append(distmat)

        all_distances.append(distances)
    
    all_distances = np.array(all_distances)
    np.save(f'{output_path}/distance_matrix_all.npy', all_distances)

    # plt.plot(sample_sizes, distances)
    # plt.savefig('distance_vs_sample_size.png')

def pc_vs_sample_size(args, output_path):
    n_pcs = [10, 20, 50, 100, 200, 500, 1000]
    random_seeds = [42, 43, 44, 45, 46]
    all_distances = []

    for n_pc in n_pcs:
        distances = []

        for seed in random_seeds:
            print(f"Computing Procruste's distance of pc dim {n_pc} at seed {seed}")
            set_seed(seed)
            extractor = LayerActivityExtractor(
                    checkpoint_path=f'{output_path}/checkpoints/{args.model}.pth',
                    image_folder=args.data_path,
                    batch_size=args.batch_size, 
                    num_workers=args.num_workers,
                    test_size=10000
            )

            activities = extractor.get_activities()
            X_train, _ = split_activity(activities, n_components=n_pc, test_size=0)
            print(np.array(X_train).shape)
            # network_names = activities.keys()

            distmat = compute_distances(X_train, X_train)
            print(distmat)
            # visualize_distance(distmat, network_names, result_path)
            distances.append(distmat)

        all_distances.append(distances)
    
    all_distances = np.array(all_distances)
    np.save(f'{output_path}/distance_matrix_pc_all.npy', all_distances)


def extract_activity_all(experiment_ls, base_path, args):
    X_train_all, X_test_all, network_names_all = [], [], []

    # extract network names, assuming all experiments has the same network names
    activities = LayerActivityExtractor(checkpoint_path=f'{base_path}/experiments/{experiment_ls[0]}/checkpoints/best_model.pth',
        image_folder=args.data_path,batch_size=args.batch_size, num_workers=args.num_workers,test_size=args.test_size
    ).get_activities()
    network_names = activities.keys()

    for exp in experiment_ls:
        output_path = f'{base_path}/experiments/{exp}'
        save_path = f'{output_path}/results/activities_pca.npz'

        print(f"Loading network representation from {exp}...")
        
        X_train, X_test = split_activity(None, save_path, n_components=1000, test_size=0)
        X_train_all.extend(X_train)
        X_test_all.extend(X_test)
        network_names_all.extend(network_names)

    return X_train_all, X_test_all, network_names_all

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def exp_to_name(exp_ls):
    name_ls = []
    param_ls = []

    symbol_map = {'temp': 'T', 'cutout_patch_size': 'C'}

    for exp in exp_ls:        
        if exp != 'clean':
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

def main():
    args = parse_args()
    set_seed(42)

    base_path = '/mnt/home/the10/ceph/results/netrep'
    output_path = f'{base_path}/experiments/{args.experiment}'
    result_path = f'{output_path}/results'
    result_path = f'{result_path}/{args.model}' if args.model != 'best_model' else result_path
    embed_path = f'{result_path}/activities.npz'

    os.makedirs(result_path, exist_ok=True)
    experiment_ls = ['clean',  'cutout/exp_resnet_cutout_patch_size_60.0', 'cutout/exp_resnet_cutout_patch_size_80.0','cutout/exp_resnet_cutout_patch_size_100.0', 'cutout/exp_resnet_cutout_patch_size_120.0']

    # distance_vs_sample_size(args, output_path)
    # dist_matrix_path = "/mnt/home/the10/ceph/results/netrep/experiments/clean/distance_matrix_pc_all.npy"
    # visualize_distance_matrices(dist_matrix_path, sample_sizes=[10, 20, 50, 100, 200, 500, 1000], seeds=[42, 43, 44, 45, 46])

    # extract activity for each layer in the network
    if 'all' not in args.experiment:
        extractor = LayerActivityExtractor(
            checkpoint_path=f'{output_path}/checkpoints/{args.model}.pth',
            image_folder=args.data_path,
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            test_size=args.test_size
        )
            
        # if not os.path.isfile(embed_path):
        print("Extracting network representation...")
        activities = extractor.get_activities()
        # np.savez_compressed(embed_path, **activities)
        # else:
        #     print("Loading network representation...")
        #     activities = np.load(embed_path)

        X_train, X_test = split_activity(activities, save_path=f'{result_path}/activities_pca.npz', test_size=0)
        network_names = activities.keys()

    # extract activity for all networks
    else:
        X_train, X_test, network_names = extract_activity_all(experiment_ls, base_path, args)

    # compute procruste's distance between pair of networks
    print(np.array(X_train).shape)
    if not os.path.isfile(f"{result_path}/distance_matrix.npy"):
        distmat = compute_distances(X_train, X_train)
    else:
        distmat = np.load(f"{result_path}/distance_matrix.npy")
    
    if 'all' in args.experiment:
        l = len(network_names) // len(experiment_ls)
        expanded_experiment_ls = [exp for exp in experiment_ls for _ in range(l)]
        name_ls, _ = exp_to_name(expanded_experiment_ls)
        visualize_distance(distmat, name_ls, result_path)

        name_ls, _ = exp_to_name(experiment_ls)
    else:
        visualize_distance(distmat, network_names, result_path)

    embedding = MDS(n_components=200, metric= True, eps = 0.00001, normalized_stress='auto', dissimilarity='precomputed', random_state=42)
    Z = embedding.fit_transform(np.abs(np.real(distmat)))
    print(embedding.stress_)

    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(Z)
    if 'all' in args.experiment:
        visualize_coordinates_all(coordinates, network_names, result_path, name_ls)
        visualize_layer_aligned(coordinates, network_names, result_path, name_ls, mds_dim=0)
        visualize_layer_aligned(coordinates, network_names, result_path, name_ls, mds_dim=1)
    else:
        visualize_coordinates(coordinates, network_names, result_path)


if __name__ == "__main__":
    main()