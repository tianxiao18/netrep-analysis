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

    parser.add_argument(
        "--edit",
        type=bool,
        default=False
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
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


def compute_across_network_distances(netA, netB):
    distmat = np.zeros((len(netA), len(netB)))

    # use procruste's metrics
    for i in tqdm(range(len(netA))):
        for j in range(len(netB)):
            metric = LinearMetric(alpha=1.0, center_columns=True, score_method='angular')
            metric.fit(netA[i], netB[j])
            dist = metric.score(netA[i], netB[j])
            distmat[i, j] = dist

    distmat += distmat.T
    return distmat

def compute_distances(X_train, X_test):
    distmat = np.zeros((len(X_train), len(X_train)))
    diff_norms = np.zeros((len(X_train), len(X_train), len(X_train[0])))
    total_pairs = len(X_train) * (len(X_train) - 1) // 2

    # use procruste's metrics
    for i, j in tqdm(itertools.combinations(range(len(X_train)), 2), total=total_pairs, desc="Computing distances"):
        metric = LinearMetric(alpha=1.0, center_columns=True, score_method='angular')
        metric.fit(X_train[i], X_train[j])
        dist = metric.score(X_test[i], X_test[j])
        distmat[i, j] = dist

        # obtain transformed representation
        if i % 4 == 2 and j % 4 == 2:
            tX = (X_test[i]-metric.mx_) @ metric.Wx_
            tY = (X_test[j]-metric.my_) @ metric.Wy_
            diff_norms[i][j] = np.linalg.norm(tX - tY, axis=1)

    distmat += distmat.T
    return distmat, diff_norms

def edit_distances(X_train, distmat, idxs):
    distmat = distmat.copy()
    print("dist before: ", np.mean(distmat[:, idxs]), np.mean(distmat[idxs, :]))

    # use procruste's metrics
    for i in idxs:
        for j in tqdm(range(len(X_train))):
            metric = LinearMetric(alpha=1.0, center_columns=True, score_method='angular')
            metric.fit(X_train[i], X_train[j])
            dist = metric.score(X_train[i], X_train[j])
            distmat[i, j] = dist
            distmat[j, i] = dist

    print("dist after: ", np.mean(distmat[:, idxs]), np.mean(distmat[idxs, :]))

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
                    seed=seed,
                    model_name=args.model_name,
                    dataset=args.dataset
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
                    test_size=10000,
                    model_name=args.model_name,
                    dataset=args.dataset
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
    print(f"Loading from {base_path}/experiments/{experiment_ls[0]}/checkpoints/final_model.pth")
    print(base_path, experiment_ls[0])
    network_names = LayerActivityExtractor(checkpoint_path=f'{base_path}/experiments/{experiment_ls[0]}/checkpoints/final_model.pth',
        image_folder=args.data_path,batch_size=args.batch_size, num_workers=args.num_workers,test_size=args.test_size, 
        model_name=args.model_name, dataset=args.dataset
    ).get_layer_names()

    for exp in experiment_ls:
        output_path = f'{base_path}/experiments/{exp}'
        os.makedirs(f'{output_path}/results', exist_ok=True)
        save_path = f'{output_path}/results/activities_pca.npz'
        # save_path = None

        if save_path and os.path.isfile(save_path):
            print(f"Loading PCA-reduced activities from {save_path}")

            with np.load(save_path, allow_pickle=False) as data:
                pca_activities = {k: data[k] for k in data.files}
                X_train = [pca_activities[name].reshape(pca_activities[name].shape[0], -1) for name in network_names]
                X_test = [pca_activities[name].reshape(pca_activities[name].shape[0], -1) for name in network_names]
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
            network_names = extractor.get_layer_names()
            activities = extractor.get_activities()
            X_train, X_test = split_activity(activities, save_path=save_path, n_components=1000, test_size=0)

        X_train_all.extend(X_train)
        X_test_all.extend(X_test)
        network_names_all.extend(network_names)

    return X_train_all, X_test_all, network_names_all

def extract_raw_activity(experiment_ls, base_path, args, n_aug1, n_aug2):
    network_names = LayerActivityExtractor(checkpoint_path=f'{base_path}/experiments/{experiment_ls[-1]}/checkpoints/final_model.pth',
        image_folder=args.data_path,batch_size=args.batch_size, num_workers=args.num_workers,test_size=args.test_size, 
        model_name=args.model_name, dataset=args.dataset
    ).get_layer_names()[-2:]
    X_all = {name: [] for name in network_names}

    for exp in experiment_ls:
        output_path = f'{base_path}/experiments/{exp}'
        save_path = f'{base_path}/experiments/{exp}/results/activities_raw.npz'

        if os.path.isfile(save_path):
            print(f"Loading raw activities from {save_path}")
            with np.load(save_path, allow_pickle=False) as data:
                pca_activities = {k: data[k] for k in data.files}

        else:
            print(f"Extracting raw activities to {save_path}")
            extractor = LayerActivityExtractor(
                checkpoint_path=f'{output_path}/checkpoints/{args.model}.pth',
                image_folder=args.data_path,
                batch_size=args.batch_size, 
                num_workers=args.num_workers,
                test_size=args.test_size,
                model_name=args.model_name,
                dataset=args.dataset
            )
            activities = extractor.get_activities()
            
            pca_activities = {}
            for key in list(activities.keys())[-2:]:
                X = np.load(activities[key], mmap_mode="r")
                X = X.astype(np.float32).reshape(X.shape[0], -1)
                pca_activities[key] = X
                print(key, X.shape)

            np.savez_compressed(save_path, **pca_activities)

        for name in network_names:
            X = pca_activities[name].reshape(pca_activities[name].shape[0], -1)
            X_all[name].append(X)

    output_path = f'{base_path}/results_aggregated'
    os.makedirs(output_path, exist_ok=True)
    for name, arr_ls in X_all.items():
        X_concat = np.stack(arr_ls, axis=0)
        X_concat = X_concat.reshape(n_aug1, n_aug2, X_concat.shape[-2], X_concat.shape[-1])
        print(X_concat.shape)
        np.savez_compressed(f'{output_path}/{name}.npz', X_concat)

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

def comparison(path1, path2, args):
    extractor1 = LayerActivityExtractor(
            checkpoint_path=path1,
            image_folder=args.data_path,
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            test_size=args.test_size
    )
    activities1 = extractor1.get_activities()

    extractor2 = LayerActivityExtractor(
            checkpoint_path=path2,
            image_folder=args.data_path,
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            test_size=args.test_size
    )
    activities2 = extractor2.get_activities()
    network_names = activities1.keys()

    X_train1, _ = split_activity(activities1, save_path='seed_42.npz', test_size=0)
    X_train2, _ = split_activity(activities2, save_path='seed_43.npz', test_size=0)

    distmat = compute_across_network_distances(X_train1, X_train2)
    visualize_distance(distmat, network_names, '.', off_diagonal_color=False)

    layer_names, exact_diffs, cos_sims, angles, p_dist = [], [], [], [], []

    for key in activities1:
        X1 = activities1[key]
        X2 = activities2[key]

        x1_flat = X1.reshape(-1)
        x2_flat = X2.reshape(-1)
        print(x1_flat.shape)
        exact_diff = np.linalg.norm(x1_flat - x2_flat)
        denom = (np.linalg.norm(x1_flat) * np.linalg.norm(x2_flat))
        cos_sim = np.dot(x1_flat, x2_flat) / denom if denom != 0 else np.nan
        angle_deg = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))

        # compute procruste's distance
        n_components = min(len(X1), 1000)
        pca = DualPCA(n_components=n_components)
        X1_pca = pca.fit_transform(X1.reshape(X1.shape[0], -1))
        pca = DualPCA(n_components=n_components)
        X2_pca = pca.fit_transform(X2.reshape(X2.shape[0], -1))

        metric = LinearMetric(alpha=1.0, center_columns=True, score_method='angular')
        metric.fit(X1_pca, X2_pca)
        dist = metric.score(X1_pca, X2_pca)

        layer_names.append(key)
        exact_diffs.append(exact_diff)
        cos_sims.append(cos_sim)
        angles.append(angle_deg)
        p_dist.append(dist)

        print(f"[{key}] exact_diff={exact_diff:.6f} | p_dist={dist:.6f} | cos_sim={cos_sim:.6f} | angle={angle_deg:.3f}°")

    fig, ax1 = plt.subplots(figsize=(10,5))

    # Plot exact diff on left y-axis
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Exact diff', color='tab:blue')
    ax1.plot(layer_names, exact_diffs, marker='o', color='tab:blue', label='Exact diff')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    plt.xticks(rotation=45, ha='right')

    # Create second y-axis for cosine similarity or angle
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cosine similarity', color='tab:orange')
    ax2.plot(layer_names, p_dist, marker='s', color='tab:orange', label='Cosine sim')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('Layer-wise differences between checkpoints')
    fig.tight_layout()
    plt.savefig('reproduce_check.png')
        

def main():
    args = parse_args()
    set_seed(42)

    base_path = '/mnt/home/the10/ceph/results/netrep'
    output_path = f'{base_path}/experiments/{args.experiment}'
    result_path = f'{output_path}/results_seed{args.seed}' if args.seed else f'{output_path}/results'
    result_path = f'{result_path}_pretrained' if args.pretrained else result_path
    result_path = f'{result_path}/{args.model}' if args.model != 'final_model' else result_path
    embed_path = f'{result_path}/activities.npz'
    print(result_path)
    print(args.dataset, args.model_name)

    os.makedirs(result_path, exist_ok=True)
    pretrain_str = "/pretrained" if args.pretrained else ""
    pretrain_str = f"{pretrain_str}_seed{args.seed}" if args.seed else pretrain_str
    # experiment_ls = [f"cutout{pretrain_str}/exp_{args.model_name}_cutout_patch_size_0.0_0.0"]
    experiment_ls = [f"cutout{pretrain_str}/exp_{args.model_name}_cutout_patch_size_{c}_{s}" for c in [4.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0] for s in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]]
    # experiment_ls = ["clean/exp_wide_resnet_clean_0.0"]+[f"cutout{pretrain_str}/exp_{args.model_name}_cutout_patch_size_{c}_{s}" for c in [12.0, 16.0, 20.0, 24.0] for s in [0.2, 0.3, 0.5, 0.8, 1.0]]
    # experiment_ls += [f"cutout_seed43/exp_wide_resnet_cutout_patch_size_16.0_0.5"]
    # experiment_ls += [f"cutout_seed44/exp_wide_resnet_cutout_patch_size_16.0_0.5"]

    # distance_vs_sample_size(args, output_path)
    # dist_matrix_path = "/mnt/home/the10/ceph/results/netrep/experiments/clean/distance_matrix_pc_all.npy"
    # visualize_distance_matrices(dist_matrix_path, sample_sizes=[10, 20, 50, 100, 200, 500, 1000], seeds=[42, 43, 44, 45, 46])
    # comparison('/mnt/home/the10/ceph/results/netrep/experiments/clean/checkpoints/final_model.pth', '/mnt/home/the10/ceph/results/netrep/experiments/clean_seed43/checkpoints/final_model.pth', args)
    # exit()

    # extract activity for each layer in the network
    if 'all' not in args.experiment:
        extractor = LayerActivityExtractor(
            checkpoint_path=f'{output_path}/checkpoints/{args.model}.pth',
            image_folder=args.data_path,
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            test_size=args.test_size,
            model_name=args.model_name,
            dataset=args.dataset
        )
        network_names = extractor.get_layer_names()
        print(network_names)

        # load pre-existing PCA representation
        save_path = f'{result_path}/activities_pca.npz'
        if save_path and os.path.isfile(save_path):
            print(f"Loading PCA-reduced activities from {save_path}")

            with np.load(save_path, allow_pickle=False) as data:
                pca_activities = {k: data[k] for k in data.files}
                X_train = [pca_activities[name].reshape(pca_activities[name].shape[0], -1) for name in network_names]
        # extract and take PCA for all layers
        else:
            activities = extractor.get_activities()
            X_train, X_test = split_activity(activities, save_path=save_path, test_size=0)

    # extract activity for all networks
    else:
        X_train, X_test, network_names = extract_activity_all(experiment_ls, base_path, args)
        # extract_raw_activity(experiment_ls, base_path, args, n_aug1=7, n_aug2=7)

    # compute procruste's distance between pair of networks
    if not os.path.isfile(f"{result_path}/distance_matrix.npy"):
        distmat, norms = compute_distances(X_train, X_train)
        np.save(f'{result_path}/distance_matrix.npy', distmat)
        np.save(f'{result_path}/diff_norms.npy', norms)
    elif args.edit:
        distmat = np.load(f"{result_path}/distance_matrix.npy")
        distmat = edit_distances(X_train, distmat, np.arange(10*4, 11*4))
        print(f"Saving editted distance matrix to {result_path}/distance_matrix.npy")
        np.save(f'{result_path}/distance_matrix.npy', distmat)
    else:
        distmat = np.load(f"{result_path}/distance_matrix.npy")
    print(len(X_train),len(X_train[0]), distmat.shape)

    if 'all' in args.experiment:
        l = len(network_names) // len(experiment_ls)
        expanded_experiment_ls = [exp for exp in experiment_ls for _ in range(l)]
        name_ls, _ = exp_to_name(expanded_experiment_ls)
        visualize_distance(distmat, name_ls, result_path, group_by_layers=True)

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