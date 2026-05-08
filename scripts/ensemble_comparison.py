import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from argparse import ArgumentParser
import random
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from netrep.models import get_model

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
        "--model_name",
        type=str,
        default="resnet50"
    )

    parser.add_argument(
        "--dataset",
        type=str
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--method",
        type=str,
        default="probs"
    )

    return parser.parse_args()

def load_checkpoint(model_name, ckpt_path, device="cuda"):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = get_model(device, model_name, pretrained=False)
    model.load_state_dict(checkpoint)
    model.eval().to(device)
    return model

def build_dataset_and_labels(image_folder, dataset):
        print(f"Evaluating on {dataset} dataset at {image_folder}")
        if dataset == "imagenet":
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
            ds = datasets.ImageFolder(image_folder, transform=transform)
            labels = [sample[1] for sample in ds.imgs]
            return transform, ds, labels
        elif dataset == "cifar10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std =[0.2023, 0.1994, 0.2010]),
            ])
            ds = datasets.CIFAR10(image_folder, train=False, transform=transform, download=True)
            labels = ds.targets
            return transform, ds, labels
        else:
            raise ValueError(f"Unknown dataset_kind: {dataset}")

def load_ensemble(model_name, ckpt_paths, device="cuda"):
    models = [load_checkpoint(model_name, p, device=device) for p in ckpt_paths]
    if len(models) == 0:
        raise ValueError("ckpt_paths is empty.")
    return models

@torch.no_grad()
def predict_ensemble(models, x, method="probs"):
    """
    method:
      - "probs": average softmax probabilities (recommended for checkpoints)
      - "logits": average logits then softmax
    """
    logits_list = [m(x) for m in models] 
    logits = torch.stack(logits_list, dim=0)

    if method == "probs":
        probs = F.softmax(logits, dim=-1)
        probs_ens = probs.mean(dim=0)
        return probs_ens
    elif method == "logits":
        avg_logits = logits.mean(dim=0)
        probs_ens = F.softmax(avg_logits, dim=-1)
        return probs_ens
    else:
        raise ValueError(f"Unknown method: {method}")
    
@torch.no_grad()
def evaluate_ensemble(model_name, ckpt_paths, image_folder, dataset_kind,
                      batch_size=256, num_workers=4, device="cuda",
                      method="probs"):
    
    _, ds, labels = build_dataset_and_labels(image_folder, dataset_kind)

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.startswith("cuda"))
    )

    models = load_ensemble(model_name, ckpt_paths, device=device)

    correct = 0
    total = 0

    for x, y in tqdm(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        probs_ens = predict_ensemble(models, x, method=method)  # [B, K]
        pred = probs_ens.argmax(dim=-1)                         # [B]

        correct += (pred == y).sum().item()
        total += y.numel()

    acc = correct / max(total, 1)
    return acc
    
def get_experiments_name_all_aug(output_path, model_name, param_dict):
    experiment_ls = [f"{output_path}/clean/exp_{model_name}_{param_dict['clean']}_0.0_0.0/checkpoints/final_model.pth"]
    experiment_ls += [f"{output_path}/rotate/exp_{model_name}_{param_dict['rotate']}_{c}_0.0/checkpoints/final_model.pth" for c in [4.0, 12.0, 18.0, 24.0]]
    experiment_ls += [f"{output_path}/sheer/exp_{model_name}_{param_dict['sheer']}_{c}_0.0/checkpoints/final_model.pth" for c in [4.0, 12.0, 18.0, 24.0]]
    experiment_ls += [f"{output_path}/jitter/exp_{model_name}_{param_dict['jitter']}_{c}_0.0/checkpoints/final_model.pth" for c in [0.1, 0.2, 0.3, 0.4]]
    experiment_ls += [f"{output_path}/grayscale/exp_{model_name}_{param_dict['grayscale']}_{c}_0.0/checkpoints/final_model.pth" for c in [0.1, 0.2, 0.3, 0.4]]
    experiment_ls += [f"{output_path}/gaussian_noise/exp_{model_name}_{param_dict['gaussian_noise']}_{c}_0.0/checkpoints/final_model.pth" for c in [0.01, 0.02, 0.03, 0.04]]
    experiment_ls += [f"{output_path}/sp_noise/exp_{model_name}_{param_dict['sp_noise']}_{c}_0.0/checkpoints/final_model.pth" for c in [0.05, 0.1, 0.15, 0.2]]
    experiment_ls += [f"{output_path}/cutout/exp_{model_name}_{param_dict['cutout']}_{c}_0.0/checkpoints/final_model.pth" for c in [2.0, 4.0, 6.0, 8.0]]
    experiment_ls += [f"{output_path}/crop/exp_{model_name}_{param_dict['crop']}_{c}_0.0/checkpoints/final_model.pth" for c in [0.6, 0.7, 0.8, 0.9]]
    return experiment_ls

def plot_synergy_profiles(S, labels):
    S = np.asarray(S, float)
    n = S.shape[0]
    S = S.copy()
    print(S.shape)

    vmin, vmax = np.nanmin(S), np.nanmax(S)

    fig, ax = plt.subplots(figsize=(6, 10))
    y_means = np.zeros(len(S))
    for i in range(n):
        y = S[i]
        y_means[i] = np.mean(y-i*0.02)
        ax.plot(range(n), y-i*0.02, color="0.8", lw=1)  # guide-to-eye
        ax.scatter(range(n), y-i*0.02, c=y, vmin=vmin, vmax=vmax, cmap="inferno",
                   s=70, edgecolors="k", linewidths=0.3)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(y_means)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Partner augmentation")
    ax.set_ylabel("Delta Ensemble Performance")
    plt.colorbar(ax.collections[-1], ax=ax, label="Magnitude", fraction=0.04, pad=0.02, shrink=0.6)

    plt.tight_layout()
    plt.savefig('ensemble_synergy.png', dpi=200)
    plt.close()

def plot_ensemble_heatmap(ensemble_accuracy, aug_types, n_augs):
    # cmin = np.min(ensemble_accuracy[~np.eye(n_augs, dtype=bool)])
    # cmax = np.max(ensemble_accuracy)
    plt.figure()
    plt.imshow(ensemble_accuracy, origin="upper", aspect="auto", cmap='inferno')
    # plt.clim(cmin-0.002, cmax)
    plt.colorbar(label="Accuracy")
    plt.title(f"Accuracy Heatmap")
    plt.xticks(ticks=np.arange(n_augs), labels=aug_types, rotation=45)
    plt.yticks(ticks=np.arange(n_augs),labels=aug_types)
    plt.tight_layout()
    plt.savefig('ensemble_acc.png', dpi=200)
    plt.close()

def plot_combined_accuracy_heatmap(angle_values, ensemble_values, title):
    m, b = np.polyfit(angle_values, ensemble_values, 1)

    x_line = np.linspace(angle_values.min(), angle_values.max(), 100)
    y_line = m * x_line + b
    r = np.corrcoef(angle_values, ensemble_values)[0, 1]

    plt.figure(figsize=(5, 5))    
    plt.scatter(angle_values, ensemble_values)
    plt.plot(x_line, y_line, color="red", linewidth=2, label=f"Best fit (r = {r:.3f})")
    plt.xlabel("Mean Geodesic Angle (degrees)")
    plt.ylabel("Accuracy")
    title_str = title.split('/')[-1]
    plt.title(title_str)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=200)

def permutation_test_correlation(x, y, n_permutations=10000, seed=42):
    """
    Tests whether the Pearson r between x and y is significant via permutation.

    Returns observed r, p-value (two-tailed), and the null distribution.
    """
    rng = np.random.default_rng(seed)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    r_obs = np.corrcoef(x, y)[0, 1]
    null = np.array([
        np.corrcoef(rng.permutation(x), y)[0, 1]
        for _ in range(n_permutations)
    ])
    p_value = np.mean(np.abs(null) >= np.abs(r_obs))

    return r_obs, p_value, null


def plot_permutation_test(x, y, title, n_permutations=10000, seed=42):
    r_obs, p_value, null = permutation_test_correlation(x, y, n_permutations=n_permutations, seed=seed)

    plt.figure(figsize=(5, 4))
    plt.hist(null, bins=60, color="steelblue", alpha=0.7, label="Null distribution")
    plt.axvline(r_obs,  color="red",    linewidth=2, label=f"Observed r = {r_obs:.3f}")
    plt.axvline(-r_obs, color="red",    linewidth=2, linestyle="--")
    plt.xlabel("Pearson r (permuted)")
    plt.ylabel("Count")
    title_str = title.split('/')[-1]
    plt.title(f"{title_str}\np = {p_value} ({n_permutations:,} permutations)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title}_permutation_test.png", dpi=200)
    plt.close()
    print(f"[{title}] r = {r_obs}, p = {p_value}")
    return r_obs, p_value

def get_experiments_name_two_aug(model_name, param_dict, aug1, aug2, aug1_params, aug2_params):
    experiment_ls = []
    for idx_c, c in enumerate(aug1_params):
        for idx_s, s in enumerate(aug2_params):
            if idx_c != idx_s:
                experiment_ls.append(f"{aug1[idx_c]}/exp_{model_name}_{param_dict[aug1[idx_c]]}_{c}_{aug2[idx_s]}_{param_dict[aug2[idx_s]]}{s}_0.0")
    return experiment_ls

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_args()
    set_seed(42)
    layer = 'avgpool'

    base_path = '/mnt/home/the10/ceph/results/netrep'
    results_path = '/mnt/home/the10/netrep-analysis/results'
    output_path = f'{base_path}/experiments'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args.dataset, args.model_name)

    param_dict = {"fixed_blur": "sigma", "weak_random_blur": "temp", 
                  "sp_noise": "sp_prob", "cutout": "cutout_patch_size", "clean": "clean",
                  "cutout_jitter": "cutout_patch_size", "cutout_crop": "cutout_patch_size", "rot_sheer": "rot_deg",
                  "rotate": "rot_deg", "sheer": "shear_deg","gaussian_noise": "noise_std",
                  "jitter": "cj_strength", "grayscale": "gray_prob",
                  "cutout_only": "cutout_patch_size", "crop": "extra_crop_scale"}

    experiment_ls = get_experiments_name_all_aug(output_path, args.model_name, param_dict)
    aug_types = list(dict.fromkeys(e.split('/')[-4] for e in experiment_ls[1:]))
    n_augs = len(aug_types)

    angle_matrices = np.load(f"{results_path}/{layer}_angle_matrix.npy")

    # compute n x n ensemble accuracy matrix
    if not os.path.isfile(f"{results_path}/{layer}_ensemble_accuracy.npy"):
        ensemble_accuracy = np.zeros((n_augs, n_augs))
        for i in range(n_augs):
            for j in range(n_augs):
                exps = experiment_ls[i*4+1: i*4+5] + experiment_ls[j*4+1: j*4+5]

                acc = evaluate_ensemble(
                    model_name=args.model_name,
                    ckpt_paths=exps,
                    image_folder=args.data_path,        
                    dataset_kind=args.dataset,
                    batch_size=args.batch_size,
                    device=device,
                    method=args.method,               
                )
                
                print(f"Ensemble of {aug_types[i]} and {aug_types[j]} accuracy:", acc)
                all_accs = []
                
                for exp in exps:
                    single_acc = evaluate_ensemble(model_name=args.model_name, ckpt_paths=[exp], method=args.method,
                    image_folder=args.data_path, dataset_kind=args.dataset, batch_size=args.batch_size, device=device)
                    all_accs.append(single_acc)
                
                ensemble_accuracy[i][j] = acc - sum(all_accs)/len(all_accs)

        print(f"Saving ensemble accuracy to {results_path}/{layer}_ensemble_accuracy.npy")
        np.save(f"{results_path}/{layer}_ensemble_accuracy.npy", ensemble_accuracy)
    else:
        print(f"Loading ensemble accuracy from {results_path}/{layer}_ensemble_accuracy.npy")
        ensemble_accuracy = np.load(f"{results_path}/{layer}_ensemble_accuracy.npy")

    combined_aug_types = ["rotate", "sheer", "jitter", "grayscale", "gaussian_noise", "sp_noise", "cutout", "crop"]
    combined_max_params = {
        "rotate":        ("rot_deg",          "24.0"),
        "sheer":         ("shear_deg",         "24.0"),
        "jitter":        ("cj_strength",       "0.4"),
        "grayscale":     ("gray_prob",         "0.1"),
        "gaussian_noise":("noise_std",         "0.04"),
        "sp_noise":      ("sp_prob",           "0.2"),
        "cutout":        ("cutout_patch_size",  "8.0"),
        "crop":          ("extra_crop_scale",  "0.9"),
    }
    n_combined = len(combined_aug_types)

    def get_single_aug_ckpt_path(aug):
        p1name, p1val = combined_max_params[aug]
        exp_dir = f"exp_{args.model_name}_{p1name}_{p1val}_0.0"
        return os.path.join(output_path, aug, exp_dir, "checkpoints", "final_model.pth")

    def get_combined_aug_ckpt_path(aug1, aug2):
        p1name, p1val = combined_max_params[aug1]
        p2name, p2val = combined_max_params[aug2]
        exp_dir = f"exp_{args.model_name}_{p1name}_{p1val}_{aug2}_{p2name}{p2val}_0.0"
        return os.path.join(output_path, aug1, exp_dir, "checkpoints", "final_model.pth")

    # compute n x n combined accuracy matrix
    combined_accuracy_cache_path = f"{results_path}/combined_accuracy_{layer}.npy"
    if os.path.isfile(combined_accuracy_cache_path):
        print(f"Loading combined_accuracy from cache: {combined_accuracy_cache_path}")
        combined_accuracy = np.load(combined_accuracy_cache_path)
    else:
        print(f"Computing combined_accuracy and saving to cache: {combined_accuracy_cache_path}")
        combined_accuracy = np.full((n_combined, n_combined), np.nan)
        for i, aug1 in enumerate(combined_aug_types):
            for j, aug2 in enumerate(combined_aug_types):
                ckpt = get_single_aug_ckpt_path(aug1) if i == j else get_combined_aug_ckpt_path(aug1, aug2)
                if not os.path.isfile(ckpt):
                    print(f"  [warn] missing checkpoint: {ckpt}")
                    continue
                acc = evaluate_ensemble(
                    model_name=args.model_name,
                    ckpt_paths=[ckpt],
                    image_folder=args.data_path,
                    dataset_kind=args.dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                    method=args.method,
                )
                label = f"{aug1}+{aug2}" if i != j else aug1
                print(f"  {label}: {acc:.4f}")
                combined_accuracy[i, j] = acc
        np.save(combined_accuracy_cache_path, combined_accuracy)
      
    plt.figure(figsize=(7, 6))
    combined_accuracy = combined_accuracy - 1/2* (np.diag(combined_accuracy)[:, None] + np.diag(combined_accuracy)[None, :])
    sns.heatmap(combined_accuracy, xticklabels=combined_aug_types, yticklabels=combined_aug_types,
                annot=True, cmap="viridis", fmt=".3f")
    plt.title("Accuracy for Combined Augmentation Types")
    plt.tight_layout()
    plt.savefig(f"{results_path}/{layer}_augmentation_pair_accuracy_heatmap.png")

    # plot ensemble accuracy heatmap
    plot_ensemble_heatmap(ensemble_accuracy, aug_types, n_augs)
    plot_synergy_profiles(ensemble_accuracy, aug_types)

    # plot ensemble accuracy vs angle
    ens_mask = ~np.eye(n_augs, dtype=bool)
    ens_angle_values  = angle_matrices[:, ens_mask].flatten()
    ensemble_values   = ensemble_accuracy[ens_mask].flatten()

    plot_combined_accuracy_heatmap(ens_angle_values, ensemble_values, f"{results_path}/{layer} angle vs ensemble accuracy")
    plot_permutation_test(ens_angle_values, ensemble_values, n_permutations=100000, title=f"{results_path}/{layer} angle vs ensemble accuracy")

    # plot combined accuracy vs angle matrix
    n_all = len(combined_aug_types)
    comb_mask = ~np.eye(n_all, dtype=bool)
    comb_angle_values = angle_matrices[:, comb_mask].flatten()
    combined_values   = combined_accuracy[comb_mask].flatten()

    plot_combined_accuracy_heatmap(comb_angle_values, combined_values, f"{results_path}/{layer} angle vs combined accuracy")
    plot_permutation_test(comb_angle_values, combined_values, n_permutations=100000, title=f"{results_path}/{layer} angle vs combined accuracy")
