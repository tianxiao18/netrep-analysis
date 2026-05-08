#!/usr/bin/env python3
"""Build a figure: CIFAR-10 originals, multiple global Gaussian noise levels, and optional patch Gaussian rows (fixed σ, increasing patch size)."""

import argparse
import os
import random
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

# Allow `python scripts/plot_cifar10_noise_grid.py` from repo root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def add_gaussian_noise(
    img: torch.Tensor, std: float, generator: torch.Generator
) -> torch.Tensor:
    noise = torch.randn(img.shape, generator=generator, dtype=img.dtype, device=img.device)
    return torch.clamp(img + noise * std, 0.0, 1.0)


def add_patch_gaussian(
    img: torch.Tensor,
    patch_size: int,
    sigma: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Patch Gaussian: same rule as ``AugmentationPipeline.random_cutout`` with ``sigma`` set."""
    out = img.clone()
    _, h, w = out.shape
    ps = max(1, min(int(patch_size), h, w))
    margin_y = ps // 2
    margin_x = ps // 2
    high_y = h - margin_y
    low_y = margin_y
    if low_y >= high_y:
        cy = h // 2
    else:
        cy = torch.randint(low_y, high_y, (1,), generator=generator).item()
    high_x = w - margin_x
    low_x = margin_x
    if low_x >= high_x:
        cx = w // 2
    else:
        cx = torch.randint(low_x, high_x, (1,), generator=generator).item()
    y1 = max(cy - ps // 2, 0)
    y2 = min(cy + ps // 2, h)
    x1 = max(cx - ps // 2, 0)
    x2 = min(cx + ps // 2, w)
    patch = out[:, y1:y2, x1:x2]
    noise = torch.randn(
        patch.shape,
        generator=generator,
        dtype=out.dtype,
        device=out.device,
    )
    out[:, y1:y2, x1:x2] = torch.clamp(patch + sigma * noise, 0.0, 1.0)
    return out


def plot_grid(
    data_path: str,
    *,
    train: bool,
    download: bool,
    indices: Tuple[int, int, int],
    gaussian_stds: Tuple[float, ...],
    include_patch_gaussian: bool,
    patch_sigma: float,
    patch_sizes: Tuple[int, ...],
    seed: int,
    save_path: str,
    show: bool,
) -> None:
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    dataset = datasets.CIFAR10(
        root=data_path,
        train=train,
        download=download,
        transform=transforms.ToTensor(),
    )

    originals = [dataset[i][0] for i in indices]
    rows = [originals]
    for std in gaussian_stds:
        rows.append([add_gaussian_noise(x, std, g) for x in originals])
    if include_patch_gaussian:
        for ps in patch_sizes:
            rows.append(
                [add_patch_gaussian(x, ps, patch_sigma, g) for x in originals]
            )
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 3, figsize=(9, 3 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    for r in range(nrows):
        for c in range(3):
            img = rows[r][c].permute(1, 2, 0).cpu().numpy()
            axes[r, c].imshow(img)
            axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CIFAR-10 grid: clean, multiple global Gaussian std rows, optional patch-Gaussian rows at fixed --patch-sigma with varying --patch-sizes (see netrep/augment.py)."
    )
    p.add_argument(
        "--data_path",
        type=str,
        default="/mnt/home/the10/ceph/dataset/cifar10",
        help="Root directory passed to torchvision CIFAR10 (contains cifar-10-batches-py/ or will download).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default="cifar10_gaussian_noise_grid.png",
        help="Where to save the figure.",
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="Use training split (default: test split).",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download; require dataset already under data_path.",
    )
    p.add_argument(
        "--indices",
        type=str,
        default="24,25,26",
        help="Three comma-separated dataset indices for the columns (default: 0,1,2).",
    )
    p.add_argument(
        "--gaussian-stds",
        type=str,
        default="0.02,0.05,0.075,0.1",
        help="Comma-separated global Gaussian noise stds; one subplot row per value after originals.",
    )
    p.add_argument(
        "--no-patch",
        action="store_true",
        help="Omit patch Gaussian rows (otherwise add one row per --patch-sizes value).",
    )
    p.add_argument(
        "--patch-sigma",
        type=float,
        default=0.35,
        help="Gaussian std inside each random patch (same for every patch-Gaussian row). Ignored with --no-patch.",
    )
    p.add_argument(
        "--patch-sizes",
        type=str,
        default="8,12,16,20",
        help="Comma-separated square patch side lengths in pixels (one patch-Gaussian row per value). Ignored with --no-patch.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", action="store_true", help="Open an interactive window after saving.")
    return p.parse_args()


def _parse_float_list(s: str, name: str) -> Tuple[float, ...]:
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if not parts:
        raise SystemExit(f"{name} must list at least one float, comma-separated.")
    try:
        return tuple(float(x) for x in parts)
    except ValueError as e:
        raise SystemExit(f"{name}: invalid float in {s!r}") from e


def _parse_int_list(s: str, name: str) -> Tuple[int, ...]:
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if not parts:
        raise SystemExit(f"{name} must list at least one integer, comma-separated.")
    try:
        return tuple(int(x) for x in parts)
    except ValueError as e:
        raise SystemExit(f"{name}: invalid integer in {s!r}") from e


def main() -> None:
    args = parse_args()
    parts = [x.strip() for x in args.indices.split(",")]
    if len(parts) != 3:
        raise SystemExit("--indices must list exactly three integers, e.g. 0,1,2")
    idx = tuple(int(x) for x in parts)
    gaussian_stds = _parse_float_list(args.gaussian_stds, "--gaussian-stds")
    patch_sizes = _parse_int_list(args.patch_sizes, "--patch-sizes")

    plot_grid(
        args.data_path,
        train=args.train,
        download=not args.no_download,
        indices=idx,
        gaussian_stds=gaussian_stds,
        include_patch_gaussian=not args.no_patch,
        patch_sigma=args.patch_sigma,
        patch_sizes=patch_sizes,
        seed=args.seed,
        save_path=args.output,
        show=args.show,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
