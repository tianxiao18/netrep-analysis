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
        default=4096
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

    extractor = LayerActivityExtractor(
        checkpoint_path=f'{output_path}/checkpoints/best_model.pth',
        image_folder='/mnt/gpuxl/scc/AI_DATASETS/ImageNet/2012/imagenet/val',
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        test_size=args.test_size
    )

    activities = extractor.get_activities()
    np.savez_compressed(f'{output_path}/results/activities.npz', **activities)
    print(activities)

    proc_metric = LinearMetric(alpha=1.0, center_columns=True)


if __name__ == "__main__":
    main()