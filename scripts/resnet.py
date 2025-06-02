import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from netrep.datasets import get_dataloaders
from netrep.models import get_model
from netrep.optimize import train, evaluate, LabelSmoothingCrossEntropy, exclude_from_weight_decay, get_lr_scheduler
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="PyTorch Resnet Trainer")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/gpuxl/scc/AI_DATASETS/ImageNet/2012/imagenet"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="/mnt/home/the10/netrep-analysis/experiments/exp_resnet"
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

    parser.add_argument(
        "--epochs", 
        type=int, 
        default=90, 
        help="Number of training epochs"
    )

    parser.add_argument(
        "--base_lr", 
        type=float, 
        default=0.256, 
        help="Base learning rate (for batch size 256)"
    )
    
    parser.add_argument(
        "--momentum", 
        type=float, 
        default=0.875, 
        help="SGD momentum"
    )

    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=1 / 32768, 
        help="Weight decay"
    )
    
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor"
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int, 
        default=5, 
        help="Warmup epochs (used if batch >= 512)"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    warmup_epochs = args.warmup_epochs if args.batch_size >= 512 else 0

    print("Setting up dataloader...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(args.data_path, args.batch_size, args.num_workers)
    model = get_model(device)

    criterion = LabelSmoothingCrossEntropy(args.label_smoothing)
    optimizer = optim.SGD(
        exclude_from_weight_decay(model.named_parameters(), args.weight_decay),
        lr=args.base_lr * (args.batch_size / 256),
        momentum=args.momentum
    )
    lr_scheduler = get_lr_scheduler(optimizer, warmup_epochs, args.epochs)
    best_val_acc = 0

    print("Training model...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        lr_scheduler.step()

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{args.output_path}/checkpoints/best_model.pth")
            print(f"Model saved to {args.output_path}/checkpoints/best_model.pth")

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    main()