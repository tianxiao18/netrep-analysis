import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import wandb

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from netrep.datasets import get_dataloaders
from netrep.models import get_model
from netrep.optimize import train, evaluate, LabelSmoothingCrossEntropy, exclude_from_weight_decay, get_lr_scheduler
from netrep.augment import AugmentationPipeline
from argparse import ArgumentParser
from torchvision import transforms
import random
import numpy as np

def parse_args():
    parser = ArgumentParser(description="PyTorch Resnet Trainer")

    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/gpuxl/scc/AI_DATASETS/ImageNet/2012/imagenet"
    )

    parser.add_argument(
        "--aug_type",
        type=str,
        default="weak_random_blur"
    )

    parser.add_argument(
        "--aug_param",
        type=float,
        default=None,
        help="params for augmentation (ex. sigma for fixed_blur, temp for random_blur, prob for sp noise)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=32
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

    parser.add_argument(
        "--model",
        type=str,
        default="resnet"
    )

    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(43)
    warmup_epochs = args.warmup_epochs if args.batch_size >= 512 else 0
    param_dict = {"fixed_blur": "sigma", "weak_random_blur": "temp", 
                  "sp_noise": "sp_prob", "cutout": "cutout_patch_size", "clean": "clean"}
    param_name = param_dict[args.aug_type]

    print("Setting up dataloader...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {"crop": True, "flip": False, param_name: args.aug_param}  # crop has to be true to ensure same shape input
    if "blur" in args.aug_type: config["blur"] = args.aug_type
    train_tf = AugmentationPipeline(config)
    eval_tf = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
    print(config)

    train_loader, val_loader = get_dataloaders(args.data_path, args.batch_size, args.num_workers, train_tf, eval_tf)
    model = get_model(device, model_name=args.model)

    criterion = LabelSmoothingCrossEntropy(args.label_smoothing)
    optimizer = optim.SGD(
        exclude_from_weight_decay(model.named_parameters(), args.weight_decay),
        lr=args.base_lr * (args.batch_size / 256),
        momentum=args.momentum
    )
    lr_scheduler = get_lr_scheduler(optimizer, warmup_epochs, args.epochs)
    best_val_acc = 0
    title = f"{args.model}_{args.aug_type}" if args.aug_param is None else f"{args.model}_{args.aug_type}_{param_name}{args.aug_param}"

    wandb.init(
        project=f"{args.model}-imagenet", 
        name=title,          
        config={                             
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "base_lr": args.base_lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            param_name: args.aug_param
        }
    )
    output_path = f"/mnt/home/the10/ceph/results/netrep/experiments/{args.aug_type}"
    if args.aug_param is not None:
        output_path = os.path.join(output_path, f"exp_{args.model}_{param_name}_{args.aug_param}")
    os.makedirs(os.path.join(output_path, "checkpoints"), exist_ok=True)

    print("Training model...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        lr_scheduler.step()

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]['lr']})
        
        if epoch < 45 or epoch % 3 == 0:
            torch.save(model.state_dict(), f"{output_path}/checkpoints/epoch{epoch}.pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{output_path}/checkpoints/best_model.pth")
            print(f"Model saved to {output_path}/checkpoints/best_model.pth")

    # Final test evaluation
    torch.save(model.state_dict(), f"{output_path}/checkpoints/final_model.pth")
    test_loss, test_acc = evaluate(model, val_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    main()