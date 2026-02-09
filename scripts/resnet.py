import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import wandb

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from netrep.datasets import get_dataloaders, get_cifar_dataloaders
from netrep.models import get_model
from netrep.optimize import train, evaluate, LabelSmoothingCrossEntropy, exclude_from_weight_decay, get_lr_scheduler, get_onecycle_lr_scheduler
from netrep.augment import AugmentationPipeline
from netrep.visualize import show_cifar_images, show_imagenet_images
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
        "--dataset",
        type=str,
        default="imagenet"
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
        default=0.0,
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

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--op_param",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None
    )

    parser.add_argument(
        "--pretrained",
        action="store_true"
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
    set_seed(args.seed)
    # warmup_epochs = args.warmup_epochs if args.batch_size >= 512 else 0
    warmup_epochs = args.warmup_epochs
    param_dict = {"fixed_blur": "sigma", "weak_random_blur": "temp", 
                  "sp_noise": "sp_prob", "cutout": "cutout_patch_size", "clean": "clean",
                  "cutout_jitter": "cutout_patch_size", "cutout_crop": "cutout_patch_size", "rot_sheer": "rot_deg",
                  "rotate": "rot_deg", "sheer": "shear_deg", "gaussian_noise": "noise_std",
                  "jitter": "cj_strength", "grayscale": "gray_prob",
                  "cutout_only": "cutout_patch_size", "crop": "extra_crop_scale"}
    param_name = param_dict[args.aug_type]
    print(args.aug_type, args.aug_param, args.seed, args.checkpoint)

    print("Setting up dataloader...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # here crop has to be true to ensure same shape input
    # if pretrained, we have to scale image to 224 to avoid changing pretrained weights
    config = {"crop": True, "flip": True, param_name: args.aug_param, "dataset": args.dataset, "to_224": args.pretrained}
    if "blur" in args.aug_type: config["blur"] = args.aug_type
    if args.aug_type == "cutout": config["sigma"] = args.op_param
    if args.aug_type == "cutout_jitter": config["cj_strength"] = args.op_param
    if args.aug_type == "cutout_crop": config["extra_crop_scale"] = args.op_param
    if args.aug_type == "rot_sheer": config["shear_deg"] = args.op_param

    if args.dataset == 'imagenet' or args.pretrained:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        image_crop = [transforms.Resize(256),transforms.CenterCrop(224)]
    elif args.dataset == 'cifar10':
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        image_crop = []

    train_tf = AugmentationPipeline(config)
    normalize = transforms.Normalize(mean=mean, std=std)
    eval_tf = transforms.Compose(image_crop + [transforms.ToTensor(), normalize])
    print(config)
    print(train_tf.transforms)

    # show_cifar_images([12, 16, 20, 24, 28], [0.2, 0.3, 0.5, 0.8, 1.0], config, args, device)
    # show_imagenet_images([20, 30, 50, 100, 150], [0.2, 0.3, 0.5, 0.8, 1.0], config, args, device)
    if args.dataset == 'imagenet':
        train_loader, val_loader = get_dataloaders(args.data_path, args.batch_size, args.num_workers, train_tf, eval_tf)
    elif args.dataset == 'cifar10':
        train_loader, val_loader = get_cifar_dataloaders(args.data_path, args.batch_size, args.num_workers, train_tf, eval_tf)

    model = get_model(device, model_name=args.model, checkpoint=args.checkpoint, pretrained=args.pretrained)

    criterion = LabelSmoothingCrossEntropy(args.label_smoothing)
    if args.model == 'vit':
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=args.base_lr * (args.batch_size / 256),
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.base_lr * (args.batch_size / 256),
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    # lr_scheduler = get_lr_scheduler(optimizer, warmup_epochs, args.epochs)
    lr_scheduler = get_onecycle_lr_scheduler(optimizer, len(train_loader), warmup_epochs, args)

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
    if args.checkpoint is not None or args.pretrained:
        output_path = os.path.join(output_path, "pretrained")
    if args.seed != 42:
        output_path = output_path+f"_seed{args.seed}"
    if args.aug_param is not None:
        output_path = os.path.join(output_path, f"exp_{args.model}_{param_name}_{args.aug_param}")
    if args.op_param is not None:
        output_path = output_path+f"_{args.op_param}"
    os.makedirs(os.path.join(output_path, "checkpoints"), exist_ok=True)

    print("Training model...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, lr_scheduler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        # lr_scheduler.step()

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]['lr']})
        
        # if epoch % 10 == 0:
        #     torch.save(model.state_dict(), f"{output_path}/checkpoints/epoch{epoch}.pth")

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