import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset, random_split
import numpy as np

def get_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # normalize
    ])
    return train_transform, eval_transform

def get_dataloaders(data_dir, batch_size, num_workers, train_tf, eval_tf):
    # train_dataset, val_dataset = get_split_datasets(
    #     train_dir="/mnt/gpuxl/scc/AI_DATASETS/ImageNet/2012/imagenet/train",
    #     val_ratio=val_ratio,
    #     seed=seed,
    #     train_transform=train_tf,
    #     val_transform=eval_tf
    # )
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

def get_split_datasets(train_dir, val_ratio=0.1, seed=42, train_transform=None, val_transform=None):
    full_dataset = datasets.ImageFolder(train_dir)

    # Get indices for each class and split
    targets = np.array(full_dataset.targets)
    train_indices = []
    val_indices = []

    np.random.seed(seed)
    for class_idx in np.unique(targets):
        class_indices = np.where(targets == class_idx)[0]
        np.random.shuffle(class_indices)

        n_val = int(len(class_indices) * val_ratio)
        val_indices.extend(class_indices[:n_val])
        train_indices.extend(class_indices[n_val:])

    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Apply transforms if needed
    if train_transform:
        train_dataset.dataset.transform = train_transform
    if val_transform:
        val_dataset.dataset.transform = val_transform

    return train_dataset, val_dataset

