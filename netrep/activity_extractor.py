import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from collections import OrderedDict
import torch.nn as nn
from tqdm import tqdm
from netrep.models import get_model
from sklearn.model_selection import train_test_split
import numpy as np

class LayerActivityExtractor:
    def __init__(self, checkpoint_path, image_folder, batch_size=256, 
                num_workers=32, device='cuda', test_size=1000, seed=42, model_name="resnet50", dataset="imagenet"):
        self.checkpoint_path = checkpoint_path
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.device = device if torch.cuda.is_available() else torch.device('cpu')

        # Load model
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        self.model = get_model(self.device, model_name)
        self.model.load_state_dict(checkpoint)
        self.model.eval().to(self.device)
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.model = self.model.module
        
        # Prepare dataset and dataloader
        self.transform, self.dataset, labels = self._build_dataset_and_labels(
            image_folder, self.dataset
        )
        print("Test set length: ", len(self.dataset))
        if test_size >= len(np.unique(labels)) and test_size < len(labels):
            _, stratified_indices = train_test_split(np.arange(len(self.dataset)), test_size=test_size, stratify=labels, random_state=seed)
        elif test_size < len(np.unique(labels)):
            _, stratified_indices = train_test_split(np.arange(len(self.dataset)), test_size=test_size, random_state=seed)
        else:
            stratified_indices = np.arange(len(self.dataset))
        print(len(np.unique(labels)), test_size)
        stratified_dataset = Subset(self.dataset, stratified_indices)
        self.dataloader = DataLoader(stratified_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

        # Register hooks after each residual block (after relu) and activation before final classifier
        self.features = defaultdict(list)
        self._batch_feats = {}
        self.target_layers = self._get_layers_by_type(self.model)
        self._register_hooks()

    def _build_dataset_and_labels(self, image_folder, dataset):
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
                                    std =[0.2470, 0.2435, 0.2616]),
            ])
            ds = datasets.CIFAR10(image_folder, train=False, transform=transform, download=True)
            labels = ds.targets
            return transform, ds, labels
        else:
            raise ValueError(f"Unknown dataset_kind: {self.dataset}")

    def _get_layers_by_type(self, model):
        """
        Returns a dict {layer_name: layer_module} for all layers of the specified types.
        """
        layers = {}
        for i in range(3, 5):
            stage = getattr(model, f"layer{i}", None)
            if stage is not None and len(stage) > 0:
                layers[f"layer{i}.{len(stage)-1}"] = stage[-1] 

        if hasattr(model, "avgpool"):
            layers["avgpool"] = model.avgpool
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            layers["fc"] = model.fc
        return layers

    def _register_hooks(self):
        for name, layer in self.target_layers.items():
            layer.register_forward_hook(self._hook_factory(name))

    def _hook_factory(self, name):
        def hook(module, input, output):
            # self.features[name].append(output.detach().cpu())
            self._batch_feats[name] = output.detach().to(dtype=torch.float16, device="cpu")
        return hook
    
    def get_layer_names(self):
        return list(self.target_layers.keys())

    def get_activities(self, output_dir="/tmp/memmap"):
        total = len(self.dataloader.dataset)
        writer = MemmapWriter(output_dir)
        print(f"Saving memmap to {output_dir}")
        first_shapes = {}

        with torch.no_grad():
            pbar = tqdm(self.dataloader, desc="Extracting", leave=False)
            for inputs, _ in pbar:
                self._batch_feats.clear()
                _ = self.model(inputs.to(self.device))

                for name, feats in self._batch_feats.items():
                    if name not in first_shapes:
                        # writer.ensure(name, total_len=feats.shape[0], feat_shape=feats.shape[1:], dtype=np.float16)
                        writer.ensure(name, total_len=total, feat_shape=feats.shape[1:], dtype=np.float16)
                        first_shapes[name] = feats.shape[1:]
                    writer.write_batch(name, feats.numpy())
                
        writer.close()
        return {name: os.path.join(output_dir, f"{name}.npy") for name in first_shapes}

    # def get_activities(self):
    #     # Clear previously collected activations
    #     for k in self.features:
    #         self.features[k] = []

    #     with torch.no_grad():
    #         pbar = tqdm(self.dataloader, desc="Extracting", leave=False)
    #         for inputs, _ in pbar:
    #             inputs = inputs.to(self.device)
    #             _ = self.model(inputs)
    #             torch.cuda.empty_cache()

    #     # Format activities
    #     activity_matrices = {}
    #     for name, feats in self.features.items():
    #         activations = torch.cat(feats, dim=0)
    #         activity_matrices[name] = activations.numpy()
    #     return activity_matrices

class MemmapWriter:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.arrs, self.ptr = {}, defaultdict(int)

    def ensure(self, key, total_len, feat_shape, dtype=np.float16):
        if key in self.arrs: 
            return
        path = os.path.join(self.root, f"{key}.npy")
        self.arrs[key] = np.lib.format.open_memmap(
            path, mode="w+", dtype=dtype, shape=(total_len, *feat_shape)
        )

    def write_batch(self, key, batch_np):
        m = self.arrs[key]
        i = self.ptr[key]
        m[i:i+len(batch_np)] = batch_np
        self.ptr[key] += len(batch_np)

    def close(self):
        for m in self.arrs.values():
            m.flush()