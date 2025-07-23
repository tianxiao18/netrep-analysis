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
                num_workers=32, device='cuda', test_size=1000, seed=42):
        self.checkpoint_path = checkpoint_path
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device if torch.cuda.is_available() else torch.device('cpu')

        # Load model
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        self.model = get_model(self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval().to(self.device)

        # Prepare dataset and dataloader
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.dataset = datasets.ImageFolder(image_folder, transform=self.transform)
        labels = [sample[1] for sample in self.dataset.imgs]

        if test_size >= len(np.unique(labels)) and test_size < len(labels):
            _, stratified_indices = train_test_split(np.arange(len(self.dataset)), test_size=test_size, stratify=labels, random_state=seed)
        elif test_size < len(np.unique(labels)):
            _, stratified_indices = train_test_split(np.arange(len(self.dataset)), test_size=test_size, random_state=seed)
        else:
            stratified_indices = np.arange(len(self.dataset))

        stratified_dataset = Subset(self.dataset, stratified_indices)
        self.dataloader = DataLoader(stratified_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

        # Register hooks after each residual block (after relu) and activation before final classifier
        self.features = defaultdict(list)
        self.target_layers = self._get_layers_by_type(self.model)
        self._register_hooks()

    
    def _get_layers_by_type(self, model):
        """
        Returns a dict {layer_name: layer_module} for all layers of the specified types.
        """
        layers = {}
        for name, module in model.named_modules():
            if type(module).__name__ == 'Bottleneck':
                layers[name] = module
        return layers

    def _register_hooks(self):
        for name, layer in self.target_layers.items():
            layer.register_forward_hook(self._hook_factory(name))

    def _hook_factory(self, name):
        def hook(module, input, output):
            self.features[name].append(output.detach().cpu())
        return hook

    def get_activities(self):
        # Clear previously collected activations
        for k in self.features:
            self.features[k] = []

        with torch.no_grad():
            pbar = tqdm(self.dataloader, desc="Extracting", leave=False)
            for inputs, _ in pbar:
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
                torch.cuda.empty_cache()

        # Format activities
        activity_matrices = {}
        for name, feats in self.features.items():
            activations = torch.cat(feats, dim=0)
            activity_matrices[name] = activations.numpy()
        return activity_matrices