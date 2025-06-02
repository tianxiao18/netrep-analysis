import torch
from torchvision import models

def get_model(device):
    model = models.resnet50(weights=None)
    model = torch.nn.DataParallel(model)
    return model.to(device)