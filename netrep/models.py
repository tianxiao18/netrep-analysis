import torch
from torchvision import models
from torchvision.models import ResNet50_Weights

def get_model(device):
    model = models.resnet50(weights=None)
    # model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.DataParallel(model)
    return model.to(device)