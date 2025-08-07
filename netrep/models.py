import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
import timm

# def get_model(device):
#     model = models.resnet50(weights=None)
#     # model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#     model = torch.nn.DataParallel(model)
#     return model.to(device)

def get_model(device, model_name="resnet50"):
    if model_name == "vgg":
        model = models.vgg16_bn(weights=None, num_classes=1000)
    elif model_name == "vit":
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1000)
    else:  # ResNet
        model = models.resnet50(weights=None, num_classes=1000)
    model = torch.nn.DataParallel(model)
    return model.to(device)