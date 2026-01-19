import torch
from torchvision import models
from torchvision.models import ResNet50_Weights, wide_resnet50_2, resnet18
import timm
import torch.nn as nn

# def get_model(device):
#     model = models.resnet50(weights=None)
#     # model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#     model = torch.nn.DataParallel(model)
#     return model.to(device)

def get_model(device, model_name="resnet50", checkpoint=None, pretrained=False):
    if model_name == "vgg":
        model = models.vgg16_bn(weights=None, num_classes=1000)
    elif model_name == "vit":
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=1000)
    elif model_name == 'wide_resnet':
        model = wide_resnet50_2(num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # CIFAR stem
        model.maxpool = nn.Identity()
    elif model_name == "resnet18":
        if pretrained: 
            print("Using pretrained weights from Imagenet...")
            model = resnet18(weights="DEFAULT")
        else:
            model = resnet18(weights=None)
        model.conv1.stride = (1, 1)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:  # ResNet50
        model = models.resnet50(weights=None, num_classes=1000)
    model = torch.nn.DataParallel(model)

    if checkpoint:
        print(f"Loading from pretrained checkpoint at {checkpoint}")
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
    return model.to(device)