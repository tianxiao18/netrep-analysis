import torch
from torchvision import models
from .densenet import densenet40
from torchvision.models import ResNet50_Weights, wide_resnet50_2, resnet18
import timm
import torch.nn as nn

# def get_model(device):
#     model = models.resnet50(weights=None)
#     # model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#     model = torch.nn.DataParallel(model)
#     return model.to(device)

def get_model(device, model_name="resnet50", checkpoint=None, pretrained=False, num_classes=10):
    if model_name == "vgg":
        model = models.vgg11_bn(weights=None)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(nn.Linear(512, num_classes))
    elif model_name == "vit":
        if pretrained:
            model = timm.create_model(
                "vit_tiny_patch16_224",
                pretrained=True,
                num_classes=num_classes
            )
        else:
            model = timm.create_model(
                "vit_tiny_patch16_224",
                pretrained=False,
                num_classes=num_classes,
                img_size=32,
                patch_size=4,
            )
    elif model_name == 'wide_resnet':
        model = wide_resnet50_2(num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # CIFAR stem
        model.maxpool = nn.Identity()
    elif model_name == 'densenet':
        model = densenet40()
        # model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # model.features.pool0 = nn.Identity()
    elif model_name == "resnet18":
        if pretrained: 
            print("Using pretrained weights from Imagenet...")
            model = resnet18(weights="DEFAULT")
        else:
            model = resnet18(weights=None)
        model.conv1.stride = (1, 1)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # ResNet50
        if pretrained:
            model = models.resnet50(weights="DEFAULT")
        else:
            model = models.resnet50(weights=None)
            if num_classes != 1000:
                # Use CIFAR-friendly stem only for small images
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = torch.nn.DataParallel(model)

    if checkpoint:
        print(f"Loading from pretrained checkpoint at {checkpoint}")
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
    return model.to(device)