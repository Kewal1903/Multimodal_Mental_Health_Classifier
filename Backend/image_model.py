import torch
import torch.nn as nn
from torchvision import models  # Make sure torchvision is imported
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# --- 1. Model Architecture ---
# This is the get_resnet50_1ch function from your teammate's notebook
def get_resnet50_1ch(num_classes=7):
    # load pretrained weights
    try:
        res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # modify first conv: kernel 3 stride 1 better for 48x48; change in_channels=1
        res.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        res.maxpool = nn.Identity()  # prevent too much downsampling
        res.fc = nn.Linear(res.fc.in_features, num_classes)
        # Note: conv1 weights reinitialized randomly; other layers keep pretrained initialization
    except Exception:
        # if pretrained not available, fallback to random init resnet
        res = models.resnet50(weights=None)
        res.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        res.maxpool = nn.Identity()
        res.fc = nn.Linear(res.fc.in_features, num_classes)
    return res

# --- 2. Preprocessing Function ---
# This is the same transformation pipeline both models were trained with
def get_image_transforms():
    return A.Compose([
        A.Resize(48, 48),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])