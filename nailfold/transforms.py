"""Input image transforms used during training and evaluation."""

from torch import nn
from torchvision import transforms

import kornia


augmentation_fn = nn.Sequential(
    kornia.augmentation.RandomHorizontalFlip(),
    kornia.augmentation.RandomRotation(degrees=180, p=0.8),
)


normalization_fn = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
