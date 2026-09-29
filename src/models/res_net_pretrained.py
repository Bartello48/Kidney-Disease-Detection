from pathlib import Path

import torch
import torch.nn as nn

from src.models.BaseModel import BaseModel
from torchvision.models import resnet50, ResNet50_Weights


class ResNetClassifierPretrained(BaseModel):
    def __init__(self, train_data_path: Path, num_classes: int = 2) -> None:
        super(
            ResNetClassifierPretrained,
            self,
        ).__init__(
            'resnet_50',
            train_data_path,
            True
        )
        self.model_path = None
        self._model = resnet50(
            weights=ResNet50_Weights.DEFAULT
        )

        old_conv = self._model.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        with torch.no_grad():
            new_conv.weight.copy_(
                old_conv.weight.mean(dim=1, keepdim=True)
            )
        self._model.conv1 = new_conv

        in_features = self._model.fc.in_features
        self._model.fc = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self._model(x)
