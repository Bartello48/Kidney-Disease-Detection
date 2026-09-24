from pathlib import Path

import torch
import torch.nn as nn
from models.BaseModel import BaseModel
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetClassifier(BaseModel):
    def __init__(
        self,
        train_data_path: Path,
        num_classes: int = 2
    ) -> None:
        super(
            EfficientNetClassifier,
            self,
            'efficientnet_b0',
            train_data_path,
            True
        ).__init__()

        self.model_name = 'efficientnet_b0'
        self.model_path = None
        self.train_data_path = None
        self.pretrained = True
        self._model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )
        # change 3-channel RGB to 1-channel grayscale
        old_conv = self._model.features[0][0]
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
        self._model.features[0][0] = new_conv

        # replace classifier with 2-classes classifier
        in_features = self._model.classifier[1].in_features
        self._model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self._model(x)
