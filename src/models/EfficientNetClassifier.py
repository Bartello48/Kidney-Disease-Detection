import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super(EfficientNetClassifier, self).__init__()
        self.train_data_path = None

        self._model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        # change 3-channel RGB to 1-channel grayscale
        old_conv = self._model.features[0][0]
        self._model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # replace classifier with 2-classes classifier
        in_features = self._model.classifier[1].in_features
        self._model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self._model(x)
