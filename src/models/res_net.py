import torch
import torch.nn as nn

from src.models.base_model import BaseModel
from torchvision.models import resnet50, ResNet50_Weights


class ResNetClassifier(BaseModel):
    def __init__(
        self,
        pretrained: bool = False,
        slices: int = 1,
        num_classes: int = 2
    ) -> None:
        super(
            ResNetClassifier,
            self,
        ).__init__(
            'resnet_50',
            False
        )

        if slices not in (1, 3, 5, 7):
            raise ValueError(f"Unsuppoerted number of slices: {slices}")
        self.slices = slices

        if pretrained:
            self._model = resnet50(ResNet50_Weights)
        else:
            self._model = resnet50()

        old_conv = self._model.conv1
        new_conv = nn.Conv2d(
            in_channels=self.slices,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
            padding_mode=old_conv.padding_mode
        )
        with torch.no_grad():
            # distributes averaged weight evenly across all slices
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(
                mean_weight.repeat(1, self.slices, 1, 1)
                * (old_conv.in_channels / self.slices)
            )
            # ---

            # set all dimentions to 0, exept for middle one
            # new_conv.weight.zero_()

            # center = self.slices // 2

            # new_conv.weight[:, center:center + 1].copy_(
            #     old_conv.weight.sum(dim=1, keepdim=True)
            # )
            # ---

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self._model.conv1 = new_conv

        in_features = self._model.fc.in_features
        self._model.fc = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self._model(x)
