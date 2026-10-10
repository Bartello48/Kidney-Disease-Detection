import torch
import torch.nn as nn

from src.models.base_model import BaseModel
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetClassifier(BaseModel):
    def __init__(
        self,
        pretrained: bool = False,
        slices: int = 1,
        num_classes: int = 2
    ) -> None:
        super(
            EfficientNetClassifier,
            self,
        ).__init__(
            'efficientnet_b0',
            pretrained
        )

        if slices not in (1, 3, 5, 7):
            raise ValueError(f"Unsuppoerted number of slices: {slices}")
        self.model_path = None
        self.slices = slices

        if pretrained:
            self._model = efficientnet_b0(
                EfficientNet_B0_Weights
            )
        else:
            self._model = efficientnet_b0()
        # change 3-channel RGB to 1-channel grayscale
        old_conv = self._model.features[0][0]
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
            # mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            # new_conv.weight.copy_(
            #     mean_weight.repeat(1, self.slices, 1, 1)
            #     * (old_conv.in_channels / self.slices)
            # )
            # ---

            # set all dimentions to 0, exept for middle one
            new_conv.weight.zero_()

            center = self.slices // 2

            new_conv.weight[:, center:center + 1].copy_(
                old_conv.weight.sum(dim=1, keepdim=True)
            )
            # ---

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self._model.features[0][0] = new_conv

        # replace classifier with 2-classes classifier
        in_features = self._model.classifier[1].in_features
        self._model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self._model(x)
