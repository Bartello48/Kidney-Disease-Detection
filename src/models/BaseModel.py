from pathlib import Path

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    model path: should be set while saving model, an absolute path to model save file
    train_data_path: initialized, should point to split file with paths to data sets
    train time: last training session end time
    pretrained
    """
    def __init__(
        self,
        model_name: str,
        pretrained: bool
    ) -> None:
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained

        self.model_path: Path = None
        self.train_time: str = None

    def forward(self, x: torch.tensor) -> None:
        raise NotImplementedError(
            'this method must be implemented in child class'
        )
