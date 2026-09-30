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
        train_data_path: Path,
        pretrained: bool
    ) -> None:
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.train_data_path = train_data_path
        self.pretrained = pretrained

        self.model_name: str = 'default'
        self.model_path: Path = None
        self.train_data_path: Path = None
        self.train_time: str = None
        self.pretrained: bool = None
        self.criterion: torch.nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: torch.optim.lr_scheduler = None

    def forward(self, x: torch.tensor) -> None:
        raise NotImplementedError(
            'this method must be implemented in child class'
        )
