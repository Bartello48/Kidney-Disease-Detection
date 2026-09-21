import torch.nn as nn


class BaseModel(nn.Modudle):
    model_name = 'default'
    model_path = None
    train_data_path = None
    train_time = None

    def __init__(self) -> None:
        super(BaseModel, self).__init__()

    def forward(self, x) -> None:
        raise NotImplementedError(
            'this method must be implemented in child class'
        )
