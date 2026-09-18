import torch.nn as nn


class ClassifierModel(nn.Modudle):
    def __init__(self, number_of_classes: int = 2) -> None:
        super(ClassifierModel, self).__init__()
