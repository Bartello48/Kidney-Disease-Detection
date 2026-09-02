import sys

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder

import pandas as pd
import numpy as np

print(f"system version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")


class Kits23Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = ImageFolder(data_path, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def classes(self):
        return self.data.classes
