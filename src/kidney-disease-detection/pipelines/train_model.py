import os
import sys

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms

from dotenv import load_dotenv

from data.kits_dataset import Kits23Dataset

print("Environment:")
print(f"system version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")


load_dotenv(override=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = Kits23Dataset(os.getenv("DATABASE_PATH"), transform)

data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

print(len(dataset))
print(dataset.classes)
