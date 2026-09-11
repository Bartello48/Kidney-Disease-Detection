import os
import sys

import torch
import torchvision

from dotenv import load_dotenv

from data.kits_dataset import Kits23Dataset

print("--- Environment ---")
print(f"system version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")


load_dotenv(override=True)

dataset = Kits23Dataset(os.getenv("PREPROCESSED_PATH"))
print(len(dataset))
dataset.save_image(1000)
dataset.save_map(1000)
