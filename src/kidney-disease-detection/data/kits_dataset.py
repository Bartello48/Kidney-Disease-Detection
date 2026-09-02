from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import pandas as pd
import numpy as np

import nibabel as nib


class Kits23Dataset(Dataset):
    def __init__(
        self,
        data_path,
        transform=None,
        crop_margin=20,
        hu_min=-200,
        hu_max=300
    ):
        # self.data = ImageFolder(data_path, transform=transform)
        self._root_dir = Path(data_path)
        self._transform = transform
        self._crop_margin = crop_margin
        self._hu_min = hu_min
        self._hu_max = hu_max

        self._build_index()

    def _build_index(self):
        directories = sorted(self._root_dir.glob("case_*"))

        for case in directories:
            image_path = case / "imaging.nii.gz"
            mask_path = case / "segmentation.nii.gz"

            if not image_path.exists() or not mask_path.exists():
                continue

            mask = nib.load(mask_path).get_fdata()



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def classes(self):
        return self.data.classes
