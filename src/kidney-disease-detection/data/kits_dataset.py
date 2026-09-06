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

        self.samples = []

        self._build_index()

    def _build_index(self):
        directories = sorted(self._root_dir.glob("case_*"))

        for case in directories:
            image_path = case / "imaging.nii.gz"
            mask_path = case / "segmentation.nii.gz"

            if not image_path.exists() or not mask_path.exists():
                continue

            mask = nib.load(mask_path).get_fdata()

            for slice_id in range(mask.shape[2]):
                slice_mask = mask[:, :, slice_id]

                if np.any(slice_mask > 0):
                    self.samples.append({
                        "image": image_path,
                        "mask": mask_path,
                        "case_id": case.name,  # case number extracted from folder name
                        "slice_id": slice_id
                    })
        # TEMP
        print(f"Found {len(self.samples)} kidney-containing slices.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_nii = nib.load(sample["image"])
        mask_nii = nib.load(sample["mask"])

        image = image_nii.get_fdata()
        mask = mask_nii.get_fdata()

        image_slice = image[:, :, sample["slice_id"]]
        mask_slice = mask[:, :, sample["slice_id"]]

        # crop

        kidney_pixels = np.where(mask_slice > 0)

        y_min, y_max = kidney_pixels[0].min(), kidney_pixels[0].max()
        x_min, x_max = kidney_pixels[1].min(), kidney_pixels[1].max()

        margin = self._crop_margin

        y_min = max(0, y_min - margin)
        y_max = min(image_slice.shape[0], y_max + margin + 1)

        x_min = max(0, x_min - margin)
        x_max = min(image_slice.shape[1], x_max + margin + 1)

        image_slice = image_slice[y_min:y_max, x_min:x_max]
        mask_slice = mask_slice[y_min:y_max, x_min:x_max]

        # clip HU

        image_slice = np.clip(
            image_slice,
            self._hu_min,
            self._hu_max
        )

        # normalize HU

        image_slice = (image_slice - self._hu_min) / (self._hu_max - self._hu_min)

        image_tensor = torch.from_numpy(image_slice).float()
        mask_tensor = torch.from_numpy(mask_slice).long()

        image_tensor = image_tensor.unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "case_id": sample["case_id"],
            "slice_id": sample["slice_id"]
        }

    @property
    def classes(self):
        return self.samples.classes
