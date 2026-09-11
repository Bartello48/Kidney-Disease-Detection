from pathlib import Path

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


class Kits23Dataset(Dataset):
    def __init__(self, data_path: str):
        self._data_path = Path(data_path)
        self._files = self._data_path / "slices"
        if not self._files.exists():
            raise FileNotFoundError("Invalid preprocessed folder data organization")
        self._data = sorted(
            self._files.glob("*.pt")
        )
        if len(self._data) == 0:
            raise FileNotFoundError("did not find any images under given path")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor, int, int]:
        sample = torch.load(
            self._data[index],
            weights_only=True
        )
        return sample['image'], sample['mask'], sample['case_id'], sample['slice_id']

    def save_image(self, index: int, path: str = 'diagnostics/images') -> None:
        path = Path(path)
        path.mkdir(
            parents=True,
            exist_ok=True
        )
        image, _, case_id, slice_id = self[index]
        print(f"image case: {case_id}")
        path = path / f"case_{case_id}_slice_{slice_id}_image.png"
        plt.imsave(
            path,
            image.squeeze(0),
            cmap='gray'
        )
        print(f"saved image as png file at: {path}")

    def save_map(self, index: int, path: str = 'diagnostics/images') -> None:
        path = Path(path)
        path.mkdir(
            parents=True,
            exist_ok=True
        )
        _, mask, case_id, slice_id = self[index]
        path = path / f"case_{case_id}_slice_{slice_id}_mask.png"
        plt.imsave(
            path,
            mask,
            cmap='gray'
        )
        print(f"saved mask as png file at: {path}")
