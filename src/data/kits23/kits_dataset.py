from pathlib import Path

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


class Kits23Dataset(Dataset):
    def __init__(self, data_slices: list[str], slices: int) -> None:
        self._data = data_slices
        self._slices = slices

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        """
        return value:
        python tuple,
        first value (image): torch tensor of size [1, 256, 256]
        second value (label): torch tensor of size [2]
        """
        indexes = [i for i in range(index - self._slices // 2, index + self._slices // 2 + 1)]
        for i in range(len(indexes)):
            if indexes[i] < 0 or indexes[i] > len(self._data) - 1:
                indexes[i] = indexes[len(indexes) - i - 1]
            if indexes[i] < 0 or indexes[i] > len(self._data) - 1:
                indexes[i] = index
        paths = [Path(self._data[idx]) for idx in indexes]

        if len(paths) > 1:
            middle = len(paths) // 2
            case_id = int(paths[middle].stem.split('_')[1])
            for i in range(1, middle + 1):
                case_low = int(paths[middle - i].stem.split('_')[1])
                case_high = int(paths[middle + i].stem.split('_')[1])
                if case_low != case_id and case_high != case_id:
                    paths[middle - i] = paths[middle]
                    paths[middle + i] = paths[middle]
                elif case_low != case_id:
                    paths[middle - i] = paths[middle + i]
                elif case_high != case_id:
                    paths[middle + i] = paths[middle - i]

        samples = [
            torch.load(
                sample_path,
                weights_only=True
            ) for sample_path in paths
        ]
        middle = len(samples) // 2
        images = []
        for sample in samples:
            image = sample['image']
            if image.ndim == 3 and image.shape[0] == 1:
                image = image.squeeze(0)
            images.append(image)
        images = torch.stack(images, dim=0)

        tumor_present = torch.any(samples[middle]['mask'] == 2)
        cyst_present = torch.any(samples[middle]['mask'] == 3)

        labels = torch.tensor(
            [tumor_present, cyst_present],
            dtype=torch.float32
        )

        return images, labels

    def get_slice(self, index: int) -> tuple[torch.tensor, torch.tensor, int, int]:
        sample = torch.load(
            Path(self._data[index]),
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
