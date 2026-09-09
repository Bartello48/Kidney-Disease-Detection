from pathlib import Path
from argparse import ArgumentParser
import os
import gc

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as funct

from dotenv import load_dotenv
from tqdm import tqdm


def get_crop(
    margin: int,
    image: np.ndarray,
    mask: np.ndarray,
    hu_min: int
) -> tuple[np.ndarray, np.ndarray]:

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        raise ValueError("No voxels with relevant mapping found")

    max_y = max(ys)
    min_y = min(ys)
    max_x = max(xs)
    min_x = min(xs)

    size_y = max_y - min_y + 1
    size_x = max_x - min_x + 1

    size = max(size_x, size_y)
    size += 2 * margin

    center = (
        (max_y + min_y) / 2,
        (max_x + min_x) / 2
    )

    y1 = int(round(center[0] - size / 2))
    y2 = y1 + size
    x1 = int(round(center[1] - size / 2))
    x2 = x1 + size

    h, w = image.shape
    pad_top = max(0, -y1)
    pad_bot = max(0, y2 - h)
    pad_left = max(0, -x1)
    pad_right = max(0, w - x2)

    if pad_top or pad_bot or pad_left or pad_right:
        image = np.pad(
            image,
            (
                (pad_top, pad_bot),
                (pad_left, pad_right),
            ),
            mode="constant",
            constant_values=hu_min,
        )

        mask = np.pad(
            mask,
            (
                (pad_top, pad_bot),
                (pad_left, pad_right),
            ),
            mode="constant",
            constant_values=0,
        )

        y1 += pad_top
        y2 += pad_top
        x1 += pad_left
        x2 += pad_left

    image = image[y1:y2, x1:x2]
    mask = mask[y1:y2, x1:x2]

    return image, mask


def normalize_hu(
    image: np.ndarray,
    hu_min: int,
    hu_max: int
) -> np.ndarray:
    image = np.clip(image, hu_min, hu_max)
    image = (image - hu_min) / (hu_max - hu_min)
    return image.astype(np.float32)


def resize(
    image: np.ndarray,
    mask: np.ndarray,
    size: int
) -> tuple[torch.Tensor, torch.Tensor]:

    image = torch.from_numpy(image)
    image = image.unsqueeze(0).unsqueeze(0)

    image = funct.interpolate(
        image,
        size=(size, size),
        mode='bilinear',
        align_corners=False
    )
    image = image.squeeze(0)

    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0).unsqueeze(0).float()
    mask = funct.interpolate(
        mask,
        size=(size, size),
        mode='nearest'
    )
    mask = mask.squeeze(0).squeeze(0).to(torch.uint8)

    return image, mask


def process_case(
    case_path: Path,
    dest_path: Path,
    hu_min: int,
    hu_max: int,
    box_size: int,
    margin: int
) -> None:
    image_path = case_path / "imaging.nii.gz"
    mask_path = case_path / "segmentation.nii.gz"

    if not image_path.exists() or not mask_path.exists():
        return 0

    image = nib.load(image_path).get_fdata(dtype=np.float32)
    mask = nib.load(mask_path).get_fdata().astype(np.uint8)

    if image.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch in {case_path.name}: "
            f"{image.shape} vs {mask.shape}"
        )

    num_saved = 0

    for slice_id in range(image.shape[2]):
        image_slice = image[:, :, slice_id]
        mask_slice = mask[:, :, slice_id]

        if not np.any(mask_slice > 0):
            continue
        try:
            image_slice, mask_slice = get_crop(
                margin,
                image_slice,
                mask_slice,
                hu_min
            )
        # maby add some logging of failed attempts
        except ValueError as e:
            print(f"Skipping {case_path.name} slice {slice_id}: {e}")
            continue

        image_slice = normalize_hu(image_slice, hu_min, hu_max)

        image_slice, mask_slice = resize(
            image_slice,
            mask_slice,
            box_size
        )

        file_path = dest_path / f"{case_path.name}_slice_{slice_id:03d}.pt"

        torch.save(
            {
                "image": image_slice,
                "mask": mask_slice,
                "case_id": case_path.name,
                "slice_idx": slice_id,
            },
            file_path,
        )

        del (
            image_slice,
            mask_slice
        )

        num_saved += 1

    return num_saved


def preprocess_data(
    hu_min: int = -200,
    hu_max: int = 300,
    box_size: int = 256,
    margin: int = 20
) -> None:
    load_dotenv(override=True)
    source_path = Path(os.getenv("DATABASE_PATH"))
    destination_path = Path(os.getenv("PREPROCESSED_PATH"))

    destination_path = (
        destination_path
        / 'slices'
    )
    destination_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    case_dirs = sorted(
        source_path.glob("case_*")
    )

    total_slices = 0
    for case_dir in tqdm(case_dirs):
        total_slices += process_case(
            case_dir,
            destination_path,
            hu_min,
            hu_max,
            box_size,
            margin
        )
        gc.collect()

    print(f'processed {total_slices} CT slices')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('hu_min')
    parser.add_argument('hu_max')
    parser.add_argument('box_size')
    parser.add_argument('margin')
    arguments = parser.parse_args()

    preprocess_data(
        int(arguments.hu_min),
        int(arguments.hu_max),
        int(arguments.box_size),
        int(arguments.margin)
    )
