from pathlib import Path
from argparse import ArgumentParser
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from dotenv import load_dotenv
from tqdm import tqdm


def get_crop(
    margin: int,
    image: np.ndarray,
    mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pass


def normalize_hu(
    image: np.ndarray,
    hu_min: int,
    hu_max: int
) -> np.ndarray:
    pass


def resize(
    image: np.ndarray,
    mask: np.ndarray,
    size: int
) -> tuple[np.ndarray, np.ndarray]:
    pass


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

    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)

    image = image_nii.get_fdata().astype(np.float32)
    mask = mask_nii.get_fdata().astype(np.int64)

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

        image_cropped, mask_cropped = get_crop(
            margin,
            image_slice,
            mask_slice,
        )

        image_normalized = normalize_hu(image_cropped, hu_min, hu_max)

        image_resized, mask_resized = resize(
            image_normalized,
            mask_cropped,
            box_size
        )

        output_path = (
            dest_path
            / f"slices/{case_path.name}_slice_{slice_id:03d}.pt"
        )

        torch.save(
            {
                "image": image_resized,
                "mask": mask_resized,
                "case_id": case_path.name,
                "slice_idx": slice_id,
            },
            output_path,
        )

        num_saved += 1

    return num_saved


def preprocess_data(hu_min: int, hu_max: int, box_size: int, margin: int) -> None:
    load_dotenv(override=True)
    source_path = os.getenv("DATABASE_PATH")
    destination_path = os.getenv("PREPROCESSED_PATH")

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
