from pathlib import Path
from argparse import ArgumentParser
import os
import gc

import nibabel as nib
import numpy as np
import torch

from dotenv import load_dotenv
from tqdm import tqdm

from src.data.preprocessing import get_crop
from src.data.preprocessing import normalize_hu
from src.data.preprocessing import resize


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
                "case_id": int(case_path.name.split('_')[1]),
                "slice_id": slice_id,
                "file_path": str(file_path)
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
    hu_min: int,  # -200
    hu_max: int,  # 300
    box_size: int,  # 256
    margin: int,  # 20
    folder_name: str
) -> None:
    load_dotenv(override=True)
    source_path = Path(os.getenv("DATABASE_PATH"))
    destination_path = Path(os.getenv("PREPROCESSED_PATH"))

    destination_path = (
        destination_path
        / f'{folder_name}'
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

    parser.add_argument(
        'hu_min',
        type=int,
        default=-200
    )
    parser.add_argument(
        'hu_max',
        type=int,
        default=300
    )
    parser.add_argument(
        'box_size',
        type=int,
        default=256
    )
    parser.add_argument(
        'margin',
        type=int,
        default=20
    )
    parser.add_argument(
        'folder_name',
        type=str,
        help='name folder to put preprocessed slices'
    )
    arguments = parser.parse_args()

    preprocess_data(
        arguments.hu_min,
        arguments.hu_max,
        arguments.box_size,
        arguments.margin,
        arguments.folder_name
    )
