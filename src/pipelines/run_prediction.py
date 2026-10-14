import os
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

import torch
from tqdm import tqdm
import nibabel as nib
from dotenv import load_dotenv
from totalsegmentator.python_api import totalsegmentator

from src.models.utils import ModelStorage
from src.data.preprocessing import get_crop
from src.data.preprocessing import normalize_hu
from src.data.preprocessing import resize


def predict(
    save_name: str,
    file_path: str = None,
    case_id: int = None,
    hu_min: int = -200,
    hu_max: int = 300,
    margin: int = 20,
    box_size: int = 256,
    log: bool = None
) -> None:
    load_dotenv(override=True)
    if file_path in None and case_id is None:
        raise ValueError('No scan to run inference on')

    if file_path is not None:
        case_path = Path(file_path)
    else:
        source_path = Path(os.getenv("DATABASE_PATH"))
        case_path = source_path / f'case_{case_id:>05}' / "imaging.nii.gz"

    if not case_path.is_file():
        raise FileNotFoundError(f'couldnt found ct scan {case_path}')
    image = nib.load(case_path).get_fdata(dtype=np.float32)

    seg_image = totalsegmentator(
        image,
        roi_subset=["kidney_left", "kidney_right"],
        device='gpu',
        quiet=not log
    )

    mask = np.asarray(seg_image.dataobj)

    slices = []
    for slice_id in tqdm(range(image.shape[2]), 'preprocessing image'):
        image_slice = image[:, :, slice_id]
        mask_slice = mask[:, :, slice_id]

        if not any(mask_slice > 0):
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

        slices.append(image_slice)
        del (
            image_slice,
            mask_slice
        )

    model = ModelStorage.load_model(save_name, model_type='latest')
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    config = ModelStorage.load_config(save_name)
    input_slices = config.slices

    results = []
    for i in tqdm(range(slices), 'running prediction'):
        indexes = [j for j in range(i - input_slices // 2, i + input_slices // 2 + 1)]

        for j in range(len(indexes)):
            if indexes[j] < 0 or indexes[j] > len(slices) - 1:
                indexes[j] = indexes[len(indexes) - j - 1]
            if indexes[j] < 0 or indexes[j] > len(slices) - 1:
                indexes[j] = i

        payload = []
        for j in range(indexes):
            if slices[j].ndim == 3 and slices[j].shape[0] == 1:
                slices[j].squeeze(0)
            payload.append(slices[j])
        payload = torch.stack(payload, dim=0)

        with torch.no_grad():
            results.append(model.forward(payload))

    print(results)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'SAVE_NAME',
        type=str,
        help='Podaj nazwę modelu do testu'
    )
    parser.add_argument(
        '--file-path',
        type=str,
        help='Provide path to nifti file containing CT scan'
    )
    parser.add_argument(
        '--kits-case-number',
        type='Alternatively provide number of kits case to run prediction on'
    )
    parser.add_argument(
        '--hu-min',
        type=int,
        help='default hu_min: -200 can be changen by providing this argument followed by new value'
    )
    parser.add_argument(
        '--hu-max',
        type=int,
        help='default hu_min: -200 can be changen by providing this argument followed by new value'
    )
    parser.add_argument(
        '--margin',
        type=int,
        help='margin is the ammount of border pixels saved around the kidney in each image'
        'default margin: 20px can be changen by providing this argument followed by new value'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        help='specifies size of models input image. Recommended to leave it to default,'
        ' unless Your model has different input image size, default size=256x256'
    )
    args = parser.parse_args()
    predict(
        args.save_name,
        file_path=args.file_path,
        case_id=args.kits_case_number,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        margin=args.margin,
        box_size=args.image_size,
        log=True
    )
