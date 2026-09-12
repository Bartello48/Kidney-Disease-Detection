import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torchvision

from dotenv import load_dotenv

from data.kits_dataset import Kits23Dataset
from pipelines.split_cases import split_cases


def main(file_name: str) -> None:
    load_dotenv(override=True)
    data_path = Path(str(os.getenv("PREPROCESSED_PATH")))
    file_path = data_path / file_name
    split = None
    if not file_path.exists():
        split = split_cases(
            data_path=data_path,
            file_name=file_name,
            train_ratio=0.7,
            validation_ratio=0.15,
            seed=int(os.getenv("SEED"))
        )
    else:
        with Path.open(file_path, "r") as fh:
            split = json.load(fh)
    # dataset = Kits23Dataset(os.getenv("PREPROCESSED_PATH"))
    # print(len(dataset))
    # dataset.save_image(1000)
    # dataset.save_map(1000)


if __name__ == '__main__':
    print("--- Environment ---")
    print(f"system version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'sets_name',
        type=str,
        help="provaide a name of a file"
    )
    args = parser.parse_args()
    main(args.sets_name)
