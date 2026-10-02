import os
import json
from pathlib import Path

from dotenv import load_dotenv

from torch.utils.data import DataLoader

from src.data.kits_dataset import Kits23Dataset


def get_data_loaders(
    split_name: Path,
    get_train_loader: bool = False,
    train_batch_size: int = 0,
    get_validation_loader: bool = False,
    validation_batch_size: int = 0,
    get_test_loader: bool = False,
    test_batch_size: int = 0,
):
    if (
        not get_train_loader and
        not get_validation_loader and
        not get_test_loader
    ):
        return None, None, None
    load_dotenv(override=True)
    data_path = os.getenv('PREPROCESSED_PATH')
    split_path = Path(data_path) / split_name
    if not split_path.exists or split_path.is_dir():
        raise FileNotFoundError('Requested split not present')
    else:
        with Path.open(split_path, 'r+') as fh:
            split = json.load(fh)

    if get_train_loader:
        train_set = Kits23Dataset(split['train_slices'])
        train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            shuffle=True
        )
    else:
        train_loader = None
    if get_validation_loader:
        validation_set = Kits23Dataset(split['validation_slices'])
        validation_loader = DataLoader(
            validation_set,
            batch_size=validation_batch_size,
            shuffle=True
        )
    else:
        validation_loader = None
    if get_test_loader:
        test_set = Kits23Dataset(split['test_slices'])
        test_loader = DataLoader(
            test_set,
            batch_size=test_batch_size,
            shuffle=True
        )
    else:
        test_loader = None

    return train_loader, validation_loader, test_loader
