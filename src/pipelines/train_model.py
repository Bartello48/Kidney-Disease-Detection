import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from dotenv import load_dotenv
from models.utils.model_trainer import ModelTrainer
from models.EfficientNetClassifier import EfficientNetClassifier
from src.models.utils.model_storage import ModelStorage
from data.kits_dataset import Kits23Dataset
from pipelines.split_cases import split_cases


MODELS = {
    'efficientnet': EfficientNetClassifier
}


def train_model(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    model: nn.Module,
    learning_rate: float = 0.001,
    epochs: int = 5
) -> tuple[list, list]:
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(
        [
            {
                "params": model._model.features.parameters(),
                "lr": learning_rate
            },
            {
                "params": model._model.classifier.parameters(),
                "lr": learning_rate / 10
            }
        ]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=1,
        min_lr=1e-7
    )

    trainer = ModelTrainer(
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        scheduler
    )

    results = trainer.train_model(
        model,
        epochs,
        log=True
    )

    results['parameters'] = {
        'epochs': epochs,
        'learning rate': learning_rate,
        'adaptive lr': True,
        'criterion': "BCEWithLogitsLoss",
        'optimizer': "Adam",
        'scheduler': {
            'mode': 'min',
            'factor': 0.5,
            'patience': 1,
            'min_lr': 1e-7
        }
    }

    return results


def main(
    file_name: str,
    model_name: str,
    save_name: str,
    learning_rate: float,
    epochs: int
) -> None:
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

    if not split:
        raise AttributeError("No split generated / found")

    model = MODELS.get(model_name)(num_classes=2)
    if model is None:
        raise ValueError(f"No model matching provieded name: {model_name}")

    train_set = Kits23Dataset(split['train_slices'])
    validation_set = Kits23Dataset(split['validation_slices'])

    # test dataset
    # print(f'dataset shape image: {train_set[1000][0].shape},
    # labels: {train_set[1000][1].shape}')
    # ----------------------

    train_loader = DataLoader(
        train_set,
        batch_size=int(os.getenv("TRAIN_BATCH_SIZE")),
        shuffle=True
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=int(os.getenv("VALIDATION_BATCH_SIZE")),
        shuffle=False
    )

    # test loader
    # l_image, l_label = next(iter(train_loader))
    # print(f"loader shape image: {l_image.shape}, label: {l_label.shape}")
    # print(f"loader labels: {l_label}")
    # test model
    # print(f"test model: {model(l_image)}, \nshape:{model(l_image).shape}")
    # -------------------------------

    training_data = train_model(
        train_loader,
        validation_loader,
        model,
        learning_rate,
        epochs
    )

    model.train_data_path = file_path
    now = datetime.now()
    now = now.strftime("%d-%m-%Y_%H-%M-%S")
    model.train_time = now
    ModelStorage.create_save(
        save_name,
        model,
        training_data[0],  # train loss
        training_data[1],  # validation loss
        training_data[2],  # validation test scores
        training_data[3]  # training parameters
    )


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
    parser.add_argument(
        'model_name',
        type=str,
        help=f'choose one from available models: {MODELS.keys()}'
    )
    parser.add_argument(
        'save_name',
        type=str,
        help='choose a name to save model'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='provide learning rate for the model',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='specify ammount of training epochs'
    )
    args = parser.parse_args()
    main(args.sets_name, args.model_name, args.save_name, args.lr, args.epochs)
