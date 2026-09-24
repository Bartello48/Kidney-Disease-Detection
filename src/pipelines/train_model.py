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

import models
from models.utils.model_trainer import ModelTrainer
from models.utils.model_storage import ModelStorage
from data.training_parameters import TrainingParameters
from data.kits_dataset import Kits23Dataset
from pipelines.split_cases import split_cases


MODELS = {
    'efficientnet': models.EfficientNetClassifier,
    'efficientnet_pretrained': models.EfficientNetClassifierPretrained,
}


def train_model(
    file_name: str,
    model_name: str,
    save_name: str,
    learning_rate: float,
    epochs: int,
) -> None:
    """
    File path
    """
    # load dataset
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

    train_set = Kits23Dataset(split['train_slices'])
    validation_set = Kits23Dataset(split['validation_slices'])

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

    # load model
    model = MODELS.get(model_name)(
        num_classes=2,
        train_data_path=file_path,
    )
    if model is None:
        raise ValueError(f"No model matching provieded name: {model_name}")

    # prepare training parameters
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [1.56, 6.30],  # cancer: 9021 / 5793 = 1.56, cyst: 13239 / 2103 = 6.30
            dtype=torch.float32,
        )
    )

    features_lr = learning_rate
    classifier_lr = learning_rate / 10
    optimizer = optim.Adam(
        [
            {
                "params": model._model.features.parameters(),
                "lr": features_lr
            },
            {
                "params": model._model.classifier.parameters(),
                "lr": classifier_lr
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
    model.criterion = criterion
    model.optimizer = optimizer
    model.scheduler = scheduler

    # train model
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

    results['parameters'] = TrainingParameters.get_training_parameters(
        epochs,
        learning_rate,
        True,
        model.pretrained,
        criterion,
        optimizer,
        scheduler
    )

    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    model.train_time = now
    ModelStorage.create_save(
        save_name,
        model,
        results  # train loss, validation loss, validation scores, training params
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
    train_model(
        args.sets_name,
        args.model_name,
        args.save_name,
        args.lr,
        args.epochs,
    )
