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
from tqdm import tqdm

from models.EfficientNetClassifier import EfficientNetClassifier
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
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # test criterion
    # image, label = next(iter(train_loader))
    # image, label = image.to(device), label.to(device)
    # print(f"image shape: {image.shape}")
    # print(f"labels: {label}, labels shape: {label.shape}")
    # output = model(image)
    # print(f"model output: {output}, shape: {output.shape}")
    # print(criterion(output, label))
    # -------------------

    train_losses, validation_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, 'train loop'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(validation_loader, 'validation loop'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
        validation_loss = running_loss / len(validation_loader.dataset)
        validation_losses.append(validation_loss)

        # TEMPORARY
        print(
            f"""Epoch {epoch + 1}|{epochs} - train loss: {train_loss},
            validation loss: {validation_loss}"""
        )


def main(
    file_name: str,
    model_name: str,
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
    # print(f'dataset shape image: {train_set[1000][0].shape}, labels: {train_set[1000][1].shape}')
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

    train_model(
        train_loader,
        validation_loader,
        model,
        learning_rate,
        epochs
    )

    # save model
    model.train_data_path = file_path
    model_path = Path(os.getenv("TRAINED_MODELS"))
    model_path.mkdir(
        parents=True,
        exist_ok=True
    )
    now = datetime.now()
    now = now.strftime("%d-%m-%Y_%H-%M-%S")
    model_path = model_path / f"{model_name}_{now}.model"
    with Path.open(model_path, 'wb') as fh:
        torch.save(model, fh)


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
    main(args.sets_name, args.model_name, args.lr, args.epochs)
