import os

import torch
from torch.utils.data import DataLoader

from src.models.BaseModel import BaseModel
from src.models.utils.model_tester import ModelTester

from tqdm import tqdm


class ModelTrainer:
    def __init__(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
    ):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def train_model(
        self,
        model: BaseModel,
        criterion,
        optimizer,
        scheduler,
        epochs: int = 5,
        log: bool = True
    ):
        if not log:
            os.environ["TQDM_DISABLE"] = "1"
        try:
            model.to(self.device)
            criterion.to(self.device)
            train_losses = []
            validation_losses = []
            statistics = []

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                for images, labels in tqdm(self.train_loader, 'train loop'):
                    images, labels = images.to(self.device), labels.to(self.device)

                    optimizer.zero_grad(set_to_none=True)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # TEMP TEMP TEMP TEMP TEMP TEMP
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite training loss: {loss.item()}")
                    if not torch.isfinite(outputs).all():
                        raise RuntimeError("Non-finite model output")
                    # -----------------------------

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images.size(0)
                train_loss = running_loss / len(self.train_loader.dataset)
                train_losses.append(train_loss)

                results = ModelTester.test_model(
                    model,
                    self.validation_loader,
                    self.device,
                    criterion=criterion,
                    log=False  # no doubled results print
                )
                validation_loss = results.loss
                validation_losses.append(validation_loss)

                scheduler.step(validation_loss)
                current_lr = optimizer.param_groups[0]['lr']

                if log:
                    print(
                        f"""Epoch {epoch + 1}|{epochs} - train loss: {train_loss},
                        validation loss: {validation_loss}
                        new learning rate: {current_lr}"""
                    )
                    results.print_results()
                statistics.append(results.payload)

            return {
                'train losses': train_losses,
                'validation losses': validation_losses,
                'statistics': statistics
            }

        finally:
            os.environ["TQDM_DISABLE"] = "0"
