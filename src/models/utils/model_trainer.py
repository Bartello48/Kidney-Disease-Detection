import os
import copy

import torch
from torch.utils.data import DataLoader

from src.models.base_model import BaseModel
from src.models.utils.model_tester import ModelTester

from tqdm import tqdm


class ModelTrainer:
    def __init__(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        save_f1: bool
    ):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.save_f1 = save_f1

    def train_model(
        self,
        model: BaseModel,
        criterion,
        optimizer,
        scheduler,
        epochs: int = 5,
        log: bool = True,
        best_loss: float = float("inf"),
        best_f1: float = 0.0
    ) -> dict[str, list]:  # train lossed, validation losses, stats
        if not log:
            os.environ["TQDM_DISABLE"] = "1"
        try:
            best_loss_model = None
            best_f1_model = None
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
                    'latest',
                    self.validation_loader,
                    self.device,
                    criterion=criterion,
                    log=False  # no doubled results print
                )
                validation_loss = results.loss
                if validation_loss < best_loss:
                    best_loss_model = copy.deepcopy(model)
                if self.save_f1:
                    if (results.f1[0].item() + results.f1[1].item()) / 2 > best_f1:
                        best_f1_model = copy.deepcopy(model)
                validation_losses.append(validation_loss)
                results_dict = results.payload
                results_dict['learning_rate'] = optimizer.param_groups[0]['lr']

                scheduler.step(validation_loss)
                current_lr = optimizer.param_groups[0]['lr']

                if log:
                    print(
                        f"""Epoch {epoch + 1}|{epochs} - train loss: {train_loss},
                        validation loss: {validation_loss}
                        new learning rate: {current_lr}"""
                    )
                    results.print_results()
                statistics.append(results_dict)

            return {
                'train losses': train_losses,
                'validation losses': validation_losses,
                'statistics': statistics
            }, best_loss_model, best_f1_model

        finally:
            os.environ["TQDM_DISABLE"] = "0"
