import os
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dotenv import load_dotenv
from tqdm import tqdm

from data.kits_dataset import Kits23Dataset
from models.BaseModel import BaseModel
from models.utils.evaluation_metric import EvaluationMetric
from models.utils.model_storage import ModelStorage


class ModelTester:
    def __init__(self):
        pass

    def test_model(
        self,
        model: BaseModel,
        test_data: DataLoader,
        device: torch.device,
        threshold_cancer: float = 0.5,
        threshold_cyst: float = 0.5,
        log: bool = True,
        criterion: torch.nn.Module = None,
    ) -> dict:
        calculate_loss = True
        calculate_loss = criterion is not None

        model.to(device)
        model.eval()

        threshold = torch.tensor(
            [threshold_cancer, threshold_cyst],
            device=device
        )

        # index 0: cancer, index 1: cyst
        tp = torch.zeros(2, dtype=torch.long, device=device)
        tn = torch.zeros(2, dtype=torch.long, device=device)
        fp = torch.zeros(2, dtype=torch.long, device=device)
        fn = torch.zeros(2, dtype=torch.long, device=device)

        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(test_data):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                if calculate_loss:
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * images.size(0)

                predictions = torch.sigmoid(outputs) >= threshold

                tp += ((predictions == 1) & (labels == 1)).sum(dim=0)
                tn += ((predictions == 0) & (labels == 0)).sum(dim=0)
                fp += ((predictions == 1) & (labels == 0)).sum(dim=0)
                fn += ((predictions == 0) & (labels == 1)).sum(dim=0)

        evaluation = None
        if calculate_loss:
            loss = running_loss / len(test_data.dataset)
            evaluation = EvaluationMetric(
                tp, fn, fp, tn,
                threshold_cancer,
                threshold_cyst,
                loss=loss
            )
        else:
            evaluation = EvaluationMetric(
                tp, fn, fp, tn,
                threshold_cancer,
                threshold_cyst
            )

        if log:
            evaluation.print_results()

        return evaluation

    def test_from_mem(
        self,
        save_name: str,
        threshold_cancer: float = 0.5,
        threshold_cyst: float = 0.5,
        log: bool = True
    ):
        load_dotenv(override=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        model = ModelStorage.load_model(save_name)

        split_path = Path(model.train_data_path)
        with Path.open(split_path, 'r') as fh:
            split = json.load(fh)
        test_set = Kits23Dataset(split['test_slices'])
        test_loader = DataLoader(
            test_set,
            batch_size=int(os.getenv("TEST_BATCH_SIZE")),
            shuffle=False
        )

        return self.test_model(
            model,
            test_loader,
            device,
            threshold_cancer,
            threshold_cyst,
            log=log
        )
