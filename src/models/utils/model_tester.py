import os
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

from data.kits_dataset import Kits23Dataset
from models.BaseModel import BaseModel
from models.utils.evaluation_metric import EvaluationMetric
from models.utils.model_storage import ModelStorage


class ModelTester:
    @staticmethod
    def _run_prediction(
        model: BaseModel,
        device: torch.device,
        test_data: DataLoader
    ) -> list:
        calculate_loss = model.criterion is not None
        running_loss = 0.0

        model.to(device)
        model.eval()

        results = []

        with torch.no_grad():
            for images, labels in tqdm(test_data):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                if calculate_loss:
                    loss = model.criterion(outputs, labels)
                    running_loss += loss.item() * images.size(0)

                for output, label in zip(outputs, labels, strict=True):
                    results.append((output, label))

        if calculate_loss:
            loss = running_loss / len(test_data.dataset)
            return (loss, results)
        else:
            return (None, results)

    @staticmethod
    def _get_results(
        prediction: tuple[float, list],
        device: torch.device,
        threshold_cancer: float,
        threshold_cyst: float
    ) -> EvaluationMetric:
        loss, results = prediction
        predictions = torch.stack([p for p, _ in results])
        labels = torch.stack([y for _, y in results])

        thresholds = torch.tensor(
            [threshold_cancer, threshold_cyst],
            device=device,
        )

        predicted = predictions > thresholds

        tp = ((predicted == 1) & (labels == 1)).sum(dim=0)
        tn = ((predicted == 0) & (labels == 0)).sum(dim=0)
        fp = ((predicted == 1) & (labels == 0)).sum(dim=0)
        fn = ((predicted == 0) & (labels == 1)).sum(dim=0)

        if loss is not None:
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
        return evaluation

    @staticmethod
    def test_model(
        model: BaseModel,
        test_data: DataLoader,
        device,
        threshold_cancer: float = 0.5,
        threshold_cyst: float = 0.5,
        log: bool = True
    ) -> EvaluationMetric:
        results = ModelTester._run_prediction(
            model,
            device,
            test_data
        )

        evaluation = ModelTester._get_results(
            results,
            device,
            threshold_cancer,
            threshold_cyst
        )

        if log:
            evaluation.print_results()

        return evaluation

    @staticmethod
    def get_test_data(split_path: Path) -> DataLoader:
        with Path.open(split_path, 'r') as fh:
            split = json.load(fh)
            test_set = Kits23Dataset(split['test_slices'])
            test_loader = DataLoader(
                test_set,
                batch_size=int(os.getenv("TEST_BATCH_SIZE")),
                shuffle=False
            )
        return test_loader

    @staticmethod
    def test_model_memory(
        save_name: str,
        threshold_cancer: float = 0.5,
        threshold_cyst: float = 0.5,
        log: bool = True
    ) -> EvaluationMetric:
        load_dotenv(override=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        model = ModelStorage.load_model(save_name)
        model.to(device)
        test_loader = ModelTester.get_test_data(model.train_data_path)

        return ModelTester.test_model(
            model,
            test_loader,
            device,
            threshold_cancer,
            threshold_cyst,
            log
        )

    @staticmethod
    def plot_pr_roc(save_name) -> None:
        load_dotenv(override=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        model = ModelStorage.load_model(save_name)
        save_path = Path(os.getenv('TRAINED_MODELS'))
        save_path = save_path / save_name
        model.to(device)
        test_loader = ModelTester.get_test_data(model.train_data_path)
        _, results = ModelTester._run_prediction(
            model,
            device,
            test_loader
        )
        predictions = torch.stack([p for p, _ in results]).cpu().numpy()
        labels = torch.stack([y for _, y in results]).cpu().numpy()

        for class_idx, class_name in enumerate(["cancer", "cyst"]):
            y_score = predictions[:, class_idx]
            y_true = labels[:, class_idx]

            # PR
            precision, recall, _ = precision_recall_curve(
                y_true,
                y_score,
            )

            pr_auc = auc(recall, precision)

            plt.plot(
                recall,
                precision,
                label=f"{class_name}, AUC={pr_auc:.3}"
            )
            plt.xlabel("Recall")
            plt.ylabel("Precission")
            plt.title("Precision-Recall Curve")
            plt.legend()
            plt.grid()
            plt.savefig(save_path / f"{class_name}_pr-auc")
            plt.close()

            # ROC
            fpr, tpr, thresholds = roc_curve(
                y_true,
                y_score,
            )

            roc_auc = auc(fpr, tpr)

            # Youden's J statistic
            j = tpr - fpr
            best_idx = j.argmax()
            best_threshold = thresholds[best_idx]

            plt.plot(
                fpr,
                tpr,
                label=f"{class_name}, AUC={roc_auc:.3}, "
                f"best threshold={best_threshold:.4}"
            )
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.grid()
            plt.savefig(save_path / f"{class_name}_roc")
            plt.close()

            print(
                f"{class_name}: "
                f"PR AUC={pr_auc:.4f}, "
                f"ROC AUC={roc_auc:.4f}"
            )
