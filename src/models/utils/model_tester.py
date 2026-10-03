from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

from src.models.base_model import BaseModel
from src.models.utils import ModelStorage, EvaluationMetric
from src.data.utils import get_data_loaders


class ModelTester:
    @staticmethod
    def _run_prediction(
        model: BaseModel,
        device: torch.device,
        test_data: DataLoader,
        criterion=None
    ) -> list:
        calculate_loss = (criterion is not None)
        running_loss = 0.0

        model.to(device)
        if calculate_loss:
            criterion.to(device)
        model.eval()

        results = []

        with torch.no_grad():
            for images, labels in tqdm(test_data, 'test model'):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                if calculate_loss:
                    loss = criterion(outputs, labels)
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
        predictions = torch.sigmoid(predictions)
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
        device=None,
        criterion=None,
        threshold_cancer: float = 0.5,
        threshold_cyst: float = 0.5,
        log: bool = True,
        plot_curves: bool = False,
        save_name: str = None
    ) -> EvaluationMetric:
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        results = ModelTester._run_prediction(
            model,
            device,
            test_data,
            criterion=criterion
        )

        evaluation = ModelTester._get_results(
            results,
            device,
            threshold_cancer,
            threshold_cyst
        )

        if log:
            evaluation.print_results()

        if plot_curves:
            ModelTester._plot_pr_roc(results[1], ModelStorage.get_save_path(save_name))

        return evaluation

    @staticmethod
    def test_model_memory(
        save_name: str,
        threshold_cancer: float = 0.5,
        threshold_cyst: float = 0.5,
        log: bool = True,
        plot_curves: bool = False
    ) -> EvaluationMetric:
        model = ModelStorage.load_model(save_name)
        config = ModelStorage.load_config(save_name)
        _, _, test_loader = get_data_loaders(
            config.dataset,
            get_test_loader=True, test_batch_size=config.test_batch_size
        )

        return ModelTester.test_model(
            model,
            test_data=test_loader,
            threshold_cancer=threshold_cancer,
            threshold_cyst=threshold_cyst,
            log=log,
            plot_curves=plot_curves,
            save_name=save_name
        )

    @staticmethod
    def _get_image_save_path(save_path: Path, suffix: str) -> Path:
        save_path = save_path / 'images'
        save_path.mkdir(parents=True, exist_ok=True)

        path = save_path / f'{suffix}.png'
        if path.exists():
            counter = 1
            while (save_path / f'{suffix}_{counter}.png').exists():
                counter += 1
            return (save_path / f'{suffix}_{counter}.png')
        return path

    @staticmethod
    def _plot_pr_roc(results: list, save_path: Path) -> None:  # ABSOLUTE SAVE PATH
        predictions = torch.stack([p for p, _ in results])
        predictions = torch.sigmoid(predictions).cpu().numpy()
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
            path = ModelTester._get_image_save_path(
                save_path,
                f"{class_name}_pr-auc"
            )
            plt.savefig(path)
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
            path = ModelTester._get_image_save_path(
                save_path,
                f"{class_name}_roc"
            )
            plt.savefig(path)
            plt.close()

            print(
                f"{class_name}: "
                f"PR AUC={pr_auc:.4f}, "
                f"ROC AUC={roc_auc:.4f}"
            )
