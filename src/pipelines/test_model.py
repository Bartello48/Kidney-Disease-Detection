import os
import json
from pathlib import Path

from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from models.ModelStorage import ModelStorage
from models.EvaluationMetric import EvaluationMetric
from data.kits_dataset import Kits23Dataset


def test_model(
    save_name: str,
    threshold_cancer: float = 0.5,
    threshold_cyst: float = 0.5
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    # load model
    model = ModelStorage.load_model(save_name)

    model.to(device)
    model.eval()

    # load test data
    split_path = Path(model.train_data_path)
    with Path.open(split_path, 'r') as fh:
        split = json.load(fh)
    test_set = Kits23Dataset(split['test_slices'])
    test_loader = DataLoader(
        test_set,
        batch_size=int(os.getenv("TEST_BATCH_SIZE")),
        shuffle=False
    )

    # test model performance
    threshold = torch.tensor(
        [threshold_cancer, threshold_cyst],
        device=device
    )

    # index 0: cancer, index 1: cyst
    tp = torch.zeros(2, dtype=torch.long, device=device)
    tn = torch.zeros(2, dtype=torch.long, device=device)
    fp = torch.zeros(2, dtype=torch.long, device=device)
    fn = torch.zeros(2, dtype=torch.long, device=device)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, 'test loop'):
            images, labels = images.to(device), labels.to(device)

            predictions = torch.sigmoid(model(images)) >= threshold

            tp += ((predictions == 1) & (labels == 1)).sum(dim=0)
            tn += ((predictions == 0) & (labels == 0)).sum(dim=0)
            fp += ((predictions == 1) & (labels == 0)).sum(dim=0)
            fn += ((predictions == 0) & (labels == 1)).sum(dim=0)

    precission = tp.float() / (tp + fp).clamp_min(1)
    recall = tp.float() / (tp + fn).clamp_min(1)
    f1 = (
        2 * precission * recall
        / (precission + recall).clamp_min(1e-8)
    )
    accuracy = (tp + tn).float() / (tp + tn + fp + fn).clamp_min(1)
    # cancer
    print("--- CANCER ---")
    print(f"tp: {tp[0].item()}\tfn: {fn[0].item()}")
    print(f"fp: {fp[0].item()}\ttn: {tn[0].item()}")
    print(f"Accuracy: {accuracy[0].item()}")
    print(f"Precission: {precission[0].item()}")
    print(f"Recall: {recall[0].item()}")
    print(f"F1 score: {f1[0].item()}")
    # cyst
    print("--- CYST ---")
    print(f"tp: {tp[1].item()}\tfn: {fn[1].item()}")
    print(f"fp: {fp[1].item()}\ttn: {tn[1].item()}")
    print(f"Accuracy: {accuracy[1].item()}")
    print(f"Precission: {precission[1].item()}")
    print(f"Recall: {recall[1].item()}")
    print(f"F1 score: {f1[1].item()}")

    metric = EvaluationMetric(
        tp,
        fn,
        fp,
        tn,
        accuracy,
        precission,
        recall,
        f1
    )
    ModelStorage.add_test_results(metric.payload)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('save_name')
    parser.add_argument(
        "--threshold_cancer",
        default=0.5,
        type=float,
        help='choose threshold for predictions, smaller predictions'
        ' will be treated as negative, resst as positive'
    )
    parser.add_argument(
        "--threshold_cyst",
        default=0.5,
        type=float,
        help='choose threshold for predictions, smaller predictions'
        ' will be treated as negative, resst as positive'
    )
    args = parser.parse_args()
    test_model(
        str(args.model_name),
        float(args.threshold_cancer),
        float(args.threshold_cyst)
    )
