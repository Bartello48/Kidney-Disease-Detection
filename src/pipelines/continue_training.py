import os
import json
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from torch.utils.data import DataLoader

from data.training_parameters import TrainingParameters
from data.kits_dataset import Kits23Dataset
from models.utils.model_trainer import ModelTrainer
from models.utils.model_storage import ModelStorage


def continue_training(save_name: str, learning_rate: float, epochs: int):
    model = ModelStorage.load_model(save_name)
    split_path = model.train_data_path
    with Path.open(split_path, 'r+') as fh:
        split = json.load(fh)
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

    trainer = ModelTrainer(
        train_loader,
        validation_loader,
        criterion=model.criterion,
        optimizer=model.optimizer,
        scheduler=model.scheduler
    )

    training_results = trainer.train_model(
        model,
        epochs,
        log=True
    )

    training_results['parameters'] = TrainingParameters.get_training_parameters(
        epochs,
        learning_rate,
        True,
        model.pretrained,
        model.criterion,
        model.optimizer,
        model.scheduler,
    )

    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    model.train_time = now
    ModelStorage.add_test_results(
        save_name,
        training_results
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('save_name', help='provide name of model folder')
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
    continue_training(args.save_name, args.lr, args.epochs)
