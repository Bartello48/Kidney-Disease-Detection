import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.models import BaseModel
from src.models import EfficientNetClassifier, EfficientNetClassifierPretrained
from src.models import ResNetClassifier, ResNetClassifierPretrained


class ConfigReader():
    models = {
        'efficientnet': EfficientNetClassifier,
        'efficientnet_pretrained': EfficientNetClassifierPretrained,
        'resnet': ResNetClassifier,
        'resnet_pretrained': ResNetClassifierPretrained
    }
    criterions = {
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss
    }
    optimizers = {
        'Adam': optim.Adam
    }
    schedulers = {
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau
    }

    def __init__(self, config_file: Path) -> None:
        base_config = None
        job_conifg = None

        base_path = Path("config.yaml")
        if not base_path.exists() or base_path.is_dir():
            self._restore_config()
        with Path.open(base_path, 'r+') as fh:
            base_config = yaml.load(fh)
        if base_config is None:
            raise FileExistsError("Coulld not find nor rebuild base config file")

        if not config_file.exists() or config_file.is_dir():
            raise FileExistsError("Could not find job config file")
        with Path.open(config_file, 'r+') as fh:
            job_conifg = yaml.load(fh)
        if job_conifg is None:
            raise FileExistsError("Couldnt load config file from Path provided")

        self.data = {**base_config, **job_conifg}

    def get_model(self) -> BaseModel:
        return self.models[self.data['model']]

    def get_criterion(self) -> tuple:
        criterion = self.criterions[self.data['criterion']['name']]
        kwargs = self.data['criterion']['kwargs']
        if "pos_weight" in kwargs:
            kwargs["pos_weight"] = torch.tensor(
                kwargs["pos_weight"],
                dtype=torch.float32,
            )
        return criterion,

    def get_optimizer(self) -> tuple:
        optimizer = self.optimizers[self.data['optimizer']['name']]
        kwargs = self.data['optimizer']['kwargs']
        return optimizer, kwargs

    def get_scheduler(self) -> tuple:
        scheduler = self.schedulers[self.data['scheduler']['name']]
        kwargs = self.data['scheduler']['kwargs']
        return scheduler, kwargs

    @property
    def seed(self):
        return self.data['seed']

    @property
    def train_batch_size(self):
        return self.data['train_batch_size']

    @property
    def validation_batch_size(self):
        return self.data['validation_batch_size']

    @property
    def test_batch_size(self):
        return self.data['test_batch_size']

    @property
    def learning_rate(self):
        return self.data['learning_rate']

    @property
    def epochs(self):
        return self.data['epochs']

    @property
    def dataset(self):
        return self.data['dataset']

    @property
    def model(self):
        return self.data['model']

    def _restore_config(self):
        data = {
            'seed': 2187,
            'train_batch_size': 16,
            'validation_batch_size': 16,
            'test_batch_size': 16,
            'learning_rate': 0.0001,
            'epochs': 5,
            'dataset': 'first-split.json',
            'model': 'efficientnet',
            'criterion': {
                'name': 'BCEWithLogitsLoss',
                'kwargs': {
                    'pos_weight': [1.56, 6.30]
                }
            },
            'optimizer': {
                'name': 'Adam',
                'kwargs': {}
            },
            'scheduler': {
                'name': 'ReduceLROnPlateau',
                'kwargs': {
                    'mode': 'min',
                    'factor': 0.5,
                    'patience': 1,
                    'min_lr': 1e-7
                }
            }
        }
        with Path.open("config.yaml", 'w+') as fh:
            yaml.dump(data, fh)
