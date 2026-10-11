import os
import json
import torch
from pathlib import Path

from dotenv import load_dotenv

from src.models.base_model import BaseModel
from src.config_reader import ConfigReader


class ModelStorage:
    @staticmethod
    def create_save(
        save_name: str,
        model: BaseModel,
        training_stats: dict,
        config: ConfigReader,
        best_loss: BaseModel = None,
        best_f1: BaseModel = None
    ) -> None:
        load_dotenv(override=True)
        models_folder = Path(os.getenv("TRAINED_MODELS"))
        model_folder = models_folder / save_name
        model_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # prepare model file

        model_path = model_folder / "latest_model.model"
        ModelStorage.save_model(model, model_path)

        if best_loss is not None:
            loss_model_path = model_folder / "best_model.model"
            ModelStorage.save_model(best_loss, loss_model_path)

        if best_f1 is not None:
            f1_model_path = model_folder / 'f1_model.model'
            ModelStorage.save_model(best_f1, f1_model_path)

        # prepare data file

        data_path = model_folder / "data.json"
        payload = dict()
        payload['data_path'] = str(data_path)
        payload['adaptive_lr'] = True
        payload['training'] = [training_stats]
        payload['tests'] = []

        ModelStorage.save_data(payload)

        # prepare config file

        config_path = model_folder / 'config.yaml'
        config.save_config(config_path)

    @staticmethod
    def save_models(
        save_name: str,
        latest_model: BaseModel,
        best_loss: BaseModel = None,
        best_f1: BaseModel = None
    ) -> None:
        load_dotenv(override=True)
        model_folder = Path(os.getenv("TRAINED_MODELS")) / save_name

        model_path = model_folder / "latest_model.model"
        ModelStorage.save_model(latest_model, model_path)

        if best_loss is not None:
            loss_model_path = model_folder / "best_model.model"
            ModelStorage.save_model(best_loss, loss_model_path)

        if best_f1 is not None:
            f1_model_path = model_folder / 'f1_model.model'
            ModelStorage.save_model(best_f1, f1_model_path)

    @staticmethod
    def save_model(model: BaseModel, save_path: Path) -> None:
        with save_path.open('wb') as fh:
            torch.save(model, fh)

    @staticmethod
    def load_model(save_name: str, model_type: str = 'latest') -> BaseModel:
        """
        model type:
        latest: loads the model saved after last epoch
        best: loads the model with minimal validation loss
        best_f1: loads the model with minimal average f1 score among classes
        """
        save_path = ModelStorage.get_save_path(save_name)
        if model_type == 'latest':
            model_path = save_path / 'latest_model.model'
            # support for legacy saves
            if not model_path.exists():
                model_path = model_path.parent / 'model.model'
        elif model_type == 'best':
            model_path = save_path / 'best_model.model'
        elif model_type == 'best_f1':
            model_path = save_path / 'f1_model.model'
        else:
            raise ValueError('Unidentified model type')
        if not model_path.exists():
            raise FileExistsError('Model does not exist')
        with Path.open(model_path, 'rb') as fh:
            model = torch.load(fh, weights_only=False)

        return model

    @staticmethod
    def save_data(data: dict) -> None:
        with Path.open(Path(data['data_path']), 'w+') as fh:
            json.dump(
                data,
                fh,
                indent=4
            )

    @staticmethod
    def load_data(save_name: str) -> dict:
        save_path = ModelStorage.get_save_path(save_name)
        data_path = save_path / 'data.json'

        with Path.open(data_path, 'r') as fh:
            data = json.load(fh)

        return data

    @staticmethod
    def get_save_path(save_name: str) -> Path:
        load_dotenv(override=True)
        models_path = Path(os.getenv('TRAINED_MODELS'))
        model_path = models_path / save_name
        if not model_path.exists() and not model_path.is_dir():
            raise FileNotFoundError('cannot find folder with provided name')
        return model_path

    @staticmethod
    def add_test_results(save_name: str, results: dict) -> None:
        """
        After running tests, wrap results into a dict
        results should contain:
        train loss: list, validation loss: list, validation scores, training params
        """
        data = ModelStorage.load_data(save_name)
        data['tests'].append(results)
        ModelStorage.save_data(data)

    @staticmethod
    def add_training(save_name: str, training_stats: dict) -> None:
        data = ModelStorage.load_data(save_name)
        data['training'].append(training_stats)
        ModelStorage.save_data(data)

    @staticmethod
    def load_config(save_name) -> ConfigReader:
        save_path = ModelStorage.get_save_path(save_name)
        config_path = save_path / 'config.yaml'

        return ConfigReader(config_path)

    @staticmethod
    def save_config(config: ConfigReader, save_name: str):
        save_path = ModelStorage.get_save_path(save_name)
        config_file = save_path / 'config.yaml'
        config.save_config(config_file)
