import os
import json
import torch
from pathlib import Path

from dotenv import load_dotenv

from models.BaseModel import BaseModel


class ModelStorage:
    @staticmethod
    def create_save(
        save_name: str,
        model: BaseModel,
        training_stats: dict
    ) -> None:
        load_dotenv(override=True)
        models_folder = Path(os.getenv("TRAINED_MODELS"))
        model_folder = models_folder / save_name
        model_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # prepare model file

        model_path = model_folder / "model.model"
        model.model_path = model_path

        ModelStorage.save_model(model)

        # prepare data file

        data_path = model_folder / "data.json"
        payload = dict()
        payload['data_path'] = str(data_path)
        payload['train_data_file'] = str(model.train_data_path)
        payload['adaptive_lr'] = True
        payload['training'] = training_stats
        payload['tests'] = []

        ModelStorage.save_data(payload)

    @staticmethod
    def save_model(model: BaseModel) -> None:
        with Path.open(model.model_path, 'wb') as fh:
            torch.save(model, fh)

    @staticmethod
    def load_model(save_name: str) -> BaseModel:
        save_path = ModelStorage.get_save_path(save_name)
        model_path = save_path / 'model.model'

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
            data = json.loads(fh)

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
        """
        data = ModelStorage.load_data(save_name)
        data['tests'].append(results)
        ModelStorage.save_data(data)
