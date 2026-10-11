import os
import sys
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from dotenv import load_dotenv

import torch
import torchvision

from src.config_reader import ConfigReader
from src.models.utils import ModelTrainer, ModelTester, ModelStorage
from src.data.kits23 import get_data_loaders


def get_configs(config_dir: Path) -> list[ConfigReader]:
    jobs = []
    for file in config_dir.glob('*.yaml'):
        jobs.append(ConfigReader(file))
    return jobs


def run_jobs(save_f1: bool, jobs_path: str = None) -> None:
    load_dotenv(override=True)
    if jobs_path is not None:
        jobs_dir = Path(jobs_path)
        if not jobs_dir.exists() and not jobs_dir.is_dir():
            raise FileNotFoundError('path does not point to folder with job configs')
    else:
        jobs_dir = Path(os.getenv('JOBS_PATH'))

    for job in get_configs(jobs_dir):  # job is a description in a config file
        try:
            train_loader, validation_loader, test_loader = get_data_loaders(
                job.dataset,
                job.slices,
                True, job.train_batch_size,
                True, job.validation_batch_size,
                True, job.test_batch_size
            )

            model = job.get_model()
            criterion, kwargs = job.get_criterion()
            criterion = criterion(**kwargs)
            optimizer, kwargs = job.get_optimizer()
            optimizer = optimizer(model.parameters(), **kwargs)
            scheduler, kwargs = job.get_scheduler()
            scheduler = scheduler(optimizer, **kwargs)

            trainer = ModelTrainer(
                train_loader,
                validation_loader,
                save_f1
            )

            results, best_loss_model, best_f1_model = trainer.train_model(
                model=model,
                epochs=job.epochs,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                log=job.log,
            )

            now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            model.train_time = now

            ModelStorage.create_save(
                job.save_name,
                model,
                results,  # train loss, validation loss, validation scores, training params
                job,
                best_loss=best_loss_model,
                best_f1=best_f1_model
            )

            evaluation = ModelTester.test_model(
                model,
                'latest',
                test_loader,
                device=None,
                criterion=criterion,
                log=True,
                plot_curves=True,
                save_name=job.save_name
            )

            ModelStorage.add_test_results(job.save_name, evaluation.payload)
        except Exception as e:
            print(f'During the work on {job.save_name} model, an error has occured')
            print(e)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--jobs-path',
        type=str,
        default=None,
        help='Provide alternative path to jobs folder'
    )
    parser.add_argument(
        '--save-best-f1',
        action='store_true',
        help='If present the training module will also save the model with best F1 score'
    )
    args = parser.parse_args()

    print("--- Environment ---")
    print(f"system version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    run_jobs(args.save_best_f1, args.jobs_path)
