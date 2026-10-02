import os
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from dotenv import load_dotenv

from src.config_reader import ConfigReader
from src.models.utils import ModelTrainer, ModelTester, ModelStorage
from src.data.utils.get_data_loaders import get_data_loaders


def get_configs(config_dir: Path) -> list[ConfigReader]:
    jobs = []
    for file in config_dir.glob('*.yaml'):
        jobs.append(ConfigReader(file))
    return jobs


def run_jobs(jobs_path: str = None) -> None:
    load_dotenv(override=True)
    if jobs_path is not None:
        jobs_dir = Path(jobs_path)
        if not jobs_dir.exists() and not jobs_dir.is_dir():
            raise FileNotFoundError('path does not point to folder with job configs')
    else:
        jobs_dir = Path(os.getenv('JOBS_PATH'))

    for job in get_configs(jobs_dir):  # job is a description in a config file
        train_loader, validation_loader, test_loader = get_data_loaders(
            job.dataset,
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
        )
        results = trainer.train_model(
            model=model,
            epochs=job.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            log=job.log
        )

        now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        model.train_time = now
        ModelStorage.create_save(
            job.save_name,
            model,
            results,  # train loss, validation loss, validation scores, training params
            job
        )

        evaluation = ModelTester.test_model(
            model,
            test_loader,
            device=None,
            criterion=criterion,
            log=True,
            plot_curves=True,
            save_path=job.save_name
        )

        ModelStorage.add_test_results(job.save_name, evaluation.payload)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--jobs-path',
        type=str,
        default=None
    )
    args = parser.parse_args()
    run_jobs(args.jobs_path)
