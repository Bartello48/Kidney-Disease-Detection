import os
from pathlib import Path
from argparse import ArgumentParser

from dotenv import load_dotenv

from src.config_reader import ConfigReader
from src.models.utils import ModelTrainer, ModelTester
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

    for job in get_configs(jobs_dir):
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
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler
        )
        trainer.train_model(
            model=model,
            epochs=job.epochs,
            log=job.log
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--jobs-path',
        type=str,
        default=None
    )
    args = parser.parse_args()
    run_jobs(args.jobs_path)
