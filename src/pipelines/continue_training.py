from datetime import datetime
from argparse import ArgumentParser

from src.models.utils.model_trainer import ModelTrainer
from src.models.utils.model_storage import ModelStorage
from src.data.utils import get_data_loaders


def continue_training(save_name: str, learning_rate: float, epochs: int):
    model = ModelStorage.load_model(save_name)
    config = ModelStorage.load_config(save_name)
    train_loader, validation_loader, _ = get_data_loaders(
        config.dataset,
        get_train_loader=True, train_batch_size=config.train_batch_size,
        get_validation_loader=True, validation_batch_size=config.validation_batch_size,
    )
    criterion, kwargs = config.get_criterion()
    criterion = criterion(**kwargs)
    optimizer, kwargs = config.get_optimizer(learning_rate=learning_rate)
    optimizer = optimizer(model.parameters(), **kwargs)
    scheduler, kwargs = config.get_scheduler()
    scheduler = scheduler(optimizer, **kwargs)

    trainer = ModelTrainer(
        train_loader,
        validation_loader,
    )

    training_results = trainer.train_model(
        model=model,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        log=True
    )

    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    model.train_time = now
    ModelStorage.add_training(
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
