from datetime import datetime
from argparse import ArgumentParser

from src.models.utils.model_trainer import ModelTrainer
from src.models.utils.model_storage import ModelStorage
from src.data.kits23 import get_data_loaders


def continue_training(
    save_name: str,
    learning_rate: float,
    epochs: int,
    save_f1: bool
) -> None:
    model = ModelStorage.load_model(save_name)
    config = ModelStorage.load_config(save_name)
    train_loader, validation_loader, _ = get_data_loaders(
        config.dataset,
        slices=config.slices,
        get_train_loader=True, train_batch_size=config.train_batch_size,
        get_validation_loader=True, validation_batch_size=config.validation_batch_size,
    )
    criterion, kwargs = config.get_criterion()
    criterion = criterion(**kwargs)
    optimizer, kwargs = config.get_optimizer(learning_rate=learning_rate)
    optimizer = optimizer(model.parameters(), **kwargs)
    scheduler, kwargs = config.get_scheduler()
    scheduler = scheduler(optimizer, **kwargs)

    data = ModelStorage.load_data(save_name=args.save_name)
    best_loss = min(
        [loss for run in data['training'] for loss in run['validation losses']]
    )
    worst_loss = max(
        [loss for run in data['training'] for loss in run['validation losses']]
    )
    print(f'best loss: {best_loss}, worst loss: {worst_loss}')
    best_f1 = max(
        [
            (entry["cancer"]["f1"] + entry["cyst"]["f1"]) / 2
            for run in data["training"]
            for entry in run["statistics"]
        ]
    )
    print(f'best_f1: {best_f1}')

    trainer = ModelTrainer(
        train_loader,
        validation_loader,
        save_f1
    )

    training_results, best_loss_model, best_f1_model = trainer.train_model(
        model=model,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        log=True,
        best_loss=best_loss,
        best_f1=best_f1
    )

    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    model.train_time = now
    ModelStorage.save_models(
        save_name,
        model,
        best_loss=best_loss_model,
        best_f1=best_f1_model
    )
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
        help='provide learning rate for the model',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        required=True,
        help='specify ammount of training epochs'
    )
    parser.add_argument(
        '--save-best-f1',
        action='store_true',
        help='If present the training module will also save the model with best F1 score'
    )
    args = parser.parse_args()
    if args.lr is not None:
        continue_training(args.save_name, args.lr, args.epochs, args.save_best_f1)
    else:
        data = ModelStorage.load_data(save_name=args.save_name)
        lraning_rate = data['training'][-1]['statistics'][-1]['learning_rate']
        continue_training(args.save_name, lraning_rate, args.epochs, args.save_best_f1)
