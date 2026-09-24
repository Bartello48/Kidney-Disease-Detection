import torch


class TrainingParameters:
    @staticmethod
    def get_training_parameters(
        epochs: int,
        learning_rate: float,
        adaptive_lr: bool,
        pretrained: bool,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler
    ) -> dict:
        return {
        'epochs': epochs,
        'learning rate': learning_rate,
        'adaptive lr': adaptive_lr,
        'pretrained': pretrained,
        'criterion': {
            "name": type(criterion).__name__,
            "pos_weight": criterion.pos_weight.tolist()
        },
        'optimizer': {
            "name": type(optimizer).__name__,
            "features_lr": optimizer.param_groups[0]["lr"],
            "classifier_lr": optimizer.param_groups[1]["lr"]
        },
        'scheduler': {
            'mode': scheduler.mode,
            'factor': scheduler.factor,
            'patience': scheduler.patience,
            'min_lr': scheduler.min_lrs[0]
        }
    }
