from argparse import ArgumentParser

from src.models.utils.model_storage import ModelStorage
from src.models.utils.model_tester import ModelTester


def test_model(
    save_name: str,
    latest: bool,
    best_loss: bool,
    best_f1: bool,
    threshold_cancer: float = 0.5,
    threshold_cyst: float = 0.5,
    plot_curves: bool = False
):
    try:
        if (not latest and not best_loss and not best_f1) or latest:
            print('testing latest')
            results = ModelTester.test_model_memory(
                save_name,
                'latest',
                threshold_cancer,
                threshold_cyst,
                log=True,
                plot_curves=plot_curves
            )
            results_dict = results.payload
            results_dict['model'] = 'latest_model'
            ModelStorage.add_test_results(save_name, results_dict)
    except FileExistsError as e:
        print(f'could not find latest model, error message: {e}')

    try:
        if best_loss:
            print('testing best')
            results = ModelTester.test_model_memory(
                save_name,
                'best',
                threshold_cancer,
                threshold_cyst,
                log=True,
                plot_curves=plot_curves
            )
            results_dict = results.payload
            results_dict['model'] = 'smallest_loss'
            ModelStorage.add_test_results(save_name, results_dict)
    except FileExistsError as e:
        print(f'could not find best model, error message: {e}')

    try:
        if best_f1:
            print('testing best f1')
            results = ModelTester.test_model_memory(
                save_name,
                'best_f1',
                threshold_cancer,
                threshold_cyst,
                log=True,
                plot_curves=plot_curves
            )
            results_dict = results.payload
            results_dict['model'] = 'best combined f1'
            ModelStorage.add_test_results(save_name, results_dict)
    except FileExistsError as e:
        print(f'could not find best f1 model, error message: {e}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('save_name')
    parser.add_argument(
        '--latest',
        action='store_true',
        help='specify to test latest model. If all model specification flags'
        '(--latest, --best-loss, --best-f1) will be absent this evaluation will run as well'
    )
    parser.add_argument(
        '--best-loss',
        action='store_true',
        help='specify to test model with the minimal loss'
    )
    parser.add_argument(
        '--best-f1',
        action='store_true',
        help='specify to test model with the minimal combined f1 score'
    )
    parser.add_argument(
        "--threshold-cancer",
        default=0.5,
        type=float,
        help='choose threshold for predictions, smaller predictions'
        ' will be treated as negative, resst as positive'
    )
    parser.add_argument(
        "--threshold-cyst",
        default=0.5,
        type=float,
        help='choose threshold for predictions, smaller predictions'
        ' will be treated as negative, resst as positive'
    )
    parser.add_argument(
        "--plot-curves",
        action='store_true',
        default=False,
        help='use when you want to generate pr auc and roc auc curves'
    )
    args = parser.parse_args()
    test_model(
        str(args.save_name),
        args.latest,
        args.best_loss,
        args.best_f1,
        float(args.threshold_cancer),
        float(args.threshold_cyst),
        args.plot_curves
    )
