from argparse import ArgumentParser

from models.utils.model_storage import ModelStorage
from models.utils.model_tester import ModelTester


def test_model(
    save_name: str,
    threshold_cancer: float = 0.5,
    threshold_cyst: float = 0.5,
    plot_curves: bool = False
):
    results = ModelTester.test_model_memory(
        save_name,
        threshold_cancer,
        threshold_cyst,
        log=True
    )

    ModelStorage.add_test_results(save_name, results.payload)

    if plot_curves:
        ModelTester.plot_pr_roc(save_name)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('save_name')
    parser.add_argument(
        "--threshold_cancer",
        default=0.5,
        type=float,
        help='choose threshold for predictions, smaller predictions'
        ' will be treated as negative, resst as positive'
    )
    parser.add_argument(
        "--threshold_cyst",
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
        float(args.threshold_cancer),
        float(args.threshold_cyst),
        args.plot_curves
    )
