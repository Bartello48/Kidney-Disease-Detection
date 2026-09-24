from argparse import ArgumentParser

from models.utils.model_storage import ModelStorage
from models.utils.model_tester import ModelTester


def test_model(
    save_name: str,
    threshold_cancer: float = 0.5,
    threshold_cyst: float = 0.5
):
    tester = ModelTester()
    results = tester.test_from_mem(
        save_name,
        threshold_cancer,
        threshold_cyst,
        log=True
    )

    ModelStorage.add_test_results(save_name, results.payload)


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
    args = parser.parse_args()
    test_model(
        str(args.save_name),
        float(args.threshold_cancer),
        float(args.threshold_cyst)
    )
