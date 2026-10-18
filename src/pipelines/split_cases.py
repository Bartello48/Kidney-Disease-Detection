import os
import json
import random
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from dotenv import load_dotenv


def split_cases(
    data_folder: str,
    file_name: str,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 31415926321589723746264
) -> dict:
    load_dotenv(override=True)
    data_path = Path(os.getenv("PREPROCESSED_PATH"))
    if train_ratio + validation_ratio > 1:
        raise ArithmeticError("sets ratio exceeds 100%")
    slices_path = data_path / data_folder

    # now not using dictionary with slices count, may be raplaced by just counting
    case_slice_count = dict()  # case_id: slices_count
    for path in slices_path.glob("*.pt"):
        case_id = path.name.split("_slice_")[0]
        case_slice_count[case_id] = case_slice_count.get(case_id, 0) + 1
    cases = list(case_slice_count.keys())
    n_cases = len(case_slice_count.keys())

    n_train_cases = int(n_cases * train_ratio)
    n_validation_cases = int(n_cases * validation_ratio)
    n_test_cases = n_cases - n_train_cases - n_validation_cases

    rng = random.Random(seed)
    rng.shuffle(cases)

    train_cases = cases[:n_train_cases]
    validate_cases = cases[n_train_cases:n_train_cases + n_validation_cases]
    test_cases = cases[
        n_train_cases + n_validation_cases:
        n_train_cases + n_validation_cases + n_train_cases
    ]

    train_slices = []
    validation_slices = []
    test_slices = []
    for path in slices_path.glob("*.pt"):
        case_id = path.name.split("_slice_")[0]
        if case_id in train_cases:
            train_slices.append(str(path))
        elif case_id in validate_cases:
            validation_slices.append(str(path))
        elif case_id in test_cases:
            test_slices.append(str(path))
        else:
            raise IndexError("No said case present")

    now = datetime.now()
    payload = {
        "seed": seed,
        "timestamp": now.strftime("%d/%m/%Y, %H:%M:%S"),
        "n_train_cases": n_train_cases,
        "n_train_slices": len(train_slices),
        "n_validation_cases": n_validation_cases,
        "n_validation_slices": len(validation_slices),
        "n_test_cases": n_test_cases,
        "n_test_slices": len(test_slices),
        "train_slices": train_slices,
        "validation_slices": validation_slices,
        "test_slices": test_slices
    }

    with Path.open(data_path / file_name, '+w') as fh:
        json.dump(
            payload,
            fh,
            indent=4
        )

    return payload


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'data_folder',
        type=str
    )
    parser.add_argument(
        'split_name',
        type=str
    )
    args = parser.parse_args()
    split_cases(
        data_folder=args.data_folder,
        file_name=args.split_name
    )
