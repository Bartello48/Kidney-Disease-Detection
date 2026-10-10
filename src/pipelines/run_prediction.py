from argparse import ArgumentParser


def predict(
    save_name: str,
    file_path: str = None,
    case_id: int = None
) -> None:
    pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'SAVE_NAME',
        type=str,
        help='Podaj nazwę modelu do testu'
    )
    parser.add_argument(
        '--file-path',
        type=str,
        help='Provide path to nifti file containing CT scan'
    )
    parser.add_argument(
        '--kits-case-number',
        type='Alternatively provide number of kits case to run prediction on'
    )
    args = parser.parse_args()
    predict(
        args.save_name,
        file_path=args.file_path,
        case_id=args.kits_case_number
    )
