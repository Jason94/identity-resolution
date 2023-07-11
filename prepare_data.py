import logging

from model_cli import make_data_args, make_universal_args
from data import ContactDataModule, lookup_field


def main():
    logging.basicConfig()

    parser = make_universal_args(make_data_args(needs_source_file=True))
    parser.add_argument(
        "--corrections",
        type=str,
        help="CSV file in the data directory storing rows of corrected data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared data file if it exists.",
    )
    args = parser.parse_args()

    data_module = ContactDataModule(
        data_lists=args.source_files,
        prepared_file=args.prepared_data,
        train_file=args.training_data,
        val_file=args.eval_data,
        corrections_file=args.corrections,
        fields=[lookup_field(f_name) for f_name in args.field_names],
    )
    data_module.prepare_data(
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
