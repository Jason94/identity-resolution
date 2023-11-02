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
    parser.add_argument(
        "--overwrite-train-val",
        action="store_true",
        help="Overwrite existing train & val data file if it exists. Note that if"
        " --overwrite is provided, the training and validation data will also be overwritten.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Don't balance the classes in the prepared dataset.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomize assignment of data to training and validation. Does nothing if not"
        " writing to train & val data.",
    )
    args = parser.parse_args()

    print("Loading " + str(args.source_files))

    # TODO: Add option for serialized JSON field string here. It's a pain to type into the console, but it's also the
    #       only place left in the code that requires the field object to be hard-coded in data.py (I think).
    data_module = ContactDataModule(
        data_lists=args.source_files,
        prepared_file=args.prepared_data,
        train_file=args.training_data,
        val_file=args.eval_data,
        corrections_file=args.corrections,
        fields=[lookup_field(f_name) for f_name in args.field_names],
        balance_classes=not args.no_balance,
    )
    data_module.prepare_data(
        overwrite=args.overwrite,
        overwrite_train_val=args.overwrite_train_val,
        shuffle=args.shuffle,
    )


if __name__ == "__main__":
    main()
