import logging

from model_cli import make_data_args, make_universal_args
from data import ContactDataModule, lookup_field


def main():
    logging.basicConfig()

    parser = make_universal_args(make_data_args(needs_source_file=True))
    args = parser.parse_args()

    data_module = ContactDataModule(
        data_lists=args.source_files,
        prepared_file=args.prepared_data,
        train_file=args.training_data,
        val_file=args.eval_data,
        fields=[lookup_field(f_name) for f_name in args.field_names],
    )
    data_module.prepare_data(
        overwrite=True,
    )


if __name__ == "__main__":
    main()
