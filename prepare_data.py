import logging

from model_cli import make_data_args
from data import ContactDataModule


def main():
    logging.basicConfig()

    parser = make_data_args(needs_source_file=True)
    args = parser.parse_args()

    data_module = ContactDataModule()
    data_module.prepare_data(
        overwrite=True,
        data_lists=args.source_files,
        writefile=args.prepared_data,
        train_file=args.training_data,
        val_file=args.eval_data,
    )


if __name__ == "__main__":
    main()
