import os
import sys
from typing import List, Optional
import logging
import torch
import lightning.pytorch as pl

from parsons.databases.redshift import Redshift
from parsons import Table

from utils import init_rs_env, get_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import ContactSingletonDataModule  # noqa:E402

if __name__ == "__main__":
    import importlib.util

    dotenv_package = importlib.util.find_spec("dotenv")
    if dotenv_package is not None:
        print("Loading dotenv")
        import dotenv

        dotenv.load_dotenv(override=True)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LOAD_DATA_QUERY = os.environ["LOAD_DATA_QUERY"]
PRIMARY_KEY = os.environ["PRIMARY_KEY"]

DATA_PATH = "data.csv"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))

OUTPUT_TABLE = os.environ["OUTPUT_TABLE"]
LIMIT = 10_000_000


def save_data() -> Table:
    init_rs_env()
    rs = Redshift()

    data = rs.query(LOAD_DATA_QUERY) or Table()

    if data.num_rows == 0:
        logger.info("No rows found. Exiting.")
        sys.exit()
    elif data.num_rows > LIMIT:
        logger.info("{data.num_rows} greater than limit {LIMIT}.")
        sys.exit()

    logger.info(f"Found {data.num_rows} rows.")

    data.to_csv(DATA_PATH, encoding="utf8")

    return data


def main():
    save_data().to_dicts()

    pl_data = ContactSingletonDataModule(
        data_dir="",
        data_lists=[DATA_PATH],
        batch_size=BATCH_SIZE,
        return_record=True,
        preserve_text_fields=False,
    )
    logger.info("Preparing data.")
    pl_data.prepare_data(overwrite=True)
    pl_trainer = pl.Trainer(enable_progress_bar=False)
    pl_model = get_model()

    logger.info("Running model. This will take a while!")
    results: Optional[List[torch.Tensor]] = pl_trainer.predict(
        pl_model,
        pl_data,
    )  # type: ignore

    if results is None:
        raise RuntimeError("Could not retrieve results.")

    logger.info("Preparing results for upload.")
    result_lists = []

    for tensor, record in results:
        for pkey, embedding in zip(record[PRIMARY_KEY].tolist(), tensor.tolist()):  # type: ignore
            result_lists.append([pkey, *embedding])  # type: ignore

    embedding_dim = results[0][0].shape[1]
    uploads = Table(
        [[PRIMARY_KEY, *[str(x) for x in range(0, embedding_dim)]], *result_lists]
    )

    logger.info("Uploading results.")
    rs = Redshift()
    rs.copy(uploads, OUTPUT_TABLE, if_exists="drop")


if __name__ == "__main__":
    logging.basicConfig()
    main()
