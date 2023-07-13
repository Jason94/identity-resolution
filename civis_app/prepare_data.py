import os
import sys
import requests
import logging

from parsons.databases.redshift import Redshift
from parsons import Table

from utils import init_rs_env

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import PlContactEncoder  # noqa:E402
from data import ContactSingletonDataModule  # noqa:E402
import torch  # noqa:E402
import lightning.pytorch as pl  # noqa:E402

if __name__ == "__main__":
    import importlib.util

    dotenv_package = importlib.util.find_spec("dotenv")
    if dotenv_package is not None:
        print("Loading dotenv")
        import dotenv

        dotenv.load_dotenv(override=True)


logger = logging.getLogger()
logger.setLevel(logging.INFO)

LOAD_DATA_QUERY = os.environ["LOAD_DATA_QUERY"]
PRIMARY_KEY = os.environ["PRIMARY_KEY"]

DATA_PATH = "data.csv"

MODEL_URL = os.environ["MODEL_URL"]
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
SAVE_PATH = "model.pt"

OUTPUT_TABLE = os.environ["OUTPUT_TABLE"]
LIMIT = 10000


def save_data():
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

    data.to_csv(DATA_PATH)


def get_model():
    # Send a GET request to download the model
    response = requests.get(MODEL_URL, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        with open(SAVE_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        logger.info("Model downloaded successfully!")
    else:
        raise ConnectionError(
            f"Failed to download model. Response code: {response.status_code}"
        )


def main():
    logger.info("Loading model.")
    get_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_data = ContactSingletonDataModule(
        data_dir="",
        prepared_file=DATA_PATH,
        batch_size=BATCH_SIZE,
        preserve_text_fields=False,
    )
    pl_model: pl.LightningModule = PlContactEncoder.load_from_checkpoint(
        SAVE_PATH, map_location=device
    )
    pl_trainer = pl.Trainer()

    results = pl_trainer.predict(pl_model, pl_data)

    print(results)


if __name__ == "__main__":
    main()
