import os
import sys
import requests
import torch
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import PlContactEncoder  # noqa:E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def init_rs_env():
    """Match the environment variables Civis generates with the names Parsons expects."""
    os.environ["REDSHIFT_DB"] = os.environ["REDSHIFT_DATABASE"]
    os.environ["REDSHIFT_USERNAME"] = os.environ["REDSHIFT_CREDENTIAL_USERNAME"]
    os.environ["REDSHIFT_PASSWORD"] = os.environ["REDSHIFT_CREDENTIAL_PASSWORD"]
    os.environ["S3_TEMP_BUCKET"] = os.environ["S3_TEMP_BUCKET"]


SAVE_PATH = os.path.join(os.path.dirname(__file__), "model.pt")


def download_model():
    model_url = os.environ["MODEL_URL"]

    # Send a GET request to download the model
    response = requests.get(model_url, stream=True)

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


def get_model() -> PlContactEncoder:
    if not os.path.exists(SAVE_PATH):
        logger.info("Loading model.")
        download_model()
    else:
        logger.info("Found model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Found device {device}")

    model = PlContactEncoder.load_from_checkpoint(SAVE_PATH, map_location=device)
    print(model.hparams)
    print(model.hparams.metric)  # type: ignore

    return model
