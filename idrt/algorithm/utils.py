import os
import sys
from typing import Type, TypeVar, List
import requests
import torch
import lightning.pytorch as pl
import logging

from pypika import Table as SQLTable, Schema, Database, Query
from pypika.queries import QueryBuilder

from parsons.databases.redshift import Redshift

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def init_rs_env():
    """Match the environment variables Civis generates with the names Parsons expects."""
    os.environ["REDSHIFT_DB"] = os.environ["REDSHIFT_DATABASE"]
    os.environ["REDSHIFT_USERNAME"] = os.environ["REDSHIFT_CREDENTIAL_USERNAME"]
    os.environ["REDSHIFT_PASSWORD"] = os.environ["REDSHIFT_CREDENTIAL_PASSWORD"]
    os.environ["S3_TEMP_BUCKET"] = os.environ["S3_TEMP_BUCKET"]


def download_model(model_url: str, model_filename: str = "model.pt"):
    model_path = os.path.join(os.path.dirname(__file__), model_filename)

    # Send a GET request to download the model
    response = requests.get(model_url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        with open(model_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        logger.info("Model downloaded successfully!")
    else:
        raise ConnectionError(
            f"Failed to download model. Response code: {response.status_code}"
        )


M = TypeVar("M", bound=pl.LightningModule)


def get_model(
    model_class: Type[M], model_url: str, model_filename: str = "model.pt", **kwargs
) -> M:
    model_path = os.path.join(os.path.dirname(__file__), model_filename)

    if not os.path.exists(model_path):
        logger.info("Loading model.")
        download_model(model_url, model_filename)
    else:
        logger.info("Found model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Found device {device}")

    model = model_class.load_from_checkpoint(model_path, map_location=device, **kwargs)
    logger.info(model.hparams)

    return model


logged_keys = []


def log_once(logger: logging.Logger, level: int, key: str, message: str):
    if key not in logged_keys:
        logged_keys.append(key)
        logger.log(level, message)


def check_encoder_uuid(rs: Redshift, encoder_uuid: str, output_table: SQLTable):
    if rs.table_exists(output_table.get_sql()):
        query = Query.select(output_table.encoder_uuid).distinct().from_(output_table)
        existing_uuids: List[str] = rs.query(query.get_sql())["encoder_uuid"]  # type: ignore

        if len(existing_uuids) > 1 or (
            encoder_uuid not in existing_uuids and len(existing_uuids) > 0
        ):
            logger.error(f"Detecting existing encoder model UUIDs: {existing_uuids}")
            logger.error(
                "Please clear all IDR results not calculated with the current model"
                f" from {output_table}"
            )
            raise RuntimeError("Cannot use conflicting models for encoding.")


def table_from_full_path(path: str) -> SQLTable:
    """Build a SQLTable from a full sql path.

    The path must contain a tablename and may contain a schema and database.
    """
    parts = path.split(".")

    if len(parts) == 3:
        db = Database(parts[0])
        schema = Schema(parts[1], parent=db)
        return SQLTable(parts[2], schema=schema)
    elif len(parts) == 2:
        schema = Schema(parts[0])
        return SQLTable(parts[1], schema=schema)
    elif len(parts) == 1:
        db = None
        schema = None
        return SQLTable(parts[0])
    else:
        raise RuntimeError(
            f"SQL table '{path}' must contain a table name and optionall a schema and db."
        )


def combine_queries(*queries: QueryBuilder) -> str:
    """Combine multiple PyPika queries into one query string."""
    queries_sql = [q.get_sql() for q in queries]
    return ";\n\n".join(queries_sql) + ";"
