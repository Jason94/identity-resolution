import os
import sys
from typing import List, Optional
import logging
import torch
import lightning.pytorch as pl

from parsons.databases.redshift import Redshift
from parsons import Table

from utils import init_rs_env, get_model, check_encoder_uuid

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from idrt.train import PlContactEncoder  # noqa:E402
from idrt.data import Field, ContactSingletonDataModule  # noqa:E402

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

DATA_PATH = "data.csv"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))

SCHEMA = os.environ["OUTPUT_SCHEMA"]
TOKENS_TABLE = SCHEMA + ".idr_tokens"
OUTPUT_TABLE = SCHEMA + ".idr_out"
LIMIT = int(os.getenv("LIMIT", str(500_000)))


def load_data_conditionally(
    rs: Redshift, load_query: str, output_table: str, fields: List[Field]
) -> str:
    subfields: List[str] = [
        subfield for field in fields for subfield in field.subfield_labels
    ]
    subfield_selects: List[str] = [
        f"LOWER(COALESCE(temp.{subfield}, '')) as {subfield}" for subfield in subfields
    ]
    subfield_select_clause: str = ",\n".join(subfield_selects)
    if rs.table_exists(output_table):
        logger.info("Output table detected.")
        temp_table_query = f"CREATE TEMP TABLE temp_load_data AS ({load_query});"

        complete_query = f"""
            {temp_table_query}

            SELECT
                temp.primary_key,
                temp.contact_timestamp,
                {subfield_select_clause},
                RIGHT(REGEXP_REPLACE(LOWER(COALESCE(temp.phone, '')), '[^0-9]', ''), 10) as phone,
                temp.pool
            FROM temp_load_data AS temp
            LEFT JOIN {output_table} AS output
            ON temp.primary_key = output.primary_key
            WHERE output.primary_key IS NULL OR temp.contact_timestamp > output.contact_timestamp
            LIMIT {LIMIT};
        """
    else:
        logger.info("Output table does not exist.")
        temp_table_query = f"CREATE TEMP TABLE temp_load_data AS ({load_query});"

        complete_query = f"""
            {temp_table_query}

            SELECT
                temp.primary_key,
                temp.contact_timestamp,
                {subfield_select_clause},
                temp.pool
            FROM temp_load_data AS temp;
        """
    return complete_query


def save_data(rs: Redshift, fields: List[Field]) -> Table:
    query = load_data_conditionally(
        rs, LOAD_DATA_QUERY.rstrip(" ;"), OUTPUT_TABLE, fields
    )
    logger.info(f"Executing: {query}")
    data = rs.query(query) or Table()

    if data.num_rows == 0:
        logger.info("No rows found. Exiting.")
        sys.exit()
    elif data.num_rows > LIMIT:
        logger.info("{data.num_rows} greater than limit {LIMIT}.")
        sys.exit()

    logger.info(f"Found {data.num_rows} rows.")

    logger.debug("Queried data:")
    logger.debug(data)

    data.to_csv(DATA_PATH, encoding="utf8")

    return data


def upload_prepared_data(rs: Redshift, pl_data: ContactSingletonDataModule):
    logger.info("Saving tokens")
    data = Table.from_csv(pl_data.prepared_file)
    logger.debug("Saved prepared data:")
    logger.debug(data)
    rs.upsert(
        table_obj=data,
        target_table=TOKENS_TABLE,
        primary_key="primary_key",
        vacuum=False,
    )


def main():
    init_rs_env()
    rs = Redshift()

    pl_trainer = pl.Trainer(enable_progress_bar=False)
    pl_model = get_model(PlContactEncoder, os.environ["MODEL_URL"])
    encoder_uuid: str = pl_model.hparams.uuid  # type: ignore

    check_encoder_uuid(rs, encoder_uuid, OUTPUT_TABLE)

    fields = pl_model.fields()

    logger.info(f"Using {len(fields)} fields:")
    for f in fields:
        logger.info(f"\t{f}")

    save_data(rs, fields)

    pl_data = ContactSingletonDataModule(
        data_dir="",
        data_lists=[DATA_PATH],
        batch_size=BATCH_SIZE,
        return_record=True,
        preserve_text_fields=False,
    )
    logger.info("Preparing data.")
    pl_data.prepare_data(overwrite=True)

    upload_prepared_data(rs, pl_data)

    logger.info("Running model. This will take a while!")
    results: Optional[List[torch.Tensor]] = pl_trainer.predict(
        pl_model,
        pl_data,
    )  # type: ignore

    if results is None:
        raise RuntimeError("Could not retrieve results.")

    logger.info("Preparing results for upload.")
    result_lists = []

    logger.debug("Results example:")
    logger.debug(results[0])

    for tensor, record in results:
        data = zip(
            record["primary_key"].tolist(),  # type: ignore
            record["contact_timestamp"],  # type: ignore
            record["pool"],  # type: ignore
            tensor.tolist(),
        )
        for pkey, timestamp, pool, embedding in data:
            result_lists.append([pkey, timestamp, pool, *embedding])  # type: ignore

    embedding_dim = results[0][0].shape[1]
    uploads = Table(
        [
            [
                "primary_key",
                "contact_timestamp",
                "pool",
                *[str(x) for x in range(0, embedding_dim)],
            ],
            *result_lists,
        ]
    )

    uploads.add_column("encoder_uuid", encoder_uuid)

    logger.info("Uploading results.")
    logger.debug(uploads)
    rs = Redshift()
    rs.upsert(uploads, OUTPUT_TABLE, primary_key="primary_key", vacuum=False)


if __name__ == "__main__":
    logging.basicConfig()
    main()
