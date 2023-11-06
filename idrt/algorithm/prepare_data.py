import os
import sys
from typing import List, Optional
import logging
import torch
import lightning.pytorch as pl

from parsons.databases.redshift import Redshift
from parsons import Table

from pypika import Table as SQLTable, Query, Order, functions as fn

from utils import (
    init_rs_env,
    get_model,
    check_encoder_uuid,
    table_from_full_path,
    combine_queries,
)

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

DATA_TABLE = table_from_full_path(os.environ["DATA_TABLE"])

DATA_PATH = "data.csv"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))

SCHEMA = os.environ["OUTPUT_SCHEMA"]
TOKENS_TABLE = table_from_full_path(SCHEMA + ".idr_tokens")
OUTPUT_TABLE = table_from_full_path(SCHEMA + ".idr_out")
LIMIT = int(os.getenv("LIMIT", str(500_000)))


def load_data_conditionally(
    rs: Redshift,
    data_table: SQLTable,
    output_table: SQLTable,
    fields: List[Field],
) -> str:
    temp_table = SQLTable("temp_load_data")
    temp_table_query = (
        Query.create_table(temp_table)
        .as_select(
            Query.from_(data_table)
            .select(data_table.star)
            .orderby(data_table.contact_timestamp, order=Order.asc)
        )
        .temporary()
    )

    subfields: List[str] = [
        subfield for field in fields for subfield in field.subfield_labels
    ]
    subfield_selects = [
        fn.Lower(fn.Coalesce(temp_table.field(subfield), "")).as_(subfield)
        for subfield in subfields
    ]

    load_temp_data = (
        Query.select(
            temp_table.primary_key,
            temp_table.contact_timestamp,
            temp_table.pool,
            *subfield_selects,
        )
        .from_(temp_table)
        .orderby(temp_table.contact_timestamp, order=Order.asc)
        .limit(LIMIT)
    )

    if rs.table_exists(output_table.get_sql()):
        logger.info("Output table detected.")

        load_temp_data = (
            load_temp_data.left_join(output_table)
            .on(temp_table.primary_key == output_table.primary_key)
            .where(
                output_table.primary_key.isnull()
                | (temp_table.contact_timestamp > output_table.contact_timestamp)
            )
        )
    else:
        logger.info("Output table does not exist.")

    return combine_queries(temp_table_query, load_temp_data)


def save_data(rs: Redshift, fields: List[Field]) -> Table:
    query = load_data_conditionally(rs, DATA_TABLE, OUTPUT_TABLE, fields)
    logger.info(f"Executing: {query}")
    data = rs.query(query) or Table()

    if data.num_rows == 0:
        logger.info("No rows found. Exiting.")
        sys.exit()
    elif data.num_rows > LIMIT:
        logger.info(f"Rows found: {data.num_rows} greater than limit {LIMIT}.")
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
        target_table=TOKENS_TABLE.get_sql(),
        primary_key="primary_key",
        vacuum=False,
    )


def main():
    init_rs_env()
    rs = Redshift()

    pl_trainer = pl.Trainer(enable_progress_bar=False)
    pl_model = get_model(PlContactEncoder, os.environ["MODEL_URL"])
    encoder_uuid: str = pl_model.hparams.uuid  # type: ignore

    check_encoder_uuid(rs, encoder_uuid, OUTPUT_TABLE.get_sql())

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
    rs.upsert(uploads, OUTPUT_TABLE.get_sql(), primary_key="primary_key", vacuum=False)


if __name__ == "__main__":
    logging.basicConfig()
    main()
