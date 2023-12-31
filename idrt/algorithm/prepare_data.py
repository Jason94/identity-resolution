import sys
import os
from typing import List, Optional
import logging
import torch
import lightning.pytorch as pl

import petl as etl
from pypika import Table as SQLTable, Query, Order, functions as fn

from idrt.train import PlContactEncoder
from idrt.data import Field, ContactSingletonDataModule

from idrt.algorithm.database_adapter import DatabaseAdapter, EtlTable
from idrt.algorithm.utils import (
    get_model,
    check_encoder_uuid,
    combine_queries,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DATA_PATH = "data.csv"


def load_data_conditionally(
    db: DatabaseAdapter,
    data_table: SQLTable,
    invalid_table: SQLTable,
    output_table: SQLTable,
    fields: List[Field],
    limit: int,
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
        .limit(limit)
    )

    if db.table_exists(output_table):
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

    if db.table_exists(invalid_table):
        logger.info("Cached invalid rows detected.")

        load_temp_data = load_temp_data.left_join(invalid_table).on(
            temp_table.primary_key == invalid_table.primary_key
        )

    return combine_queries(temp_table_query, load_temp_data)


def save_data(
    db: DatabaseAdapter,
    fields: List[Field],
    data_table: SQLTable,
    invalid_table: SQLTable,
    output_table: SQLTable,
    limit: int,
) -> EtlTable:
    query = load_data_conditionally(
        db, data_table, invalid_table, output_table, fields, limit
    )
    logger.info(f"Executing: {query}")
    data = db.execute_query(query) or []
    n_rows = etl.nrows(data)

    if n_rows == 0:
        logger.info("No rows found. Exiting.")
        sys.exit()

    logger.info(f"Found {n_rows} rows.")

    logger.debug("Queried data:")
    logger.debug(data)

    etl.tocsv(data, DATA_PATH, encoding="utf8")

    return data


def upload_prepared_data(
    db: DatabaseAdapter, pl_data: ContactSingletonDataModule, tokens_table: SQLTable
):
    logger.info("Saving tokens")
    data = etl.fromcsv(pl_data.prepared_file)
    logger.debug("Saved prepared data:")
    logger.debug(data)
    db.upsert(
        tokens_table,
        data,
        "primary_key",
    )


def save_invalid_rows(
    db: DatabaseAdapter,
    invalid_table: SQLTable,
    invalid_rows_filename: str,
    clean: bool = True,
):
    invalid_data = etl.fromcsv(invalid_rows_filename)

    # If there are no invalid rows, the CSV will have 1 row with no column headers
    if len(invalid_data) > 0 and "primary_key" in etl.header(invalid_data):
        invalid_data = etl.cut(invalid_data, "primary_key", "pool")

        logger.info(
            f"Uploading {etl.nrows(invalid_data)} invalid rows to {invalid_rows_filename}"
        )
        db.upsert(invalid_table, invalid_data, primary_key=["primary_key", "pool"])

    if clean:
        os.remove(invalid_rows_filename)


def step_1_encode_contacts(
    db: DatabaseAdapter,
    batch_size: int,
    data_table: SQLTable,
    tokens_table: SQLTable,
    output_table: SQLTable,
    invalid_table: SQLTable,
    limit: int,
    encoder_path: str,
    enable_progress_bar: bool = True,
):
    pl_trainer = pl.Trainer(enable_progress_bar=enable_progress_bar)
    pl_model = get_model(PlContactEncoder, encoder_path)
    encoder_uuid: str = pl_model.hparams.uuid  # type: ignore
    invalid_rows_filename = os.path.join(os.getcwd(), "invalid_rows.csv")

    check_encoder_uuid(db, encoder_uuid, output_table)

    fields = pl_model.fields()

    logger.info(f"Using {len(fields)} fields:")
    for f in fields:
        logger.info(f"\t{f}")

    save_data(db, fields, data_table, invalid_table, output_table, limit)

    pl_data = ContactSingletonDataModule(
        data_dir="",
        data_lists=[DATA_PATH],
        batch_size=batch_size,
        return_record=True,
        preserve_text_fields=False,
    )
    logger.info("Preparing data.")
    pl_data.prepare_data(overwrite=True, invalid_rows_filename=invalid_rows_filename)

    upload_prepared_data(db, pl_data, tokens_table)
    save_invalid_rows(db, invalid_table, invalid_rows_filename)

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
    uploads = [
        [
            "primary_key",
            "contact_timestamp",
            "pool",
            *[str(x) for x in range(0, embedding_dim)],
        ],
        *result_lists,
    ]

    uploads = etl.addfield(uploads, "encoder_uuid", encoder_uuid)

    logger.info("Uploading results.")
    logger.debug(uploads)

    db.upsert(output_table, uploads, "primary_key")
