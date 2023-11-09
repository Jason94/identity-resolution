import sys
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

    return combine_queries(temp_table_query, load_temp_data)


def save_data(
    db: DatabaseAdapter,
    fields: List[Field],
    data_table: SQLTable,
    output_table: SQLTable,
    limit: int,
) -> EtlTable:
    query = load_data_conditionally(db, data_table, output_table, fields, limit)
    logger.info(f"Executing: {query}")
    data = db.execute_query(query) or []
    n_rows = etl.nrows(data)

    if n_rows == 0:
        logger.info("No rows found. Exiting.")
        sys.exit()
    elif n_rows > limit:
        logger.info(f"Rows found: {n_rows} greater than limit {limit}.")
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


def step_1_encode_contacts(
    db: DatabaseAdapter,
    batch_size: int,
    data_table: SQLTable,
    tokens_table: SQLTable,
    output_table: SQLTable,
    limit: int,
    encoder_path: str,
    enable_progress_bar: bool = True,
):
    pl_trainer = pl.Trainer(enable_progress_bar=enable_progress_bar)
    pl_model = get_model(PlContactEncoder, encoder_path)
    encoder_uuid: str = pl_model.hparams.uuid  # type: ignore

    check_encoder_uuid(db, encoder_uuid, output_table)

    fields = pl_model.fields()

    logger.info(f"Using {len(fields)} fields:")
    for f in fields:
        logger.info(f"\t{f}")

    save_data(db, fields, data_table, output_table, limit)

    pl_data = ContactSingletonDataModule(
        data_dir="",
        data_lists=[DATA_PATH],
        batch_size=batch_size,
        return_record=True,
        preserve_text_fields=False,
    )
    logger.info("Preparing data.")
    pl_data.prepare_data(overwrite=True)

    upload_prepared_data(db, pl_data, tokens_table)

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
