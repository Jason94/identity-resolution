import os
import sys
from typing import Dict, List, Optional, Tuple
from annoy import AnnoyIndex
import logging
from uuid import uuid4
import torch
import lightning.pytorch as pl
from datetime import datetime
from enum import Enum, auto
from pypika import Table as SQLTable, Query, Order, AliasedQuery
from pypika.queries import QueryBuilder
from pypika.terms import LiteralValue

from parsons import Table
from parsons.databases.redshift import Redshift

from utils import init_rs_env, get_model, log_once, table_from_full_path

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from idrt.metric import Metric
from idrt.train import PlContactEncoder
from idrt.train_classifier import PlContactsClassifier
from idrt.data import ContactDataModule, Field
from utilities import transpose_dict_of_lists

from algorithm.utils import check_encoder_uuid
from algorithm.database_adapter import DatabaseAdapter
from algorithm.redshift_db_adapter import RedshiftDbAdapter

if __name__ == "__main__":
    import importlib.util

    dotenv_package = importlib.util.find_spec("dotenv")
    if dotenv_package is not None:
        print("Loading dotenv")
        import dotenv

        dotenv.load_dotenv(
            override=True, dotenv_path=os.path.join(os.path.dirname(__file__), ".env")
        )

SCHEMA = os.environ["OUTPUT_SCHEMA"]
SOURCE_TABLE = table_from_full_path(SCHEMA + ".idr_out")
TOKENS_TABLE = table_from_full_path(SCHEMA + ".idr_tokens")
DUP_CANDIDATE_TABLE = table_from_full_path(SCHEMA + ".idr_candidates")
DUP_OUTPUT_TABLE = table_from_full_path(SCHEMA + ".idr_dups")

THRESHOLD = os.getenv("THRESHOLD")
CLASSIFIER_THRESHOLD = os.getenv("CLASSIFIER_THRESHOLD")
N_CLOSEST = int(os.getenv("N_CLOSEST", 2))
N_TREES = int(os.getenv("N_TREES", 10))
SEARCH_K = int(os.getenv("SEARCH_K", -1))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))

SEARCH_POOL = os.getenv("SEARCH_POOL")
SOURCE_POOL = os.getenv("SOURCE_POOL")

ENCODER_URL = os.environ["MODEL_URL"]
CLASSIFIER_URL = os.environ["CLASSIFIER_URL"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Mode(Enum):
    Pooled = auto()
    # PooledReflective means that we have the same source and search pool.
    PooledReflective = auto()
    Unpooled = auto()


def determine_mode() -> Mode:
    if SEARCH_POOL is None and SOURCE_POOL is None:
        return Mode.Unpooled
    elif SEARCH_POOL == SOURCE_POOL:
        return Mode.PooledReflective
    else:
        return Mode.Pooled


def load_raw_unpooled_data(db: DatabaseAdapter, source_table: SQLTable) -> Table:
    query = (
        Query.select(source_table.star)
        .from_(source_table)
        .orderby(source_table.primary_key, order=Order.asc)
    )

    return db.execute_query(query) or Table()


def load_raw_pooled_data(
    db: DatabaseAdapter, source_table: SQLTable, pool: str
) -> Table:
    query = (
        Query.select(source_table.star)
        .from_(source_table)
        .where(source_table.pool == pool)
        .orderby(source_table.primary_key, order=Order.asc)
    )

    return db.execute_query(query) or Table()


def shape_raw_data(
    raw_data: Table, embedding_dim: int
) -> Tuple[List[List[float]], Dict[int, int]]:
    vectors = []
    index_id_map = {}
    i = 0

    for row in raw_data:
        index_id_map[i] = row["primary_key"]
        vectors.append([row[f"x_{n}"] for n in range(0, embedding_dim)])

        i += 1
        if i % 100_000 == 0:
            logger.info(f"{i} / {raw_data.num_rows}")

    return vectors, index_id_map


def load_data(
    embedding_dim: int,
    db: DatabaseAdapter,
    pool: Optional[str] = None,
) -> Tuple[List[List[float]], Dict[int, int]]:
    if pool is not None:
        raw_data = load_raw_pooled_data(db, SOURCE_TABLE, pool)
    else:
        raw_data = load_raw_unpooled_data(db, SOURCE_TABLE)

    if raw_data is None:
        raise ConnectionError("Error retrieving data from the database.")

    logger.info(f"Found {raw_data.num_rows} rows to parse. Reshaping data.")

    return shape_raw_data(raw_data, embedding_dim)


def find_duplicates(
    search_vectors: List[List[float]],
    source_vectors: List[List[float]],
    search_index_pkey_map: Dict[int, int],
    source_index_pkey_map: Dict[int, int],
    metric: Metric,
    mode: Mode,
) -> Dict[Tuple[int, int], float]:
    """Find duplicate candidates

    Args:
        search_vectors: Vectors to search through.
        source_vectors: Vectors to try to match.
        metric

    Returns:
        Dict[(source_vector_index:int, search_vector_index:int), distance:float]
    """
    f = len(search_vectors[0])
    logger.info(f"Searching under metric {metric.annoy_metric}")
    t = AnnoyIndex(f, metric.annoy_metric)  # type: ignore

    for i in range(len(search_vectors)):
        t.add_item(i, search_vectors[i])

    logger.info("Building vector tree")
    t.build(N_TREES)

    logger.info(
        f"Searching for duplicates with neighborhood threshold {metric.threshold:0.4f}"
    )
    pairs_with_distance = {}
    for i in range(len(source_vectors)):
        source_vector = source_vectors[i]

        nbrs, distances = t.get_nns_by_vector(
            source_vector, N_CLOSEST + 1, search_k=SEARCH_K, include_distances=True
        )

        if mode == Mode.Unpooled or Mode.PooledReflective:
            # If we are in unpooled mode or reflective mode, then the search vector is in the
            # source pool. Remove the element from its own neighbors
            nbrs = nbrs[1:]
            distances = distances[1:]

        for nbr_i, dist in zip(nbrs, distances):
            if metric.annoy_metric == "angular":
                # Convert Annoy's angular distance to cosine similarity
                dist = (2 - (dist**2)) / 2

            if metric.distance_matches(dist):
                source_pkey = source_index_pkey_map[i]
                search_pkey = search_index_pkey_map[nbr_i]
                pair = [source_pkey, search_pkey]

                if logger.level == logging.DEBUG:
                    log_once(
                        logger,
                        logging.DEBUG,
                        "source_pkey",
                        f"source_pkey example: {source_pkey}",
                    )
                    log_once(
                        logger,
                        logging.DEBUG,
                        "search_pkey",
                        f"search_pkey example: {search_pkey}",
                    )
                    log_once(logger, logging.DEBUG, "pair", f"pair example: {pair}")

                if mode == Mode.Unpooled or mode == Mode.PooledReflective:
                    # If we are in unpooled or reflective mode, then the match will get reciprocated
                    # when we come to searching for nbr's duplicates. Sort so that we don't store
                    # them twice.
                    pair.sort()
                    pair_tuple = (pair[0], pair[1])
                else:
                    # If we are in pooled mode, then the source vector is not in the search pool
                    # and it's important to keep the source index in the first slot and the search
                    # index in the second slot!
                    pair_tuple = tuple(pair)

                if logger.level == logging.DEBUG:
                    log_once(
                        logger, logging.DEBUG, "key", f"pair key example: {pair_tuple}"
                    )
                pairs_with_distance[pair_tuple] = dist

        if i % 50_000 == 0:
            logger.info(f"{i} / {len(source_vectors)}")

    return pairs_with_distance


def upload_duplicate_candidates(
    db: DatabaseAdapter,
    candidates: Dict[Tuple[int, int], float],
    candidates_table: SQLTable,
):
    logger.info("Preparing data for upload.")
    item_classes = []
    for pair, dist in candidates.items():
        class_id = uuid4()
        i = 0
        for item in pair:
            item_classes.append(
                {
                    "primary_key": item,
                    "class": class_id,
                    "class_index": i,
                    "metric": dist,
                }
            )
            i += 1

    upload_data = Table(item_classes)

    if upload_data.num_rows > 0:
        logger.info(f"Found {upload_data.num_rows} duplicate candidates.")
        logger.info("Uploading data.")
        db.bulk_upload(candidates_table, upload_data)
    else:
        logger.info("No duplicates identified.")


def check_candidate_uuids(
    db: DatabaseAdapter,
    classifier_uuid: str,
    classifier_encoder_uuid: str,
    encoder_uuid: str,
    duplicates_table: SQLTable,
):
    if encoder_uuid != classifier_encoder_uuid:
        logging.error(f"Encoder model submitted with uuid: {encoder_uuid}")
        logging.error(
            f"Classifier model submitted was trained with encoder uuid: {classifier_encoder_uuid}"
        )
        raise RuntimeError("Cannot use inconsistent classifier and encoder models")
    if db.table_exists(duplicates_table):
        query = (
            Query()
            .select(duplicates_table.classifier_uuid)
            .distinct()
            .from_(duplicates_table)
        )
        existing_uuids: List[str] = db.execute_query(query)["classifier_uuid"]  # type: ignore

        if len(existing_uuids) > 1 or (
            classifier_uuid not in existing_uuids and len(existing_uuids) > 0
        ):
            logger.error(f"Detecting existing classifier model UUIDs: {existing_uuids}")
            logger.error(
                "Please clear all IDR results not calculated with the current model"
                f" from {duplicates_table.get_sql()}"
            )
            raise RuntimeError("Cannot use conflicting models for classifying.")


def generate_candidates(
    db: DatabaseAdapter, model: PlContactEncoder, candidates_table: SQLTable
):
    if THRESHOLD is not None:
        model.hparams.metric.threshold = float(THRESHOLD)  # type: ignore

    metric: Metric = model.hparams.metric  # type: ignore
    embedding_dim: int = model.hparams.output_embedding_dim  # type: ignore

    execution_mode = determine_mode()
    logger.debug(f"Execution mode: {execution_mode}")

    if execution_mode == Mode.Unpooled:
        logger.info("Loading data from source table.")
        vectors, index_pkey_map = load_data(embedding_dim, db)

        source_vectors = vectors
        search_vectors = vectors

        source_index_pkey_map = index_pkey_map
        search_index_pkey_map = index_pkey_map

    else:  # execution_mode == Mode.Pooled or execution_mode == Mode.PooledReflective
        logger.info(f"Loading source data in pool {SOURCE_POOL}")
        source_vectors, source_index_pkey_map = load_data(
            embedding_dim, db, pool=SOURCE_POOL
        )

        if execution_mode == Mode.PooledReflective:
            logger.info(f"Using source pool '{SOURCE_POOL}' as search pool.")
            search_vectors, search_index_pkey_map = (
                source_vectors,
                source_index_pkey_map,
            )
        else:
            logger.info(f"Loading search data in pool {SEARCH_POOL}")
            search_vectors, search_index_pkey_map = load_data(
                embedding_dim, db, pool=SEARCH_POOL
            )

        if logger.level == logging.DEBUG:
            logger.debug("Source primary_keys sample:")
            logger.debug(list(source_index_pkey_map.values())[0:15])

            logger.debug("Search primary_keys sample:")
            logger.debug(list(search_index_pkey_map.values())[0:15])

    duplicate_candidates = find_duplicates(
        source_vectors=source_vectors,
        search_vectors=search_vectors,
        source_index_pkey_map=source_index_pkey_map,
        search_index_pkey_map=search_index_pkey_map,
        metric=metric,
        mode=execution_mode,
    )

    if logger.level == logging.DEBUG:
        logger.debug("Candidate pairs sample:")
        logger.debug(list(duplicate_candidates.keys())[0:15])

    upload_duplicate_candidates(db, duplicate_candidates, candidates_table)


def load_candidates_data_query(
    exists: bool,
    tokens_table: SQLTable,
    candidate_table: SQLTable,
    dups_table: SQLTable,
    fields: List[Field],
) -> QueryBuilder:
    def field_subquery(class_index: int) -> QueryBuilder:
        selects = []
        n = class_index + 1

        for f in fields:
            selects.append(
                tokens_table.field(f"{f.field}_tokens").as_(f"{f.field}_tokens{n}")
            )
            selects.append(
                tokens_table.field(f"{f.field}_length").as_(f"{f.field}_length{n}")
            )

        return (
            Query.select(
                candidate_table.primary_key,
                tokens_table.pool,
                candidate_table.field("class"),
                tokens_table.contact_timestamp,
                *selects,
            )
            .from_(candidate_table)
            .join(tokens_table)
            .on(tokens_table.primary_key == candidate_table.primary_key)
            .where(candidate_table.class_index == class_index)
        )

    def field_selects(tbl: AliasedQuery, n: int) -> List[QueryBuilder]:
        selects = []

        for f in fields:
            selects.append(tbl.field(f"{f.field}_tokens{n}"))
            selects.append(tbl.field(f"{f.field}_length{n}"))

        return selects

    left_candidates = field_subquery(0)
    left_tbl = AliasedQuery("left_candidates")
    right_candidates = field_subquery(1)
    right_tbl = AliasedQuery("right_candidates")

    load_query = (
        Query.with_(left_candidates, left_tbl.alias)
        .with_(right_candidates, right_tbl.alias)
        .select(
            left_tbl.primary_key.as_("pkey1"),
            left_tbl.pool.as_("pool1"),
            *field_selects(left_tbl, 1),
            right_tbl.primary_key.as_("pkey2"),
            right_tbl.pool.as_("pool2"),
            *field_selects(right_tbl, 2),
            LiteralValue(1).as_("label"),
        )
        .from_(left_tbl)
        .join(right_tbl)
        .on(left_tbl.field("class") == right_tbl.field("class"))
        .orderby(left_tbl.pool, order=Order.asc)
        .orderby(right_tbl.pool, order=Order.asc)
        .orderby(left_tbl.primary_key, order=Order.asc)
        .orderby(right_tbl.primary_key, order=Order.asc)
    )

    if exists:
        load_query = (
            load_query.left_join(dups_table)
            .on(
                (
                    (dups_table.pkey1 == left_tbl.primary_key)
                    & (dups_table.pkey2 == right_tbl.primary_key)
                )
                | (
                    (dups_table.pkey1 == right_tbl.primary_key)
                    & (dups_table.pkey2 == left_tbl.primary_key)
                )
            )
            .where(
                dups_table.pkey1.isnull()
                | (dups_table.comparison_timestamp < left_tbl.contact_timestamp)
                | (dups_table.comparison_timestamp < right_tbl.contact_timestamp)
            )
        )

    return load_query


def evaluate_candidates(
    db: DatabaseAdapter,
    pl_encoder: PlContactEncoder,
    classifier_model: PlContactsClassifier,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Found device {device}")

    classifier_model.to(device)

    fields = pl_encoder.fields()

    query = load_candidates_data_query(
        db.table_exists(DUP_CANDIDATE_TABLE),
        TOKENS_TABLE,
        DUP_CANDIDATE_TABLE,
        DUP_OUTPUT_TABLE,
        fields,
    )

    logger.debug(f"Executing query: {query}")
    data = db.execute_query(query) or Table()
    logger.info(f"Loaded {data.num_rows} new candidate pairs.")

    if data.num_rows == 0:
        logger.info("No new candidate pairs found. Exiting.")
        sys.exit()

    logger.info("Saving candidates data to disk.")
    data_filename = "prepared_data.csv"
    data.to_csv(data_filename)

    logger.info(f"Assembling data module using batch size {BATCH_SIZE}.")
    pl_data = ContactDataModule(
        data_dir="",
        prepared_file=data_filename,
        val_file=data_filename,
        train_file=data_filename,
        batch_size=BATCH_SIZE,
        fields=classifier_model.fields(),
        preserve_text_fields=False,
        return_predict_record=True,
    )
    pl_data.prepare_data()
    pl_data.setup("predict")

    trainer = pl.Trainer(enable_progress_bar=False)

    logger.info("Running classification model. This will take a while!")
    all_evaluated_pairs = []
    for classification_score, labels, data in trainer.predict(  # type: ignore
        classifier_model, pl_data
    ):
        all_evaluated_pairs.extend(
            transpose_dict_of_lists(
                {
                    "pkey1": data["pkey1"],
                    "pkey2": data["pkey2"],
                    "pool1": data["pool1"],
                    "pool2": data["pool2"],
                    "classification_score": classification_score,
                }
            )
        )

    logger.info("Parsing classification results.")
    classification_threshold = float(
        CLASSIFIER_THRESHOLD or classifier_model.hparams.classification_threshold  # type: ignore
    )
    for pair in all_evaluated_pairs:
        pair["pkey1"] = pair["pkey1"].item()
        pair["pkey2"] = pair["pkey2"].item()
        pair["pool1"] = pair["pool1"]
        pair["pool2"] = pair["pool2"]
        pair["classification_score"] = pair["classification_score"].item()
        pair["matches"] = pair["classification_score"] >= classification_threshold

    logger.info("Uploading results.")
    upload_data = Table(all_evaluated_pairs)
    upload_data.add_column("comparison_timestamp", datetime.now())

    classifier_uuid: str = classifier_model.hparams.uuid  # type: ignore
    upload_data.add_column("classifier_uuid", classifier_uuid)

    logger.debug(upload_data)

    db.upsert(
        DUP_OUTPUT_TABLE,
        upload_data,
        primary_key=["pkey1", "pkey2"],
    )


def main(db: DatabaseAdapter):
    encoder = get_model(PlContactEncoder, ENCODER_URL)
    encoder_uuid: str = encoder.hparams.uuid  # type: ignore
    check_encoder_uuid(db, encoder_uuid, SOURCE_TABLE)

    generate_candidates(db, encoder, DUP_CANDIDATE_TABLE)

    classifier_model = get_model(
        PlContactsClassifier,
        CLASSIFIER_URL,
        "classifier.pt",
        encoder=encoder.encoder,
    )
    classifier_encoder_uuid: str = classifier_model.hparams.encoder_uuid  # type: ignore
    classifier_uuid: str = classifier_model.hparams.uuid  # type: ignore

    check_candidate_uuids(
        db, classifier_uuid, classifier_encoder_uuid, encoder_uuid, DUP_OUTPUT_TABLE
    )

    evaluate_candidates(db, encoder, classifier_model)


if __name__ == "__main__":
    logging.basicConfig()

    init_rs_env()
    db = RedshiftDbAdapter(Redshift())

    main(db)
