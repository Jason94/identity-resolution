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

from parsons import Table
from parsons.databases.redshift import Redshift

from utils import init_rs_env, get_model, log_once

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metric import Metric  # noqa:E402
from train import PlContactEncoder  # noqa:E402
from train_classifier import PlContactsClassifier  # noqa:E402
from data import ContactDataModule  # noqa:E402
from utilities import transpose_dict_of_lists  # noqa:E402

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
SOURCE_TABLE = SCHEMA + ".idr_out"
TOKENS_TABLE = SCHEMA + ".idr_tokens"
DUP_CANDIDATE_TABLE = SCHEMA + ".idr_candidates"
DUP_OUTPUT_TABLE = SCHEMA + ".idr_dups"

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


def load_raw_unpooled_data(rs: Redshift) -> Table:
    return rs.query(f"SELECT * FROM {SOURCE_TABLE} ORDER BY primary_key;") or Table()


def load_raw_pooled_data(rs: Redshift, pool: str) -> Table:
    return (
        rs.query(
            f"""
            SELECT *
            FROM {SOURCE_TABLE}
            WHERE pool = %s
            ORDER BY primary_key;
            """,
            [pool],
        )
        or Table()
    )


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
    rs: Redshift,
    pool: Optional[str] = None,
) -> Tuple[List[List[float]], Dict[int, int]]:
    if pool is not None:
        raw_data = load_raw_pooled_data(rs, pool)
    else:
        raw_data = load_raw_unpooled_data(rs)

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
    rs: Redshift,
    candidates: Dict[Tuple[int, int], float],
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
        rs.copy(upload_data, DUP_CANDIDATE_TABLE, if_exists="drop")
    else:
        logger.info("No duplicates identified.")


def generate_candidates(rs: Redshift, model: PlContactEncoder):
    if THRESHOLD is not None:
        model.hparams.metric.threshold = float(THRESHOLD)  # type: ignore

    metric: Metric = model.hparams.metric  # type: ignore
    embedding_dim: int = model.hparams.output_embedding_dim  # type: ignore

    execution_mode = determine_mode()
    logger.debug(f"Execution mode: {execution_mode}")

    if execution_mode == Mode.Unpooled:
        logger.info("Loading data from source table.")
        vectors, index_pkey_map = load_data(embedding_dim, rs)

        source_vectors = vectors
        search_vectors = vectors

        source_index_pkey_map = index_pkey_map
        search_index_pkey_map = index_pkey_map

    else:  # execution_mode == Mode.Pooled or execution_mode == Mode.PooledReflective
        logger.info(f"Loading source data in pool {SOURCE_POOL}")
        source_vectors, source_index_pkey_map = load_data(
            embedding_dim, rs, pool=SOURCE_POOL
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
                embedding_dim, rs, pool=SEARCH_POOL
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

    upload_duplicate_candidates(rs, duplicate_candidates)


def evaluate_candidates(rs: Redshift, pl_encoder: PlContactEncoder):
    classifier_model = get_model(
        PlContactsClassifier,
        CLASSIFIER_URL,
        "classifier.pt",
        encoder=pl_encoder.encoder,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Found device {device}")

    classifier_model.to(device)

    # TODO: Base this on the fields array in the classifier model
    query = f"""
            with a as (
                select
                    idr_candidates.primary_key as pkey,
                    class,
                    contact_timestamp,
                    name_tokens as name_tokens1,
                    name_length as name_length1,
                    email_tokens as email_tokens1,
                    email_length as email_length1,
                    phone_tokens as phone_tokens1,
                    phone_length as phone_length1,
                    state_tokens as state_tokens1,
                    state_length as state_length1
                from {DUP_CANDIDATE_TABLE}
                join {TOKENS_TABLE} t
                    on t.primary_key = idr_candidates.primary_key
                where class_index = 0
            ), b as (
                select
                    idr_candidates.primary_key as pkey,
                    class,
                    contact_timestamp,
                    name_tokens as name_tokens2,
                    name_length as name_length2,
                    email_tokens as email_tokens2,
                    email_length as email_length2,
                    phone_tokens as phone_tokens2,
                    phone_length as phone_length2,
                    state_tokens as state_tokens2,
                    state_length as state_length2
                from {DUP_CANDIDATE_TABLE}
                join {TOKENS_TABLE} t
                    on t.primary_key = idr_candidates.primary_key
                where class_index = 1
            )
            select
                a.pkey as pkey1,
                a.name_tokens1,
                a.name_length1,
                a.email_tokens1,
                a.email_length1,
                a.phone_tokens1,
                a.phone_length1,
                a.state_tokens1,
                a.state_length1,
                b.pkey as pkey2,
                b.name_tokens2,
                b.name_length2,
                b.email_tokens2,
                b.email_length2,
                b.phone_tokens2,
                b.phone_length2,
                b.state_tokens2,
                b.state_length2,
                1 as label
            from a
            join b
                on a.class = b.class
            """
    if rs.table_exists(DUP_OUTPUT_TABLE):
        query += f"""
            left join {DUP_OUTPUT_TABLE} dups
                on (
                    dups.pkey1 = a.pkey
                    and dups.pkey2 = b.pkey
                ) or (
                    dups.pkey1 = b.pkey
                    and dups.pkey2 = a.pkey
                )
            where dups.pkey1 is null
                or dups.comparison_timestamp < a.contact_timestamp
                or dups.comparison_timestamp < b.contact_timestamp
        """

    logger.debug(f"Executing query: {query}")
    data = rs.query(query) or Table()
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
        pair["classification_score"] = pair["classification_score"].item()
        pair["matches"] = pair["classification_score"] >= classification_threshold

    logger.info("Uploading results.")
    upload_data = Table(all_evaluated_pairs)
    upload_data.add_column("comparison_timestamp", datetime.now())

    logger.debug(upload_data)

    rs.upsert(upload_data, DUP_OUTPUT_TABLE, primary_key=["pkey1", "pkey2"])


def main():
    init_rs_env()
    rs = Redshift()
    encoder = get_model(PlContactEncoder, ENCODER_URL)

    generate_candidates(rs, encoder)
    evaluate_candidates(rs, encoder)


if __name__ == "__main__":
    logging.basicConfig()
    main()
