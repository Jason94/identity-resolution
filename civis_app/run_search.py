import os
import sys
from typing import Dict, List, Set, Tuple
from annoy import AnnoyIndex
import logging
from uuid import uuid4
import torch
import lightning.pytorch as pl

from parsons import Table
from parsons.databases.redshift import Redshift

from utils import init_rs_env, get_model

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

SOURCE_TABLE = os.environ["SOURCE_TABLE"]
PRIMARY_KEY = os.environ["PRIMARY_KEY"]

THRESHOLD = os.getenv("THRESHOLD")
CLASSIFIER_THRESHOLD = os.getenv("CLASSIFIER_THRESHOLD")
N_CLOSEST = int(os.getenv("N_CLOSEST", 2))
N_TREES = int(os.getenv("N_TREES", 10))
SEARCH_K = int(os.getenv("SEARCH_K", -1))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))

ENCODER_URL = os.environ["MODEL_URL"]
CLASSIFIER_URL = os.environ["CLASSIFIER_URL"]

DUP_CANDIDATE_TABLE = os.environ["DUP_CANDIDATE_TABLE"]
DUP_OUTPUT_TABLE = os.environ["DUP_OUTPUT_TABLE"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data(
    pl_model: PlContactEncoder, rs: Redshift
) -> Tuple[List[List[float]], Dict[int, int]]:
    embedding_dim: int = pl_model.hparams.output_embedding_dim  # type: ignore
    raw_data = rs.query(f"SELECT * FROM {SOURCE_TABLE} ORDER BY {PRIMARY_KEY};")

    if raw_data is None:
        raise ConnectionError("Error retrieving data from the database.")

    logger.info(f"Found {raw_data.num_rows} rows to parse. Reshaping data.")

    vectors = []
    index_id_map = {}
    i = 0

    for row in raw_data:
        index_id_map[i] = row[PRIMARY_KEY]
        vectors.append([row[f"x_{n}"] for n in range(0, embedding_dim)])

        i += 1
        if i % 100_000 == 0:
            logger.info(f"{i} / {raw_data.num_rows}")

    return vectors, index_id_map


def find_duplicates(vectors, metric: Metric) -> Dict[Set[int], float]:
    f = len(vectors[0])
    logger.info(f"Searching under metric {metric.annoy_metric}")
    t = AnnoyIndex(f, metric.annoy_metric)  # type: ignore

    for i in range(len(vectors)):
        t.add_item(i, vectors[i])

    logger.info("Building vector tree")
    t.build(N_TREES)

    logger.info("Searching for duplicates")
    pairs_to_check = set()
    pairs_with_distance = {}
    for i in range(len(vectors)):
        nbrs, distances = t.get_nns_by_item(
            i, N_CLOSEST + 1, search_k=SEARCH_K, include_distances=True
        )

        # Remove the element from its own neighbors
        nbrs = nbrs[1:]
        distances = distances[1:]

        new_pairs = []
        for nbr, dist in zip(nbrs, distances):
            if metric.annoy_metric == "angular":
                # Convert Annoy's angular distance to cosine similarity
                dist = (2 - (dist**2)) / 2

            if metric.distance_matches(dist):
                pair = frozenset([i, nbr])
                new_pairs.append(pair)
                pairs_with_distance[(i, nbr)] = dist

        pairs_to_check = pairs_to_check.union(new_pairs)

        if i % 50_000 == 0:
            logger.info(f"{i} / {len(vectors)}")

    return pairs_with_distance


def upload_duplicate_candidates(
    rs: Redshift, candidates: Dict[Set[int], float], index_pkey_map: Dict[int, int]
):
    logger.info("Preparing data for upload.")
    item_classes = []
    for pair, dist in candidates.items():
        class_id = uuid4()
        i = 0
        for item in pair:
            item_classes.append(
                {
                    PRIMARY_KEY: index_pkey_map[item],
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

    logger.info("Loading data from source table.")
    vectors, index_pkey_map = load_data(model, rs)

    logger.info("Identifying duplicates.")
    duplicate_candidates = find_duplicates(vectors, metric)

    upload_duplicate_candidates(rs, duplicate_candidates, index_pkey_map)


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
    data = (
        rs.query(
            f"""
            with a as (
                select
                    idr_candidates.{PRIMARY_KEY} as pkey1,
                    class,
                    name_tokens as name_tokens1,
                    name_length as name_length1,
                    email_tokens as email_tokens1,
                    email_length as email_length1,
                    phone_tokens as phone_tokens1,
                    phone_length as phone_length1,
                    state_tokens as state_tokens1,
                    state_length as state_length1
                from indivisible_test.idr_candidates
                join indivisible_test.idr_tokens t
                    on t.user_id = idr_candidates.user_id
                where class_index = 0
            ), b as (
                select
                    idr_candidates.{PRIMARY_KEY} as pkey2,
                    class,
                    name_tokens as name_tokens2,
                    name_length as name_length2,
                    email_tokens as email_tokens2,
                    email_length as email_length2,
                    phone_tokens as phone_tokens2,
                    phone_length as phone_length2,
                    state_tokens as state_tokens2,
                    state_length as state_length2
                from indivisible_test.idr_candidates
                join indivisible_test.idr_tokens t
                    on t.{PRIMARY_KEY} = idr_candidates.{PRIMARY_KEY}
                where class_index = 1
            )
            select
                a.pkey1,
                a.name_tokens1,
                a.name_length1,
                a.email_tokens1,
                a.email_length1,
                a.phone_tokens1,
                a.phone_length1,
                a.state_tokens1,
                a.state_length1,
                b.pkey2,
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
        )
        or Table()
    )
    logger.info(f"Found {data.num_rows} candidate pairs.")

    logger.info("Saving candidates data to disk.")
    data_filename = "prepared_data.csv"
    data.to_csv(data_filename)

    logger.info("Assembling data module.")
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

    trainer = pl.Trainer()

    logger.info("Running classification model. This will take a while!")
    all_evaluated_pairs = []
    for classification_score, labels, data in trainer.predict(classifier_model, pl_data):  # type: ignore
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
    rs.copy(upload_data, DUP_OUTPUT_TABLE, if_exists="drop")


def main():
    init_rs_env()
    rs = Redshift()
    encoder = get_model(PlContactEncoder, ENCODER_URL)

    generate_candidates(rs, encoder)
    evaluate_candidates(rs, encoder)


if __name__ == "__main__":
    logging.basicConfig()
    main()
