import os
import sys
from typing import Dict, List, Tuple
from annoy import AnnoyIndex
import logging
from uuid import uuid4

from parsons import Table
from parsons.databases.redshift import Redshift

from utils import init_rs_env, get_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metric import Metric  # noqa:E402

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
EMBEDDING_DIM = int(os.environ["EMBEDDING_DIM"])
PRIMARY_KEY = os.environ["PRIMARY_KEY"]

THRESHOLD = float(os.getenv("THRESHOLD", 0.5))
N_TREES = int(os.getenv("N_TREES", 10))
SEARCH_K = int(os.getenv("SEARCH_K", -1))

OUTPUT_TABLE = os.environ["DUP_OUTPUT_TABLE"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data(rs: Redshift) -> Tuple[List[List[float]], Dict[int, int]]:
    raw_data = rs.query(f"SELECT * FROM {SOURCE_TABLE} ORDER BY {PRIMARY_KEY};")

    if raw_data is None:
        raise ConnectionError("Error retrieving data from the database.")

    logger.info(f"Found {raw_data.num_rows} rows to parse. Reshaping data.")

    vectors = []
    index_id_map = {}
    i = 0

    for row in raw_data:
        index_id_map[i] = row[PRIMARY_KEY]
        vectors.append([row[f"x_{n}"] for n in range(0, EMBEDDING_DIM)])

        i += 1
        if i % 100_000 == 0:
            logger.info(i)

    return vectors, index_id_map


def find_duplicates(vectors, threshold, metric: str) -> List[List[int]]:
    f = len(vectors[0])
    t = AnnoyIndex(f, "euclidean")

    for i in range(len(vectors)):
        t.add_item(i, vectors[i])

    logger.info("Building vector tree")
    t.build(N_TREES)

    logger.info("Searching for duplicates")
    handled_indexes = set()
    equivalence_classes = []
    for i in range(len(vectors)):
        if i not in handled_indexes:
            similar_items = t.get_nns_by_item(
                i, 2, search_k=SEARCH_K, include_distances=True
            )

            closest = similar_items[0][1]
            dist = similar_items[1][1]

            if dist <= threshold and closest not in handled_indexes:
                pair = [i, closest]
                handled_indexes = handled_indexes.union(pair)
                equivalence_classes.append(pair)

    return equivalence_classes


def main():
    init_rs_env()
    rs = Redshift()

    model = get_model()
    metric: Metric = model.hparams.metric  # type: ignore

    logger.info("Loading data from source table.")
    vectors, index_pkey_map = load_data(rs)

    logger.info("Identifying duplicates.")
    duplicates = find_duplicates(vectors, THRESHOLD, metric.annoy_metric)

    logger.info("Preparing data for upload.")
    item_classes = []
    for equivalence_class in duplicates:
        if len(equivalence_class) > 1:
            class_id = uuid4()
            for item in equivalence_class:
                item_classes.append(
                    {PRIMARY_KEY: index_pkey_map[item], "class": class_id}
                )

    upload_data = Table(item_classes)

    if upload_data.num_rows > 0:
        logger.info("Uploading data.")
        rs.copy(upload_data, OUTPUT_TABLE, if_exists="drop")
    else:
        logger.info("No duplicates identified.")


if __name__ == "__main__":
    logging.basicConfig()
    main()
