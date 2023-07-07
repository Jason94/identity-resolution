import os

DUPLICATE_CLASS = 1
DISTINCT_CLASS = -1

### --- IO & Logging

MODEL_DIR = "models"


def model_path(fname: str) -> str:
    return os.path.join(MODEL_DIR, fname)
