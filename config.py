import os
from contrastive_metric import ContrastiveLoss, is_duplicate

DUPLICATE_CLASS = 1
DISTINCT_CLASS = -1

### ---Hyperparameters

## Model
MAX_NAME_LENGTH = 50
MAX_EMAIL_LENGTH = 35

## Training
EVAL_BATCH_SIZE = 64

## Behavior
# SIMILARITY_METRIC = nn.CosineSimilarity

### --- IO & Logging

MODEL_DIR = "models"


def model_path(fname: str) -> str:
    return os.path.join(MODEL_DIR, fname)
