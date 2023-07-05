import os
from contrastive_metric import ContrastiveLoss, is_duplicate

DUPLICATE_CLASS = 1
DISTINCT_CLASS = -1

### ---Hyperparameters

## Model
MAX_NAME_LENGTH = 50
MAX_EMAIL_LENGTH = 35

## Training
SAVED_MODEL_DIR = "models/"
SAVED_MODEL_PATH = "model"
SAVED_MODEL_FNAME = os.path.join(SAVED_MODEL_DIR, f"{SAVED_MODEL_PATH}.pth")
SAVED_MODEL_CONFIG_FNAME = f"config_{SAVED_MODEL_PATH}.json"
SAVED_MODEL_CONFIG_PATH = os.path.join(SAVED_MODEL_DIR, SAVED_MODEL_CONFIG_FNAME)

EVAL_BATCH_SIZE = 64

## Behavior
# SIMILARITY_METRIC = nn.CosineSimilarity
SIMILARITY_METRIC = is_duplicate
SIMILARITY_THRESHOLD = 0.5

MARGIN = 2.0
LOSS_FUNCTION = ContrastiveLoss
