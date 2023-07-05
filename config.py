import torch.nn as nn

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
EVAL_BATCH_SIZE = 8

## Behavior
# SIMILARITY_METRIC = nn.CosineSimilarity
SIMILARITY_METRIC = is_duplicate
SIMILARITY_THRESHOLD = 0.5

MARGIN = 2.0
LOSS_FUNCTION = ContrastiveLoss
