import torch.nn as nn

from contrastive_metric import ContrastiveLoss, is_duplicate

### ---Hyperparameters

## Training
SAVED_MODEL_DIR = "models/"
SAVED_MODEL_PATH = "models/model.pth"
EVAL_BATCH_SIZE = 2

## Behavior
# SIMILARITY_METRIC = nn.CosineSimilarity
SIMILARITY_METRIC = is_duplicate
MARGIN = 0.5
LOSS_FUNCTION = ContrastiveLoss
