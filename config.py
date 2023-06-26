import torch.nn as nn

### ---Hyperparameters

## Training
SAVED_MODEL_PATH = "models/model.pth"
EVAL_BATCH_SIZE = 2

## Behavior
SIMILARITY_METRIC = nn.CosineSimilarity
