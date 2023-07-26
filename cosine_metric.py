from typing import Callable, Tuple
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import cosine_similarity
from metric import Metric


def is_duplicate_cosine(threshold, return_distance=True):
    """
    Create a function to determine if two embeddings represent the same class (a duplicate)
    or not, using cosine similarity.

    Args:
        threshold (float): The similarity threshold for determining duplicates. If the cosine
          similarity between the embeddings is more than this value, the inputs are considered
          duplicates.

    Returns:
        callable: A function that takes two embeddings and returns True if the inputs are
          duplicates (i.e., the cosine similarity between the embeddings is more than the
          threshold), False otherwise.
    """

    def _is_duplicate(embedding1, embedding2):
        """
        Determine if two embeddings represent the same class (a duplicate) or not, using
        cosine similarity.

        Args:
            embedding1, embedding2 (torch.Tensor): The two embeddings to compare.

        Returns:
            tensor[bool]: True if the inputs are duplicates (i.e., the cosine similarity
              between the embeddings is more than the threshold), False otherwise.
        """
        similarity = cosine_similarity(embedding1, embedding2)

        if not return_distance:
            return similarity > threshold
        else:
            return similarity > threshold, similarity

    return _is_duplicate


class CosineMetric(Metric):
    def __init__(self, margin: float, threshold: float):
        super().__init__()
        self.margin = margin
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        # TODO: Remove this. Old versions of the model saved the metric without the _threshold
        #       private attribute. The getattribute can be removed when a new model is trained.
        return getattr(self, "_threshold", None) or getattr(self, "threshold")

    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value

    @property
    def loss(self) -> nn.Module:
        return nn.CosineEmbeddingLoss(margin=self.margin)

    @property
    def similarity_function(self) -> Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        return is_duplicate_cosine(self.threshold, True)

    @property
    def annoy_metric(self) -> str:
        return "angular"

    def __repr__(self):
        return f"CosineMetric(margin={self.margin}, threshold={self.threshold})"

    def distance_matches(self, dist: float) -> bool:
        if dist < -1.0 or dist > 1.0:
            raise ValueError(f"Invalid cosine distance {dist}")

        return dist >= self.threshold
