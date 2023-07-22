from typing import Callable, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import pairwise_distance
from metric import Metric


def is_duplicate(threshold, return_distance=True):
    """
    Create a function to determine if two embeddings represent the same class (a duplicate) or not.

    Args:
        threshold (float): The distance threshold for determining duplicates.
            If the distance between the embeddings is less than this value,
            the inputs are considered duplicates.

    Returns:
        callable: A function that takes two embeddings and returns True if the
            inputs are duplicates (i.e., the distance between the embeddings is
            less than the threshold), False otherwise.
    """

    def _is_duplicate(embedding1, embedding2):
        """
        Determine if two embeddings represent the same class (a duplicate) or not.

        Args:
            embedding1, embedding2 (torch.Tensor): The two embeddings to compare.

        Returns:
            tensor[bool]: True if the inputs are duplicates (i.e., the distance between
                the embeddings is less than the threshold), False otherwise.
        """
        distance = pairwise_distance(embedding1, embedding2)

        if not return_distance:
            return distance < threshold
        else:
            return distance < threshold, distance

    return _is_duplicate


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Takes embeddings of two samples and a target label == 1 if samples are from the same class
        and label == -1 otherwise.

    This class inherits from PyTorch's Module class.
    """

    def __init__(self, margin):
        """
        Initialize the ContrastiveLoss instance.

        Args:
            margin (float): Margin for the contrastive loss. This is a hyperparameter
                that determines the minimum distance between the embeddings of distinct pairs.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        # PairwiseDistance computes the pairwise distance matrix with two given matrices.
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, output1, output2, target):
        """
        Forward pass of the Contrastive Loss function.

        Args:
            output1, output2 (torch.Tensor): Embeddings for the two input samples.
            target (torch.Tensor): Contains the target labels. 1 indicates the samples
                are from the same class, -1 indicates they are from different classes.
        """
        # Calculate the Euclidean distance between the two output embeddings.
        distances = self.pdist(output1, output2)

        # Implement the contrastive loss formula.
        # If the target == 1 (samples from the same class), the first term of the loss
        # becomes active and the algorithm attempts to minimize the distance between
        # the two embeddings.
        same_class_loss = (target == 1).float() * distances

        # If the target == -1 (samples from different classes), the second term of the
        # loss becomes active and the algorithm attempts to maximize the distance between
        # the two embeddings, up to a certain margin.
        distinct_class_loss = (target == -1).float() * torch.relu(
            self.margin - distances
        ).float()

        # Calculate the total loss by adding the two terms
        losses = same_class_loss + distinct_class_loss

        # Return the mean of the losses.
        return losses.mean()


class ContrastiveMetric(Metric):
    def __init__(self, margin: float, threshold: float):
        super().__init__()
        self.margin = margin
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def loss(self) -> nn.Module:
        return ContrastiveLoss(self.margin)

    @property
    def similarity_function(self) -> Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        return is_duplicate(self.threshold, True)

    @property
    def annoy_metric(self) -> str:
        return "euclidean"

    def __repr__(self):
        return f"ContrastiveMetric(margin={self.margin}, threshold={self.threshold})"

    def distance_matches(self, dist: float) -> bool:
        if dist < 0.0:
            raise ValueError(f"Invalid euclidean distance {dist}")
        return dist <= self.threshold
