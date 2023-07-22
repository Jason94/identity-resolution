from abc import ABC, abstractmethod
from typing import Callable, Tuple
from torch import Tensor
import torch.nn as nn


class Metric(ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def loss(self) -> nn.Module:
        pass

    @property
    @abstractmethod
    def similarity_function(
        self,
    ) -> Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        pass

    @property
    @abstractmethod
    def annoy_metric(self) -> str:
        pass

    @abstractmethod
    def distance_matches(self, dist: float) -> bool:
        pass

    @property
    @abstractmethod
    def threshold(self) -> float:
        pass
