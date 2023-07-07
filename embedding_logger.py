from typing import Any, List, Optional, Union
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from lightning.fabric.utilities.types import _PATH


class TensorBoardEmbeddingLogger(TensorBoardLogger):
    def __init__(
        self,
        save_dir: _PATH,
        name: Optional[str] = "lightning_logs",
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: Optional[_PATH] = None,
        metadata_header: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            save_dir=save_dir,
            name=name,
            version=version,
            log_graph=log_graph,
            default_hp_metric=default_hp_metric,
            prefix=prefix,
            sub_dir=sub_dir,
        )
        # This is a terrible hack
        self.epoch = 0

        self._features = None
        self._metadata = None
        self._label_img = None

        self.metadata_headers = metadata_header

    @rank_zero_only
    def log_embeddings(
        self,
        features: torch.Tensor,
        metadata: Optional[List[List[str]]] = None,
        label_img: Optional[torch.Tensor] = None,
    ):
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        if self._features is None:
            self._features = features
        else:
            self._features = torch.cat([self._features, features], dim=0)

        if metadata is not None:
            metadata_flat = metadata
            if self._metadata is None:
                self._metadata = metadata_flat
            else:
                self._metadata.extend(metadata_flat)

        if label_img:
            if self._label_img is None:
                self._label_img = label_img
            else:
                self._label_img = torch.cat([self._label_img, label_img], dim=0)

    @rank_zero_only
    def save(self):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        # self.experiment.
        super().save()
        if self._features is not None:
            self.experiment.add_embedding(
                self._features,
                metadata=self._metadata,
                label_img=self._label_img,
                global_step=self.epoch,
                tag=f"{self.name}_{self.version}",
                metadata_header=self.metadata_headers,
            )

            from torch.utils.tensorboard._embedding import make_tsv
            from torch.utils.tensorboard.writer import SummaryWriter

            self._features = None
            self._metadata = None
            self._label_img = None

            self.epoch += 1
