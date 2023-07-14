from typing import Any, Optional, List
import torch
from torch import optim
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score
from argparse import Namespace
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.loggers.logger import Logger as PlLogger
import logging

from model import ContactEncoder
from config import *  # noqa: F403
from data import ContactDataModule, Field, lookup_field
from model_cli import *  # noqa: F403
from embedding_logger import TensorBoardEmbeddingLogger
from utilities import transpose_dict_of_lists, split_field_dict
from metric import Metric
from cosine_metric import CosineMetric  # noqa:F401
from contrastive_metric import ContrastiveMetric  # noqa:F401


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_bool_tensor(tensor):
    ones = torch.ones_like(tensor, dtype=torch.float32)
    minus_ones = -1 * ones
    converted_tensor = torch.where(tensor, ones, minus_ones)
    return converted_tensor


class PlContactEncoder(pl.LightningModule):
    def __init__(
        self,
        field_names: List[str],
        vocab_size: int,
        checkpoint_path: Optional[str] = None,
        prepared_data: Optional[str] = None,
        source_files: Optional[List[str]] = None,
        training_data: Optional[str] = None,
        eval_data: Optional[str] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        num_epochs: Optional[int] = None,
        p_dropout: Optional[float] = None,
        version_name: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        n_heads_attn: Optional[int] = None,
        attn_dim: Optional[int] = 180,
        norm_eps: Optional[float] = None,
        output_mlp_layers: Optional[int] = None,
        output_embedding_dim: Optional[int] = None,
        metric: Optional[Metric] = None,
        encoder: Optional[ContactEncoder] = None,
    ):
        super().__init__()

        # --- Evaluation Performance Data
        self.validation_labels = []
        self.validation_preds = []

        self.save_hyperparameters(ignore=["encoder", "example_input"])

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = ContactEncoder.from_namespace(Namespace(**self.hparams))

        self.example_input_array = (
            *self.encoder.example_tensor(),
            *self.encoder.example_tensor(),
            torch.tensor([-1]),
        )

    def fields(self) -> List[Field]:
        field_names: List[str] = self.hparams.field_names  # type: ignore
        return [lookup_field(f_name) for f_name in field_names]  # type: ignore

    def training_step(self, batch, batch_idx):
        metric: Optional[Metric] = self.hparams.metric  # type: ignore
        if not metric:
            raise RuntimeError("Not configured for training!")

        (tokens1, lengths1, tokens2, lengths2, labels) = batch

        # Forward pass through the model
        output1 = self.encoder(tokens1, lengths1)
        output2 = self.encoder(tokens2, lengths2)

        # Calculate loss
        loss = metric.loss(output1, output2, labels)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=self.hparams.batch_size,  # type: ignore
        )

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        metric: Optional[Metric] = self.hparams.metric  # type: ignore
        if not metric:
            raise RuntimeError("Not configured for evaluating!")

        (tokens1, lengths1, tokens2, lengths2, labels, field_data) = batch

        # Forward pass through the model
        output1 = self.encoder(tokens1, lengths1)
        output2 = self.encoder(tokens2, lengths2)

        # Compute the loss
        loss = metric.loss(output1, output2, labels)

        pred_raw, dists = metric.similarity_function(output1, output2)
        pred = convert_bool_tensor(pred_raw)

        if isinstance(self.logger, TensorBoardEmbeddingLogger):
            field_data1, field_data2 = split_field_dict(
                self.fields(), transpose_dict_of_lists(field_data)  # type: ignore
            )

            field_data1 = [list(d.values()) for d in field_data1]
            field_data2 = [list(d.values()) for d in field_data2]

            self.logger.log_embeddings(output1, self.global_step, metadata=field_data1)
            self.logger.log_embeddings(output2, self.global_step, metadata=field_data2)

        # Log
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,  # type: ignore
        )
        self.validation_labels.append(labels.cpu())
        self.validation_preds.append(pred.cpu())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        tensors, lengths, *data = batch
        return self.encoder(tensors, lengths), *data

    def on_validation_epoch_end(self) -> None:
        all_labels = torch.cat(self.validation_labels).numpy()
        all_preds = torch.cat(self.validation_preds).numpy()

        precision = float(precision_score(all_labels, all_preds, zero_division=0))
        recall = float(recall_score(all_labels, all_preds))
        f1 = float(f1_score(all_labels, all_preds))

        self.log_dict({"val_precision": precision, "val_recall": recall, "val_f1": f1})

        # Free validation logging memory
        self.validation_labels.clear()
        self.validation_preds.clear()

    def configure_optimizers(self) -> Any:
        lr: float = self.hparams.learning_rate  # type: ignore
        decay: float = self.hparams.weight_decay  # type: ignore
        optimizer = optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=decay,
        )
        return optimizer

    def forward(
        self, tokens1: List[torch.Tensor], lengths1: List[torch.Tensor], _, __, ___
    ):
        return self.encoder(tokens1, lengths1)


def train(
    args: Namespace,
    data_module: ContactDataModule,
    lightning_logger: Optional[PlLogger] = None,
):
    checkpoint_path: Optional[str] = args.checkpoint_path

    if checkpoint_path is not None:
        logger.info(f"Loading model from {checkpoint_path}")
        lightning_model = PlContactEncoder.load_from_checkpoint(checkpoint_path)
        metric = type(lightning_model.hparams.metric)(  # type: ignore
            margin=lightning_model.hparams.metric.margin,  # type: ignore
            threshold=args.threshold,
        )  # type: ignore
        lightning_model.save_hyperparameters(
            {
                "batch_size": args.batch_size,
                "training_data": args.training_data,
                "eval_data": args.eval_data,
                "metric": metric,
                "version_name": (
                    args.version_name or lightning_model.hparams.version_name  # type: ignore
                ),
                "learning_rate": args.learning_rate,
            }
        )
        print(lightning_model.hparams)
    else:
        args.metric = globals()[args.metric](
            margin=args.margin, threshold=args.threshold
        )
        delattr(args, "margin")
        delattr(args, "threshold")
        lightning_model = PlContactEncoder(
            **{
                **vars(args),
                "vocab_size": len(data_module.vocabulary),
            }
        )
        lightning_model._save_to_state_dict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found device {device}")

    lightning_model.to(device)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=4,
        monitor="val_f1",
        mode="max",
        filename="{epoch:02d}---{val_loss:.4f}-{val_f1:.4f}",
        every_n_epochs=2,
        save_last=True,
    )

    if lightning_logger is None:
        lightning_logger = TensorBoardEmbeddingLogger(
            save_dir="",
            metadata_header=[f.field for f in data_module.fields],
            maximum_embeddings_to_save=10000,
            version=lightning_model.hparams.version_name,  # type: ignore
            log_graph=True,
        )
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=[ModelSummary(max_depth=-1), checkpoint_callback],
        logger=lightning_logger,
    )
    trainer.fit(
        model=lightning_model, datamodule=data_module, ckpt_path=checkpoint_path
    )


def margin_experiment(args: Namespace):
    start = 1.5
    end = 4.0
    for margin in [start + x / 2 for x in range(0, int(end - start) * 2)]:
        scenario_args = Namespace(**{**vars(args), "margin": margin})

        for i in range(0, 3):
            print(f"Experiment {i}: margin={margin}")
            lightning_logger = TensorBoardLogger(
                save_dir="", version=f"margin_{margin:0.2f}__{i}"
            )
            lightning_logger.experiment.add_embedding()
            # Load the data
            train(scenario_args, data_module)


def embedding_experiment(args: Namespace, data_module: ContactDataModule):
    start = 80
    end = 120
    step = 20
    example_size = 4
    hparam = "embedding_dim"

    for value in [start + x * step for x in range(0, int((end - start) / step + 1))]:
        scenario_args = Namespace(**{**vars(args), hparam: int(value)})

        for i in range(0, example_size):
            print(f"Experiment {i}: {hparam}={value}")
            lightning_logger = TensorBoardLogger(
                save_dir="", version=f"exp_{hparam}_{value:0.5f}__{i}"
            )
            train(scenario_args, data_module, lightning_logger)  # type: ignore


if __name__ == "__main__":
    logging.basicConfig()

    parser = make_universal_args()
    make_model_io_args(parser)
    make_data_args(parser)
    make_model_args(parser)
    make_training_args(parser)

    args = parser.parse_args()

    data_module = ContactDataModule(
        batch_size=args.batch_size,
        return_eval_fields=True,
        train_file=args.training_data,
        val_file=args.eval_data,
        fields=[lookup_field(f_name) for f_name in args.field_names],
    )
    args.vocab_size = len(data_module.vocabulary)

    # margin_experiment(args)
    # embedding_experiment(args, data_module)

    train(args, data_module)
