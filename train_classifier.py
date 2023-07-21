from typing import Any, List, Optional
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.nn as nn
from torch import optim
import logging
from torchmetrics.classification import F1Score, Precision, Recall

from data import ContactDataModule, Field, lookup_field
from model import ContactEncoder
from model_classifier import ContactsClassifier
from model_cli import *
from train import PlContactEncoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def labels_to_probs(tensor: torch.Tensor) -> torch.Tensor:
    tensor[tensor == -1.0] = 0
    tensor[tensor == 1.0] = 1
    return tensor.float()


class PlContactsClassifier(pl.LightningModule):
    def __init__(
        self,
        field_names: List[str],
        encoder: ContactEncoder,
        classification_threshold: float,
        pre_pool_mlp_layers: Optional[int] = None,
        pool_mlp_layers: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        training_data: Optional[str] = None,
        eval_data: Optional[str] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        num_epochs: Optional[int] = None,
        p_dropout: Optional[float] = None,
        version_name: Optional[str] = None,
        norm_eps: Optional[float] = None,
        classifier: Optional[ContactsClassifier] = None,
    ):
        super().__init__()

        # TODO: Find a better way of handling the encoder
        self.save_hyperparameters(ignore=["classifier", "encoder"])
        self.encoder = encoder

        if classifier:
            self.classifier = classifier
        else:
            self.classifier = ContactsClassifier.from_encoder(
                encoder,
                pre_pool_mlp_layers=pre_pool_mlp_layers,
                pool_mlp_layers=pool_mlp_layers,
            )

        self.val_f1 = F1Score(task="binary", threshold=classification_threshold)
        self.val_precision = Precision(
            task="binary", threshold=classification_threshold
        )
        self.val_recall = Recall(task="binary", threshold=classification_threshold)

        self.example_input_array = (
            *encoder.example_tensor(),
            *encoder.example_tensor(),
            torch.tensor([-1]),
        )

    def fields(self) -> List[Field]:
        field_names: List[str] = self.hparams.field_names  # type: ignore
        return [lookup_field(f_name) for f_name in field_names]  # type: ignore

    def on_train_start(self) -> None:
        encoder: ContactEncoder = self.encoder  # type: ignore
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

    def training_step(self, batch, batch_idx):
        encoder: ContactEncoder = self.encoder  # type: ignore
        (tokens1, lengths1, tokens2, lengths2, labels) = batch
        labels = labels_to_probs(labels)

        # Forward pass through the encoder
        _, attn_out1, _ = encoder.forward(tokens1, lengths1)
        _, attn_out2, _ = encoder.forward(tokens2, lengths2)

        # Forward pass through the classifier
        classification = self.classifier.forward(attn_out1, attn_out2)

        loss = nn.BCELoss()(classification, labels)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=self.hparams.batch_size,  # type: ignore
        )

        return loss

    def validation_step(self, batch, batch_idx):
        encoder: ContactEncoder = self.encoder  # type: ignore
        classification_threshold: float = self.hparams.classification_threshold  # type: ignore
        (tokens1, lengths1, tokens2, lengths2, labels, _) = batch
        labels = labels_to_probs(labels)

        # Forward pass through the encoder
        _, attn_out1, _ = encoder.forward(tokens1, lengths1)
        _, attn_out2, _ = encoder.forward(tokens2, lengths2)

        # Forward pass through the classifier
        classification = self.classifier.forward(attn_out1, attn_out2)
        loss = nn.BCELoss()(classification, labels)

        # Calculate metrics
        preds = (classification > classification_threshold).float()
        self.val_f1.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,  # type: ignore
        )

        return preds, classification

    def on_validation_epoch_end(self) -> None:
        f1_score = self.val_f1.compute()
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()

        self.log("val_f1", f1_score)
        self.log("val_precision", precision)
        self.log("val_recall", recall)

        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        encoder: ContactEncoder = self.encoder  # type: ignore

        tokens1, lengths1, tokens2, lengths2, labels, *data = batch

        # Forward pass through the encoder
        _, attn_out1, _ = encoder.forward(tokens1, lengths1)
        _, attn_out2, _ = encoder.forward(tokens2, lengths2)

        # Forward pass through the classifier
        return self.classifier.forward(attn_out1, attn_out2), labels, *data

    def configure_optimizers(self) -> Any:
        lr: float = self.hparams.learning_rate  # type: ignore
        decay: float = self.hparams.weight_decay  # type: ignore
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=decay)
        return optimizer

    def forward(
        self,
        tokens1: List[torch.Tensor],
        lengths1: List[torch.Tensor],
        tokens2: List[torch.Tensor],
        lengths2: List[torch.Tensor],
        _,
    ):
        encoder: ContactEncoder = self.encoder  # type: ignore

        _, attn_out1, _ = encoder.forward(tokens1, lengths1)
        _, attn_out2, _ = encoder.forward(tokens2, lengths2)

        return self.classifier.forward(attn_out1, attn_out2)


if __name__ == "__main__":
    logging.basicConfig()

    parser = make_universal_args(mode="classifier")
    make_model_io_args(parser)
    make_data_args(parser)
    make_training_args(parser, mode="classifier")

    args = parser.parse_args()

    data_module = ContactDataModule(
        batch_size=args.batch_size,
        return_eval_fields=True,
        train_file=args.training_data,
        val_file=args.eval_data,
        fields=[lookup_field(f_name) for f_name in args.field_names],
    )

    checkpoint_path: Optional[str] = args.checkpoint_path

    encoder = PlContactEncoder.load_from_checkpoint(args.encoder_path).encoder
    delattr(args, "encoder_path")

    if checkpoint_path is not None:
        logger.info(f"Loading model from {checkpoint_path}")
        lightning_model = PlContactsClassifier.load_from_checkpoint(
            checkpoint_path, encoder=encoder
        )
        lightning_model.save_hyperparameters(
            {
                "batch_size": args.batch_size,
                "version_name": (
                    args.version_name or lightning_model.hparams.version_name  # type: ignore
                ),
                "learning_rate": args.learning_rate,
            }
        )
        print(lightning_model.hparams)
    else:
        lightning_model = PlContactsClassifier(
            **{
                **vars(args),
                "encoder": encoder,
            }
        )

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

    lightning_logger = TensorBoardLogger(
        save_dir="", version=lightning_model.hparams.version_name  # type: ignore
    )

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=[ModelSummary(max_depth=-1), checkpoint_callback],
        logger=lightning_logger,
    )
    trainer.fit(
        model=lightning_model, datamodule=data_module, ckpt_path=checkpoint_path
    )
