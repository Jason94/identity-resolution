from typing import Any, Callable, Optional, Tuple
import torch
from torch import optim
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score

from model import ContactEncoder
from config import *
from data import ContactDataModule

N_EPOCHS = 10
TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 0.00005


def convert_bool_tensor(tensor):
    ones = torch.ones_like(tensor, dtype=torch.float32)
    minus_ones = -1 * ones
    converted_tensor = torch.where(tensor, ones, minus_ones)
    return converted_tensor


class PlContactEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder: ContactEncoder,
        loss_function: Optional[
            Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
        ] = None,
        similarity_function: Optional[
            Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        lr: float = 1e-5,
    ):
        super().__init__()
        self.encoder = encoder
        self.loss_function = loss_function
        self.similarity_function = similarity_function
        self.lr = lr

        # --- Evaluation Performance Data
        self.validation_labels = []
        self.validation_preds = []

    def training_step(self, batch, batch_idx):
        if not self.loss_function:
            raise RuntimeError("Not configured for training!")

        (tokens1, lengths1, tokens2, lengths2, labels) = batch

        # Forward pass through the model
        output1 = self.encoder(tokens1, lengths1)
        output2 = self.encoder(tokens2, lengths2)

        # Calculate loss
        loss = self.loss_function(output1, output2, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        if not self.loss_function or not self.similarity_function:
            raise RuntimeError("Not configured for evaluating!")

        (tokens1, lengths1, tokens2, lengths2, labels) = batch

        # Forward pass through the model
        output1 = self.encoder(tokens1, lengths1)
        output2 = self.encoder(tokens2, lengths2)

        # Compute the loss
        loss = self.loss_function(output1, output2, labels)

        pred_raw, dists = self.similarity_function(output1, output2)
        pred = convert_bool_tensor(pred_raw)

        # Log
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.validation_labels.append(labels.cpu())
        self.validation_preds.append(pred.cpu())

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
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    # Load the data
    data_module = ContactDataModule(batch_size=TRAIN_BATCH_SIZE)

    # Create model instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found device {device}")

    model = ContactEncoder(len(data_module.vocabulary), p_dropout=0.0)
    print("Initializing random weights.")
    model.to(device)

    criterion = LOSS_FUNCTION(margin=MARGIN)

    lightning_model = PlContactEncoder(
        model,
        criterion,
        SIMILARITY_METRIC(0.5, return_distance=True),
        LEARNING_RATE,
    )
    lightning_model.to(device)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1, filename="{epoch:02d}-{val_f1:.4f}", every_n_epochs=2
    )

    trainer = pl.Trainer(
        max_epochs=N_EPOCHS, callbacks=[ModelSummary(max_depth=-1), checkpoint_callback]
    )
    trainer.fit(model=lightning_model, datamodule=data_module)
