import os
from typing import Any, Callable, List, Optional, Tuple
import torch
from torch import optim
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelSummary
from tqdm import tqdm
import json
from sklearn.metrics import precision_score, recall_score, f1_score

from model import ContactEncoder
from eval import eval_model
from config import *
from data import ContactDataModule

N_EPOCHS = 4
TRAIN_BATCH_SIZE = 2
LEARNING_RATE = 0.00005

CHECKPOINT_PERIOD = 2


def convert_bool_tensor(tensor):
    ones = torch.ones_like(tensor, dtype=torch.float32)
    minus_ones = -1 * ones
    converted_tensor = torch.where(tensor, ones, minus_ones)
    return converted_tensor


class PlContactEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder: ContactEncoder,
        loss_function: Callable[[torch.Tensor, torch.Tensor, float], float],
        similarity_function: Callable[
            [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
        ],
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
        (tokens1, lengths1, tokens2, lengths2, labels) = batch

        # Forward pass through the model
        output1 = self.encoder(tokens1, lengths1)
        output2 = self.encoder(tokens2, lengths2)

        # Calculate loss
        loss = self.loss_function(output1, output2, labels)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        (tokens1, lengths1, tokens2, lengths2, labels) = batch

        # Forward pass through the model
        output1 = self.encoder(tokens1, lengths1)
        output2 = self.encoder(tokens2, lengths2)

        # Compute the loss
        loss = criterion(output1, output2, labels)

        pred, dists = self.similarity_function(output1, output2)
        pred = convert_bool_tensor(pred)

        # Log
        self.log("val_loss", loss)
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


def get_training_config(optimizer, criterion, current_epoch: int, n_epochs: int):
    config = {
        "current_epoch": current_epoch,
        "n_epochs": n_epochs,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "margin": MARGIN,
        "optimizer": str(optimizer),
        "criterion": str(criterion),
    }
    return config


def get_eval_config(eval_loss, precision, recall, f1):
    config = {
        "eval_loss": eval_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return config


def package_configs(
    epoch: int,
    n_epochs: int,
    model,
    input_data,
    optimizer,
    criterion,
    eval_loss,
    precision,
    recall,
    f1,
):
    # model_config = summary(model, input_data=input_data)
    model_config = ""
    training_config = get_training_config(optimizer, criterion, epoch, n_epochs)
    eval_config = get_eval_config(eval_loss, precision, recall, f1)
    return {
        "model_config": model_config,
        "training_config": training_config,
        "eval_config": eval_config,
    }


def save_configs(configs, epoch: Optional[int] = None, fname: Optional[str] = None):
    if fname is None and epoch is not None:
        fname = f"config_chkpt_{epoch}.json"
    elif fname is None:
        raise ValueError("Must supply epoch or fname")
    with open(os.path.join(SAVED_MODEL_DIR, fname), "w") as f:
        json.dump(
            configs,
            f,
            indent=4,
        )


def train_model(
    model,
    data_loader: DataLoader,
    eval_data_loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device = torch.device("cpu"),
    n_epochs=N_EPOCHS,
    start_epoch=0,
):
    n_epochs = n_epochs + start_epoch

    if start_epoch == 0:
        eval_loss, precision, recall, f1 = eval_model(
            model,
            device,
            eval_data_loader,
            criterion,
            SIMILARITY_METRIC(0.5, return_distance=True),
        )
        torch.save(model.state_dict(), f"{SAVED_MODEL_DIR}/chkpt_0.pth")
        config = package_configs(
            0,
            n_epochs,
            model,
            eval_data_loader.dataset[0],
            optimizer,
            criterion,
            eval_loss,
            precision,
            recall,
            f1,
        )
        save_configs(config, 0)
        print(
            f"Preliminary: Eval Loss = {eval_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}"
        )

    config = None
    for epoch in range(start_epoch, n_epochs):
        model.train()
        total_loss = 0.0

        for (
            (name1_tensor, len1, email1_tensor, len_email1),
            (name2_tensor, len2, email2_tensor, len_email2),
            label,
        ) in tqdm(data_loader, leave=False):
            # Move tensors to the device
            name1_tensor = name1_tensor.to(device)
            name2_tensor = name2_tensor.to(device)
            email1_tensor = email1_tensor.to(device)
            email2_tensor = email2_tensor.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            # Forward pass through the model
            output1 = model(name1_tensor, len1, email1_tensor, len_email1)
            output2 = model(name2_tensor, len2, email2_tensor, len_email2)

            # Calculate loss
            loss = criterion(output1, output2, label.float())
            total_loss += loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Evaluate the model
        avg_train_loss = total_loss / len(data_loader)
        eval_loss, precision, recall, f1 = eval_model(
            model,
            device,
            eval_data_loader,
            criterion,
            SIMILARITY_METRIC(0.5, return_distance=True),
        )
        tqdm.write(
            f"Epoch {epoch+1} / {n_epochs}: Avg Train Loss = {avg_train_loss:.4f}; Eval Loss = {eval_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}"
        )

        if (epoch + 1) % CHECKPOINT_PERIOD == 0 or (epoch + 1) == n_epochs:
            torch.save(model.state_dict(), f"{SAVED_MODEL_DIR}/chkpt_{epoch+1}.pth")
            config = package_configs(
                epoch + 1,
                n_epochs,
                model,
                eval_data_loader.dataset[0],
                optimizer,
                criterion,
                eval_loss,
                precision,
                recall,
                f1,
            )
            save_configs(config, epoch + 1)

    return config


if __name__ == "__main__":
    # Load the data
    data_module = ContactDataModule(batch_size=TRAIN_BATCH_SIZE)

    # Create model instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found device {device}")

    model = ContactEncoder(len(data_module.vocabulary))
    if os.path.exists(SAVED_MODEL_FNAME):
        print("Found existing model weights. Starting from there.")
        model.load_state_dict(torch.load(SAVED_MODEL_FNAME))
        with open(SAVED_MODEL_CONFIG_PATH) as conf:
            model_config = json.load(conf)
    else:
        print("Initializing random weights.")
        model_config = {}
    model.to(device)

    criterion = LOSS_FUNCTION(margin=MARGIN)

    # lightning_model = PlContactEncoder(
    #     model, criterion, SIMILARITY_METRIC(0.5, return_distance=True), LEARNING_RATE
    # )
    # lightning_model.to(device)

    # trainer = pl.Trainer(max_epochs=N_EPOCHS, callbacks=[ModelSummary(max_depth=-1)])
    # trainer.fit(
    #     model=lightning_model,
    #     datamodule=data_module,
    # )

    # Train the model
    start_epoch = model_config.get("training_config", {}).get("current_epoch", 0)
    final_config = train_model(
        model,
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        optim.Adam(self.parameters(), lr=LEARNING_RATE),
        criterion,
        device,
        start_epoch=start_epoch,
    )

    # print("Saving model weights")
    # torch.save(model.state_dict(), SAVED_MODEL_FNAME)
    # if final_config:
    #     save_configs(final_config, fname=SAVED_MODEL_CONFIG_FNAME)
