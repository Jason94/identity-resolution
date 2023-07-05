import os
from typing import List, Optional
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
import json

from model import create_char_to_int, ContactEncoder
from eval import eval_model
from config import *
from data import NameDataset

N_EPOCHS = 6
TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 0.00005

CHECKPOINT_PERIOD = 2


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
    saved_model_fname = os.path.join(SAVED_MODEL_DIR, f"{SAVED_MODEL_PATH}.pth")
    saved_model_config_fname = f"config_{SAVED_MODEL_PATH}.json"
    saved_model_config_path = os.path.join(SAVED_MODEL_DIR, saved_model_config_fname)

    char_to_int, chars = create_char_to_int()

    # Create model instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found device {device}")

    model = ContactEncoder(len(chars))
    if os.path.exists(saved_model_fname):
        print("Found existing model weights. Starting from there.")
        model.load_state_dict(torch.load(saved_model_fname))
        with open(saved_model_config_path) as conf:
            model_config = json.load(conf)
    else:
        print("Initializing random weights.")
        model_config = {}
    model.to(device)

    # Define the optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = LOSS_FUNCTION(margin=MARGIN)

    # Create the DataLoader
    data_loader = DataLoader(
        NameDataset("data/training.csv", char_to_int),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )
    eval_data_loader = DataLoader(
        NameDataset("data/eval.csv", char_to_int, debug=True),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    # Train the model
    start_epoch = model_config.get("training_config", {}).get("current_epoch", 0)
    final_config = train_model(
        model,
        data_loader,
        eval_data_loader,
        optimizer,
        criterion,
        device,
        start_epoch=start_epoch,
    )

    print("Saving model weights")
    torch.save(model.state_dict(), saved_model_fname)
    if final_config:
        save_configs(final_config, fname=saved_model_config_fname)
