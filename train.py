import os
from typing import List
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

N_EPOCHS = 10
TRAIN_BATCH_SIZE = 8
LEARNING_RATE = 0.00005

CHECKPOINT_PERIOD = 5


def get_training_config(optimizer, criterion):
    config = {
        "n_epochs": N_EPOCHS,
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


def save_configs(
    epoch, model, input_data, optimizer, criterion, eval_loss, precision, recall, f1
):
    # model_config = summary(model, input_data=input_data)
    model_config = ""
    training_config = get_training_config(optimizer, criterion)
    eval_config = get_eval_config(eval_loss, precision, recall, f1)
    with open(f"{SAVED_MODEL_DIR}/config_chkpt_{epoch}.json", "w") as f:
        json.dump(
            {
                "model_config": model_config,
                "training_config": training_config,
                "eval_config": eval_config,
            },
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
):
    eval_loss, precision, recall, f1 = eval_model(
        model, device, eval_data_loader, criterion, SIMILARITY_METRIC(0.5)
    )
    torch.save(model.state_dict(), f"{SAVED_MODEL_DIR}/chkpt_0.pth")
    save_configs(
        0,
        model,
        eval_data_loader.dataset[0],
        optimizer,
        criterion,
        eval_loss,
        precision,
        recall,
        f1,
    )
    print(
        f"Preliminary: Eval Loss = {eval_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}"
    )

    for epoch in range(n_epochs):
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
            model, device, eval_data_loader, criterion, SIMILARITY_METRIC(0.5)
        )
        tqdm.write(
            f"Epoch {epoch+1} / {n_epochs}: Avg Train Loss = {avg_train_loss:.4f}; Eval Loss = {eval_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}"
        )

        if (epoch + 1) % CHECKPOINT_PERIOD == 0:
            torch.save(model.state_dict(), f"{SAVED_MODEL_DIR}/chkpt_{epoch+1}.pth")
            save_configs(
                epoch + 1,
                model,
                eval_data_loader.dataset[0],
                optimizer,
                criterion,
                eval_loss,
                precision,
                recall,
                f1,
            )


if __name__ == "__main__":
    char_to_int, chars = create_char_to_int()

    # Create model instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found device {device}")

    model = ContactEncoder(len(chars))
    if os.path.exists(SAVED_MODEL_PATH):
        print("Found existing model weights. Starting from there.")
        model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    else:
        print("Initializing random weights.")
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
    train_model(model, data_loader, eval_data_loader, optimizer, criterion, device)

    print("Saving model weights")
    torch.save(model.state_dict(), SAVED_MODEL_PATH)
