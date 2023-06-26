import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import create_char_to_int, ContactEncoder
from eval import eval_model
from config import *
from data import NameDataset

N_EPOCHS = 15
TRAIN_BATCH_SIZE = 32
LEARNING_RATE = 0.1


def train_model(
    model,
    data_loader: DataLoader,
    eval_data_loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device = torch.device("cpu"),
    n_epochs=N_EPOCHS,
):
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        for (name1_tensor, len1), (name2_tensor, len2), label in tqdm(data_loader):
            # Move tensors to the device
            name1_tensor = name1_tensor.to(device)
            name2_tensor = name2_tensor.to(device)
            label = label.to(device)

            # Forward pass through the model
            output1 = model(name1_tensor, len1)
            output2 = model(name2_tensor, len2)

            # Calculate loss
            loss = criterion(output1, output2, label.float())
            total_loss += loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model
        avg_train_loss = total_loss / len(data_loader)
        eval_loss, precision, recall, f1 = eval_model(
            model, device, eval_data_loader, criterion, SIMILARITY_METRIC()
        )
        print(
            f"Epoch {epoch}: Avg Train Loss = {avg_train_loss:.4f}; Eval Loss = {eval_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}"
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
    criterion = nn.CosineEmbeddingLoss()

    # Create the DataLoader
    data_loader = DataLoader(
        NameDataset("data/training.csv", char_to_int),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )
    eval_data_loader = DataLoader(
        NameDataset("data/eval.csv", char_to_int),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    # Train the model
    train_model(model, data_loader, eval_data_loader, optimizer, criterion, device)

    print("Saving model weights")
    torch.save(model.state_dict(), SAVED_MODEL_PATH)
