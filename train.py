import os
import torch
from torch import nn, optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import create_char_to_int, ContactEncoder

### ---Hyperparameters

## Training
N_EPOCHS = 1
SAVED_MODEL_PATH = "models/model.pth"


class NameDataset(Dataset):
    def __init__(self, csv_file, char_to_int):
        # Load the dataset
        self.data = pd.read_csv(csv_file)
        self.data[
            ["first_name_1", "last_name_1", "first_name_2", "last_name_2"]
        ] = self.data[
            ["first_name_1", "last_name_1", "first_name_2", "last_name_2"]
        ].astype(
            str
        )

        # Initialize the character-to-integer mapping
        self.char_to_int = char_to_int

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Get names and preprocess them
        name1_tensor, len1 = ContactEncoder.preprocess_names(
            row["first_name_1"], row["last_name_1"], self.char_to_int
        )
        name2_tensor, len2 = ContactEncoder.preprocess_names(
            row["first_name_2"], row["last_name_2"], self.char_to_int
        )

        # Get label
        label = row["label"]

        return (name1_tensor, len1), (name2_tensor, len2), label


def train_model(
    model,
    data_loader,
    optimizer,
    criterion,
    device: torch.device = torch.device("cpu"),
    n_epochs=N_EPOCHS,
):
    for epoch in range(n_epochs):
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

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(data_loader)}")


if __name__ == "__main__":
    char_to_int, chars = create_char_to_int()

    # Create model instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found device {device}")

    model = ContactEncoder(len(chars))
    if os.path.exists(SAVED_MODEL_PATH):
        model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    model.to(device)

    # Define the optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CosineEmbeddingLoss()

    # Create the DataLoader
    data_loader = DataLoader(
        NameDataset("data/training.csv", char_to_int), batch_size=32, shuffle=True
    )

    # Train the model
    train_model(model, data_loader, optimizer, criterion, device)

    torch.save(model.state_dict(), SAVED_MODEL_PATH)
