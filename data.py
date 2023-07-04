import torch
from torch.utils.data import Dataset
import pandas as pd

from config import MAX_INPUT_LENGTH, MAX_EMAIL_LENGTH
from model import ContactEncoder


class NameDataset(Dataset):
    @staticmethod
    def preprocess_names(
        first_name: str, last_name: str, char_to_int, max_len=MAX_INPUT_LENGTH
    ):
        # Concatenate first name and last name
        try:
            name = first_name + " " + last_name
        except Exception as e:
            print(f"Error processing '{first_name}' & '{last_name}'")
            raise e

        # Truncate or pad name to max_len
        name = name[:max_len].ljust(max_len, ContactEncoder.PAD_CHARACTER)

        # Convert name to tensor of character indices
        name_tensor = torch.tensor([char_to_int[char] for char in name])

        # Count the number of non-pad characters
        non_pad_count = (
            (name_tensor != char_to_int[ContactEncoder.PAD_CHARACTER]).sum().item()
        )

        # Return name tensor and its length
        return name_tensor, non_pad_count

    @staticmethod
    def preprocess_emails(email: str, char_to_int, max_len=MAX_EMAIL_LENGTH):
        # Count the number of non-pad characters
        non_pad_count = len(email)

        # Truncate or pad emali to max_len
        email = email[:max_len].ljust(max_len, ContactEncoder.PAD_CHARACTER)

        # Convert email to tensor of character indices
        email_tensor = torch.tensor([char_to_int[char] for char in email])

        return email_tensor, non_pad_count

    def __init__(self, csv_file, char_to_int, debug=False):
        # Load the dataset
        self.data = pd.read_csv(csv_file)
        self.data[
            [
                "first_name_1",
                "last_name_1",
                "email1",
                "first_name_2",
                "last_name_2",
                "email2",
            ]
        ] = self.data[
            [
                "first_name_1",
                "last_name_1",
                "email1",
                "first_name_2",
                "last_name_2",
                "email2",
            ]
        ].astype(
            str
        )
        self.debug = debug

        # Initialize the character-to-integer mapping
        self.char_to_int = char_to_int

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Get names and preprocess them
        name1_tensor, len1 = NameDataset.preprocess_names(
            row["first_name_1"], row["last_name_1"], self.char_to_int
        )
        name2_tensor, len2 = NameDataset.preprocess_names(
            row["first_name_2"], row["last_name_2"], self.char_to_int
        )

        # Get emails and preprocess them
        email1_tensor, email_len1 = NameDataset.preprocess_emails(
            row["email1"], self.char_to_int
        )
        email2_tensor, email_len2 = NameDataset.preprocess_emails(
            row["email2"], self.char_to_int
        )

        # Get label
        label = row["label"]

        if self.debug:
            return (
                (name1_tensor, len1, email1_tensor, email_len1),
                (name2_tensor, len2, email2_tensor, email_len2),
                label,
                (row["first_name_1"], row["last_name_1"], row["email1"]),
                (row["first_name_2"], row["last_name_2"], row["email2"]),
            )
        else:
            return (
                (name1_tensor, len1, email1_tensor, email_len1),
                (name2_tensor, len2, email2_tensor, email_len2),
                label,
            )
