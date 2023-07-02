from torch.utils.data import Dataset
import pandas as pd

from model import ContactEncoder


class NameDataset(Dataset):
    def __init__(self, csv_file, char_to_int, debug=False):
        # Load the dataset
        self.data = pd.read_csv(csv_file)
        self.data[
            ["first_name_1", "last_name_1", "first_name_2", "last_name_2"]
        ] = self.data[
            ["first_name_1", "last_name_1", "first_name_2", "last_name_2"]
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
        name1_tensor, len1 = ContactEncoder.preprocess_names(
            row["first_name_1"], row["last_name_1"], self.char_to_int
        )
        name2_tensor, len2 = ContactEncoder.preprocess_names(
            row["first_name_2"], row["last_name_2"], self.char_to_int
        )

        # Get label
        label = row["label"]

        if self.debug:
            return (
                (name1_tensor, len1),
                (name2_tensor, len2),
                label,
                (row["first_name_1"], row["last_name_1"]),
                (row["first_name_2"], row["last_name_2"]),
            )
        else:
            return (name1_tensor, len1), (name2_tensor, len2), label
