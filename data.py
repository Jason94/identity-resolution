from dataclasses import dataclass
from typing import Any, Callable, List
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning.pytorch as pl
import pandas as pd
import csv
import os
import logging
import string
import random

from config import MAX_NAME_LENGTH, MAX_EMAIL_LENGTH

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


PAD_CHARACTER = "\0"


def create_char_tokenizer():
    # Create a list of all ASCII printable characters.
    chars = list(string.printable) + [PAD_CHARACTER]

    # Create a dictionary that maps each character to a unique integer.
    char_to_int = {char: i for i, char in enumerate(chars)}

    return char_to_int, chars


@dataclass
class Field:
    field: str
    subfield_labels: List[str]
    max_length: int


CompositeNameField = Field(
    field="name",
    subfield_labels=["first_name", "last_name"],
    max_length=MAX_NAME_LENGTH,
)

EmailField = Field(
    field="email", subfield_labels=["email"], max_length=MAX_EMAIL_LENGTH
)


class ContactDataset(Dataset):
    def __init__(self, data: List[dict], fields: List[Field]):
        self.data = data
        self.fields = fields

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """Return (list of tokens 1, list of lengths 1, list of tokens 2, list of lengths 2, labels) for each field idx.

        The 1 and 2 record are in different tuple elements."""
        tokens1 = []
        lengths1 = []
        tokens2 = []
        lengths2 = []
        record = self.data[idx]

        label = record["label"]

        for f in self.fields:
            tokens1.append(record[f"{f.field}_tokens1"])
            lengths1.append(record[f"{f.field}_length1"])

            tokens2.append(record[f"{f.field}_tokens2"])
            lengths2.append(record[f"{f.field}_length2"])

        return tokens1, lengths1, tokens2, lengths2, torch.tensor([label])


class ContactDataModule(pl.LightningDataModule):
    @staticmethod
    def _is_valid(string: str) -> bool:
        # Remove non-ascii entries
        for c in string:
            if not (0 <= ord(c) <= 127):
                return False

        return True

    @staticmethod
    def _valid_row(row: dict) -> bool:
        for s in row.values():
            if not ContactDataModule._is_valid(s):
                return False

        return True

    @staticmethod
    def _read_data(filename):
        with open(filename, "r", encoding="utf8") as file:
            reader = csv.DictReader(file)
            data = [row for row in reader if ContactDataModule._valid_row(row)]
        return data

    @staticmethod
    def _preprocess_field_text(field_text: str) -> str:
        return field_text.lower()

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 16,
        data_lists: List[str] = ["duplicates", "distincts"],
        fields: List[Field] = [CompositeNameField, EmailField],
        balance_classes: bool = True,
        preserve_text_fields: bool = True,
        p_validation: float = 0.2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_lists = data_lists
        self.fields = fields
        self.balance_classes = balance_classes
        self.tokenize, self.vocabulary = create_char_tokenizer()
        self.preserve_text_fields = preserve_text_fields
        self.p_validation = p_validation

    def _tokenize_field(self, field: Field) -> Callable[[dict], pd.Series]:
        def _tokenize(row: dict) -> pd.Series:
            data = []
            indices = []
            for i in ["1", "2"]:
                full_text = " ".join(
                    [row[subfield + i] for subfield in field.subfield_labels]
                )
                full_text = self._preprocess_field_text(full_text)
                non_pad_length = len(full_text)

                # Truncate or pad to max_length
                full_text = full_text[: field.max_length].ljust(
                    field.max_length, PAD_CHARACTER
                )

                # Convert to a list of character tokens
                tokens = [self.tokenize[c] for c in full_text]

                # Convert to a '|' delimited list for saving
                tokens = "|".join([str(t) for t in tokens])

                data.extend([tokens, non_pad_length])
                indices.extend([f"{field.field}_tokens{i}", f"{field.field}_length{i}"])

            return pd.Series(data, index=indices)

        return _tokenize

    def prepare_data(
        self, writefile: str = "prepared_data.csv", overwrite: bool = False
    ) -> None:
        writepath = os.path.join(self.data_dir, writefile)
        if os.path.exists(writepath) and not overwrite:
            logger.info(f"Prepared data already found at {writepath}.")
            return

        logger.info(f"Preparing {len(self.data_lists)} lists")
        data = []
        for list in self.data_lists:
            filename = os.path.join(self.data_dir, f"{list}.csv")
            file_data = self._read_data(filename)
            logger.info(f"Found {len(file_data)} valid rows in {list}.csv")
            data.extend(file_data)

        df = pd.DataFrame(data, dtype="string").astype({"label": int})

        if self.balance_classes:
            # count the number of instances for each class
            class_counts = df["label"].value_counts()

            # get the class with the least number of instances
            min_class = class_counts.idxmin()
            min_count = class_counts.min()

            logger.info(
                f"Found least represented class {min_class} with {min_count} rows. Balancing."
            )

            # under-sample the other class
            df_balanced = pd.concat(
                [
                    df[df["label"] == label].sample(min_count, random_state=0)
                    for label in class_counts.index
                ]
            )

            df = df_balanced

        logger.info(f"Saving {len(df)} rows.")

        for field in self.fields:
            logger.info(f"Tokenizing {field.field} field")
            df = df.join(df.apply(self._tokenize_field(field), axis=1))

            if not self.preserve_text_fields:
                for i in ["1", "2"]:
                    df = df.drop(columns=[sf + i for sf in field.subfield_labels])

        df.to_csv(writepath)

    def setup(self, stage: str, readfile: str = "prepared_data.csv") -> None:
        readpath = os.path.join(self.data_dir, readfile)
        if stage == "fit" or stage == "validate":
            data = pd.read_csv(readpath).to_dict(orient="records")

            for row in data:
                for f in self.fields:
                    for i in ["1", "2"]:
                        column = f"{f.field}_tokens{i}"
                        row[column] = torch.tensor(
                            [int(t) for t in row[column].split("|")]
                        )

            dataset = ContactDataset(data, self.fields)

            n_validation = int(len(data) * self.p_validation)
            n_training = len(data) - n_validation
            self.train_dataset, self.val_dataset = random_split(
                dataset, [n_training, n_validation]
            )

        else:
            raise NotImplementedError(f"Have not implemented data stage {stage}")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        token_tensors1, length_tensors1, token_tensors2, length_tensors2, labels = batch
        return (
            [t.to(device) for t in token_tensors1],
            [t.to(device) for t in length_tensors1],
            [t.to(device) for t in token_tensors2],
            [t.to(device) for t in length_tensors2],
            labels.to(device),
        )


if __name__ == "__main__":
    logging.basicConfig()
    data_module = ContactDataModule(batch_size=4)
    data_module.prepare_data(overwrite=True)
