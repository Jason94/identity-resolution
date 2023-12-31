from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import pandas as pd
import numpy as np
import csv
import os
import logging
import string
import sys
import json


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


PAD_CHARACTER = "\0"


def update_label(df1: pd.DataFrame, df2: pd.DataFrame, fields: List[str]):
    """
    Updates the 'label' value in df1 with the 'label' value in df2
    for rows where all columns in 'fields' list match in both dataframes.

    Parameters:
    df1 (pandas.DataFrame): The dataframe to be updated.
    df2 (pandas.DataFrame): The dataframe that contains the new label values.
    fields (list of str): The fields to be matched in both dataframes.

    Returns:
    pandas.DataFrame: The updated dataframe.
    """
    merged = pd.merge(df1, df2, on=fields, how="left", suffixes=("", "_y"))
    df1["label"] = np.where(
        pd.notna(merged["label_y"]), merged["label_y"], df1["label"]
    )
    return df1


def create_char_tokenizer() -> Tuple[Dict[str, int], List[str]]:
    # Create a list of all ASCII printable characters.
    vocabulary = list(string.printable) + [PAD_CHARACTER]

    # Create a dictionary that maps each character to a unique integer.
    tokenizer = {char: i for i, char in enumerate(vocabulary)}

    return tokenizer, vocabulary


@dataclass
class Field:
    field: str
    subfield_labels: List[str]
    max_length: int

    def to_string(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_string(cls, s: str) -> "Field":
        return cls(**json.loads(s))


CompositeNameField = Field(
    field="name",
    subfield_labels=["first_name", "last_name"],
    max_length=50,
)

EmailField = Field(field="email", subfield_labels=["email"], max_length=35)

PhoneField = Field(field="phone", subfield_labels=["phone"], max_length=10)

StateField = Field(field="state", subfield_labels=["state"], max_length=2)

ZipField = Field(field="zip", subfield_labels=["zip"], max_length=5)

ALL_FIELDS = [CompositeNameField, EmailField, PhoneField, StateField, ZipField]


def lookup_field(name: str) -> Field:
    for f in ALL_FIELDS:
        if f.field == name:
            return f

    raise NotImplementedError(f"Unrecognized field {name}")


def smart_parse_field(field: str) -> Field:
    """Detect if the field is a serialized field or a field name, and return field object."""
    if "{" in field:
        return Field.from_string(field)
    else:
        return lookup_field(field)


class ContactDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
        fields: List[Field],
        return_field_values: bool = False,
        return_record: bool = False,
    ):
        if return_field_values and return_record:
            raise ValueError("Cannot return field values and record")

        self.data = data
        self.fields = fields
        self.return_field_values = return_field_values
        self.return_record = return_record

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """Return (list of tokens 1, list of lengths 1, list of tokens 2, list of lengths 2, labels)
           for each field idx.

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

        if self.return_field_values:
            return (
                tokens1,
                lengths1,
                tokens2,
                lengths2,
                label,
                self.get_field_values(idx),
            )
        elif self.return_record:
            return (
                tokens1,
                lengths1,
                tokens2,
                lengths2,
                label,
                record,
            )

        else:
            return tokens1, lengths1, tokens2, lengths2, label

    def get_field_values(self, idx: int) -> dict:
        field_values = {}

        for f in self.fields:
            for i in ["1", "2"]:
                field_values[f.field + i] = self.data[idx][f.field + i]

        return field_values


class ContactDataModule(pl.LightningDataModule):
    @staticmethod
    def is_valid(vocabulary: List[str], string: str) -> bool:
        # Remove non-ascii entries
        for c in string:
            if c not in vocabulary or c == PAD_CHARACTER:
                return False

        return True

    @staticmethod
    def valid_row(vocabulary: List[str], row: dict) -> bool:
        for s in row.values():
            if not ContactDataModule.is_valid(vocabulary, s):
                return False

        return True

    @staticmethod
    def _read_data(
        vocabulary: List[str], filename: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Returns:
            Tuple[List[Dict], List[Dict]]: (List of valid rows, list of invalivd rows)
        """
        with open(filename, "r", encoding="utf8") as file:
            reader = csv.DictReader(file)
            valid: List[dict] = []
            invalid: List[dict] = []
            for row in reader:
                if ContactDataModule.valid_row(vocabulary, row):
                    valid.append(row)
                else:
                    invalid.append(row)

        return valid, invalid

    @staticmethod
    def _preprocess_field_text(field_text: str) -> str:
        return field_text.lower()

    @staticmethod
    def field_tokenizer(
        field: Field, tokenizer: Dict[str, int]
    ) -> Callable[[dict], pd.Series]:
        def _tokenize(row: dict) -> pd.Series:
            data = []
            indices = []
            max_length = field.max_length
            pad_token = tokenizer[PAD_CHARACTER]

            for i in ["1", "2"]:
                full_text = " ".join(
                    [row[subfield + i] for subfield in field.subfield_labels]
                )
                full_text = ContactDataModule._preprocess_field_text(full_text)

                full_text = full_text[: field.max_length]
                non_pad_length = len(full_text)
                padding_length = max_length - non_pad_length
                tokens = [tokenizer[c] for c in full_text]
                tokens.extend([pad_token] * padding_length)

                data.extend([tokens, non_pad_length])
                indices.extend([f"{field.field}_tokens{i}", f"{field.field}_length{i}"])

            return pd.Series(data, index=indices)

        return _tokenize

    def __init__(
        self,
        data_dir: str = "data",
        data_lists: List[str] = [],
        prepared_file: str = "prepared_data.csv",
        train_file: str = "prepared_train_data.csv",
        val_file: str = "prepared_val_data.csv",
        corrections_file: Optional[str] = None,
        batch_size: int = 16,
        fields: List[Field] = ALL_FIELDS,
        balance_classes: bool = True,
        preserve_text_fields: bool = True,
        p_validation: float = 0.2,
        return_eval_fields: bool = False,
        return_predict_fields: bool = False,
        return_predict_record: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_lists = data_lists
        self.prepared_file = prepared_file
        self.train_file = train_file
        self.val_file = val_file
        self.corrections_file = corrections_file
        self.batch_size = batch_size
        self.fields = fields
        self.balance_classes = balance_classes
        self.tokenize, self.vocabulary = create_char_tokenizer()
        self.preserve_text_fields = preserve_text_fields
        self.p_validation = p_validation
        self.return_eval_fields = return_eval_fields
        self.return_predict_fields = return_predict_fields
        self.return_predict_record = return_predict_record

        self._val_dataloader = None

    def prepare_data(
        self,
        overwrite: bool = False,
        overwrite_train_val: bool = False,
        shuffle: bool = False,
        invalid_rows_filename: Optional[str] = None,
    ) -> None:
        writepath = os.path.join(self.data_dir, self.prepared_file)
        write_prepared_file = not os.path.exists(writepath) or overwrite

        trainpath = os.path.join(self.data_dir, self.train_file)
        valpath = os.path.join(self.data_dir, self.val_file)
        write_train_val = (
            not os.path.exists(valpath)
            or not os.path.exists(trainpath)
            or overwrite_train_val
        )

        if write_prepared_file:
            logger.info("Assembling prepared data.")

            logger.info(f"Preparing {len(self.data_lists)} lists")
            valid_data = []
            invalid_data = []
            for list in self.data_lists:
                filename = os.path.join(self.data_dir, list)
                valid_file_data, invalid_file_data = ContactDataModule._read_data(
                    self.vocabulary, filename
                )
                logger.info(f"Found {len(valid_file_data)} valid rows in {list}.csv")
                valid_data.extend(valid_file_data)
                if invalid_rows_filename:
                    invalid_data.extend(invalid_file_data)

            df = pd.DataFrame(valid_data, dtype="string").astype({"label": int})
            if invalid_rows_filename:
                invalid_df = pd.DataFrame(invalid_data, dtype="string").astype(
                    {"label": int}
                )
            else:
                invalid_df = None

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
                df = df.join(
                    df.apply(
                        ContactDataModule.field_tokenizer(field, self.tokenize), axis=1
                    )
                )

                # Convert to a '|' delimited list for saving
                for i in ["1", "2"]:
                    col = f"{field.field}_tokens{i}"
                    df[col] = df[col].map(
                        lambda tokens: "|".join([str(t) for t in tokens])
                    )

                if self.preserve_text_fields:
                    for i in ["1", "2"]:
                        if len(field.subfield_labels) > 1:
                            indexed_subfields = [sf + i for sf in field.subfield_labels]
                            df[field.field + i] = df[indexed_subfields].apply(
                                lambda x: " ".join(x), axis=1
                            )
                        df[field.field + i] = df[field.field + i].str.slice(
                            0, field.max_length
                        )

                # If we aren't keeping the text fields, or if we've already combined them, drop.
                if not self.preserve_text_fields or len(field.subfield_labels) > 1:
                    for i in ["1", "2"]:
                        df = df.drop(columns=[sf + i for sf in field.subfield_labels])
        else:
            logger.info(f"Loading existing prepared data from {writepath}")
            df = pd.read_csv(writepath, keep_default_na=False)
            invalid_df = None

        if self.corrections_file is not None and (
            write_prepared_file or write_train_val
        ):
            logger.info(f"Loading corrections from {self.corrections_file}")
            corrections = pd.read_csv(
                os.path.join(self.data_dir, self.corrections_file),
                keep_default_na=False,
            )
            df = update_label(
                df,
                corrections,
                [f.field + "1" for f in self.fields]
                + [f.field + "2" for f in self.fields],
            )

        if write_prepared_file:
            df.to_csv(writepath, index=False)
            logger.info(f"Wrote prepared data to {self.prepared_file}")

            if invalid_df is not None and invalid_rows_filename:
                invalid_df.to_csv(invalid_rows_filename, index=False)
                logger.info(f"Wrote invalid rows to {invalid_rows_filename}")

        if write_train_val or write_prepared_file:
            if shuffle:
                logger.info("Randomizing prepared data")
                df = df.sample(frac=1)

            df_val = df.sample(frac=self.p_validation)
            df_train = df.drop(df_val.index)

            df_train.to_csv(os.path.join(self.data_dir, self.train_file), index=False)
            logger.info(f"Wrote prepared training data to {self.train_file}")
            df_val.to_csv(os.path.join(self.data_dir, self.val_file), index=False)
            logger.info(f"Wrote prepared validation data to {self.val_file}")
        else:
            logger.info(
                f"Loading existing training and validation data from {self.train_file}"
                f" and {self.val_file}"
            )

    def _read_prepared_data(self, filepath: str, **dataset_args) -> ContactDataset:
        df = pd.read_csv(filepath, keep_default_na=False)
        data = df.to_dict(orient="records")

        for row in data:
            for f in self.fields:
                for i in ["1", "2"]:
                    column = f"{f.field}_tokens{i}"
                    row[column] = torch.tensor([int(t) for t in row[column].split("|")])

        return ContactDataset(data, self.fields, **dataset_args)

    def setup(
        self,
        stage: str,
    ) -> None:
        if stage == "fit" or stage == "validate":
            if stage == "fit":
                self.train_dataset = self._read_prepared_data(
                    os.path.join(self.data_dir, self.train_file)
                )

            self.val_dataset = self._read_prepared_data(
                os.path.join(self.data_dir, self.val_file),
                return_field_values=self.return_eval_fields,
            )
        elif stage == "predict":
            self.predict_dataset = self._read_prepared_data(
                os.path.join(self.data_dir, self.val_file),
                return_field_values=self.return_predict_fields,
                return_record=self.return_predict_record,
            )
        else:
            raise NotImplementedError(f"Have not implemented data stage {stage}")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(
        self,
        memoize: bool = False,
    ) -> EVAL_DATALOADERS:
        if not self._val_dataloader or not memoize:
            self._val_dataloader = DataLoader(
                self.val_dataset, batch_size=self.batch_size
            )
        return self._val_dataloader

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        (
            token_tensors1,
            length_tensors1,
            token_tensors2,
            length_tensors2,
            labels,
            *rem,
        ) = batch
        return (
            [t.to(device) for t in token_tensors1],
            [t.to(device) for t in length_tensors1],
            [t.to(device) for t in token_tensors2],
            [t.to(device) for t in length_tensors2],
            labels.to(device),
            *rem,
        )


class ContactSingletonDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
        fields: List[Field],
        return_field_values: bool = False,
        return_record: bool = False,
    ):
        if return_field_values and return_record:
            raise ValueError("Cannot return field values and full record.")

        self.data = data
        self.fields = fields
        self.return_field_values = return_field_values
        self.return_record = return_record

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """Return (list of tokens 1, list of lengths 1) for each field idx."""
        tokens = []
        lengths = []
        record = self.data[idx]

        for f in self.fields:
            tokens.append(record[f"{f.field}_tokens"])
            lengths.append(record[f"{f.field}_length"])

        # TODO: Clean this up a little. Probably have an enum for what retrun you want.
        if self.return_field_values:
            return (
                tokens,
                lengths,
                self.get_field_values(idx),
            )
        elif self.return_record:
            return (tokens, lengths, record)
        else:
            return tokens, lengths

    def get_field_values(self, idx: int) -> dict:
        field_values = {}

        for f in self.fields:
            field_values[f.field] = self.data[idx][f.field]

        return field_values


class ContactSingletonDataModule(pl.LightningDataModule):
    @staticmethod
    def field_tokenizer(
        field: Field, tokenizer: Dict[str, int]
    ) -> Callable[[dict], pd.Series]:
        def _tokenize(row: dict) -> pd.Series:
            data = []
            indices = []
            full_text = " ".join([row[subfield] for subfield in field.subfield_labels])
            full_text = ContactDataModule._preprocess_field_text(full_text)
            non_pad_length = len(full_text)

            # Truncate or pad to max_length
            full_text = full_text[: field.max_length].ljust(
                field.max_length, PAD_CHARACTER
            )

            # Convert to a list of character tokens
            tokens = [tokenizer[c] for c in full_text]

            data.extend([tokens, non_pad_length])
            indices.extend([f"{field.field}_tokens", f"{field.field}_length"])

            return pd.Series(data, index=indices)

        return _tokenize

    def __init__(
        self,
        data_dir: str = "data",
        data_lists: List[str] = [],
        prepared_file: str = "prepared_data.csv",
        batch_size: int = 16,
        fields: List[Field] = ALL_FIELDS,
        preserve_text_fields: bool = True,
        return_record: bool = False,
        return_eval_fields: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_lists = data_lists
        self.prepared_file = prepared_file
        self.batch_size = batch_size
        self.fields = fields
        self.tokenize, self.vocabulary = create_char_tokenizer()
        self.preserve_text_fields = preserve_text_fields
        self.return_record = return_record
        self.return_eval_fields = return_eval_fields

        self._val_dataloader = None

    def prepare_data(
        self,
        overwrite: bool = False,
        overwrite_train_val: bool = False,
        shuffle: bool = False,
        invalid_rows_filename: Optional[str] = None,
    ) -> None:
        writepath = os.path.join(self.data_dir, self.prepared_file)
        write_prepared_file = not os.path.exists(writepath) or overwrite

        if write_prepared_file:
            logger.info("Assembling prepared data.")

            logger.info(f"Preparing {len(self.data_lists)} lists")
            data = []
            invalid_data = []
            for list in self.data_lists:
                filename = os.path.join(self.data_dir, list)
                file_data, invalid_file_data = ContactDataModule._read_data(
                    self.vocabulary, filename
                )
                logger.info(f"Found {len(file_data)} valid rows in {list}.csv")
                data.extend(file_data)
                if invalid_rows_filename:
                    invalid_data.extend(invalid_file_data)

            if len(data) == 0:
                logger.info("No valid rows found. Exiting.")
                sys.exit()

            df = pd.DataFrame(data, dtype="string")
            if invalid_rows_filename:
                invalid_df = pd.DataFrame(invalid_data, dtype="string")
            else:
                invalid_df = None

            logger.info(f"Saving {len(df)} rows.")

            for field in self.fields:
                logger.info(f"Tokenizing {field.field} field")
                df = df.join(
                    df.apply(
                        ContactSingletonDataModule.field_tokenizer(
                            field, self.tokenize
                        ),
                        axis=1,
                    )
                )

                # Convert to a '|' delimited list for saving
                col = f"{field.field}_tokens"
                df[col] = df[col].map(lambda tokens: "|".join([str(t) for t in tokens]))

                if self.preserve_text_fields or self.return_record:
                    if len(field.subfield_labels) > 1:
                        indexed_subfields = [sf for sf in field.subfield_labels]
                        df[field.field] = df[indexed_subfields].apply(
                            lambda x: " ".join(x), axis=1
                        )
                    df[field.field] = df[field.field].str.slice(0, field.max_length)

                # If we aren't keeping the text fields, or if we've already combined them, drop.
                if not self.return_record and (
                    not self.preserve_text_fields or len(field.subfield_labels) > 1
                ):
                    df = df.drop(columns=[sf for sf in field.subfield_labels])
        else:
            logger.info(f"Loading existing prepared data from {writepath}")
            df = pd.read_csv(writepath, keep_default_na=False)
            invalid_df = None

        if write_prepared_file:
            df.to_csv(writepath, index=False)
            logger.info(f"Wrote prepared data to {self.prepared_file}")

            if invalid_df is not None and invalid_rows_filename:
                invalid_df.to_csv(invalid_rows_filename, index=False)
                logger.info(f"Wrote invalid rows to {invalid_rows_filename}")

    def _read_prepared_data(
        self, filepath: str, **dataset_args
    ) -> ContactSingletonDataset:
        df = pd.read_csv(filepath, keep_default_na=False)
        data = df.to_dict(orient="records")

        for row in data:
            for f in self.fields:
                column = f"{f.field}_tokens"
                row[column] = torch.tensor([int(t) for t in row[column].split("|")])

        return ContactSingletonDataset(data, self.fields, **dataset_args)

    def setup(
        self,
        stage: str,
    ) -> None:
        if stage == "predict":
            self.predict_dataset = self._read_prepared_data(
                os.path.join(self.data_dir, self.prepared_file),
                return_field_values=self.preserve_text_fields
                and not self.return_record,
                return_record=self.return_record,
            )

        else:
            raise NotImplementedError(f"Have not implemented data stage {stage}")

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        (token_tensors, length_tensors, *rem) = batch
        return (
            [t.to(device) for t in token_tensors],
            [t.to(device) for t in length_tensors],
            *rem,
        )
