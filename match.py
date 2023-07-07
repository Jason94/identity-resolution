import os
import sys
from typing import List, Tuple
import torch

from model import ContactEncoder
from train import PlContactEncoder
from config import *
from data import (
    create_char_tokenizer,
    CompositeNameField,
    EmailField,
    ContactDataModule,
    lookup_field,
    Field,
    ALL_FIELDS,
)

if __name__ == "__main__":
    tokenizer, vocabulary = create_char_tokenizer()

    if len(sys.argv) <= 1:
        print([f.subfield_labels for f in ALL_FIELDS])
        sys.exit()

    # Create model instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found device {device}")
    model_fname = model_path("model.ckpt")
    if not os.path.exists(model_fname):
        print("Could not find model.")
        sys.exit()

    pl_model = PlContactEncoder.load_from_checkpoint(
        model_fname, loss_function=ContrastiveLoss, similarity_function=is_duplicate
    )

    model = pl_model.encoder
    model.to(device)
    model.eval()

    fields = [lookup_field(f) for f in pl_model.hparams_initial.fields]  # type: ignore

    i = 1
    data1 = {}
    data2 = {}

    for f in fields:
        for sl in f.subfield_labels:
            data1[sl] = sys.argv[i].lower()
            i += 1
    for f in fields:
        for sl in f.subfield_labels:
            data2[sl] = sys.argv[i].lower()
            i += 1

    print(f"Matching '{data1}' & '{data2}'")

    record = {}

    tokens1 = []
    tokens2 = []
    for f in fields:
        for sl in f.subfield_labels:
            record[sl + "1"] = data1[sl]
    for f in fields:
        for sl in f.subfield_labels:
            record[sl + "2"] = data2[sl]

    tokenized_name = ContactDataModule.field_tokenizer(CompositeNameField, tokenizer)(
        record
    )
    tokenized_email = ContactDataModule.field_tokenizer(EmailField, tokenizer)(record)

    def convert_series_to_tensors(
        token_series, f: Field
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_tensors = []
        length_tensors = []

        for i in range(1, 3):
            token_tensors.append(torch.tensor(token_series[f"{f.field}_tokens{i}"]))
            length_tensors.append(torch.tensor([token_series[f"{f.field}_length{i}"]]))

        return torch.stack(token_tensors), torch.cat(length_tensors)

    token_tensors = []
    length_tensors = []

    for f in fields:
        tokens, lengths = convert_series_to_tensors(
            ContactDataModule.field_tokenizer(f, tokenizer)(record), f
        )
        token_tensors.append(tokens)
        length_tensors.append(lengths)

    token_tensors = [t.to(device) for t in token_tensors]
    length_tensors = [t.to(device) for t in length_tensors]

    embeddings = model.forward(token_tensors, length_tensors)

    print(embeddings)

    matches, dist = pl_model.similarity_function(embeddings[0], embeddings[1])

    print(f"Distance: {dist:.4f}, Matches: {matches}")
