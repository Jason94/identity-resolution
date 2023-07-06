import os
import sys
import torch

from model import ContactEncoder
from train import PlContactEncoder
from config import *
from data import (
    create_char_tokenizer,
    CompositeNameField,
    EmailField,
    ContactDataModule,
)

if __name__ == "__main__":
    tokenizer, vocabulary = create_char_tokenizer()

    # Create model instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found device {device}")
    model_fname = model_path("model.ckpt")
    if not os.path.exists(model_fname):
        print("Could not find model.")
        sys.exit()

    pl_model = PlContactEncoder.load_from_checkpoint(
        model_fname, encoder=ContactEncoder(len(vocabulary))
    )

    model = pl_model.encoder
    model.to(device)
    model.eval()

    first1 = sys.argv[1].lower()
    last1 = sys.argv[2].lower()
    email1 = sys.argv[3].lower()
    first2 = sys.argv[4].lower()
    last2 = sys.argv[5].lower()
    email2 = sys.argv[6].lower()

    print(f"Matching '{first1} {last1} - {email1}' & '{first2} {last2} - {email2}'")

    record = {
        f"{CompositeNameField.subfield_labels[0]}1": first1,
        f"{CompositeNameField.subfield_labels[1]}1": last1,
        f"{EmailField.subfield_labels[0]}1": email1,
        f"{CompositeNameField.subfield_labels[0]}2": first2,
        f"{CompositeNameField.subfield_labels[1]}2": last2,
        f"{EmailField.subfield_labels[0]}2": email2,
    }

    tokenized_name = ContactDataModule.field_tokenizer(CompositeNameField, tokenizer)(
        record
    )
    tokenized_email = ContactDataModule.field_tokenizer(EmailField, tokenizer)(record)

    def convert_series_to_tensors(tokenized_name, tokenized_email):
        name_token_tensors = []
        email_token_tensors = []
        name_length_tensors = []
        email_length_tensors = []

        for i in range(1, 3):  # iterate over the two groups of tokens and lengths
            # convert the name and email tokens and lengths to tensors
            name_token_tensors.append(torch.tensor(tokenized_name[f"name_tokens{i}"]))
            name_length_tensors.append(
                torch.tensor([tokenized_name[f"name_length{i}"]])
            )

            email_token_tensors.append(
                torch.tensor(tokenized_email[f"email_tokens{i}"])
            )
            email_length_tensors.append(
                torch.tensor([tokenized_email[f"email_length{i}"]])
            )

        field_tensors = [
            torch.stack(name_token_tensors),
            torch.stack(email_token_tensors),
        ]

        length_tensors = [
            torch.cat(name_length_tensors),
            torch.cat(email_length_tensors),
        ]

        return field_tensors, length_tensors

    field_tensors, length_tensors = convert_series_to_tensors(
        tokenized_name, tokenized_email
    )

    field_tensors = [t.to(device) for t in field_tensors]
    length_tensors = [t.to(device) for t in length_tensors]

    embeddings = model.forward(field_tensors, length_tensors)

    print(embeddings)

    matches, dist = is_duplicate(threshold=SIMILARITY_THRESHOLD, return_distance=True)(
        embeddings[0], embeddings[1]
    )

    print(f"Distance: {dist:.4f}, Matches: {matches}")
