import os
import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model import create_char_to_int, ContactEncoder
from config import *
from data import NameDataset
from contrastive_metric import ContrastiveLoss

if __name__ == "__main__":
    char_to_int, chars = create_char_to_int()

    # Create model instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found device {device}")

    model = ContactEncoder(len(chars))
    if os.path.exists(SAVED_MODEL_PATH):
        print("Found existing model.")
        model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    else:
        print("Could not find model.")
        sys.exit()
    model.to(device)
    model.eval()

    first1 = sys.argv[1].lower()
    last1 = sys.argv[2].lower()
    email1 = sys.argv[3].lower()
    first2 = sys.argv[4].lower()
    last2 = sys.argv[5].lower()
    email2 = sys.argv[6].lower()

    print(f"Matching '{first1} {last1} - {email1}' & '{first2} {last2} - {email2}'")

    tensor1, len1 = NameDataset.preprocess_names(first1, last1, char_to_int)
    tensor2, len2 = NameDataset.preprocess_names(first2, last2, char_to_int)

    tensor_email1, email_len1 = NameDataset.preprocess_emails(email1, char_to_int)
    tensor_email2, email_len2 = NameDataset.preprocess_emails(email2, char_to_int)

    name_tensor = torch.stack([tensor1, tensor2]).to(device)
    len_tensor = torch.tensor([len1, len2])

    email_tensor = torch.stack([tensor_email1, tensor_email2]).to(device)
    email_len_tensor = torch.tensor([email_len1, email_len2])

    print(name_tensor)

    embeddings = model.forward(name_tensor, len_tensor, email_tensor, email_len_tensor)

    print(embeddings)

    matches, dist = is_duplicate(threshold=SIMILARITY_THRESHOLD, return_distance=True)(
        embeddings[0], embeddings[1]
    )

    print(f"Distance: {dist:.4f}, Matches: {matches}")
