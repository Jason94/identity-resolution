from typing import Optional
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model import create_char_to_int, ContactEncoder
from config import *
from data import NameDataset
from report import create_html_report


def convert_bool_tensor(tensor):
    ones = torch.ones_like(tensor, dtype=torch.float32)
    minus_ones = -1 * ones
    converted_tensor = torch.where(tensor, ones, minus_ones)
    return converted_tensor


def eval_model(
    model,
    device: torch.device,
    eval_data_loader: DataLoader,
    criterion,
    similarity,
    report_filename: Optional[str] = None,
):
    """
    Evaluate the performance of a model on a given dataset.

    Args:
        model: The model to evaluate.
        device: torch.device to use to run the model.
        eval_data_loader: DataLoader providing the evaluation data.
        criterion: Loss function to use for evaluation.
        similarity: Similarity function to use for evaluation.

    Returns:
        A tuple containing:
            - Average loss over the evaluation dataset.
            - Precision: The proportion of predicted positive observations
                         in the actual positive observations. It is a measure
                         of a classifier's exactness. Low precision indicates
                         a high number of false positives.
            - Recall: The proportion of actual positive observations
                      that are correctly identified. It is a measure of a
                      classifier's completeness. Low recall indicates a high
                      number of false negatives.
            - F1 Score: The weighted average of Precision and Recall. It tries
                        to find the balance between precision and recall.
                        F1 Score reaches its best value at 1 (perfect precision
                        and recall) and worst at 0.
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in eval_data_loader:
            # Unpack batch and move to the device
            (name1_tensor, len1), (name2_tensor, len2), label = batch

            name1_tensor = name1_tensor.to(device)
            name2_tensor = name2_tensor.to(device)
            label = label.to(device)

            # Forward pass through the model
            output1 = model(name1_tensor, len1)
            output2 = model(name2_tensor, len2)

            # Compute the loss
            loss = criterion(output1, output2, label)
            total_loss += loss.item()

            pred = convert_bool_tensor(similarity(output1, output2))

            # Compute the predictions
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(pred.cpu().numpy().flatten().astype(int))

    # Compute the average loss over the entire evaluation dataset
    avg_loss = total_loss / len(eval_data_loader)

    # Compute precision, recall and F1 score
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    if report_filename:
        create_html_report(all_preds, all_labels, report_filename, "Report")

    return avg_loss, precision, recall, f1


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
        print("No model found. Exiting.")
        sys.exit()
    model.to(device)

    # Define the optimizer and criterion
    criterion = LOSS_FUNCTION(MARGIN)

    # Create the DataLoader
    eval_data_loader = DataLoader(
        NameDataset("data/eval.csv", char_to_int),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=True,
    )

    eval_loss, precision, recall, f1 = eval_model(
        model,
        device,
        eval_data_loader,
        criterion,
        SIMILARITY_METRIC(0.5),
        report_filename="report.html",
    )
    print(
        f"Eval Loss = {eval_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}"
    )
