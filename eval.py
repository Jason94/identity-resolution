from typing import Optional
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import sys
import torch
from tqdm import tqdm
import pandas as pd

from model import ContactEncoder
from config import *
from data import ContactDataModule
from report import create_html_report


def convert_bool_tensor(tensor):
    ones = torch.ones_like(tensor, dtype=torch.float32)
    minus_ones = -1 * ones
    converted_tensor = torch.where(tensor, ones, minus_ones)
    return converted_tensor


def eval_model(
    model,
    device: torch.device,
    data_module: ContactDataModule,
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
    eval_data_loader = data_module.val_dataloader(memoize=True)

    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_dists = []

    all_field_data = []

    with torch.no_grad():
        for batch in tqdm(eval_data_loader, leave=False):
            batch = data_module.transfer_batch_to_device(batch, device, 0)

            # Unpack batch and move to the device
            (tokens1, lengths1, tokens2, lengths2, labels, idx) = batch

            # Forward pass through the model
            output1 = model(tokens1, lengths1)
            output2 = model(tokens2, lengths2)

            # Compute the loss
            loss = criterion(output1, output2, labels)
            total_loss += loss.item()

            pred, dists = similarity(output1, output2)
            pred = convert_bool_tensor(pred)

            # Compute the predictions
            all_labels.append(labels.cpu())
            all_preds.append(pred.cpu())
            all_dists.append(dists.cpu())

            for i in idx:
                all_field_data.append(
                    data_module.val_dataset.get_field_values(i.item())
                )

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_dists = torch.cat(all_dists).numpy()

    all_field_data = pd.DataFrame(all_field_data)

    # Compute the average loss over the entire evaluation dataset
    avg_loss = total_loss / len(eval_data_loader)

    # Compute precision, recall and F1 score
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    if report_filename:
        create_html_report(
            all_preds,
            all_labels,
            all_dists,
            all_field_data,
            report_filename,
            "Report",
        )

    return avg_loss, precision, recall, f1


if __name__ == "__main__":
    data_module = ContactDataModule(batch_size=EVAL_BATCH_SIZE)
    data_module.prepare_data()
    data_module.setup(stage="validate")

    # Create model instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found device {device}")

    model = ContactEncoder(len(data_module.vocabulary))
    if os.path.exists(SAVED_MODEL_CONFIG_FNAME):
        print("Found existing model weights.")
        model.load_state_dict(torch.load(SAVED_MODEL_CONFIG_FNAME))
    else:
        print("No model found. Exiting.")
        sys.exit()
    model.to(device)

    # Define the optimizer and criterion
    criterion = LOSS_FUNCTION(MARGIN)

    eval_loss, precision, recall, f1 = eval_model(
        model,
        device,
        data_module.val_dataloader(),
        criterion,
        SIMILARITY_METRIC(0.5, return_distance=True),
        report_filename="report.html",
    )
    print(
        f"Eval Loss = {eval_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}"
    )
