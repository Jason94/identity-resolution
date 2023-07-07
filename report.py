from typing import Optional
import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import pandas as pd
from io import BytesIO
import argparse


from model import ContactEncoder
from config import *
from data import ContactDataModule, create_char_tokenizer
from train import PlContactEncoder
from eval import eval_model

REPORT_FILENAME = "report.html"


def embed_matplotlib_figure(fig):
    """Converts a Matplotlib figure to a base64 encoded PNG to be used in HTML reports"""
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{encoded}'/>"


def create_html_report(
    all_preds,
    all_labels,
    all_dists,
    report_df,
    title: str,
):
    report_df["distance"] = all_dists
    report_df["eval_duplicates"] = all_preds
    report_df["label_duplicates"] = all_labels

    # Create bar and pie graph for correct and incorrect predictions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    correct_mask = report_df["label_duplicates"] == report_df["eval_duplicates"]
    incorrect_mask = report_df["label_duplicates"] != report_df["eval_duplicates"]

    correct_counts = report_df.loc[correct_mask, "label_duplicates"].value_counts()
    incorrect_counts = report_df.loc[incorrect_mask, "label_duplicates"].value_counts()

    # Fill in 0 counts for any missing categories
    for cat in [-1, 1]:
        if cat not in correct_counts:
            correct_counts[cat] = 0
        if cat not in incorrect_counts:
            incorrect_counts[cat] = 0

    # Bar plots
    sns.barplot(x=correct_counts.index, y=correct_counts, ax=axes[0, 0])
    sns.barplot(x=incorrect_counts.index, y=incorrect_counts, ax=axes[0, 1])

    axes[0, 0].set_title("Labels: Correct Predictions")
    axes[0, 1].set_title("Labels: Incorrect Predictions")

    for ax in axes[0, :2]:  # Set labels for the first two bar plots
        ax.set_xticklabels(["Distinct", "Duplicate"])

    # Pie charts
    # Calculate correct and incorrect counts for duplicates and distincts
    duplicate_counts = [correct_counts[1], incorrect_counts[1]]
    distinct_counts = [correct_counts[-1], incorrect_counts[-1]]

    axes[1, 0].pie(duplicate_counts, labels=["Correct", "Incorrect"], autopct="%1.1f%%")
    axes[1, 0].set_title("Duplicates: Correct vs Incorrect Predictions")

    axes[1, 1].pie(distinct_counts, labels=["Correct", "Incorrect"], autopct="%1.1f%%")
    axes[1, 1].set_title("Distincts: Correct vs Incorrect Predictions")

    # Embed the plot as base64
    distributions_plot_base64 = embed_matplotlib_figure(fig)

    plt.close(fig)

    def highlight_correct(row):
        duplicate_correct = row["label_duplicates"] == row["eval_duplicates"]

        style_correct = "background-color: lightgreen"
        style_error = "background-color: lightcoral"

        if duplicate_correct:
            return [style_correct for _ in row]
        else:
            return [style_error for _ in row]

    styled_report = report_df.style.apply(highlight_correct, axis=1)

    styled_report.set_table_styles(
        [
            {
                "selector": "th",
                "props": [("background-color", "lightgray"), ("font-weight", "bold")],
            },
            {"selector": "td", "props": [("border", "1px solid black")]},
            {
                "selector": "tr:nth-of-type(odd)",
                "props": [("background-color", "white")],
            },
            {
                "selector": "tr:nth-of-type(even)",
                "props": [("background-color", "aliceblue")],
            },
        ]
    )

    with open(REPORT_FILENAME, "w", encoding="utf-8", errors="replace") as f:
        f.write(
            f"""
        <html>
        <head>
            <title>{title} - Model Report</title>
            <style>
                img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <h1>{title} - Model Report</h1>
            <h2>Label Prediction Distributions</h2>
            {distributions_plot_base64}
            <h2>Evaluation Dataset</h2>
            {styled_report.to_html()}
        </body>
        </html>
        """
        )


if __name__ == "__main__":
    # Add argparse for CLI interactions
    parser = argparse.ArgumentParser(description="Contact Data Evaluation Script")
    parser.add_argument(
        "--filename",
        default="prepared_val_data.csv",
        type=str,
        help="CSV filename in the data directory",
    )
    parser.add_argument(
        "--failed",
        action="store_true",
        help="Output a CSV with the rows the model got wrong",
    )
    args = parser.parse_args()

    tokenizer, vocabulary = create_char_tokenizer()

    data_module = ContactDataModule(batch_size=EVAL_BATCH_SIZE, return_eval_fields=True)
    data_module.setup(stage="validate", val_file=args.filename)

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

    # Define the optimizer and criterion
    criterion = LOSS_FUNCTION(MARGIN)

    eval_loss, precision, recall, f1, incorrect_df = eval_model(
        model,
        device,
        data_module,
        criterion,
        SIMILARITY_METRIC(0.5, return_distance=True),
        report_callback=create_html_report,
    )

    if args.failed:
        incorrect_df.to_csv(
            os.path.join(data_module.data_dir, "failed_records.csv"), index=False
        )

    print(
        f"Eval Loss = {eval_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}"
    )
