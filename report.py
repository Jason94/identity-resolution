import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import base64
from io import BytesIO


def embed_matplotlib_figure(fig):
    """Converts a Matplotlib figure to a base64 encoded PNG to be used in HTML reports"""
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{encoded}'/>"


def create_html_report(
    all_preds,
    all_labels,
    all_first1,
    all_last1,
    all_first2,
    all_last2,
    file_name,
    title: str,
):
    report_df = pd.DataFrame()
    report_df["(1) first_name"] = list(all_first1)
    report_df["(1) last_name"] = list(all_last1)
    report_df["(2) first_name"] = list(all_first2)
    report_df["(2) last_name"] = list(all_last2)

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

    with open(file_name, "w", encoding="utf-8", errors="replace") as f:
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
