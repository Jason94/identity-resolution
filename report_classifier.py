import logging
import os
from typing import Dict, List
from lightning.pytorch.loggers.logger import Logger
from lightning_utilities.core.rank_zero import rank_zero_only
import lightning.pytorch as pl
import torch
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from model_cli import *
from data import ContactDataModule
from train_classifier import PlContactsClassifier
from utilities import transpose_dict_of_lists
from report import create_html_report

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReportLogger(Logger):
    def __init__(self):
        super().__init__()
        self.metric_logs: Dict[str, List[float]] = {}

    @property
    def name(self):
        return "Report Logger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for m, val in metrics.items():
            self.metric_logs[m] = [*self.metric_logs.get(m, []), val]

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass


def main(checkpoint_path: str, data_path: str, failed: bool, batch_size: int):
    report_filename = os.path.join(
        os.path.dirname(checkpoint_path), Path(checkpoint_path).stem + ".html"
    )

    logger.info(f"Loading model from {checkpoint_path}")
    pl_model = PlContactsClassifier.load_from_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Found device {device}")

    pl_model.to(device)

    pl_data = ContactDataModule(
        prepared_file=data_path,
        val_file=data_path,
        train_file=data_path,
        batch_size=batch_size,
        fields=pl_model.fields(),
        preserve_text_fields=True,
        return_predict_fields=True,
    )
    pl_data.prepare_data()
    pl_data.setup("validate")

    # pl_logger = ReportLogger()

    trainer = pl.Trainer()  # logger=pl_logger)

    all_probs = []
    all_labels = []
    all_preds = []

    all_field_data = []

    classification_threshold: float = pl_model.hparams.classification_threshold  # type: ignore

    for classification_score, labels, data in trainer.predict(pl_model, pl_data):  # type: ignore
        probs = classification_score.clone()
        preds = classification_score.clone()
        preds[preds < classification_threshold] = -1.0
        preds[preds >= classification_threshold] = 1.0

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
        all_field_data.extend(transpose_dict_of_lists(data))

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()

    all_field_data = pd.DataFrame(all_field_data)

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    logger.info(f"Precision:\t{precision:0.6f}")
    logger.info(f"Recall:\t{recall:0.6f}")
    logger.info(f"F1 Score:\t{f1:0.6f}")

    create_html_report(
        all_preds,
        all_labels,
        all_probs,
        all_field_data,
        pl_model.hparams.version_name,  # type: ignore
        filename=report_filename,
    )


if __name__ == "__main__":
    logging.basicConfig()

    parser = make_evaluation_args(mode="classifier")
    make_data_args(parser, needs_training=False)
    make_model_io_args(parser)
    parser.add_argument(
        "--failed",
        action="store_true",
        help="Output a CSV with the rows the model got wrong",
    )
    args = parser.parse_args()

    if args.checkpoint_path is None:
        raise KeyError("Must specify model checkpoint.")

    main(args.checkpoint_path, args.eval_data, args.failed, args.batch_size)
