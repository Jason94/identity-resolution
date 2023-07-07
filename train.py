from typing import Any, Callable, Dict, Optional, Tuple, List, Union
import torch
from torch import optim
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score
from argparse import Namespace
from pytorch_lightning.loggers import TensorBoardLogger


from contrastive_metric import ContrastiveLoss, is_duplicate
from model import ContactEncoder
from config import *
from data import ContactDataModule, Field
from model_cli import make_parser
from embedding_logger import TensorBoardEmbeddingLogger


def convert_bool_tensor(tensor):
    ones = torch.ones_like(tensor, dtype=torch.float32)
    minus_ones = -1 * ones
    converted_tensor = torch.where(tensor, ones, minus_ones)
    return converted_tensor


def split_field_dict(
    fields: List[Field], data: List[dict]
) -> Tuple[List[dict], List[dict]]:
    data1 = []
    data2 = []

    for d in data:
        d1 = {}
        d2 = {}

        for f in fields:
            d1[f.field] = d[f.field + "1"]
            d2[f.field] = d[f.field + "2"]

        data1.append(d1)
        data2.append(d2)

    return data1, data2


def transpose_dict_of_lists(dict_of_lists):
    keys = dict_of_lists.keys()
    length_of_lists = len(next(iter(dict_of_lists.values())))

    list_of_dicts = []
    for i in range(length_of_lists):
        new_dict = {}
        for key in keys:
            new_dict[key] = dict_of_lists[key][i]
        list_of_dicts.append(new_dict)

    return list_of_dicts


class PlContactEncoder(pl.LightningModule):
    def __init__(
        self,
        hyperparameters: Union[Namespace, dict],
        encoder: Optional[ContactEncoder] = None,
        loss_function: Optional[
            Callable[[float], Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]]
        ] = None,
        similarity_function: Optional[
            Callable[
                [float],
                Callable[
                    [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
                ],
            ]
        ] = None,
    ):
        super().__init__()

        if isinstance(hyperparameters, dict):
            hyperparameters = Namespace(**hyperparameters)

        self.loss_func_factory = (
            hyperparameters and vars(hyperparameters).get("loss_function")
        ) or loss_function
        self.similarity_func_factory = (
            hyperparameters and vars(hyperparameters).get("similarity_function")
        ) or similarity_function

        # --- Evaluation Performance Data
        self.validation_labels = []
        self.validation_preds = []

        if encoder:
            self.encoder = encoder
            hyperparameters = Namespace(
                **{**vars(hyperparameters), **encoder.hyperparameters()}
            )
            self.save_hyperparameters(hyperparameters)
        # If we only got hyperparameters, create a new ContactEncoder.
        else:
            self.encoder = ContactEncoder.from_namespace(hyperparameters)
            self.save_hyperparameters(hyperparameters)

        # Create loss function from the margin hyperparameter
        self.loss_function = self.loss_func_factory(hyperparameters.margin)  # type: ignore
        # Create similarity function from the duplicate thershold hyperparameter
        self.similarity_function = self.similarity_func_factory(hyperparameters.threshold)  # type: ignore

    def training_step(self, batch, batch_idx):
        if not self.loss_function:
            raise RuntimeError("Not configured for training!")

        (tokens1, lengths1, tokens2, lengths2, labels) = batch

        # Forward pass through the model
        output1 = self.encoder(tokens1, lengths1)
        output2 = self.encoder(tokens2, lengths2)

        # Calculate loss
        loss = self.loss_function(output1, output2, labels)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=self.hparams.batch_size,  # type: ignore
        )

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        if not self.loss_function or not self.similarity_function:
            raise RuntimeError("Not configured for evaluating!")

        (tokens1, lengths1, tokens2, lengths2, labels, field_data) = batch

        # Forward pass through the model
        output1 = self.encoder(tokens1, lengths1)
        output2 = self.encoder(tokens2, lengths2)

        # Compute the loss
        loss = self.loss_function(output1, output2, labels)

        pred_raw, dists = self.similarity_function(output1, output2)
        pred = convert_bool_tensor(pred_raw)

        if isinstance(self.logger, TensorBoardEmbeddingLogger):
            field_data1, field_data2 = split_field_dict(
                self.hparams.fields, transpose_dict_of_lists(field_data)  # type: ignore
            )

            field_data1 = [list(d.values()) for d in field_data1]
            field_data2 = [list(d.values()) for d in field_data2]

            self.logger.log_embeddings(output1, metadata=field_data1)
            self.logger.log_embeddings(output2, metadata=field_data2)

        # Log
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,  # type: ignore
        )
        self.validation_labels.append(labels.cpu())
        self.validation_preds.append(pred.cpu())

    def on_validation_epoch_end(self) -> None:
        all_labels = torch.cat(self.validation_labels).numpy()
        all_preds = torch.cat(self.validation_preds).numpy()

        precision = float(precision_score(all_labels, all_preds, zero_division=0))
        recall = float(recall_score(all_labels, all_preds))
        f1 = float(f1_score(all_labels, all_preds))

        self.log_dict({"val_precision": precision, "val_recall": recall, "val_f1": f1})

        # Free validation logging memory
        self.validation_labels.clear()
        self.validation_preds.clear()

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)  # type: ignore
        return optimizer


def train(
    args: Namespace, data_module: ContactDataModule, logger: Optional[Any] = None
):
    # Create model instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found device {device}")

    lightning_model = PlContactEncoder(
        hyperparameters=args,
        loss_function=ContrastiveLoss,
        similarity_function=is_duplicate,
    )
    lightning_model.to(device)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_f1",
        mode="max",
        filename="{epoch:02d}---{val_loss:.4f}-{val_f1:.4f}",
        every_n_epochs=2,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=[ModelSummary(max_depth=-1), checkpoint_callback],
        logger=logger,
    )
    trainer.fit(model=lightning_model, datamodule=data_module)


def margin_experiment(args: Namespace):
    start = 1.5
    end = 4.0
    for margin in [start + x / 2 for x in range(0, int(end - start) * 2)]:
        scenario_args = Namespace(**{**vars(args), "margin": margin})

        for i in range(0, 3):
            print(f"Experiment {i}: margin={margin}")
            logger = TensorBoardLogger(
                save_dir="", version=f"margin_{margin:0.2f}__{i}"
            )
            logger.experiment.add_embedding()
            # Load the data
            train(scenario_args, data_module, logger=logger)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    data_module = ContactDataModule(batch_size=args.batch_size, return_eval_fields=True)
    args.vocab_size = len(data_module.vocabulary)

    # margin_experiment(args)
    logger = TensorBoardEmbeddingLogger(
        save_dir="",
        metadata_header=[f.field for f in data_module.fields],
        maximum_embeddings_to_save=10000,
    )
    train(args, data_module, logger=logger)
