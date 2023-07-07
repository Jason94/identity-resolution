from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Optional

from data import ALL_FIELDS


def make_universal_args(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--fields",
        nargs="+",
        help="Fields passed into the model. Note that this is breaking!",
        default=[f.field for f in ALL_FIELDS],
    )
    return parser


# TODO: Support this from train.py
def make_data_args(
    parser: Optional[ArgumentParser] = None, needs_source_file: bool = False
) -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    if needs_source_file:
        parser.add_argument(
            "--prepared_data",
            type=str,
            default="prepared_data.csv",
            help="CSV file in the data directory storing the entire prepared data.",
        )
        parser.add_argument(
            "--source_files",
            nargs="+",
            help="CSV files in the data directory with raw field data to prepare.",
            default=["duplicates.csv", "distincts.csv"],
        )

    parser.add_argument(
        "--training_data",
        type=str,
        default="prepared_train_data.csv",
        help="CSV file in the data directory to use as training data.",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default="prepared_val_data.csv",
        help="CSV file in the data directory to use as evaluation data.",
    )
    return parser


def make_training_args(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--margin",
        type=float,
        default=2.0,
        help=(
            "Margin value for the loss function. Determines how much to penalize model"
            " predictions that deviate from the actual labels."
        ),
    )
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help=(
            "Number of data points to train at once. Higher values will train faster, at the cost"
            " of using more RAM/VRAM. Start with a higher number and lower as needed if you run out"
            " of memory."
        ),
    )
    parser.add_argument(
        "--p_dropout",
        type=float,
        default=0.0,
        help=(
            "Dropout probability. This is the probability that each neuron in the network is"
            " temporarily dropped out, or turned off, during training. The purpose is to"
            " prevent overfitting."
        ),
    )
    return parser


def make_model_args(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=60,
        help=(
            "Dimension of the initial embeddings inside the model. This value won't affect"
            " the output directly. Too small values might not allow the model to capture"
            " enough patterns from the input."
        ),
    )
    parser.add_argument(
        "--n_heads_attn",
        type=int,
        default=4,
        help=(
            "Number of attention heads. In the transformer model, the attention mechanism"
            " is applied multiple times in parallel which are referred to as attention heads."
            " More attention heads means the model will jointly attend to information from"
            " different positions, providing a better understanding of the context."
        ),
    )
    parser.add_argument(
        "--attn_dim",
        type=int,
        default=180,
        help=(
            "Dimension of the attention layers. This parameter sets the size of the output vectors"
            " from the attention layers. The higher the value, the larger the output vectors, which"
            " could lead to the model capturing more complex patterns, but also increases"
            " computational cost."
        ),
    )
    parser.add_argument(
        "--norm_eps",
        type=float,
        default=1e-6,
        help=(
            "Epsilon used for normalization layers. It's a very small number that's added to"
            " the variance of the normalization layer to prevent division by zero."
            " This parameter helps with numerical stability."
        ),
    )
    parser.add_argument(
        "--output_mlp_layers",
        type=int,
        default=6,
        help=(
            "Number of layers in the output MLP (Multilayer Perceptron). The MLP is used to"
            " process the output of the attention mechanism and maps the output to the desired"
            " number of classes or values. The more layers there are, the more complex"
            " transformations the model can learn."
        ),
    )
    parser.add_argument(
        "--output_embedding_dim",
        type=int,
        default=8,
        help=(
            "Dimension of the output embeddings. This parameter sets the size of the output vector"
            " for each token after it has been processed through all layers of the model. This is"
            " effectively the final size of the representations of each token."
        ),
    )
    return parser


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    make_universal_args(parser)
    make_model_args(parser)
    make_training_args(parser)

    # -- Evaluation Arguments
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold value for the similarity function. Determines the cutoff point"
            " below which two model outputs are considered similar/duplicates."
        ),
    )

    return parser
