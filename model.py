import torch
from torch import nn
from torchvision.ops import MLP
import string
import math

from config import MAX_NAME_LENGTH, MAX_EMAIL_LENGTH


class ContactEncoder(nn.Module):
    PAD_CHARACTER = "\0"

    @staticmethod
    def init_positional_encoding(max_sequence_length, embedding_dimension):
        """
        Function to initialize positional encoding tensor.

        In the positional encoding function, we're essentially encoding each position of the sequence
        (from 0 to the max sequence length) as a unique vector. We do this using sine and cosine
        functions of different frequencies. The idea here is to create unique positional vectors that
        the model can learn to use to pay different levels of "attention" to the input data based on
        their positions in the sequence.

        We alternate between using sine and cosine to ensure that the positional encoding for position
        p and position p + k can be represented as a linear function of the encoding at position p.
        This characteristic helps the model easily learn relative positions, which is particularly
        important when dealing with sequences.

        The reason we divide by the log of 10000 is to create a range of frequencies, so we can
        capture positional information at different scales. This scaling factor ensures that the
        positional encoding for close positions don't change dramatically, making the positional
        information easier to learn for the model.

        Args:
            max_sequence_length: The maximum length of sequences.
            embedding_dimension: The dimension of the embeddings.

        Returns:
            pe: Positional encoding tensor of shape (1, max_sequence_length, embedding_dimension).
        """

        # Create an empty tensor for positional encoding
        positional_encoding = torch.zeros(max_sequence_length, embedding_dimension)

        # Create a tensor that represents position indices from 0 to max_sequence_length
        position_indices = torch.arange(max_sequence_length).unsqueeze(1).float()

        # Create a tensor that includes the divisors for the sinusoidal functions
        # It includes values for even indices from 0 to embedding_dimension
        # Dividing by the log of 10000 creates a range of frequencies to capture positional information at different scales
        divisors = torch.exp(
            -(math.log(10000.0))
            * torch.arange(0, embedding_dimension, 2).float()
            / embedding_dimension
        )

        # For even indices, use the sine function
        positional_encoding[:, 0::2] = torch.sin(position_indices * divisors)

        # For odd indices, use the cosine function
        positional_encoding[:, 1::2] = torch.cos(position_indices * divisors)

        # Add an extra dimension for the batch size
        positional_encoding = positional_encoding.unsqueeze(0)

        # Wrap positional_encoding in a PyTorch Parameter object
        # This object can be included in a model's list of parameters by assigning it as an attribute of a Module
        # The requires_grad=False argument means that this tensor doesn't need to be updated during backpropagation
        positional_encoding = nn.Parameter(positional_encoding, requires_grad=False)

        return positional_encoding

    @staticmethod
    def create_attn_mask(lengths, max_len):
        mask = torch.arange(max_len)[None, :] >= lengths[:, None]
        return mask.bool()

    def __init__(
        self,
        vocab_size,
        embedding_dim=60,
        n_heads_attn=4,
        attn_dim=180,
        norm_eps=1e-6,
        output_embedding_dim=8,
        output_mlp_layers=6,
        dropout=0.0,
    ):
        super(ContactEncoder, self).__init__()

        # --- Embedding layers
        self.embedding_dim = embedding_dim
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.LayerNorm(embedding_dim, eps=norm_eps),
        )

        # --- Attention Layer
        self.attn_dim = attn_dim
        self.fc_expand_embedding = nn.Linear(embedding_dim, attn_dim)

        self.positional_encoding = self.init_positional_encoding(
            MAX_NAME_LENGTH + MAX_EMAIL_LENGTH, attn_dim
        )

        self.n_heads_attn = n_heads_attn
        self.multihead_attn = nn.MultiheadAttention(
            attn_dim, n_heads_attn, batch_first=True
        )

        self.norm_attn = nn.LayerNorm(attn_dim, eps=norm_eps)

        # -- Attention Processing
        self.fc_proc_attn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.ReLU(),
            nn.Linear(attn_dim, attn_dim),
        )

        self.norm_proc_attn = nn.LayerNorm(attn_dim, eps=norm_eps)

        # -- Final Output Processing
        self.output_embedding_dim = output_embedding_dim
        self.fc_output = nn.Sequential(
            MLP(
                attn_dim,
                [attn_dim] * output_mlp_layers,
                norm_layer=lambda dim: nn.LayerNorm(dim, eps=norm_eps),
                # activation_layer=lambda: nn.Tanh(),
            ),
            nn.Linear(attn_dim, output_embedding_dim),
        )

    def device(self):
        return next(self.parameters()).device

    # Use *xargs as a hack to avoid torch-info bug
    def forward(self, name_tensor, lengths, email_tensor, email_lengths, *xargs):
        # Generate the initial embeddings
        embedding_name = self.embedding(name_tensor)
        embedding_email = self.embedding(email_tensor)

        expanded_name = self.fc_expand_embedding(embedding_name)
        expanded_email = self.fc_expand_embedding(embedding_email)

        # Concat field embeddings along the sequence dimension, to prepare for the attention
        combined_field_embeddings = torch.cat([expanded_name, expanded_email], dim=1)

        # Create the attention mask
        name_attn_mask = self.create_attn_mask(lengths, expanded_name.size(1)).to(
            self.device()
        )
        email_attn_mask = self.create_attn_mask(
            email_lengths, expanded_email.size(1)
        ).to(self.device())
        attn_mask = torch.cat([name_attn_mask, email_attn_mask], dim=1)

        # Add the positional information
        batch_size, _, _ = combined_field_embeddings.shape
        pad_mask = attn_mask.unsqueeze(-1).expand(-1, -1, self.attn_dim)
        positional_encodings_masked = self.positional_encoding.repeat(batch_size, 1, 1)
        positional_encodings_masked[pad_mask] = 0

        combined_field_embeddings = (
            combined_field_embeddings + positional_encodings_masked
        )

        # Pass through the multihead attention
        attn_output, attn_output_weights = self.multihead_attn(
            combined_field_embeddings,
            combined_field_embeddings,
            combined_field_embeddings,
            key_padding_mask=attn_mask,
        )

        norm_attn_output = self.norm_attn(attn_output + combined_field_embeddings)

        # Pass through the feed forward network
        ff_output = self.fc_proc_attn(norm_attn_output)

        norm_output = self.norm_proc_attn(ff_output + norm_attn_output)

        # Produce a weighted sum of the character-wise embeddings to represent the input
        final_attn_weights = attn_output_weights[:, -1, :]
        weighted_sum_output = (final_attn_weights.unsqueeze(-1) * norm_output).sum(
            dim=1
        )

        # Reshape the internal attented embedding to the output embedding
        output_embeddings = self.fc_output(weighted_sum_output)

        return output_embeddings


def create_char_to_int():
    # Create a list of all ASCII printable characters.
    chars = list(string.printable) + [ContactEncoder.PAD_CHARACTER]

    # Create a dictionary that maps each character to a unique integer.
    char_to_int = {char: i for i, char in enumerate(chars)}

    return char_to_int, chars
