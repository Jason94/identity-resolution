from typing import List, Tuple
import torch
from torch import nn
from torchvision.ops import MLP
import math
from argparse import Namespace


from data import Field, lookup_field


class ContactEncoder(nn.Module):
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
        # Dividing by the log of 10000 creates a range of frequencies to capture positional
        # information at different scales
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
    def init_field_positional_encoding(field_lengths, embedding_dimension):
        """
        Function to initialize positional encoding tensor for fields.

        In this positional encoding function, we're essentially encoding each position of the field
        (from 0 to the max number of fields) as a unique vector. We do this using the init_positional_encoding
        function. The positional encoding for each character in the same field is the same.

        Args:
            field_lengths: A list of the maximum lengths for each field.
            embedding_dimension: The dimension of the embeddings.

        Returns:
            field_positional_encoding: Positional encoding tensor of shape
                (1, total_sequence_length, embedding_dimension).
        """

        # Calculate the total sequence length by summing all field lengths
        total_sequence_length = sum(field_lengths)

        # Initialize the positional encoding for the total sequence length
        positional_encoding = ContactEncoder.init_positional_encoding(
            total_sequence_length, embedding_dimension
        )

        # Initialize an empty tensor for the field positional encoding
        field_positional_encoding = torch.zeros_like(positional_encoding)

        # Initialize the starting index for filling in the field positional encoding
        start_index = 0

        # Loop through each field length
        for field_length in field_lengths:
            # Get the positional encoding for this field
            field_enc = positional_encoding[:, start_index]

            # Repeat this positional encoding for the length of this field
            field_enc = field_enc.repeat(field_length, 1)

            # Fill in the positional encoding for this field in the field positional encoding tensor
            field_positional_encoding[
                :, start_index : start_index + field_length
            ] = field_enc

            # Update the starting index for the next field
            start_index += field_length

        # Wrap field_positional_encoding in a PyTorch Parameter object
        field_positional_encoding = nn.Parameter(
            field_positional_encoding, requires_grad=False
        )

        return field_positional_encoding

    @staticmethod
    def from_namespace(namespace: Namespace):
        fields = [lookup_field(name) for name in namespace.field_names]
        return ContactEncoder(
            vocab_size=namespace.vocab_size,
            fields=fields,
            embedding_dim=namespace.embedding_dim,
            n_heads_attn=namespace.n_heads_attn,
            attn_dim=namespace.attn_dim,
            norm_eps=namespace.norm_eps,
            output_embedding_dim=namespace.output_embedding_dim,
            output_mlp_layers=namespace.output_mlp_layers,
            p_dropout=namespace.p_dropout,
        )

    def __init__(
        self,
        vocab_size,
        fields: List[Field],
        embedding_dim=60,
        n_heads_attn=4,
        attn_dim=180,
        norm_eps=1e-6,
        output_embedding_dim=8,
        output_mlp_layers=6,
        p_dropout=0.0,
    ):
        super(ContactEncoder, self).__init__()

        self.fields = fields

        # --- Embedding layers
        self.embedding_dim = embedding_dim
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.LayerNorm(embedding_dim, eps=norm_eps),
        )

        self.attn_dim = attn_dim
        self.fc_expand_embedding = nn.Linear(embedding_dim, attn_dim)

        # --- Field-Attending Attention Layer
        self.n_heads_attn = n_heads_attn

        self.f_que_layer = nn.Linear(attn_dim, attn_dim)
        self.f_key_layer = nn.Linear(attn_dim, attn_dim)
        self.f_val_layer = nn.Linear(attn_dim, attn_dim)

        field_lengths = sum([f.max_length for f in fields])

        self.f_positional_encoding = self.init_field_positional_encoding(
            [f.max_length for f in self.fields], attn_dim
        )

        self.f_multihead_attn = nn.MultiheadAttention(
            attn_dim, n_heads_attn, batch_first=True, dropout=p_dropout
        )

        self.f_norm_attn = nn.LayerNorm(attn_dim, eps=norm_eps)

        # --- Field-Attending Attention Processing
        self.field_proc_attn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(attn_dim, attn_dim),
        )

        self.f_norm_proc_attention = nn.LayerNorm(attn_dim, eps=norm_eps)

        # --- Character-Attending Attention Layer
        self.que_layer = nn.Linear(attn_dim, attn_dim)
        self.key_layer = nn.Linear(attn_dim, attn_dim)
        self.val_layer = nn.Linear(attn_dim, attn_dim)

        self.positional_encoding = self.init_positional_encoding(
            field_lengths, attn_dim
        )

        self.multihead_attn = nn.MultiheadAttention(
            attn_dim, n_heads_attn, batch_first=True, dropout=p_dropout
        )

        self.norm_attn = nn.LayerNorm(attn_dim, eps=norm_eps)

        # --- Attention Processing
        self.fc_proc_attn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(attn_dim, attn_dim),
        )

        self.norm_proc_attn = nn.LayerNorm(attn_dim, eps=norm_eps)

        # --- Attn Combinations
        self.proc_attn_comb = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(attn_dim, attn_dim),
        )

        self.norm_proc_attn_comb = nn.LayerNorm(attn_dim, eps=norm_eps)

        # --- Final Output Processing
        self.output_embedding_dim = output_embedding_dim
        self.fc_output = nn.Sequential(
            MLP(
                attn_dim,
                [attn_dim] * output_mlp_layers,
                norm_layer=lambda dim: nn.LayerNorm(dim, eps=norm_eps),
                # activation_layer=lambda: nn.Tanh(),
            ),
            nn.Dropout(p_dropout),
            nn.Linear(attn_dim, attn_dim),
            nn.Linear(attn_dim, attn_dim),
            nn.Linear(attn_dim, output_embedding_dim),
        )

    def create_attn_mask(self, lengths, max_len):
        raw = torch.arange(max_len)[None, :].to(self.device())
        mask = raw >= lengths[:, None]
        return mask.bool()

    def hyperparameters(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "n_heads_attn": self.n_heads_attn,
            "attn_dim": self.attn_dim,
            "norm_eps": self.norm_eps,
            "output_embedding_dim": self.output_embedding_dim,
            "output_mlp_layers": self.output_mlp_layers,
            "p_dropout": self.p_dropout,
        }

    def device(self):
        return next(self.parameters()).device

    # Use *xargs as a hack to avoid torch-info bug
    def forward(
        self,
        token_tensors: List[torch.Tensor],
        length_tensors: List[torch.Tensor],
        *xargs
    ):
        # --- Initial Embeddings

        # Generate the initial embeddings
        embeddings = [self.embedding(field_tensor) for field_tensor in token_tensors]
        expanded_embeddings = [
            self.fc_expand_embedding(embedding) for embedding in embeddings
        ]

        # Concat field embeddings along the sequence dimension
        combined_field_embeddings = torch.cat(expanded_embeddings, dim=1)

        # --- Field-Attending Attention

        # Create the attention mask
        attn_masks = []
        for i in range(0, len(self.fields)):
            attn_masks.append(
                self.create_attn_mask(length_tensors[i], expanded_embeddings[i].size(1))
            )
        attn_mask = torch.cat(attn_masks, dim=1)

        batch_size, _, _ = combined_field_embeddings.shape
        pad_mask = attn_mask.unsqueeze(-1).expand(-1, -1, self.attn_dim)

        # Add the field-level positional encoding
        field_positional_encodings_masked = self.f_positional_encoding.repeat(
            batch_size, 1, 1
        )
        field_positional_encodings_masked[pad_mask] = 0

        f_attn_input = combined_field_embeddings + field_positional_encodings_masked

        # Pass through the multihead attention
        f_queries = self.f_que_layer(f_attn_input)
        f_keys = self.f_key_layer(f_attn_input)
        f_values = self.f_val_layer(f_attn_input)

        f_attn_output, f_attn_output_weights = self.f_multihead_attn(
            f_queries, f_keys, f_values, key_padding_mask=attn_mask
        )

        f_norm_attn_output = self.f_norm_attn(f_attn_output + f_attn_input)

        # Pass through feed-forward network
        f_ff_output = self.field_proc_attn(f_norm_attn_output)

        f_norm_output = self.f_norm_proc_attention(f_ff_output + f_norm_attn_output)

        # --- Character-Attending Attention

        # We can re-use the same mask as for the field-attending layer
        positional_encodings_masked = self.positional_encoding.repeat(batch_size, 1, 1)
        positional_encodings_masked[pad_mask] = 0

        attn_input = f_norm_output + positional_encodings_masked

        # Pass through the multihead attention
        queries = self.que_layer(attn_input)
        keys = self.key_layer(attn_input)
        values = self.val_layer(attn_input)

        attn_output, attn_output_weights = self.multihead_attn(
            queries,
            keys,
            values,
            key_padding_mask=attn_mask,
        )

        norm_attn_output = self.norm_attn(attn_output + attn_input)

        # Pass through the feed forward network
        ff_output = self.fc_proc_attn(norm_attn_output)

        norm_output = self.norm_proc_attn(ff_output + norm_attn_output)

        # --- Combine the attention outputs
        combined_attn_output = self.proc_attn_comb(f_norm_output + norm_output)

        norm_combined_attn_output = self.norm_proc_attn_comb(
            combined_attn_output + f_norm_output + norm_output
        )

        # --- Output

        # Produce a weighted sum of the character-wise embeddings to represent the input

        # We will take the average attention weight for the two attention layers
        mean_attn_weigths = torch.mean(
            torch.stack(
                [f_attn_output_weights[:, -1, :], attn_output_weights[:, -1, :]]
            ),
            dim=0,
        )
        weighted_sum_output = (
            mean_attn_weigths.unsqueeze(-1) * norm_combined_attn_output
        ).sum(dim=1)

        # Reshape the internal attented embedding to the output embedding
        output_embeddings = self.fc_output(weighted_sum_output)

        return output_embeddings

    def example_tensor(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        This function creates an example tensor (all 0s) with batch size 1 for this model.

        Returns:
            (token_tensors, length_tensors): a tuple of lists of tensors.
        """

        # Initialize lists to store token tensors and length tensors
        token_tensors = []
        length_tensors = []

        # Loop through each field in self.fields
        for field in self.fields:
            # Generate an example token tensor for this field
            # We'll use a tensor of zeros, with dimensions:
            # 1 (batch size) x field.max_length
            token_tensor = torch.zeros((1, field.max_length)).long()

            # The length tensor for this field is simply the max length of the field
            # We need to add an extra dimension to create a 1D tensor
            length_tensor = torch.full((1,), field.max_length)

            # Add the token and length tensors for this field to the lists
            token_tensors.append(token_tensor)
            length_tensors.append(length_tensor)

        # Return the token and length tensors as a tuple
        return token_tensors, length_tensors
