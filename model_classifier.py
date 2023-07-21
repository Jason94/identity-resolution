from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
from torchvision.ops import MLP


from data import Field
from model import ContactEncoder


# This didn't seem to help, but it's a neat idea. Maybe with less MLP layers this would help.
class AttentionPooling(nn.Module):
    def __init__(self, attn_dim, seq_length):
        super(AttentionPooling, self).__init__()
        self.context_matrix = nn.Parameter(
            torch.randn(attn_dim, seq_length), requires_grad=True
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, processed):
        # Calculate dot product between processed and context matrix
        # processed shape: (batch_size, seq_length, attn_dim)
        # context_matrix shape: (attn_dim, seq_length)
        # dot_product shape: (batch_size, seq_length, seq_length)
        dot_product = torch.matmul(processed, self.context_matrix)

        # Apply softmax to get attention weights
        # attn_weights shape: (batch_size, seq_length, seq_length)
        attn_weights = self.softmax(dot_product)

        # Multiply dot_product by attention weights and sum over attn_dim dimension
        # pooled shape: (batch_size, seq_length)
        pooled = torch.sum(attn_weights * dot_product, dim=-1)

        return pooled


class ContactsClassifier(nn.Module):
    @staticmethod
    def from_encoder(
        encoder: ContactEncoder, p_dropout: Optional[float] = None, **kwargs
    ) -> ContactsClassifier:
        return ContactsClassifier(
            encoder.fields,
            encoder.attn_dim,
            encoder.n_heads_attn,
            p_dropout=p_dropout or encoder.p_dropout,
            norm_eps=encoder.norm_eps,
            **kwargs,
        )

    def __init__(
        self,
        fields: List[Field],
        attn_dim=180,
        n_heads_attn=4,
        pre_pool_mlp_layers=8,
        pool_mlp_layers=4,
        p_dropout=0.0,
        norm_eps=1e-6,
    ):
        super().__init__()

        # --- Save hyperparameters
        self.fields = fields
        self.attn_dim = attn_dim
        self.n_heads_attn = n_heads_attn
        self.p_dropout = p_dropout

        # --- Preprocessing
        self.prerocess_layer = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.LayerNorm(attn_dim, eps=norm_eps),
            nn.Linear(attn_dim, attn_dim),
        )

        # --- Attention Layer
        self.que_layer = nn.Linear(attn_dim, attn_dim)
        self.key_layer = nn.Linear(attn_dim, attn_dim)
        self.val_layer = nn.Linear(attn_dim, attn_dim)

        field_lengths = sum([f.max_length for f in fields])
        self.positional_encodings = ContactEncoder.init_positional_encoding(
            field_lengths, attn_dim
        )

        self.multihead_attn = nn.MultiheadAttention(
            attn_dim, n_heads_attn, batch_first=True, dropout=p_dropout
        )

        self.norm_attn = nn.LayerNorm(attn_dim, eps=norm_eps)

        # --- Process Attention Outputs
        self.proc_attn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(attn_dim, attn_dim),
        )

        self.norm_proc_attn = nn.LayerNorm(attn_dim, eps=norm_eps)

        # --- Final Output Processing
        self.processing = MLP(
            attn_dim,
            [attn_dim] * pre_pool_mlp_layers,
            norm_layer=lambda dim: nn.LayerNorm(dim, eps=norm_eps),
            activation_layer=lambda: nn.Tanh(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        # self.pool = AttentionPooling(attn_dim, 2 * field_lengths)
        # self.pool = nn.MaxPool1d(1)

        self.post_pool_processing = MLP(
            field_lengths,
            [field_lengths] * pool_mlp_layers,
            norm_layer=lambda dim: nn.LayerNorm(dim, eps=norm_eps),
            activation_layer=lambda: nn.Tanh(),
        )

        self.output = nn.Sequential(
            nn.Linear(field_lengths, field_lengths),
            nn.Linear(field_lengths, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, attn_embedding1: torch.Tensor, attn_embedding2: torch.Tensor
    ) -> torch.Tensor:
        batch_size, _, _ = attn_embedding1.shape

        preprocessed1 = self.prerocess_layer(attn_embedding1)
        preprocessed2 = self.prerocess_layer(attn_embedding2)

        # --- Attention
        # TODO: Maybe have an attention mask?
        positional_encodings = self.positional_encodings.repeat(batch_size, 1, 1)

        attn_input1 = preprocessed1 + positional_encodings
        attn_input2 = preprocessed2 + positional_encodings

        # Cross-attention from attn_embedding1 to attn_embedding2
        queries1 = self.que_layer(attn_input1)
        keys1 = self.key_layer(attn_input2)
        values1 = self.val_layer(attn_input2)
        attn_output1, attn_output_weights1 = self.multihead_attn(
            queries1, keys1, values1
        )

        # Cross-attention from attn_embedding2 to attn_embedding1
        queries2 = self.que_layer(attn_input2)
        keys2 = self.key_layer(attn_input1)
        values2 = self.val_layer(attn_input1)
        attn_output2, attn_output_weights2 = self.multihead_attn(
            queries2, keys2, values2
        )

        # Concatenate or otherwise combine the outputs
        combined_attn_output = torch.max(attn_output1, attn_output2)

        # TODO? Should I do some kind of pooling of the initial embeddings and do a bypass here?
        norm_attn_output = self.norm_attn(combined_attn_output)

        # Process
        output = self.proc_attn(norm_attn_output)

        norm_output = self.norm_proc_attn(output + norm_attn_output)

        # --- Output
        processed = self.processing(norm_output) + norm_output

        pooled = self.pool(processed).squeeze(-1)

        pooled_processed = pooled + self.post_pool_processing(pooled)

        return self.output(pooled_processed).squeeze(-1)
