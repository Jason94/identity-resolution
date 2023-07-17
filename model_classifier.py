from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
from torchvision.ops import MLP


from data import Field
from model import ContactEncoder


class ContactsClassifier(nn.Module):
    @staticmethod
    def from_encoder(
        encoder: ContactEncoder, p_dropout: Optional[float] = None
    ) -> ContactsClassifier:
        return ContactsClassifier(
            encoder.fields,
            encoder.attn_dim,
            encoder.n_heads_attn,
            p_dropout=p_dropout or encoder.p_dropout,
            norm_eps=encoder.norm_eps,
        )

    def __init__(
        self,
        fields: List[Field],
        attn_dim=180,
        n_heads_attn=4,
        output_mlp_layers=8,
        p_dropout=0.0,
        norm_eps=1e-6,
    ):
        super(ContactsClassifier, self).__init__()

        # --- Save hyperparameters
        self.fields = fields
        self.attn_dim = attn_dim
        self.n_heads_attn = n_heads_attn
        self.p_dropout = p_dropout

        # --- Attention Layer
        self.que_layer = nn.Linear(attn_dim, attn_dim)
        self.key_layer = nn.Linear(attn_dim, attn_dim)
        self.val_layer = nn.Linear(attn_dim, attn_dim)

        field_lengths = sum([f.max_length for f in fields])
        self.positional_encodings = ContactEncoder.init_positional_encoding(
            2 * field_lengths, attn_dim
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
            [attn_dim] * output_mlp_layers,
            norm_layer=lambda dim: nn.LayerNorm(dim, eps=norm_eps),
            activation_layer=lambda: nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.output = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(2 * field_lengths, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, attn_embedding1: torch.Tensor, attn_embedding2: torch.Tensor
    ) -> torch.Tensor:
        batch_size, _, _ = attn_embedding1.shape
        embeddings = torch.cat([attn_embedding1, attn_embedding2], dim=1)

        # --- Attention
        # TODO: Maybe have an attention mask?
        positional_encodings = self.positional_encodings.repeat(batch_size, 1, 1)

        attn_input = embeddings + positional_encodings

        queries = self.que_layer(attn_input)
        keys = self.key_layer(attn_input)
        values = self.val_layer(attn_input)

        attn_output, attn_output_weights = self.multihead_attn(
            queries,
            keys,
            values,
        )

        norm_attn_output = self.norm_attn(attn_output + attn_input)

        # Process
        output = self.proc_attn(norm_attn_output)

        norm_output = self.norm_proc_attn(output + norm_attn_output)

        # --- Output
        processed = self.processing(norm_output)

        pooled = self.pool(processed).squeeze()

        return self.output(pooled).squeeze()
