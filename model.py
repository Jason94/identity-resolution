import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
import string


class ContactEncoder(nn.Module):
    PAD_CHARACTER = "\0"

    def __init__(
        self,
        vocab_size,
        embedding_dim=60,
        output_embedding_dim=60,
        n_heads_attn=4,
        norm_eps=1e-6,
        dropout=0.0,
    ):
        super(ContactEncoder, self).__init__()

        # --- Embedding layers
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # --- Initial Attention Layer
        self.output_embedding_dim = output_embedding_dim
        self.n_heads_attn = n_heads_attn

        self.multihead_attn = nn.MultiheadAttention(
            output_embedding_dim, n_heads_attn, batch_first=True
        )

        self.norm_attn = nn.LayerNorm(output_embedding_dim, eps=norm_eps)

        # -- Final Output Processing
        self.fc_final = nn.Sequential(
            nn.Linear(output_embedding_dim, output_embedding_dim),
            nn.ReLU(),
            nn.Linear(output_embedding_dim, output_embedding_dim),
        )

        self.norm_output = nn.LayerNorm(output_embedding_dim, eps=norm_eps)

    @staticmethod
    def create_attn_mask(lengths, max_len):
        mask = torch.arange(max_len)[None, :] >= lengths[:, None]
        return mask.bool()

    def device(self):
        return next(self.parameters()).device

    def forward(self, name_tensor, lengths, email_tensor, email_lengths):
        # Generate the initial embeddings
        embedding_name = self.embedding(name_tensor)
        embedding_email = self.embedding(email_tensor)

        # Concat field embeddings along the sequence dimension, to prepare for the attention
        combined_field_embeddings = torch.cat([embedding_name, embedding_email], dim=1)

        # Create the attention mask
        name_attn_mask = self.create_attn_mask(lengths, embedding_name.size(1)).to(
            self.device()
        )
        email_attn_mask = self.create_attn_mask(
            email_lengths, embedding_email.size(1)
        ).to(self.device())
        attn_mask = torch.cat([name_attn_mask, email_attn_mask], dim=1)

        # Pass through the multihead attention
        attn_output, attn_output_weights = self.multihead_attn(
            combined_field_embeddings,
            combined_field_embeddings,
            combined_field_embeddings,
            key_padding_mask=attn_mask,
        )

        norm_attn_output = self.norm_attn(attn_output + combined_field_embeddings)

        # Pass through the feed forward network
        ff_output = self.fc_final(norm_attn_output)

        norm_output = self.norm_output(ff_output + norm_attn_output)

        # Produce a weighted sum of the character-wise embeddings to represent the input
        final_attn_weights = attn_output_weights[:, -1, :]
        weighted_sum_output = (final_attn_weights.unsqueeze(-1) * norm_output).sum(
            dim=1
        )

        return weighted_sum_output


def create_char_to_int():
    # Create a list of all ASCII printable characters.
    chars = list(string.printable) + [ContactEncoder.PAD_CHARACTER]

    # Create a dictionary that maps each character to a unique integer.
    char_to_int = {char: i for i, char in enumerate(chars)}

    return char_to_int, chars
