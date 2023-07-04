import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
import string


class ContactEncoder(nn.Module):
    PAD_CHARACTER = "\0"

    def __init__(
        self,
        vocab_size,
        embedding_dim=50,
        hidden_dim_name=100,
        n_layers_name=8,
        hidden_dim_email=100,
        n_layers_email=8,
        output_embedding_dim=50,
        n_heads_attn=5,
        dropout=0.0,
    ):
        super(ContactEncoder, self).__init__()

        # --- Embedding layers
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # --- Name fields
        self.hidden_dim_name = hidden_dim_name
        self.n_layers_name = n_layers_name

        self.gru_name = nn.GRU(
            embedding_dim,
            hidden_dim_name,
            n_layers_name,
            dropout=(0 if n_layers_name == 1 else dropout),
            bidirectional=True,
        )
        self.fc_name = nn.Linear(hidden_dim_name * 2, output_embedding_dim)

        # --- Email field
        self.hidden_dim_email = hidden_dim_email
        self.n_layers_email = n_layers_email

        self.gru_email = nn.GRU(
            embedding_dim,
            hidden_dim_email,
            n_layers_email,
            dropout=(0 if n_layers_email == 1 else dropout),
            bidirectional=True,
        )
        self.fc_email = nn.Linear(hidden_dim_email * 2, output_embedding_dim)

        # --- Attention
        self.output_embedding_dim = output_embedding_dim
        self.n_heads_attn = n_heads_attn

        self.multihead_attn = nn.MultiheadAttention(
            output_embedding_dim * 2, n_heads_attn, batch_first=True
        )

        # --- Final Output Processing
        self.fc_final = nn.Sequential(
            nn.Linear(output_embedding_dim * 2, output_embedding_dim),
            nn.Tanh(),
            nn.Linear(output_embedding_dim, output_embedding_dim),
        )

    def _forward_name(self, name_tensor, lengths):
        # Sort name_tensor and lengths by the corresponding lengths in descending order
        sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_name_tensor = name_tensor[sorted_indices]

        # Pass name tensor through embedding layer
        embedded = self.embedding(sorted_name_tensor)

        # Pack the sequence
        packed = rnn_utils.pack_padded_sequence(
            embedded, sorted_lengths, batch_first=True, enforce_sorted=True
        )

        # Pass embeddings through layers
        packed_outputs, hidden = self.gru_name(packed)

        # Unpack the sequence
        outputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)

        # Take the output from the final time step
        out = outputs[range(outputs.shape[0]), sorted_lengths - 1, :]

        # Pass through Fully connected layer
        embedding = self.fc_name(out)

        # Unsort the output embeddings
        _, unsorted_indices = torch.sort(sorted_indices)
        unsorted_embedding = embedding[unsorted_indices]

        return unsorted_embedding

    def _forward_email(self, email_tensor, lengths):
        # Sort email_tensor and lengths by corresponding lengths in descending order
        sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_email_tensor = email_tensor[sorted_indices]

        # Pass email tensor through embedding layer
        embedded = self.embedding(sorted_email_tensor)

        # Pack the sequence
        packed = rnn_utils.pack_padded_sequence(
            embedded, sorted_lengths, batch_first=True, enforce_sorted=True
        )

        # Pass embeddings through layers
        packed_outputs, hidden = self.gru_email(packed)

        # Unpack the sequence
        outputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)

        # Take the output from the final time step
        out = outputs[range(outputs.shape[0]), sorted_lengths - 1, :]

        # Pass through Fully connected layer
        embedding = self.fc_email(out)

        # Unsort the output embeddings
        _, unsorted_indices = torch.sort(sorted_indices)
        unsorted_embedding = embedding[unsorted_indices]

        breakpoint()

        return unsorted_embedding

    def forward(self, name_tensor, lengths, email_tensor, email_lengths):
        embedding_name = self._forward_name(name_tensor, lengths)
        embedding_email = self._forward_email(email_tensor, email_lengths)

        # Concatenate name and email embeddings along the sequence dimension
        embedding = torch.cat([embedding_name, embedding_email], dim=1)

        # Pass through multihead attention
        attn_output, attn_output_weights = self.multihead_attn(
            embedding, embedding, embedding
        )

        # Pass through the final fully connected layer
        final_embedding = self.fc_final(attn_output)

        return final_embedding


def create_char_to_int():
    # Create a list of all ASCII printable characters.
    chars = list(string.printable) + [ContactEncoder.PAD_CHARACTER]

    # Create a dictionary that maps each character to a unique integer.
    char_to_int = {char: i for i, char in enumerate(chars)}

    return char_to_int, chars
