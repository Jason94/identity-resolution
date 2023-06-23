import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
import string


class ContactEncoder(nn.Module):
    PAD_CHARACTER = "\0"

    def __init__(
        self, vocab_size, embedding_dim=50, hidden_dim=100, n_layers=2, dropout=0.2
    ):
        super(ContactEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # GRU Layer
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            n_layers,
            dropout=(0 if n_layers == 1 else dropout),
            bidirectional=True,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, name_tensor):
        # Pass name tensor through embedding layer
        embedded = self.embedding(name_tensor)

        # Pass embeddings through GRU
        outputs, hidden = self.gru(embedded.view(len(name_tensor), 1, -1))

        # Take the output from the final time step
        out = outputs[-1]

        # Pass through Fully connected layer
        embedding = self.fc(out)

        return embedding

    @staticmethod
    def preprocess_names(first_name, last_name, char_to_int, max_len=50):
        # Concatenate first name and last name
        name = first_name + " " + last_name

        # Truncate or pad name to max_len
        name = name[:max_len].ljust(max_len, ContactEncoder.PAD_CHARACTER)

        # Convert name to tensor of character indices
        name_tensor = torch.tensor([char_to_int[char] for char in name])

        return name_tensor


def create_char_to_int():
    # Create a list of all ASCII printable characters.
    chars = list(string.printable) + [ContactEncoder.PAD_CHARACTER]

    # Create a dictionary that maps each character to a unique integer.
    char_to_int = {char: i for i, char in enumerate(chars)}

    return char_to_int, chars


if __name__ == "__main__":
    char_to_int, chars = create_char_to_int()

    first_name = "John"
    last_name = "Doe"
    name_tensor = ContactEncoder.preprocess_names(first_name, last_name, char_to_int)

    # Create model instance
    model = ContactEncoder(len(chars))

    # Generate embedding
    embedding = model(name_tensor)

    print(first_name, last_name)
    print(name_tensor)
    print(embedding)
