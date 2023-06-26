import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
import string


class ContactEncoder(nn.Module):
    PAD_CHARACTER = "\0"

    def __init__(
        self, vocab_size, embedding_dim=50, hidden_dim=100, n_layers=4, dropout=0.2
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

    def forward(self, name_tensor, lengths):
        # Sort name_tensor and lengths by the corresponding lengths in descending order
        sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_name_tensor = name_tensor[sorted_indices]

        # Pass name tensor through embedding layer
        embedded = self.embedding(sorted_name_tensor)

        # Pack the sequence
        packed = rnn_utils.pack_padded_sequence(
            embedded, sorted_lengths, batch_first=True, enforce_sorted=False
        )

        # Pass embeddings through GRU
        packed_outputs, hidden = self.gru(packed)

        # Unpack the sequence
        outputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)

        # Take the output from the final time step
        out = outputs[range(outputs.shape[0]), lengths - 1, :]

        # Pass through Fully connected layer
        embedding = self.fc(out)

        return embedding

    @staticmethod
    def preprocess_names(first_name, last_name, char_to_int, max_len=50):
        # Concatenate first name and last name
        try:
            name = first_name + " " + last_name
        except Exception as e:
            print(f"Error processing '{first_name}' & '{last_name}'")
            raise e

        # Truncate or pad name to max_len
        name = name[:max_len].ljust(max_len, ContactEncoder.PAD_CHARACTER)

        # Convert name to tensor of character indices
        name_tensor = torch.tensor([char_to_int[char] for char in name])

        # Count the number of non-pad characters
        non_pad_count = (
            (name_tensor != char_to_int[ContactEncoder.PAD_CHARACTER]).sum().item()
        )

        # Return name tensor and its length
        return name_tensor, non_pad_count


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
    name_tensor, lengths = ContactEncoder.preprocess_names(
        first_name, last_name, char_to_int
    )

    # Create model instance
    model = ContactEncoder(len(chars))

    # Generate embedding
    embedding = model(name_tensor.unsqueeze(0), torch.tensor([lengths]))

    print(first_name, last_name)
    print(name_tensor)
    print(embedding)
