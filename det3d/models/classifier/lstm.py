import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=5, dropout=0.01):
        """
        LSTM-based classifier for tracking data.

        Args:
            input_size (int): Size of the input feature vector.
            hidden_size (int): Number of hidden units in the LSTM.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability for LSTM and fully connected layers.
        """
        super(LSTMClassifier, self).__init__()

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,  # Set to True if bidirectional LSTM is needed
        )

        # Fully connected classifier head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        """
        Forward pass through the LSTM-based classifier.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, input_size).
                        Padded values are all equal to -500.

        Returns:
            output (Tensor): Binary classification scores for each track point 
                             (before sigmoid), shape (batch_size, seq_length, 1).
        """
        # Masking padded values for packed sequences
        _, padded_length, _ = x.shape
        mask = (x != -500).all(dim=-1)  # Shape: (batch_size, seq_length)
        lengths = mask.sum(dim=1)  # Calculate valid lengths for each sequence

        # Sort sequences by length (required for packing)
        lengths_sorted, perm_idx = lengths.sort(0, descending=True)
        x = x[perm_idx]  # Rearrange input according to sorted lengths
        mask = mask[perm_idx]  # Rearrange mask accordingly

        # Pack the sequence to ignore padding
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True)

        # LSTM forward pass
        packed_output, (h_n, _) = self.lstm(packed_input)  # h_n contains the final hidden states

        # Unpack the sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Restore original order
        _, unperm_idx = perm_idx.sort(0)
        output = output[unperm_idx]

        # Classification head for each time step
        output = self.fc(output)  # Shape: (batch_size, seq_length, 1)

        return output