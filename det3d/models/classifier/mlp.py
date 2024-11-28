import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackMLPClassifier(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=3, dropout=0):
        """
        A simple MLP for binary classification of TP/FP based on tracking data.
        Args:
            input_size (int): Size of the input feature vector (translation, size, rotation, etc.).
            hidden_size (int): Number of neurons in hidden layers.
            num_layers (int): Number of hidden layers.
            dropout (float): Dropout probability for regularization.
        """
        super(TrackMLPClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

        # self.hidden_layers = nn.ModuleList()
        # for _ in range(num_layers):
        #     self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        #     self.hidden_layers.append(nn.LayerNorm(hidden_size))

        # Output layer (binary classification)
        self.output_layer = nn.Linear(hidden_size, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x, mask=None):
        """
        Forward pass through the MLP.
        Args:
            x (Tensor): Input tensor of shape (batch_size, 4, input_size).
            mask (Tensor): Optional mask for valid track points (shape: batch_size, 4).
        Returns:
            output (Tensor): Binary classification scores for each track point (before sigmoid).
        """
        batch_size, num_tracks, input_size = x.shape
        
        # Flatten the input (batch_size * num_tracks, input_size)
        x = x.view(-1, input_size)

        # Input layer
        x_in = F.leaky_relu(self.input_layer(x))

        # Hidden layers
        # for layer in self.hidden_layers:
        #         x_out = layer(x_in)
        #         if isinstance(layer, nn.LayerNorm):
        #             x_out = F.leaky_relu(x_in)  # Apply activation after BatchNorm
        #         x_in = x_out + x_in

        for i, layer in enumerate(self.hidden_layers):
            x_out = F.leaky_relu(layer(x_in))
            x_in = x_out + x_in

        # Output layer
        x = self.output_layer(x_in)

        # Reshape output to (batch_size, num_tracks, 1)
        output = x.view(batch_size, num_tracks, 1)

        # Apply mask to ignore padded points (optional)
        if mask is not None:
            output = output * mask.unsqueeze(-1)

        return output
