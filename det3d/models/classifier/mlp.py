import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import DETECTORS

@DETECTORS.register_module
class TrackMLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        """
        A simple MLP for binary classification of TP/FP based on tracking data.
        Args:
            input_size (int): Size of the input feature vector (translation, size, rotation combined).
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
        
        # Dropout layer for regularization
        self.dropout_layer = nn.Dropout(dropout)

        self.batch_norm_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)  # Binary classification output
        
    def forward(self, x, mask=None):
        """
        Forward pass through the MLP.
        Args:
            x (Tensor): Input tensor of shape (batch_size, max_length, input_size).
            mask (Tensor): Optional mask for valid track points (shape: batch_size, max_length).
        Returns:
            output (Tensor): Binary classification scores for each track point (before sigmoid).
        """
        batch_size, max_length, input_size = x.shape
        
        # Flatten the input (batch_size * max_length, input_size)
        x = x.view(-1, input_size)
        
        # Input layer
        x = F.relu(self.input_layer(x))
        
        # Hidden layers
        for i, layer in enumerate(self.hidden_layers):
            x = F.relu(self.batch_norm_layers[i](layer(x)))
            x = self.dropout_layer(x)
    
        # Output layer
        x = self.output_layer(x)
        
        # Reshape output to (batch_size, max_length, 1)
        output = x.view(batch_size, max_length, 1)
        
        # Apply mask to ignore padded points (optional)
        if mask is not None:
            output = output * mask.unsqueeze(-1)
        
        return output