import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import DETECTORS
from Pointnet_Pointnet2_pytorch.models.pointnet_utils import PointNetEncoder

@DETECTORS.register_module
class PCTrackMLPClassifier(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=3, dropout=0, track_len=5):
        """
        A simple MLP for binary classification of TP/FP based on tracking data.
        Args:
            input_size (int): Size of the input feature vector (translation, size, rotation, etc.).
            hidden_size (int): Number of neurons in hidden layers.
            num_layers (int): Number of hidden layers.
            dropout (float): Dropout probability for regularization.
        """
        super(PCTrackMLPClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.track_len = track_len

        # PointNet
        self.pc_en = PointNetEncoder(global_feat=True, feature_transform=True, channel=5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

        # Dropout layer for regularization
        # self.dropout_layer = nn.Dropout(dropout)

        # self.batch_norm_layers = nn.ModuleList([
        #     nn.BatchNorm1d(hidden_size) for _ in range(num_layers)
        # ])

        # Output layer (binary classification)
        self.output_layer = nn.Linear(hidden_size * 2, 1)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, inputs, mask=None):
        """
        Forward pass through the MLP.
        Args:
            x (Tensor): Input tensor of shape (batch_size, 4, input_size).
            mask (Tensor): Optional mask for valid track points (shape: batch_size, 4).
        Returns:
            output (Tensor): Binary classification scores for each track point (before sigmoid).
        """
        x, pc = inputs
        # import ipdb; ipdb.set_trace()
        pc_feat, _, _ = self.pc_en(pc)
        pc = F.relu(self.bn1(self.fc1(pc_feat)))
        pc = F.relu(self.bn2(self.dropout(self.fc2(pc))))
        pc = self.fc3(pc)

        # Binary Classifier
        batch_size, num_tracks, input_size = x.shape
        
        # Flatten the input (batch_size * num_tracks, input_size)
        x = x.view(-1, input_size)

        # Input layer
        x_in = F.leaky_relu(self.input_layer(x))

        # Hidden layers
        for i, layer in enumerate(self.hidden_layers):
            x_out = F.leaky_relu(layer(x_in))
            x_in = x_out + x_in
            # x = self.dropout_layer(x)

        # Output layer
        if x_in.shape[0] != pc.shape[0]:
            diff = int(x_in.shape[0] / pc.shape[0])
            pc = pc.repeat(diff, 1)
        final_out = torch.cat([x_in, pc], axis=1)
        x = self.output_layer(final_out)

        # Reshape output to (batch_size, num_tracks, 1)
        output = x.view(batch_size, num_tracks, 1)

        # Apply mask to ignore padded points (optional)
        if mask is not None:
            output = output * mask.unsqueeze(-1)

        return output
