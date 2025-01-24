from det3d.core.sampler import preprocess as prep
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.torchie.trainer import load_checkpoint
from ..readers.voxel_encoder import VoxelFeatureExtractorV3
from ..backbones.scn import SpMiddleResNetFHD
from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.models.necks import RPN
# from det3d.torchie import Config


class TrackMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(TrackMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Input: track features of shape (batch_size, num_tracks, feature_dim)
        """
        batch_size, num_tracks, feature_dim = x.size()
        x = self.mlp(x.view(-1, feature_dim))  # Shape: (batch_size * num_tracks, output_dim)
        return x.view(batch_size, num_tracks, -1)  # Reshape to (batch_size, num_tracks, output_dim)


class CrossAttentionFusion(nn.Module):
    def __init__(self, voxel_feature_dim, mlp_feature_dim, fused_dim):
        super(CrossAttentionFusion, self).__init__()
        self.query_proj = nn.Linear(mlp_feature_dim, fused_dim)
        self.key_proj = nn.Linear(voxel_feature_dim, fused_dim)
        self.value_proj = nn.Linear(voxel_feature_dim, fused_dim)
        self.output_proj = nn.Linear(fused_dim, fused_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, track_features, voxel_features):
        """
        Inputs:
          - track_features: (batch_size, num_tracks, mlp_feature_dim)
          - voxel_features: (batch_size, voxel_feature_dim, height, width)
        """
        # Reshape voxel features to match track dimensions
        batch_size, voxel_dim, height, width = voxel_features.size()
        voxel_features_flat = voxel_features.view(batch_size, voxel_dim, -1)  # (batch_size, voxel_feature_dim, height*width)
        voxel_features_flat = voxel_features_flat.permute(0, 2, 1)  # (batch_size, height*width, voxel_feature_dim)

        queries = self.query_proj(track_features)  # (batch_size, num_tracks, fused_dim)
        keys = self.key_proj(voxel_features_flat)  # (batch_size, height*width, fused_dim)
        values = self.value_proj(voxel_features_flat)  # (batch_size, height*width, fused_dim)

        # Cross-attention
        attention = torch.matmul(queries, keys.transpose(-1, -2))  # (batch_size, num_tracks, height*width)
        attention = self.softmax(attention)
        attended_values = torch.matmul(attention, values)  # (batch_size, num_tracks, fused_dim)

        # Residual connection
        fused_features = self.output_proj(attended_values + queries)  # Residual fusion

        return fused_features

class Voxelization(object):
    def __init__(self, **kwargs):
        self.range = [-54, -54, -5.0, 54, 54, 3.0]
        self.voxel_size = [0.075, 0.075, 0.2]
        self.max_points_in_voxel = 10
        self.max_voxel_num = [120000, 160000]

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )

    def __call__(self, points):
        res = {"points": points}

        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size
        max_voxels = self.max_voxel_num[0]

        voxels, coordinates, num_points = self.voxel_generator.generate(
            res["points"], max_voxels=max_voxels 
        )
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        voxels = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
            range=pc_range,
            size=voxel_size
        )

        return voxels


class AnomalyAttention(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=3, dropout=0.1,
                 track_len=5, mlp_feature_dim=128, num_heads=16):
        """
        A simple MLP for binary classification of TP/FP based on tracking data.
        Args:
            input_size (int): Size of the input feature vector (translation, size, rotation, etc.).
            hidden_size (int): Number of neurons in hidden layers.
            num_layers (int): Number of hidden layers.
            dropout (float): Dropout probability for regularization.
        """
        super(AnomalyAttention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.track_len = track_len

        # Output layer (binary classification)
        self.output_layer = nn.Linear(hidden_size * 2, 1)

        track_input_dim=input_size
        voxel_feature_dim=512
        fused_dim=hidden_size * 2
        self.track_mlp = TrackMLP(track_input_dim, mlp_feature_dim, mlp_feature_dim)
        self.fusion_layer = CrossAttentionFusion(voxel_feature_dim, mlp_feature_dim, fused_dim)
        
        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=fused_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        self._initialize_weights()

        # Centerpoint's Voxelnet
        self.voxelizer = Voxelization()
        self.vox_reader = VoxelFeatureExtractorV3(num_input_features=5)

        self.backbone = SpMiddleResNetFHD(num_input_features=5, ds_factor=8)
        self.neck = RPN(
            layer_nums=[5, 5],
            ds_layer_strides=[1, 2],
            ds_num_filters=[128, 256],
            us_layer_strides=[1, 2],
            us_num_filters=[256, 256],
            num_input_features=256,
        )
        weights_path = './work_dirs/5_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/latest.pth'
        checkpoint = torch.load(weights_path) if torch.cuda.is_available() else torch.load(weights_path, map_location=torch.device('cpu'))
        
        backbone_checkpoint = {k.replace('backbone.', '') : v for k, v in checkpoint['state_dict'].items() if k.replace('backbone.', '') in self.backbone.state_dict()}
        self.backbone.load_state_dict(backbone_checkpoint)
        neck_checkpoint = {k.replace('neck.', '') : v for k, v in checkpoint['state_dict'].items() if k.replace('neck.', '') in self.neck.state_dict()}
        self.neck.load_state_dict(neck_checkpoint)

        # Freeze weights of specific modules
        # self.freeze_module(self.backbone)
        # self.freeze_module(self.neck)


    def freeze_module(self, module):
        """Set requires_grad=False for all parameters in the given module."""
        for param in module.parameters():
            param.requires_grad = False

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
        batch_size, num_tracks, input_size = x.shape

        # VoxelNet
        sweep_points_list = []
        # TODO do PC to Voxels outside the training loop
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().numpy()
            for i in range(len(pc)):
                sweep_points_list.append(pc[i].T)
            pc = np.concatenate(sweep_points_list, axis=0)
        
        v_data = self.voxelizer(pc)
        v_data['coordinates'] = np.pad(
                v_data['coordinates'], ((0, 0), (1, 0)), mode="constant", constant_values=i
        )
        v_data = {k: torch.tensor(v, device=x.device) if k != 'shape' else v
                  for k, v in v_data.items()}
        data = dict(
                features=v_data['voxels'],
                num_voxels=v_data["num_points"],
                coors=v_data["coordinates"],
                batch_size=batch_size,
                input_shape=v_data["shape"], # use ["shape"][0] if outside batching 
        )
    
        input_features = self.vox_reader(data["features"],
                                    data['num_voxels'])

        voxel_feat, _ = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        voxel_features = self.neck(voxel_feat)

        # Process track features
        track_features = self.track_mlp(x)  # (batch_size, 41, mlp_feature_dim)

        # Fuse features
        fused_features = self.fusion_layer(track_features, voxel_features)  # (batch_size, 41, fused_dim)

        # Temporal encoding with LSTM
        # import ipdb; ipdb.set_trace()
        lstm_output, _ = self.lstm(fused_features)

        # Optional Multi-Head Attention
        mha_output, _ = self.multihead_attention(lstm_output, lstm_output, lstm_output)

        # Classification
        logits = self.classifier(mha_output)

        return logits