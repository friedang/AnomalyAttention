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
from Pointnet_Pointnet2_pytorch.models.pointnet_utils import PointNetEncoder


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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out.transpose(1, 2)).transpose(1, 2)
        
        out += identity
        out = F.relu(out)
        return out

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim * 2,  # * 2 because of bidirectional LSTM
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Add positional encoding for transformer
        seq_len = x.size(1)
        pos_encoding = torch.arange(seq_len, device=x.device).unsqueeze(0).unsqueeze(-1)
        pos_encoding = pos_encoding.expand(x.size(0), -1, 1).float()
        
        # Concatenate position encoding
        transformer_input = torch.cat([lstm_out, pos_encoding], dim=-1)
        
        # Transformer processing
        transformer_out = self.transformer(lstm_out)
        
        return transformer_out

class EnhancedTrackToPointCloud(nn.Module):
    def __init__(self, input_features=12, output_points=256, output_features=5):
        super().__init__()
        self.output_points = output_points
        self.output_features = output_features
        
        # Temporal encoding
        self.temporal_encoder = TemporalEncoder(input_features, input_features // 6)
        
        # Feature transformation with residual connections
        self.feature_transform = nn.Sequential(
            ResidualBlock(input_features, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, output_features)
        )
        
        # Point position generator
        self.position_net = nn.Sequential(
            ResidualBlock(42, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, output_points)
        )
        
        # Final point feature refinement
        self.point_feature_refinement = nn.Sequential(
            ResidualBlock(output_features, output_features),
            nn.LayerNorm(output_features)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Process temporal information
        temporal_features = self.temporal_encoder(x)
        
        # Transform features with residual connections
        point_features = self.feature_transform(x)
        point_features = point_features + x.mean(dim=2, keepdim=True).expand(-1, -1, self.output_features)
        
        # Generate point positions using temporal features
        x_temporal = temporal_features.permute(0, 2, 1) #  temporal_features.transpose(1, 2)
        point_weights = self.position_net(x_temporal)
        point_weights = F.softmax(point_weights, dim=-1)
        point_weights = point_weights.mean(dim=1, keepdim=True).expand(-1, 42, -1)
        
        # Create point cloud
        point_cloud = torch.matmul(point_weights.permute(0, 2, 1), point_features)
        
        # Refine point features
        point_cloud = self.point_feature_refinement(point_cloud)
        
        return point_cloud

class EnhancedClassificationHead(nn.Module):
    def __init__(self, input_dim=1024, sequence_length=42):
        super().__init__()
        self.sequence_length = sequence_length
        
        # Initial projection with residual connection
        self.initial_project = ResidualBlock(input_dim, 512)
        
        # Temporal decoder
        self.temporal_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=512,
                nhead=4,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Generate query embeddings for each time step
        self.query_embed = nn.Parameter(torch.randn(1, sequence_length, 512))
        
        # Final classification layers with residual connections
        self.final_layers = nn.Sequential(
            ResidualBlock(512, 256),
            ResidualBlock(256, 128),
            ResidualBlock(128, 64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Initial projection
        x = self.initial_project(x.unsqueeze(1))
        
        # Expand query embeddings for batch size
        query = self.query_embed.expand(x.size(0), -1, -1)
        
        # Temporal decoding
        decoded = self.temporal_decoder(query, x)
        
        # Final classification
        output = self.final_layers(decoded)
        
        return output

class PCTrackMLPClassifier(nn.Module):
    def __init__(self, input_features=12, num_points=10240, output_features=5):
        super().__init__()
        self.track_to_pointcloud = EnhancedTrackToPointCloud(
            input_features=input_features,
            output_points=num_points,
            output_features=output_features
        )
        
        self.pointnet = PointNetEncoder(global_feat=True, feature_transform=True, channel=5)
        
        self.classification_head = EnhancedClassificationHead(
            input_dim=1024,  # Adjust based on your PointNet output
            sequence_length=42
        )

        hidden_size=128
        num_layers=3
        dropout=0.1

        mlp_feature_dim=400
        voxel_feature_dim=512
        fused_dim=256
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
            num_heads=16, 
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

        self.trackpc_backbone = SpMiddleResNetFHD(num_input_features=5, ds_factor=4)
        self.trackpc_rpn = RPN(
            layer_nums=[5, 5],
            ds_layer_strides=[1, 2],
            ds_num_filters=[128, 256],
            us_layer_strides=[1, 2],
            us_num_filters=[256, 256],
            num_input_features=256,
        )

        conv_size = [[512, 256], [256, 128], [128, 42]]
        self.conv = nn.Sequential()
        for i, size in enumerate(conv_size):
            ks = [3, 1, 1] if i ==0 else [3, 3, 0]
            self.conv.append(
                nn.Conv2d(size[0], size[1], kernel_size=ks[0], stride=ks[1], padding=ks[2], bias=True))
            self.conv.append(nn.BatchNorm2d(size[1]))
            self.conv.append(nn.ReLU(inplace=True)
                             )
        self.fc = nn.Linear(20 * 20, hidden_size)

        self._initialize_weights()

        # Centerpoint's Voxelnet
        self.voxelizer = Voxelization()
        self.vox_reader = VoxelFeatureExtractorV3(num_input_features=5)

        self.backbone = SpMiddleResNetFHD(num_input_features=5, ds_factor=4)
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
        
    def forward(self, x):
        # Convert track to point cloud
        x, pc_raw = x
        batch_size, num_tracks, input_size = x.shape

        # VoxelNet
        sweep_points_list = []
        # TODO do PC to Voxels outside the training loop
        pc = pc_raw
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
        voxel_feat = self.neck(voxel_feat)

        voxel_features = self.conv(voxel_feat)


        # Track branch
        pc = self.track_to_pointcloud(x)
        # pc = F.interpolate(pc.unsqueeze(0).unsqueeze(0), size=(num_tracks, pc.shape[1]), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        sweep_points_list = []
        if isinstance(pc, torch.Tensor):
            pc = pc.detach().cpu().numpy()
            for i in range(len(pc)):
                sweep_points_list.append(pc[i])
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
        trackcloud_features = self.neck(voxel_feat)
        
        # Get PointNet embedding
        # trackcloud_features, _, _ = self.pointnet(point_cloud.permute(0,2,1))        

        # trackcloud_features = F.interpolate(trackcloud_features.unsqueeze(0).unsqueeze(0), size=(num_tracks, trackcloud_features.shape[1]), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        b, n, x, y = voxel_features.shape
        voxel_features = voxel_features.reshape(b, n, x * y)
        fused_features = self.fusion_layer(voxel_features, trackcloud_features)
        
        # import ipdb; ipdb.set_trace()
        # Temporal encoding with LSTM
        lstm_output, _ = self.lstm(fused_features)

        # Optional Multi-Head Attention
        mha_output, _ = self.multihead_attention(lstm_output, lstm_output, lstm_output)

        # Classification
        logits = self.classifier(mha_output)

        return logits


# class PCTrackMLPClassifier(nn.Module):
#     def __init__(self, input_size=12, hidden_size=128, num_layers=3, dropout=0, track_len=5):
#         """
#         A simple MLP for binary classification of TP/FP based on tracking data.
#         Args:
#             input_size (int): Size of the input feature vector (translation, size, rotation, etc.).
#             hidden_size (int): Number of neurons in hidden layers.
#             num_layers (int): Number of hidden layers.
#             dropout (float): Dropout probability for regularization.
#         """
#         super(PCTrackMLPClassifier, self).__init__()

#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.track_len = track_len

#         # PointNet
#         self.pc_en = PointNetEncoder(global_feat=True, feature_transform=True, channel=5)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, hidden_size)
#         self.dropout = nn.Dropout(p=0.1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.relu = nn.ReLU()

#         # Input layer
#         self.input_layer = nn.Linear(input_size, hidden_size)

#         # Hidden layers
#         self.hidden_layers = nn.ModuleList([
#             nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
#         ])

#         # Dropout layer for regularization
#         # self.dropout_layer = nn.Dropout(dropout)

#         # self.batch_norm_layers = nn.ModuleList([
#         #     nn.BatchNorm1d(hidden_size) for _ in range(num_layers)
#         # ])

#         # Output layer (binary classification)
#         self.output_layer = nn.Linear(hidden_size * 2, 1)

#         self._initialize_weights()


#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 torch.nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
        
#     def forward(self, inputs, mask=None):
#         """
#         Forward pass through the MLP.
#         Args:
#             x (Tensor): Input tensor of shape (batch_size, 4, input_size).
#             mask (Tensor): Optional mask for valid track points (shape: batch_size, 4).
#         Returns:
#             output (Tensor): Binary classification scores for each track point (before sigmoid).
#         """
#         x, pc = inputs
#         # import ipdb; ipdb.set_trace()
#         pc_feat, _, _ = self.pc_en(pc)
#         pc = F.relu(self.bn1(self.fc1(pc_feat)))
#         pc = F.relu(self.bn2(self.dropout(self.fc2(pc))))
#         pc = self.fc3(pc)

#         # Binary Classifier
#         batch_size, num_tracks, input_size = x.shape
        
#         # Flatten the input (batch_size * num_tracks, input_size)
#         x = x.view(-1, input_size)

#         # Input layer
#         x_in = F.leaky_relu(self.input_layer(x))

#         # Hidden layers
#         for i, layer in enumerate(self.hidden_layers):
#             x_out = F.leaky_relu(layer(x_in))
#             x_in = x_out + x_in
#             # x = self.dropout_layer(x)

#         # Output layer
#         if x_in.shape[0] != pc.shape[0]:
#             diff = int(x_in.shape[0] / pc.shape[0])
#             pc = pc.repeat(diff, 1)
#         final_out = torch.cat([x_in, pc], axis=1)
#         x = self.output_layer(final_out)

#         # Reshape output to (batch_size, num_tracks, 1)
#         output = x.view(batch_size, num_tracks, 1)

#         # Apply mask to ignore padded points (optional)
#         if mask is not None:
#             output = output * mask.unsqueeze(-1)

#         return output
