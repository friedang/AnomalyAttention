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


class VoxelTrackMLPClassifier(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=3, dropout=0, track_len=5):
        """
        A simple MLP for binary classification of TP/FP based on tracking data.
        Args:
            input_size (int): Size of the input feature vector (translation, size, rotation, etc.).
            hidden_size (int): Number of neurons in hidden layers.
            num_layers (int): Number of hidden layers.
            dropout (float): Dropout probability for regularization.
        """
        super(VoxelTrackMLPClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.track_len = track_len

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

        # Output layer (binary classification)
        self.output_layer = nn.Linear(hidden_size * 2, 1)

        # VoxelNet Head
        # conv_size = [[512, 256], [256, 128], [128, 64]]
        self.voxel_conv = nn.Sequential(
            nn.Conv2d(512, 328, kernel_size=3, stride=1, padding=1, bias=True),
        )
        conv_size = [[369, 164], [164, 82], [82, 41]]
        self.conv = nn.Sequential()
        for i, size in enumerate(conv_size):
            ks = [3, 1, 1] if i == 0 else [3, 3, 0]
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
        self.freeze_module(self.backbone)
        self.freeze_module(self.neck)


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
        voxel_feat = self.neck(voxel_feat)
        
        pc = self.voxel_conv(voxel_feat) # batch_size, 128, 59, 59
        # conv_feat_for_mlp = conv_feat.view(batch_size * conv_feat.shape[1], -1)
        # conv_feat_for_mlp = conv_feat.view(batch_size * conv_feat.shape[1], 
        #                                    conv_feat.shape[2], conv_feat.shape[3])
        # import ipdb; ipdb.set_trace()
        # pc = self.fc_vox(conv_feat_for_mlp)


        # Track Branch
        # Flatten the input (batch_size * num_tracks, input_size)
        x = x.view(-1, input_size)

        # Input layer
        x_in = F.leaky_relu(self.input_layer(x))

        # Hidden layers
        for i, layer in enumerate(self.hidden_layers):
            x_out = F.leaky_relu(layer(x_in))
            x_in = x_out + x_in
            # x = self.dropout_layer(x)

        # x_in torch.Size([1230, 128]) is (batch_size * num_tracks, 128)
        x_in = x_in.view(batch_size, num_tracks, x_in.shape[1])
        x_residual = x_in
        x_in = x_in.unsqueeze(-1).repeat(1, 1, 1, x_in.shape[2])

        # Classification Head
        # Output layer
        if x_in.shape[-1] < pc.shape[-1]: 
            diff = (x_in.shape[-1] + pc.shape[-1] - 1) // x_in.shape[-1]
            diff = diff if diff != 1 else 2
            x_in = x_in.repeat(1, 1, diff, diff)
            x_in = x_in[:, :, :pc.shape[-1], :pc.shape[-1]]

        final_out = torch.cat([x_in, pc], axis=1)
        final_out = self.conv(final_out)

        final_out = final_out.view(batch_size, num_tracks, -1)
        final_out = self.fc(final_out)
        final_out = torch.cat([x_residual, final_out], axis=2)
        output = self.output_layer(final_out)

        # Reshape output to (batch_size, num_tracks, 1)
        # output = x.view(batch_size, num_tracks, 1)
        # import ipdb; ipdb.set_trace()

        return output
