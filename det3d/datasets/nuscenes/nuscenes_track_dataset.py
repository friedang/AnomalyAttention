import sys
import pickle
import json
import random
import operator
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import wandb

from ipdb import set_trace
from functools import reduce
from pathlib import Path
from copy import deepcopy

# try:
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.utils.splits import create_splits_scenes
# except:
    # print("nuScenes devkit not found!")

from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.nuscenes.utils import load_json, save_json
from det3d.datasets.nuscenes.nusc_common import (
    general_to_detection,
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main
)
from det3d.datasets.registry import DATASETS


import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Union, Tuple


class FalsePositiveAugmentation(torch.nn.Module):
    def __init__(
        self,
        center_distance_threshold: float = 4.0,
        fp_probability: float = 0.5
    ):
        """
        Initialize FP augmentation with class-independent center distance threshold.
        
        Args:
            center_distance_threshold: Maximum center distance threshold (meters)
            size_factor_range: Range for random size scaling
            fp_probability: Probability of converting a GT to FP
        """
        super().__init__()
        self.center_distance_threshold = center_distance_threshold
        self.fp_probability = fp_probability

    def generate_translation_noise(self, translation: torch.Tensor, threshold: float) -> torch.Tensor:
        """Generate noise vector that exceeds the center distance threshold."""
        # Randomly decide how many dimensions to perturb (1, 2, or all 3)
        num_dims_to_modify = torch.randint(1, 4, (1,)).item()  # Randomly choose 1, 2, or 3 dimensions

        # Create a mask for selected dimensions
        dim_indices = torch.randperm(3)[:num_dims_to_modify]  # Randomly select dimensions
        mask = torch.zeros_like(translation, dtype=torch.bool)
        mask[dim_indices] = True

        # Generate noise for the selected dimensions
        noise = torch.randn_like(translation)
        noise = noise / torch.norm(noise)
        # noise[mask] = 1

        # threshold = threshold / torch.sqrt(torch.tensor(num_dims_to_modify, dtype=torch.float))

        # Scale the noise to ensure the total displacement exceeds the threshold
        if num_dims_to_modify == 1:
            scale = random.uniform(5.6, 8) # scale = threshold * (1.15 + torch.rand(1) * 0.6) # min is 4.6, max 7 [4.6, 7]
        elif num_dims_to_modify == 2:
            scale = random.uniform(4.3, 5.75) # scale = threshold * (1 + torch.rand(1) * 0.5) # min is 4.6, max 7 [3.3, 4.75]
        else:
            scale = random.uniform(3.75, 5) # scale = threshold * (1.0 + torch.rand(1) * 0.25)  # [2.75, 4]
        
        noise[mask] += scale

        assert np.linalg.norm(noise.cpu().numpy()) > 4.5

        return noise

    def forward(self, bbox: Dict[str, Union[torch.Tensor, list]]) -> Dict[str, torch.Tensor]:
        """
        Apply FP augmentation to a single bounding box.
        """
        if torch.rand(1) > self.fp_probability:
            return bbox

        translation_noise = self.generate_translation_noise(torch.tensor(bbox['translation']), self.center_distance_threshold)
        augmented_translation = torch.tensor(bbox['translation']) + translation_noise
        bbox['translation'] = augmented_translation.tolist()
        bbox['TP'] = 0
        bbox['dist_TP'] = [0, 0, 0, 0]

        return bbox

    @torch.no_grad()
    def __call__(self, bbox: Dict[str, Union[torch.Tensor, list]]) -> Dict[str, torch.Tensor]:
        return self.forward(bbox)


class SceneDataset(Dataset):
    def __init__(self, scenes_info_path=None, track_info=None, gt_scenes_info_path=None, gt_track_info=None,
                 load_chunks_from=None, sample_ratio=1, chunk_size=42, inference=False):
        # scenes is a dict where keys are scene names, and values are lists of tensors
        self.inference = inference
        self.chunks = []
        self.chunk_size = chunk_size
        self.dummy = {'translation': [-500, -500, -500], 'size': [-500, -500, -500], 'rotation': [-500, -500, -500, -500],
                      'sample_token': 'dummy', 'TP': -500, 'num_lidar_pts': -500, 'tracking_id': 'dummy', 'detection_name': 'dummy', 'dist_TP': [0, 0, 0, 0]}
        
        detection_names = ['car','bus','trailer','truck','pedestrian','bicycle',
                                'motorcycle','construction_vehicle', 'barrier', 'traffic_cone']
        self.name_dict = {n: i+1 for n, i in zip(detection_names, range(len(detection_names)))}

        self.max_track_len = 42
        # TODO Delete dummy adding below for 42 and do this in utils
        lengths_thresh = 2
        sample_ratio = None if sample_ratio == 1 else sample_ratio

        self.fp_aug = FalsePositiveAugmentation(
            center_distance_threshold=4.0,
            fp_probability=0.7
        )

        if load_chunks_from:
            self.chunks = torch.load(load_chunks_from)
        else:
            scenes_info = json.load(open(scenes_info_path, 'r'))
            track_info = json.load(open(track_info, 'r'))
            
            # Remove tracks if their lengths is smaller threshold
            assert len([v for val in scenes_info.values() for v in val]) == len(track_info)
            print(f"Number of tracks before threshold filtering: {len(track_info)}")
            track_info = {k: v + [self.dummy] for k, v in track_info.items() if len([det for det in v if det['TP'] in [0, 1]]) > lengths_thresh}
            scenes_info = {k: [name for name in v if name in track_info.keys()] for k, v in scenes_info.items()}
            assert len([v for val in scenes_info.values() for v in val]) == len(track_info)
            print(f"Number of tracks after threshold filtering: {len(track_info)}")

            max_scene_size = max([len(v) for v in scenes_info.values()])
            print(f"Maximum Scene size is: {max_scene_size}")
            self.scene_names = list(scenes_info.keys())
            if sample_ratio:
                torch.manual_seed(random.randint(0, sys.maxsize))
                total_indices = torch.randperm(len(self.scene_names))
                split = int(sample_ratio * len(self.scene_names))
                self.scene_names = [x for _, x in sorted(zip(total_indices, self.scene_names))]
                train_scenes = self.scene_names[:split]
                val_scenes = self.scene_names[split:]
                scene_split = [train_scenes, val_scenes]
            else:
                scene_split = [self.scene_names]

            chunk_split = []
            if gt_scenes_info_path:
                    gt_scenes_info = json.load(open(gt_scenes_info_path, 'r'))
                    gt_track_info = json.load(open(gt_track_info, 'r'))
                    gt_track_info = {k: v + [self.dummy] for k, v in gt_track_info.items()}
                    max_gt_scene_size = max([len(v) for v in gt_track_info.values()])
                    print(f"Maximum GT Scene size is: {max_gt_scene_size}")
            for i, split in enumerate(scene_split):
                self.scene_names = split
                if i == 0:
                    for _ in range(2):
                        self.fill_chunks(track_info, scenes_info, max_scene_size, gt=False)
                else:
                    self.fill_chunks(track_info, scenes_info, max_scene_size, gt=False)

                if gt_scenes_info_path and i == 0:
                    self.fill_chunks(gt_track_info, gt_scenes_info, max_gt_scene_size, gt=True)
                    for _ in range(2):
                        self.fill_chunks(gt_track_info, gt_scenes_info, max_gt_scene_size, gt=True, augment=True)

                if sample_ratio:
                    chunk_split.append(self.chunks)
                    self.chunks = []

            if sample_ratio:
                torch.save(chunk_split[0], scenes_info_path.replace('scene2trackname.json', 'train_chunks.pt'))
                torch.save(chunk_split[1], scenes_info_path.replace('scene2trackname.json', 'val_chunks.pt'))
                print(f"Saved validation chunk for loading to {scenes_info_path.replace('scene2trackname.json', 'val_chunks.pt')}")
                print("Set current chunks to train chunks.")
                self.chunks = chunk_split[0]
            else:
                torch.save(self.chunks, scenes_info_path.replace('scene2trackname.json', 'inference_chunks.pt'))

    def fill_chunks(self, track_info, scenes_info, max_scene_size, gt=False, augment=False):
        self.scenes = {k: [] for k in self.scene_names}
        print(f"Number of non dummy items in track_info is {len([v for values in track_info.values() for v in values if v['TP'] != -500])}")

        for name in self.scene_names:
            track_names = scenes_info[name]
            for n in track_names:
                self.scenes[name].append(track_info[n])
            
            if len(self.scenes[name]) < max_scene_size:
               track_len_diff = max_scene_size - len(self.scenes[name])
               dummy_track = [self.dummy] * self.max_track_len
               for _ in range(track_len_diff):
                   self.scenes[name].append(dummy_track)
        
        print(f"Number of non dummy items in self.scenes is {len([v for values in self.scenes.values() for k in values for v in k if v['TP'] != -500])}")
        for name in self.scenes.keys():

            scene_data = []
            for i in range(0, self.max_track_len, self.chunk_size):
                for track in self.scenes[name]:
                    if i + self.chunk_size > self.max_track_len:
                        scene_data.append(track[i])
                        continue
                    for j in range(self.chunk_size):
                        scene_data.append(track[i+j])

            # Fill chunks
            for i in range(0, len(scene_data), self.chunk_size):
                chunk = scene_data[i:i + self.chunk_size]
                if gt and augment and not all(d['TP'] == -500 for d in chunk):
                    # Augment the chunk
                    for j, detection in enumerate(chunk):
                        if detection['TP'] == 1:
                            aug_bbox = self.fp_aug(detection)
                            chunk[j].update(aug_bbox)
                # Add dummy
                if len(chunk) != self.chunk_size:
                    len_diff  = abs(self.chunk_size - len(chunk))
                    for i in range(len_diff):
                        scene_data.append(self.dummy)

                self.chunks.append(self._collate_scene(chunk, gt=gt))

        print(f"Number of non dummy items in chunks is {len([v for values in self.chunks for v in values[1][1] if v != 'dummy'])}")

    def _collate_scene(self, scene_data, gt=False):
        """
        Converts a list of dicts for one scene into a single tensor by extracting values from each field.
        Returns a tensor where the first dimension is the number of tracks (max_scene_size), 
        and subsequent dimensions depend on the field sizes.
        """
        translations = torch.tensor([d['translation'] for d in scene_data], dtype=torch.float32)
        sizes = torch.tensor([d['size'] for d in scene_data], dtype=torch.float32)
        rotations = torch.tensor([d['rotation'] for d in scene_data], dtype=torch.float32)
        tp = torch.tensor([d['TP'] for d in scene_data], dtype=torch.float32)
        num_lidar_pts = torch.tensor([d['num_lidar_pts'] for d in scene_data], dtype=torch.float32)
        tokens = [d['sample_token'] for d in scene_data]
        if not self.inference:
            dist_tps = torch.tensor([d['dist_TP'] for d in scene_data], dtype=torch.float32)
        if gt:
            ids = [(d['sample_token'] + '_gt') if d['TP']!=-500 else 'dummy' for d in scene_data]
        else:
            ids = [(d['sample_token'] + '_' + d['tracking_id']) if ('tracking_id' in d.keys() and d['TP']!=-500) else 'dummy' for d in scene_data]

        output_keys = ['translation', 'size', 'velocity', 'rotation', 'tracking_score', 'sample_token',
                       'tracking_id', 'tracking_name', 'detection_name', 'detection_score', 'attribute_name']
        
        # TODO repalce all of this with json Encoder for tensors in test_ad
        for d in scene_data:
            for k in output_keys:
                if k not in d.keys():
                    d[k] = 'dummy'
                elif isinstance(d[k], list):
                    for i in range(len(d[k])):
                        if isinstance(d[k][i], torch.Tensor):
                            d[k][i] = float(d[k][i])
                elif isinstance(d[k], torch.Tensor):
                    d[k] = float(d[k])
                
        # dict_keys(['translation', 'size', 'velocity', 'rotation', 'tracking_score', 'sample_token', 'tracking_id', 'tracking_name', 'detection_name', 'detection_score', 'attribute_name', 'TP', 'num_lidar_pts'])
        det_names = [dic['detection_name'] if dic['detection_name']!=-500 else 'dummy' for dic in scene_data]
        name_number = torch.tensor([self.name_dict[d] if (isinstance(d, str) and d!='dummy') else -500 for d in det_names], dtype=torch.float32)

        if self.inference:
            return (tp.unsqueeze(-1),
                    (tokens, ids, scene_data),
                    torch.cat([translations, sizes, rotations, num_lidar_pts.unsqueeze(-1), name_number.unsqueeze(-1)], dim=-1))
        else:
            return (tp.unsqueeze(-1),
                    (tokens, ids, dist_tps),
                    torch.cat([translations, sizes, rotations, num_lidar_pts.unsqueeze(-1), name_number.unsqueeze(-1)], dim=-1))


    def __len__(self):
        # return len(self.scene_names)
        return len(self.chunks)

    def __getitem__(self, idx):
        # Return all tensors from a specific scene
        return self.chunks[idx]
