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
        center_distance_threshold: float = 2.0,
        size_factor_range: Tuple[float, float] = (0.5, 1.5),
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
        self.size_factor_range = size_factor_range
        self.fp_probability = fp_probability

    def generate_translation_noise(self, translation: torch.Tensor, threshold: float) -> torch.Tensor:
        """Generate noise vector that exceeds the center distance threshold."""
        # Generate random direction
        noise = torch.randn_like(translation)
        noise = noise / torch.norm(noise)
        
        # Scale noise to exceed threshold
        scale = threshold * (1.0 + torch.rand(1) * 0.5)  # 1.0-1.5 times threshold
        
        # scale = threshold * (1.0 + torch.rand(1) * 0.5)  # 1.0-1.5 times threshold
        # noise_dim = torch.randint(0, 2, (1,))
        # noise[noise_dim] = noise[noise_dim] + scale

        return noise + scale

    def forward(self, bbox: Dict[str, Union[torch.Tensor, list]]) -> Dict[str, torch.Tensor]:
        """
        Apply FP augmentation to a single bounding box.
        
        Args:
            bbox: Dictionary containing 'translation', 'size', and 'rotation' keys
            
        Returns:
            Augmented bounding box dictionary
        """
        if torch.rand(1) > self.fp_probability:
            return bbox

        # Convert inputs to tensors if they're lists
        # bbox = {k: torch.tensor(v) if isinstance(v, list) else v for k, v in bbox.items()}
        
        # Augment translation only
        translation_noise = self.generate_translation_noise(torch.tensor(bbox['translation']), self.center_distance_threshold)
        augmented_transloation = torch.tensor(bbox['translation']) + translation_noise
        bbox['translation'] = augmented_transloation.tolist()
        bbox['TP'] = 0
        
        return bbox

    @torch.no_grad()
    def __call__(self, bbox: Dict[str, Union[torch.Tensor, list]]) -> Dict[str, torch.Tensor]:
        return self.forward(bbox)


class SceneDataset(Dataset):
    def __init__(self, scenes_info_path=None, track_info=None, gt_scenes_info_path=None, gt_track_info=None,
                 load_chunks_from=None, sample_ratio=1, chunk_size=5):
        # scenes is a dict where keys are scene names, and values are lists of tensors
        self.chunks = []
        self.chunk_size = 5
        self.dummy = {'translation': [-500, -500, -500], 'size': [-500, -500, -500], 'rotation': [-500, -500, -500, -500],
                      'sample_token': 'dummy', 'TP': -500, 'num_lidar_pts': -500, 'tracking_id': 'dummy', 'detection_name': 'dummy'}
        
        detection_names = ['car','bus','trailer','truck','pedestrian','bicycle',
                                'motorcycle','construction_vehicle', 'barrier', 'traffic_cone']
        self.name_dict = {n: i+1 for n, i in zip(detection_names, range(len(detection_names)))}

        self.max_track_len = 41
        max_scene_size = 2400 # 3690 2350 #
        max_gt_scene_size = 300
        sample_ratio = None if sample_ratio == 1 else sample_ratio

        self.fp_aug = FalsePositiveAugmentation(
            center_distance_threshold=2.0,
            size_factor_range=(0.5, 1.5),
            fp_probability=0.7
        )

        if load_chunks_from:
            self.chunks = torch.load(load_chunks_from)
        else:
            if scenes_info_path:
                scenes_info = json.load(open(scenes_info_path, 'r'))
                track_info = json.load(open(track_info, 'r'))
                self.scene_names = list(scenes_info.keys())
                self.fill_chunks(track_info, scenes_info, max_scene_size, gt=False)
            if gt_scenes_info_path:
                gt_scenes_info = json.load(open(gt_scenes_info_path, 'r'))
                gt_track_info = json.load(open(gt_track_info, 'r'))
                self.scene_names = list(gt_scenes_info.keys())
                # for _ in range(2):
                # self.fill_chunks(gt_track_info, gt_scenes_info, max_gt_scene_size, gt=True)
                # for _ in range(5):
                #     self.fill_chunks(gt_track_info, gt_scenes_info, max_gt_scene_size, gt=True, augment=True)

            if sample_ratio:
                # TODO Move this to the top and sample scene names instead
                torch.manual_seed(random.randint(0, sys.maxsize))
                total_indices = torch.randperm(len(self.chunks))
                split = int(sample_ratio * len(self.chunks))
                self.chunks = [x for _, x in sorted(zip(total_indices, self.chunks))]
                train_chunk = self.chunks[:split]
                val_chunk = self.chunks[split:]
                torch.save(train_chunk, scenes_info_path.replace('scene2trackname.json', 'train_chunks.pt'))
                torch.save(val_chunk, scenes_info_path.replace('scene2trackname.json', 'val_chunks.pt'))

                print(f"Saved validation chunk for loading to {scenes_info_path.replace('scene2trackname.json', 'val_chunks.pt')}")
                print("Set current chunks to train chunks.")
                self.chunks = train_chunk

        self.chunks = self.chunks
        if not sample_ratio and not load_chunks_from:
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
               dummy_track = [self.dummy] * 41
               for _ in range(track_len_diff):
                   self.scenes[name].append(dummy_track)
        
        print(f"Number of non dummy items in self.scenes is {len([v for values in self.scenes.values() for k in values for v in k if v['TP'] != -500])}")
        for name in self.scenes.keys():
            # scene_data = [det for track in self.scenes[name] for det in track]
            # scene_data = []
            # for i in range(self.max_track_len):
            #     for track in self.scenes[name]:
            #         scene_data.append(track[i])

            scene_data = []
            for i in range(0, self.max_track_len, self.chunk_size):
                for track in self.scenes[name]:
                    if i + self.chunk_size > self.max_track_len:
                        scene_data.append(track[i])
                        continue
                    for j in range(self.chunk_size):
                        scene_data.append(track[i+j])

            if all(d['TP'] == -500 for d in scene_data):
                continue

            if gt and augment:
                # Augment GT detections
                for i in range(0, len(scene_data), self.chunk_size):
                    chunk = scene_data[i:i + self.chunk_size]
                    if all(d['TP'] == -500 for d in chunk):
                        continue
                    if all(d['TP'] == 1 for d in chunk):
                        # Augment the chunk
                        for j, detection in enumerate(chunk):
                            if detection['TP'] == 1:
                                # Augment the detection
                                aug_bbox = self.fp_aug(detection)
                                chunk[j].update(aug_bbox)
                    self.chunks.append(self._collate_scene(chunk, gt=True))
            else:
                # Process non-augmented chunks
                for i in range(0, len(scene_data), self.chunk_size):
                    # filter dummy-only chunks
                    if all(d['TP'] == -500 for d in scene_data[i:i + self.chunk_size]):
                        continue
                    else:
                        if len(scene_data[i:i + self.chunk_size]) < self.chunk_size:
                            len_diff  = self.chunk_size - len(scene_data[i:i + self.chunk_size])
                            for i in range(len_diff):
                                scene_data.append(self.dummy)
                        self.chunks.append(self._collate_scene(scene_data[i:i + self.chunk_size], gt=gt))

        print(f"Number of non dummy items in chunks is {len([v for values in self.chunks for v in values[1][1] if v != 'dummy'])}")

    def replace_tensor_values_in_tuples(self, data):
        # Iterate over the list of tuples
        for i, (first_tensor, _, last_tensor) in enumerate(data):
            # Replace -500 with -5 in the first tensor
            data[i] = (torch.where(first_tensor == -500, torch.tensor(-5), first_tensor),
                       data[i][1],  # Keep the middle element unchanged
                       torch.where(last_tensor == -500, torch.tensor(-5), last_tensor))
            
        print(f"Number of -5 dummy items in chunks is {len([v for values in data for v in values[0] if v == -5])}")
        return data

    # def fill_chunks(self, track_info, scenes_info, max_scene_size, gt=False):
        self.scenes = {k: [] for k in self.scene_names}
        print(f"Number of non dummy items in track_info is {len([v for values in track_info.values() for v in values if v['TP'] != -500])}")
        for name in self.scene_names:
            track_names = scenes_info[name]
            for n in track_names:
                self.scenes[name].append(track_info[n])
            
            if len(self.scenes[name]) < max_scene_size:
               track_len_diff = max_scene_size - len(self.scenes[name])
               dummy_track = [self.dummy] * 41
               for _ in range(track_len_diff):
                   self.scenes[name].append(dummy_track)
        
        print(f"Number of non dummy items in self.scenes is {len([v for values in self.scenes.values() for k in values for v in k if v['TP'] != -500])}")
        # Convert each scene's list of dicts into a tensor
        for name in self.scenes.keys():
            scene_data = [det for track in self.scenes[name] for det in track]

            if all(d['TP'] == -500 for d in scene_data):
                continue
            elif self.chunk_size == self.max_track_len:
                if len(scene_data) <= self.chunk_size:
                    len_diff  = self.chunk_size - len(scene_data)
                    for i in range(len_diff):
                        scene_data.append(self.dummy)
                self.chunks.append(self._collate_scene(scene_data, gt))
                continue
            
            for i in range(0, len(scene_data), self.chunk_size):
                # filter dummy-only chunks
                if all(d['TP'] == -500 for d in scene_data[i:i + self.chunk_size]):
                    continue
                else:
                    if len(scene_data[i:i + self.chunk_size]) < self.chunk_size:
                        len_diff  = self.chunk_size - len(scene_data[i:i + self.chunk_size])
                        for i in range(len_diff):
                            scene_data.append(self.dummy)
                    self.chunks.append(
                        self._collate_scene(scene_data[i:i + self.chunk_size], gt))
                    
        print(f"Number of non dummy items in chunks is {len([v for values in self.chunks for v in values[1][1] if v != 'dummy'])}")

    def _collate_scene(self, scene_data, gt=False):
        """
        Converts a list of dicts for one scene into a single tensor by extracting values from each field.
        Returns a tensor where the first dimension is the number of tracks (max_scene_size), 
        and subsequent dimensions depend on the field sizes.
        """
        # set_trace()
        translations = torch.tensor([d['translation'] for d in scene_data], dtype=torch.float32)
        sizes = torch.tensor([d['size'] for d in scene_data], dtype=torch.float32)
        rotations = torch.tensor([d['rotation'] for d in scene_data], dtype=torch.float32)
        tp = torch.tensor([d['TP'] for d in scene_data], dtype=torch.float32)
        num_lidar_pts = torch.tensor([d['num_lidar_pts'] for d in scene_data], dtype=torch.float32)
        tokens = [d['sample_token'] for d in scene_data]
        if gt:
            ids = [(d['sample_token'] + '_gt') if d['TP']!=-500 else 'dummy' for d in scene_data]
        else:
            ids = [(d['sample_token'] + '_' + d['tracking_id']) if ('tracking_id' in d.keys() and d['TP']!=-500) else 'dummy' for d in scene_data]

        # set_trace()
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
        # TODO bring back class number as feature
        name_number = torch.tensor([self.name_dict[d] if (isinstance(d, str) and d!='dummy') else -500 for d in det_names], dtype=torch.float32)
        # set_trace()
        # Stack all tensors together into one large tensor for the scene
        return (tp.unsqueeze(-1),
                (tokens, ids, scene_data),
                torch.cat([translations, sizes, rotations, num_lidar_pts.unsqueeze(-1)], dim=-1)) # , name_number.unsqueeze(-1)], dim=-1))

    def __len__(self):
        # return len(self.scene_names)
        return len(self.chunks)

    def __getitem__(self, idx):
        # Return all tensors from a specific scene
        return self.chunks[idx]
    

# scenes_info = json.load(open('/workspace/CenterPoint/work_dirs/immo/cp_5_seed_2hz/scene2trackname.json', "r"))
# track_info = json.load(open('/workspace/CenterPoint/work_dirs/immo/cp_5_seed_2hz/track_info_tp_padded_tracks.json', "r"))
# gt_scenes_info = json.load(open('/workspace/CenterPoint/work_dirs/immo/cp_5_seed_2hz/gt_scene2trackname.json', "r"))
# gt_track_info = json.load(open('/workspace/CenterPoint/work_dirs/immo/cp_5_seed_2hz/gt_track_info_tp_padded_tracks.json', "r"))

# chunks = torch.load('/workspace/CenterPoint/work_dirs/immo/cp_5_seed_2hz/chunks.pt')
# from ipdb import launch_ipdb_on_exception, set_trace
# with launch_ipdb_on_exception():
#     test = SceneDataset(scenes_info, track_info, gt_scenes_info, gt_track_info)
#     set_trace()
#     print(1)