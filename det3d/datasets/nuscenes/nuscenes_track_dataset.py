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

class SceneDataset(Dataset):
    def __init__(self, scenes_info, track_info):
        # scenes is a dict where keys are scene names, and values are lists of tensors
        scenes_info = scenes_info
        track_info = track_info
        self.scene_names = list(scenes_info.keys())

        self.chunks = []
        chunk_size = 4

        dummy = {'translation': [-500, -500, -500], 'size': [-500, -500, -500], 'rotation': [-500, -500, -500, -500],
                      'sample_token': 'dummy', 'TP': -500, 'num_lidar_pts': -500}
        max_track_len = 41
        max_scene_size = 2348

        self.scenes = {k: [] for k in self.scene_names}
        for name in self.scene_names:
            track_names = scenes_info[name]
            for n in track_names:
                self.scenes[name].append(track_info[n])
            
            if len(self.scenes[name]) < max_scene_size:
               track_len_diff = max_scene_size - len(self.scenes[name])
               dummy_track = [dummy] * 41
               for _ in range(track_len_diff):
                   self.scenes[name].append(dummy_track)

        # Convert each scene's list of dicts into a tensor
        for name in self.scenes.keys():
            scene_data = self.scenes[name]
            for i in range(0, max_track_len*chunk_size - 1, chunk_size):
                scene_data = self.scenes[name][0] + self.scenes[name][1] + self.scenes[name][2] + self.scenes[name][3]
                if all(d == dummy for d in scene_data[i:i + chunk_size]):
                    continue
                else:
                    self.chunks.append(
                        self._collate_scene(scene_data[i:i + chunk_size]))
            # self.scenes[name] = self._collate_scene(self.scenes[name])

    def _collate_scene(self, scene_data):
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

        # Stack all tensors together into one large tensor for the scene
        return (tp.unsqueeze(-1),
                tokens,
                torch.cat([translations, sizes, rotations, num_lidar_pts.unsqueeze(-1)], dim=-1))

    def __len__(self):
        # return len(self.scene_names)
        return len(self.chunks)

    def __getitem__(self, idx):
        # Return all tensors from a specific scene
        return self.chunks[idx]
    

import json
scenes_info = json.load(open('/workspace/CenterPoint/work_dirs/immo/cp_5_seed_2hz/scene2trackname_tp.json', "r"))
track_info = json.load(open('/workspace/CenterPoint/work_dirs/immo/cp_5_seed_2hz/tp_padded_tracks.json', "r"))

from ipdb import launch_ipdb_on_exception, set_trace
with launch_ipdb_on_exception():
    test = SceneDataset(scenes_info, track_info)


@DATASETS.register_module
class NuScenesTrackDataset(PointCloudDataset):
    NumPointFeatures = 5  # x, y, z, intensity, ring_index

    def __init__(
        self,
        info_path,
        root_path,
        nsweeps=0, # here set to zero to catch unset nsweep
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        version="v1.0-trainval",
        load_interval=1,
        sample_ratio=1,
        load_indices=None,
        **kwargs,
    ):
        self.load_interval = load_interval 
        super(NuScenesTrackDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names, sample_ratio=sample_ratio,
        )

        self.nsweeps = nsweeps
        assert self.nsweeps > 0, "At least input one sweep please!"
        print(self.nsweeps)

        self._info_path = info_path
        self._class_names = class_names

        self.load_infos(self._info_path, sample_ratio, load_indices)
        self.flag = np.ones(len(self), dtype=np.uint8)

        self._num_point_features = NuScenesTrackDataset.NumPointFeatures
        self._name_mapping = general_to_detection

        self.virtual = kwargs.get('virtual', False)
        if self.virtual:
            self._num_point_features = 16 

        self.version = version
        self.eval_version = "detection_cvpr_2019"

    def reset(self):
        self.logger.info(f"re-sample {self.frac} frames from full set")
        random.shuffle(self._nusc_infos_all)
        self._nusc_infos = self._nusc_infos_all[: self.frac]

    def load_infos(self, info_path, sample_ratio=1, load_indices=None):
        with open(self._info_path, "rb") as f:
            _nusc_infos_all = load_json(f)

        _nusc_infos_all = _nusc_infos_all[::self.load_interval]

        self._nusc_infos = list(_nusc_infos_all.keys())

    def __len__(self):

        if not hasattr(self, "_nusc_infos"):
            self.load_infos(self._info_path)

        return len(self._nusc_infos)

    @property
    def ground_truth_annotations(self):
        if "gt_boxes" not in self._nusc_infos[0]:
            return None
        cls_range_map = config_factory(self.eval_version).serialize()['class_range']
        gt_annos = []
        for info in self._nusc_infos:
            gt_names = np.array(info["gt_names"])
            gt_boxes = info["gt_boxes"]
            mask = np.array([n != "ignore" for n in gt_names], dtype=np.bool_)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            # det_range = np.array([cls_range_map[n] for n in gt_names_mapped])
            det_range = np.array([cls_range_map[n] for n in gt_names])
            det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
            mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
            mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)
            N = int(np.sum(mask))
            gt_annos.append(
                {
                    "bbox": np.tile(np.array([[0, 0, 50, 50]]), [N, 1]),
                    "alpha": np.full(N, -10),
                    "occluded": np.zeros(N),
                    "truncated": np.zeros(N),
                    "name": gt_names[mask],
                    "location": gt_boxes[mask][:, :3],
                    "dimensions": gt_boxes[mask][:, 3:6],
                    "rotation_y": gt_boxes[mask][:, 6],
                    "token": info["token"],
                }
            )
        return gt_annos

    def get_sensor_data(self, idx):

        info = self._nusc_infos[idx]

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "nsweeps": self.nsweeps,
                # "ground_plane": -gp[-1] if with_gp else None,
                "annotations": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "virtual": self.virtual 
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def evaluation(self, detections, output_dir=None, testset=False, train=False):
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "train" if train else "val",
            "v1.0-test": "test",
        }

        # if not testset:
        #     dets = []
        #     gt_annos = self.ground_truth_annotations
        #     assert gt_annos is not None

        #     miss = 0
        #     for gt in gt_annos:
        #         try:
        #             dets.append(detections[gt["token"]])
        #         except Exception:
        #             miss += 1

        #     assert miss == 0
        # else:
        #     dets = [v for _, v in detections.items()]
        #     assert len(detections) == 6008

        # nusc_annos = {
        #     "results": {},
        #     "meta": None,
        # }

        nusc = NuScenes(version=self.version, dataroot=str(self._root_path), verbose=True)

        mapped_class_names = []
        for n in self._class_names:
            if n in self._name_mapping:
                mapped_class_names.append(self._name_mapping[n])
            else:
                mapped_class_names.append(n)

        # for det in dets:
        #     annos = []
        #     boxes = _second_det_to_nusc_box(det)
        #     boxes = _lidar_nusc_box_to_global(nusc, boxes, det["metadata"]["token"])
        #     for i, box in enumerate(boxes):
        #         name = mapped_class_names[box.label]
        #         if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
        #             if name in [
        #                 "car",
        #                 "construction_vehicle",
        #                 "bus",
        #                 "truck",
        #                 "trailer",
        #             ]:
        #                 attr = "vehicle.moving"
        #             elif name in ["bicycle", "motorcycle"]:
        #                 attr = "cycle.with_rider"
        #             else:
        #                 attr = None
        #         else:
        #             if name in ["pedestrian"]:
        #                 attr = "pedestrian.standing"
        #             elif name in ["bus"]:
        #                 attr = "vehicle.stopped"
        #             else:
        #                 attr = None

        #         nusc_anno = {
        #             "sample_token": det["metadata"]["token"],
        #             "translation": box.center.tolist(),
        #             "size": box.wlh.tolist(),
        #             "rotation": box.orientation.elements.tolist(),
        #             "velocity": box.velocity[:2].tolist(),
        #             "detection_name": name,
        #             "detection_score": box.score,
        #             "attribute_name": attr
        #             if attr is not None
        #             else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[
        #                 0
        #             ],
        #         }
        #         annos.append(nusc_anno)
        #     nusc_annos["results"].update({det["metadata"]["token"]: annos})

        # nusc_annos["meta"] = {
        #     "use_camera": False,
        #     "use_lidar": True,
        #     "use_radar": False,
        #     "use_map": False,
        #     "use_external": False,
        # }

        # name = self._info_path.split("/")[-1].split(".")[0]
        # res_path = str(Path(output_dir) / Path(name + ".json"))
        # with open(res_path, "w") as f:
        #     json.dump(nusc_annos, f)

        # print(f"Finish generate predictions for testset, save to {res_path}")

        output_dir = "/workspace/CenterPoint/work_dirs/immo/results_custom_pass/"
        res_path = Path("/workspace/CenterPoint/work_dirs/immo/results_custom_pass/results.json") # TODO add json result pathing option
        
        if not testset:
            eval_main(
                nusc,
                self.eval_version,
                res_path,
                eval_set_map[self.version],
                output_dir,
            )

            with open(Path(output_dir) / "metrics_summary.json", "r") as f:
                metrics = json.load(f)

            detail = {}
            result = f"Nusc {self.version} Evaluation\n"
            for name in mapped_class_names:
                detail[name] = {}
                for k, v in metrics["label_aps"][name].items():
                    detail[name][f"dist@{k}"] = v
                threshs = ", ".join(list(metrics["label_aps"][name].keys()))
                scores = list(metrics["label_aps"][name].values())
                mean = sum(scores) / len(scores)
                scores = ", ".join([f"{s * 100:.2f}" for s in scores])
                result += f"{name} Nusc dist AP@{threshs}\n"
                result += scores
                result += f" mean AP: {mean}"
                result += "\n"
            res_nusc = {
                "results": {"nusc": result},
                "detail": {"nusc": detail},
            }
        else:
            res_nusc = None

        if res_nusc is not None:
            res = {
                "results": {"nusc": res_nusc["results"]["nusc"],},
                "detail": {"eval.nusc": res_nusc["detail"]["nusc"],},
            }
        else:
            res = None

        log_out = {k: np.mean(list(v.values()))
            for k,v in res['detail']['eval.nusc'].items()
        }
        wandb.log(log_out)

        return res, None
