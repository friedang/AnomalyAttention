import sys
import pickle
import json
import random
import operator
import matplotlib.pyplot as plt
import numpy as np
import os
from pyquaternion import Quaternion
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
from det3d.datasets.nuscenes.nusc_common import (
    general_to_detection,
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main,
    quaternion_yaw
)
from det3d.datasets.registry import DATASETS


@DATASETS.register_module
class NuScenesDataset(PointCloudDataset):
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
        hz20=False,
        **kwargs,
    ):
        self.load_interval = load_interval 
        super(NuScenesDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names, sample_ratio=sample_ratio,hz20=hz20,
        )

        self.nsweeps = nsweeps
        assert self.nsweeps > 0, "At least input one sweep please!"
        print(self.nsweeps)

        self._info_path = info_path
        self._class_names = class_names

        self.load_infos(self._info_path, sample_ratio, load_indices, hz20)
        self.flag = np.ones(len(self), dtype=np.uint8)

        self._num_point_features = NuScenesDataset.NumPointFeatures
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
    
    def map_timestamps_to_sequences(self, nusc, train_scenes, timestamps):
        timestamp_to_sequence = {}

        for scene in nusc.scene:
            if scene['name'] not in train_scenes:
                continue
            timestamp_to_sequence[scene['name']] = []
            # Get first and last sample tokens
            first_sample_token = scene['first_sample_token']
            last_sample_token = scene['last_sample_token']
            
            # Get start and end timestamps from the first and last samples
            start_time = nusc.get('sample', first_sample_token)['timestamp']
            end_time = nusc.get('sample', last_sample_token)['timestamp']
            
            # Map the provided timestamps to the scene (sequence)
            for timestamp in timestamps:
                if start_time <= float(str(timestamp).replace('.','')) <= end_time:
                    timestamp_to_sequence[scene['name']] += [timestamp]
        
        return timestamp_to_sequence

    def get_class_distribution(self, nusc, scene_names, classes):
        from collections import Counter
        import tqdm
        class_counter = Counter()

        for scene in tqdm.tqdm(nusc.scene):
            if scene['name'] not in scene_names:
                continue

            sample = nusc.get('sample', scene['first_sample_token'])
            while sample:
                for ann_token in sample['anns']:
                    ann = nusc.get('sample_annotation', ann_token)
                    if any([c for c in classes if c in ann['category_name']]):
                        class_counter[ann['category_name']] += 1
                
                if sample['next'] == '':
                    break
                sample = nusc.get('sample', sample['next'])

        dis = {}
        dis_cat_names = dict(class_counter)

        for c in classes:
            dis[c] = 0
            vals = {k: v for k,v in dis_cat_names.items() if c in k}
            dis[c] += np.sum(list(vals.values()))

        return dis

    def load_infos(self, info_path, sample_ratio=1, load_indices=None, hz20=True):

        with open(self._info_path, "rb") as f:
            _nusc_infos_all = pickle.load(f)

        _nusc_infos_all = _nusc_infos_all[::self.load_interval]

        if sample_ratio != 1 or load_indices != None:
            ts_all = [_nusc_infos_all[i]['timestamp'] for i in range(len(_nusc_infos_all))]
            nusc = NuScenes(version='v1.0-trainval', dataroot=str(self._root_path), verbose=True)
            train_scenes = create_splits_scenes()['train']
            timestamp_to_sequence = self.map_timestamps_to_sequences(nusc, train_scenes, ts_all)
            train_scene_names = list(timestamp_to_sequence.keys())

            if load_indices:
                self.train_indices = torch.load(load_indices)
                print(f"Loaded indices from {load_indices}")         
            else:
                print(f"Sample {sample_ratio*100}% of the dataset.")
                torch.manual_seed(random.randint(0, sys.maxsize))
                total_indices = torch.randperm(len(train_scene_names))
                split = int(sample_ratio * len(train_scene_names))
                train_scene_names = [x for _, x in sorted(zip(total_indices, train_scene_names))]
                self.train_indices = train_scene_names[:split]
                self.pseudo_indices = train_scene_names[split:]
                
                save_path = f"/workspace/CenterPoint/work_dirs/cp_{int(sample_ratio*100)}"
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                torch.save(self.train_indices, f"{save_path}/train_indices.pth")
                torch.save(self.pseudo_indices, f"{save_path}/pseudo_indices.pth")
            
            classes = ["car", "truck", "bus", "trailer", "vehicle.construction", "pedestrian",
                       "motorcycle", "bicycle", "trafficcone", "barrier"]
            class_distribution = self.get_class_distribution(nusc, self.train_indices, classes)
            print(class_distribution) 

            print(f"Original # of Frames is {len(_nusc_infos_all)}")
            print(f"Original # of Scenes is {len(train_scenes)}")
            scenes = {k: v for k, v in timestamp_to_sequence.items() if k in self.train_indices}
            timestamps = [t for ts in scenes.values() for t in ts]
            _nusc_infos_all = [i for i in _nusc_infos_all if i['timestamp'] in timestamps]
            print(f"New # of Frames is {len(_nusc_infos_all)}")
            print(f"New # of Scenes is {len(self.train_indices)}")

            # nk_frames = []
            # for scene in nusc.scene:
            #     if scene['name'] not in self.train_indices:
            #         continue
            #     sample_token = scene['last_sample_token']
            #     sample = nusc.get('sample', sample_token)
            #     sd_token = sample['data']['LIDAR_TOP']
            #     while sd_token:
            #         sd_record = nusc.get('sample_data', sd_token)
            #         if sd_record['is_key_frame']:
            #             sd_token = sd_record['prev']
            #             continue
                    

            #         sd_token = sd_record['prev']


            if hz20:
                from det3d.datasets.nuscenes.nusc_common import _fill_non_keyframe_infos
                nk_frames = _fill_non_keyframe_infos(nusc, _nusc_infos_all)
                # nk_frames = pickle.load(open("./work_dirs/5_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/nk_seed5.pkl", 'rb'))
                for f in nk_frames:
                    try:
                        f['sample_token'] = f['sample_token'][0]
                        print('fixed token')
                    except:
                        c=1
                    
                pickle.dump(nk_frames, open("./work_dirs/5_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/nk_seed5.pkl"))
                    

                    # also fix empty predictions
                    # fix ignore gts
                    

            # if False:
                # nk_frames = []
                # k_frames = 0
                # for info in _nusc_infos_all:
                #     if nusc.get('sample_data', info['sweeps'][8]['sample_data_token'])['token'] == nusc.get('sample_data', info['sweeps'][2]['sample_data_token'])['token']:
                #         continue
                #     sample = nusc.get('sample', info['token'])
                #     sd_token = sample['data']['LIDAR_TOP']
                #     nk_info = info
                #     for sweep in info['sweeps']:
                #         set_trace()
                #         nk = [sweep] * 9
                #         nk_info['sweeps'] = nk
                #         data = nusc.get('sample_data', sweep['sample_data_token'])
                #         if data['is_key_frame']:
                #             k_frames += 1
                #             continue
                #             set_trace()
                #         # nk_info.update({k: v for k,v in data.items() if k in nk_info.keys()})
                #         sample = nusc.get('sample', nusc.get('sample_data', info['sweeps'][8]['sample_data_token'])['sample_token'])
                #         sd_token = sample['data']['LIDAR_TOP']
                #         sd_record = nusc.get('sample_data', sd_token)
                #         if sd_record['is_key_frame']:
                #             continue

                        

                #         sd_token = sd_record['prev']
                #         if not sd_token:
                #             break

                #         nk_info.update({k: v for k,v in sweep.items() if k in nk_info.keys()})
                        
                #         boxes = nusc.get_boxes(data['token'])
                #         sd_record = nusc.get("sample_data", data['token'])
                #         cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
                #         pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])
                #         # Make list of Box objects including coord system transforms.
                #         ref_boxes = []
                #         for box in boxes:
                #             box.velocity = nusc.box_velocity(box.token)
                #             # Move box to ego vehicle coord system
                #             box.translate(-np.array(pose_record["translation"]))
                #             box.rotate(Quaternion(pose_record["rotation"]).inverse)

                #             #  Move box to sensor coord system
                #             box.translate(-np.array(cs_record["translation"]))
                #             box.rotate(Quaternion(cs_record["rotation"]).inverse)

                #             ref_boxes.append(box)

                #         locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
                #         dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)
                #         # rots = np.array([b.orientation.yaw_pitch_roll[0] for b in ref_boxes]).reshape(-1, 1)
                #         velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
                #         rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(
                #             -1, 1
                #         )
                #         names = np.array([b.name for b in ref_boxes])
                #         tokens = np.array([b.token for b in ref_boxes])
                #         gt_boxes = np.concatenate(
                #             [locs, dims, velocity[:, :2], -rots - np.pi / 2], axis=1
                #         )

                #         nk_info["gt_boxes"] = gt_boxes
                #         nk_info["gt_boxes_velocity"] = velocity
                #         nk_info["gt_names"] = np.array([general_to_detection[name] for name in names])
                #         nk_info["gt_boxes_token"] = tokens

                #         nk_frames.append(nk_info)

        # _nusc_infos_all += nk_frames
        # print(f"New # of Frames is {len(_nusc_infos_all)}")
        # toks = [k['token'] for k in _nusc_infos_all]
        # print(len(set(toks)))
        # print(k_frames)
        
        if not self.test_mode:  # if training
            self.frac = int(len(_nusc_infos_all) * 0.25)

            _cls_infos = {name: [] for name in self._class_names}
            for info in _nusc_infos_all:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
            _cls_dist = {k: len(v) / max(duplicated_samples, 1) for k, v in _cls_infos.items()}

            self._nusc_infos = []

            frac = 1.0 / len(self._class_names)
            ratios = [frac / v for v in _cls_dist.values()]

            for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
                self._nusc_infos += np.random.choice(
                    cls_infos, int(len(cls_infos) * ratio)
                ).tolist()

            _cls_infos = {name: [] for name in self._class_names}
            for info in self._nusc_infos:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            _cls_dist = {
                k: len(v) / len(self._nusc_infos) for k, v in _cls_infos.items()
            }
        else:
            if isinstance(_nusc_infos_all, dict):
                self._nusc_infos = []
                for v in _nusc_infos_all.values():
                    self._nusc_infos.extend(v)
            else:
                self._nusc_infos = _nusc_infos_all

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

    def plot_results(self, output_dir, x, y):
        detailed_results_path = output_dir + '/metrics_details.json'
        with open(detailed_results_path, 'r') as f:
            detailed_results = json.load(f)
        objects = detailed_results.keys()
        classes = []
        for obj in objects:
            if obj[:-4] not in classes:
                classes.append(obj[:-4])

        colors = [
            "#FF5733",  # Red
            "#33FF57",  # Green
            "#3357FF",  # Blue
            "#33FFFF",  # Cyan
            "#FF33FF",  # Magenta
            "#FFFF33",  # Yellow
            ]

        for c in classes:
            # Filter objects by class
            obj_by_c = [obj for obj in objects if c in obj]

            # Create a new figure for this class
            plt.figure(figsize=(8, 6))

            # Loop over each subclass object and plot the PR curve with different colors
            for idx, o in enumerate(obj_by_c):
                obj_results = detailed_results[o]

                x_quantity = obj_results[f"{x}"]
                y_quantity = obj_results[f"{y}"]

                plt.plot(x_quantity, y_quantity, label=o, marker='o', linestyle='-', color=colors[idx])

            plt.title(f'{y}-{x} curve for {c} class')
            plt.xlabel(f'{x}')
            plt.ylabel(f'{y}')
            plt.grid(True)
            if x == 'confidence':
                plt.xlim([1, 0])
            else:
                plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.legend(title='Subclasses')
            plt.savefig(f"{detailed_results_path.replace('metrics_details.json', 'plots/' + y + '-' + x + '_' + c)}.png")
            plt.close()


    def evaluation(self, detections, output_dir=None, testset=False, train=False, res_path=None, filter_ad=False):
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "train" if train else "val",
            "v1.0-test": "test",
        }

        nusc = NuScenes(version=self.version, dataroot=str(self._root_path), verbose=True)

        mapped_class_names = []
        for n in self._class_names:
            if n in self._name_mapping:
                mapped_class_names.append(self._name_mapping[n])
            else:
                mapped_class_names.append(n)

        if not res_path:
            if not testset:
                dets = []
                gt_annos = self.ground_truth_annotations
                assert gt_annos is not None

                miss = 0
                for gt in gt_annos:
                    try:
                        dets.append(detections[gt["token"]])
                    except Exception:
                        miss += 1

                assert miss == 0
            else:
                dets = [v for _, v in detections.items()]
                # assert len(detections) == 6008

            nusc_annos = {
                "results": {},
                "meta": None,
            }

            for det in dets:
                annos = []
                boxes = _second_det_to_nusc_box(det)
                boxes = _lidar_nusc_box_to_global(nusc, boxes, det["metadata"]["token"])
                counter = 1
                for i, box in enumerate(boxes):
                    name = mapped_class_names[box.label]
                    if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                        if name in [
                            "car",
                            "construction_vehicle",
                            "bus",
                            "truck",
                            "trailer",
                        ]:
                            attr = "vehicle.moving"
                        elif name in ["bicycle", "motorcycle"]:
                            attr = "cycle.with_rider"
                        else:
                            attr = None
                    else:
                        if name in ["pedestrian"]:
                            attr = "pedestrian.standing"
                        elif name in ["bus"]:
                            attr = "vehicle.stopped"
                        else:
                            attr = None

                    nusc_anno = {
                        "sample_token": det["metadata"]["token"],
                        "translation": box.center.tolist(),
                        "size": box.wlh.tolist(),
                        "rotation": box.orientation.elements.tolist(),
                        "velocity": box.velocity[:2].tolist(),
                        "detection_name": name,
                        "detection_score": box.score,
                        "attribute_name": attr
                        if attr is not None
                        else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[
                            0
                        ],
                    }
                    annos.append(nusc_anno)
                
                token = det["metadata"]["token"]
                # while token in nusc_annos["results"].keys():
                #     token = token[:-1] + f"{counter}" if '_nk' in token[-4:] else token[:-1] + f"_nk{counter}"
                #     counter += 1
                if token in nusc_annos["results"].keys():
                    print("OVERWRITING DETECTION RESULTS!!!")
                nusc_annos["results"].update({token: annos})

            nusc_annos["meta"] = {
                "use_camera": False,
                "use_lidar": True,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            }

            name = self._info_path.split("/")[-1].split(".")[0]
            res_path = str(Path(output_dir) / Path(name + ".json"))
            counter = 1
            while os.path.exists(res_path):
                res_path = res_path[:-7] + f"_{counter}.json"
                counter += 1
            with open(res_path, "w") as f:
                json.dump(nusc_annos, f)

            print(f"Finish generate predictions for testset, save to {res_path}")
        
        if not testset:
            eval_main(
                nusc,
                self.eval_version,
                res_path,
                eval_set_map[self.version],
                output_dir,
                filter_ad
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
            self.plot_results(output_dir, 'confidence', 'precision')
            self.plot_results(output_dir, 'confidence', 'recall')
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
        try:
            wandb.log(log_out)
        except:
            print('WandB deactivated')

        return res, None
