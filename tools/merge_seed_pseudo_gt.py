import os

import pickle as pkl
import numpy as np
import tqdm
from pyquaternion import Quaternion

import json
from typing import Dict, Tuple

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.common.loaders import load_prediction, load_gt


def tracking_to_detection(track_path):
    with open(track_path, "r") as f:
        data = json.load(f)

    results = data['results']
    for k, v in results.items():
        detections = []
        for det in v:
            det['detection_name'] = det['tracking_name']
            del det['tracking_name']
            det['detection_score'] = det['tracking_score']
            del det['tracking_score']
            det['attribute_name'] = ''    
            detections.append(det)
        
        results[k] = detections
    
    data['results'] = results

    with open(track_path.replace('tracking_result', 'track_to_det_results'), "w") as f:
        json.dump(data, f)


def update_data(data_path, updates_path):
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    
    with open(updates_path, "rb") as f:
        updates = pkl.load(f)

    # gts, _ = load_gt(nusc, "detection_cvpr_2019", DetectionBox, verbose=True)
    # preds, meta = load_prediction(updates_path, max_boxes_per_sample=500, box_cls=DetectionBox, verbose=True)
    
    counter = 0
    update_tokens = list(updates.keys())
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    for frame in tqdm.tqdm(data):
        if frame['token'] not in update_tokens:
            continue
        ups = updates[frame['token']]
        frame['gt_boxes'] = ups['box3d_lidar'].detach().cpu().numpy()
        frame['gt_boxes_velocity'] = np.zeros(len(ups['box3d_lidar']), dtype=frame['gt_boxes_velocity'].dtype)
        frame['gt_boxes_token'] = np.array((['pseudo'] for i in range(len(ups['box3d_lidar']))), dtype='<U32')
        frame['gt_names'] = np.array(class_names)[ups['label_preds'].detach().cpu().numpy()]
        counter += 1


    print(f"Updated {counter} of {len(updates)} possible updates for {data_path} with {len(data)} entries")
    return data


def extract_pseudo_gt(data_path, updates_path):
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    
    with open(updates_path, "rb") as f:
        updates = pkl.load(f)

    # gts, _ = load_gt(nusc, "detection_cvpr_2019", DetectionBox, verbose=True)
    # preds, meta = load_prediction(updates_path, max_boxes_per_sample=500, box_cls=DetectionBox, verbose=True)
    
    counter = 0
    update_tokens = list(updates.keys())
    pseudo_gt = []

    for frame in tqdm.tqdm(data):
        if frame['token'] in update_tokens:
            pseudo_gt.append(frame)
            counter += 1

    print(f"Extracted {len(pseudo_gt)} GTs for {len(updates)} Pseudos from {data_path} with {len(data)} entries")
    return pseudo_gt


# List of pickle files to merge
pickle_files = ['./data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl', '/workspace/CenterPoint/work_dirs/5_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/prediction_1.pkl']

# Directory to save the merged file
output_dir = pickle_files[1].replace('prediction_1.pkl', '')
output_file_path = os.path.join(output_dir, 'seed_and_pseudo_gt.pkl')

# Merge data and save
from ipdb import launch_ipdb_on_exception, set_trace
with launch_ipdb_on_exception():
    tracking_to_detection("/workspace/CenterPoint/work_dirs/immo/results_flaseNMS_sc0/results.json")

    # data = update_data(pickle_files[0], pickle_files[1])

    # # output_file_path = os.path.join(output_dir, 'gt_for_pseudo.pkl')
    # # data = extract_pseudo_gt(pickle_files[0], pickle_files[1])
    # with open(output_file_path, 'wb') as file:
    #     pkl.dump(data, file)