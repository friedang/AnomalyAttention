import os

import pickle as pkl
import numpy as np
import torch
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
from ipdb import set_trace


def tracking_to_detection(track_path):
    with open(track_path, "r") as f:
        data = json.load(f)

    results = data['results']
    for k, v in results.items():
        detections = []
        for det in v:
            det['detection_name'] = det['tracking_name']
            # del det['tracking_name']
            det['detection_score'] = det['tracking_score']
            # del det['tracking_score']
            det['attribute_name'] = ''    
            detections.append(det)
        
        results[k] = detections
    
    data['results'] = results

    with open(track_path.replace('tracking_result', 'track_to_det_results'), "w") as f:
        json.dump(data, f)


def detection_to_tracking(track_path):
    with open(track_path, "r") as f:
        data = json.load(f)

    results = data['results']
    for k, v in results.items():
        detections = []
        for det in v:
            det['tracking_name'] = det['detection_name']
            det['tracking_id'] = det['detection_name'] + '_' + det['sample_token']
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
    
    counter = 0
    update_tokens = list(updates.keys())
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    for frame in tqdm.tqdm(data):
        if frame['token'] not in update_tokens or updates[frame['token']]['label_preds'].nelement() == 0:
            continue
        ups = updates[frame['token']]
        frame['gt_boxes'] = ups['box3d_lidar'].detach().cpu().numpy()
        frame['gt_boxes_velocity'] = np.zeros(len(ups['box3d_lidar']), dtype=frame['gt_boxes_velocity'].dtype)
        frame['gt_boxes_token'] = np.array((['pseudo'] for i in range(len(ups['box3d_lidar']))), dtype='<U32')
        frame['gt_names'] = np.array(class_names)[ups['label_preds'].detach().cpu().numpy()]
        counter += 1


    print(f"Updated {counter} of {len(updates)} possible updates for {data_path} with {len(data)} entries")
    return data


def update_results(cp_det_path, results_path):
    with open(cp_det_path, "r") as f:
        data = json.load(f)

    with open(results_path, "r") as f:
        results = json.load(f)

    res = results if 'results' not in results.keys() else results['results']
    length = [1 for t in res.values() if t != []]
    print("Before update:")
    print(f"Number of frames with detections is {len(length)}")

    print(f"Number of Detections before filtering is {len([v for values in res.values() for v in values])}") # if v['TP'] == 1])}")
    counter = 0
    
    print(f"Filtered out {counter} Detections labeled as FP")
    print(f"Number of Detections after filtering is {len([v for values in res.values() for v in values])}")

    
    for k in data['results'].keys():
        if k not in res.keys() or res[k] == []:
            res[k] = data['results'][k]

    length = [1 for t in res.values() if t != []]
    print("After update:")
    print(f"Number of frames with detections is {len(length)}")
    print(f"Number of Detections after updating is {len([v for values in res.values() for v in values])}")

    data['results'] = res

    with open(results_path.replace('results', 'merged_results'), 'w') as f:
        json.dump(data, f)

def extract_pseudo_gt(data_path, updates_path):
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    
    with open(updates_path, "rb") as f:
        updates = pkl.load(f)
    
    counter = 0
    update_tokens = list(updates.keys())
    pseudo_gt = []

    for frame in tqdm.tqdm(data):
        if frame['token'] in update_tokens:
            pseudo_gt.append(frame)
            counter += 1

    print(f"Extracted {len(pseudo_gt)} GTs for {len(updates)} Pseudos from {data_path} with {len(data)} entries")
    return pseudo_gt


def merge_2hz_and_20hz_files(keyframe_file, non_keyframe_file, output_file):
    # Load 2Hz (keyframes) results
    with open(keyframe_file, 'r') as f:
        keyframe_data = json.load(f)
    
    # Load 20Hz (non-keyframes) results
    with open(non_keyframe_file, 'r') as f:
        non_keyframe_data = json.load(f)
    
    # Merged results will be stored here
    merged_results = {
        'meta': keyframe_data['meta'],  # Assuming metadata can be taken from either file
        'results': {}
    }
    
    # First, add all the non-keyframes to the merged results
    merged_results['results'].update(non_keyframe_data['results'])
    
    # Then, add the keyframes, ensuring no duplication
    for sample_token, detections in keyframe_data['results'].items():
        if sample_token not in merged_results['results']:
            merged_results['results'][sample_token] = detections

    # Save the merged result to the output file
    with open(output_file, 'w') as f:
        json.dump(merged_results, f)

    print(f'Merged results saved to {output_file}')


def _global_nusc_box_to_lidar(nusc, boxes, sample_token):
    """Reverse the global box transformation to LiDAR coordinates."""
    try:
        s_record = nusc.get("sample", sample_token)
        sample_data_token = s_record["data"]["LIDAR_TOP"]
    except:
        sample_data_token = sample_token

    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    box_list = []
    for box in boxes:
        # Undo global to ego vehicle coord system
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)
        # Undo ego vehicle to LiDAR coord system
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)
        box_list.append(box)
    return box_list


def json_to_dets(nusc, json_file, output_pickle):
    """Load JSON predictions and reverse transform to pickle format."""
    with open(json_file, "r") as f:
        nusc_annos = json.load(f)
    
    dets = {}

    detection_names = ['car','bus','trailer','truck','pedestrian','bicycle',
                        'motorcycle','construction_vehicle', 'barrier', 'traffic_cone']
    name_dict = {n: i for n, i in zip(detection_names, range(len(detection_names)))}

    for token, annos in nusc_annos["results"].items():
        if token not in dets.keys():
            dets.update({token: {"metadata": {"token": token}}})
        boxes = []
        # scores = []
        labels = []
        for anno in annos:
            box = Box(
                center=np.array(anno["translation"]),
                size=np.array(anno["size"]),
                orientation=Quaternion(anno["rotation"]),
                label=name_dict[anno["detection_name"][0] if isinstance(anno["detection_name"], list) else anno["detection_name"]],  # Adjust mapping if necessary
                score=anno["detection_score"],
                velocity=(*anno["velocity"], 0.0)
            )
            boxes.append(box)
            # scores.append(anno["detection_score"])
            labels.append(name_dict[anno["detection_name"][0] if isinstance(anno["detection_name"], list) else anno["detection_name"]])  # Map to numeric label if required
        
        boxes = _global_nusc_box_to_lidar(nusc, boxes, token)
        
        # Convert boxes back to tensors
        box3d_lidar = []
        for box in boxes:
            yaw = -box.orientation.yaw_pitch_roll[0] - np.pi / 2
            box3d_lidar.append([
                *box.center, *box.wlh, yaw, box.velocity[0], box.velocity[1]
            ])

        if 'box3d_lidar' not in dets[token].keys():
            dets[token]['box3d_lidar'] = torch.tensor(box3d_lidar)
            dets[token]['label_preds'] = torch.tensor(labels)
        else:
            dets[token]['box3d_lidar'] = torch.vstack(dets[token]['box3d_lidar'], np.array(box3d_lidar))
            dets[token]['label_preds'] = torch.hstack(dets[token]['label_preds'], labels)
    
    with open(output_pickle, "wb") as f:
        pkl.dump(dets, f)
    
    print(f"Saved detections to {output_pickle}")


# # List of pickle files to merge
# pickle_files = ['./data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl', '/workspace/CenterPoint/work_dirs/ad_pc_mlp_05/seed10_voxel_at_org01th_c42_exCarPed/cp_10_pseudo_bal_ad/inference_merged_results.pkl'] # '/workspace/CenterPoint/work_dirs/5_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/prediction_1.pkl']

# # Directory to save the merged file
# output_dir = pickle_files[1].replace('prediction_1.pkl', '')
# output_file_path = os.path.join(output_dir, 'seed_and_pseudo_gt.pkl')

# keyframe_json_path = "/workspace/CenterPoint/work_dirs/immo/cp_5_seed_2hz/results_tp.json"
# non_keyframe_json_path = "/workspace/CenterPoint/work_dirs/immo/cp_5_seed_20hz/results_tp.json"
# output_json_path = "/workspace/CenterPoint/work_dirs/immo/cp_5_seed_2hz/merged_results_tp.json"

# Merge data and save
from ipdb import launch_ipdb_on_exception, set_trace
with launch_ipdb_on_exception():
    # Do after ImmoTracker results generation - Add detection_name
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nusc_data_info_path', type=str, default=None)
    parser.add_argument('--pseudo_path', type=str, default=None)
    parser.add_argument('--workdir', type=str, default=None)

    parser.add_argument('--res_path', type=str, default=None)
    parser.add_argument('--cp_det_path', type=str, default=None)
    parser.add_argument('--tracking_to_detection', type=str, default=None)
    args = parser.parse_args()

    if args.nusc_data_info_path and args.pseudo_path:
        # merge seed and pseudo
        data = update_data(args.nusc_data_info_path, args.pseudo_path)
        output_file_path = os.path.join(args.workdir, 'gt_for_pseudo.pkl')
        with open(output_file_path, 'wb') as file:
            pkl.dump(data, file)

    if args.tracking_to_detection:
        tracking_to_detection(args.tracking_to_detection) #inference_results
    
    ## Add tracking_name from detection_name
    # detection_to_tracking("/workspace/CenterPoint/work_dirs/immo/results/results.json")
    # merge_2hz_and_20hz_files(keyframe_json_path, non_keyframe_json_path, output_json_path)

    if args.res_path:
        update_results(cp_det_path=args.cp_det_path, 
                       results_path=args.res_path)
                                # args.res_path


    ## extract pseudo gt for evaluation/validation
    # # output_file_path = os.path.join(output_dir, 'gt_for_pseudo.pkl')
    # # data = extract_pseudo_gt(pickle_files[0], pickle_files[1])

    # with open(output_file_path, 'wb') as file:
    #     pkl.dump(data, file)
