from pathlib import Path
import os
import json
from typing import Dict, List
import cupy as cp  # Use CuPy for GPU processing
import open3d as o3d
import tqdm
import torch
from functools import reduce
import numpy as np
from pyquaternion import Quaternion
from det3d.datasets.pipelines.loading import read_sweep

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix, points_in_box
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import boxes_to_sensor

# Set CUDA device to GPU 0
if torch.cuda.is_available():
    cp.cuda.Device(0).use()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return float(obj)
        return super(NpEncoder, self).default(obj)


def farthest_point_sampling(points, num_samples):
    """
    Perform Farthest Point Sampling (FPS) on a point cloud.

    Args:
    - points (cp.ndarray): The input point cloud of shape (N, 4), where N is the number of points.
    - num_samples (int): The maximum number of points to sample.

    Returns:
    - sampled_points (cp.ndarray): The sampled point cloud of shape (num_samples, 4).
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(cp.asnumpy(points[:, :3]))  # Only take the xyz coordinates

    samples = point_cloud.farthest_point_down_sample(num_samples)
    samples = cp.asarray(samples.points)  # Convert back to CuPy array
    
    mask = cp.isclose(points[:, None, :3], samples[None, :, :], atol=1e-500).all(-1)
    matching_indices = cp.where(mask.any(1))[0]

    if len(matching_indices) > num_samples:
        matching_indices = cp.random.choice(matching_indices, num_samples, replace=False)

    return points[matching_indices, :]


def get_sweep_data(nusc, lidar_data):
    # Get the path to the LIDAR_TOP point cloud data file (in .pcd.bin format)
    point_cloud_path = os.path.join(nusc.dataroot, lidar_data['filename'])

    curr_sd_rec = nusc.get("sample_data", lidar_data["prev"])
    ref_time = 1e-6 * lidar_data["timestamp"]
    time_lag = ref_time - 1e-6 * curr_sd_rec["timestamp"]
    
    ref_cs_rec = nusc.get(
            "calibrated_sensor", lidar_data["calibrated_sensor_token"])
    current_cs_rec = nusc.get(
        "calibrated_sensor", curr_sd_rec["calibrated_sensor_token"])
    ref_pose_rec = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    current_pose_rec = nusc.get("ego_pose", curr_sd_rec["ego_pose_token"])


    ref_from_car = transform_matrix(
            ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True
    )
    car_from_current = transform_matrix(
                    current_cs_rec["translation"],
                    Quaternion(current_cs_rec["rotation"]),
                    inverse=False,
    )
    # Homogeneous transformation matrix from global to _current_ ego car frame
    car_from_global = transform_matrix(
        ref_pose_rec["translation"],
        Quaternion(ref_pose_rec["rotation"]),
        inverse=True,
    )
    global_from_car = transform_matrix(
                    current_pose_rec["translation"],
                    Quaternion(current_pose_rec["rotation"]),
                    inverse=False,
    )
    tm = reduce(np.dot,
                [ref_from_car, car_from_global, global_from_car,
                 car_from_current],
                )

    sweep = dict(
        lidar_path=point_cloud_path,
        transform_matrix=tm,
        time_lag=time_lag,
    )

    return sweep


def save_point_cloud(token, output_path, nusc, max_points=15000):
    """
    Loads the point cloud for a given sample token and saves it as an .npy file.

    Args:
    - token (str): The sample token to load.
    - output_path (str): Directory to save the .npy file.
    """
    # Get the sample data for the LIDAR_TOP sensor
    sample = nusc.get('sample', token)
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    
    point_cloud_path = os.path.join(nusc.dataroot, lidar_data['filename'])
    sweep = get_sweep_data(nusc, lidar_data) if lidar_data['prev'] != "" else dict(time_lag=0, transform_matrix=None, lidar_path=point_cloud_path)
    points_sweep, times_sweep = read_sweep(sweep, virtual=False)
    point_cloud = np.hstack([points_sweep, times_sweep])

    # Load the point cloud
    # point_cloud = LidarPointCloud.from_file(point_cloud_path)

    # Convert the point cloud to a CuPy array
    points = cp.asarray(point_cloud)  # Shape (N, 4) if you want to include intensity
    # TODO add voxel arg?
    # if len(points) > max_points:
    #     points = farthest_point_sampling(points, max_points)
    # else:
    #     print(f"Frame {token} has less than {max_points} points with only {len(points)} points")

    # Define the output file path
    output_file = os.path.join(output_path, f"{token}.npy")

    # Save the point cloud as an .npy file (convert back to NumPy array for saving)
    cp.save(output_file, points)
    # print(f"Saved point cloud to {output_file}")


def create_tracks(detection_results, nusc, num_lidar_pts=False, sub_path='train'):

    tracks = {} # box, tp_label, sample_token

    pointcloud_cache = {}

    for sample_token, preds in tqdm.tqdm(detection_results['results'].items()):
        try:
            sample = nusc.get('sample', sample_token)
        except:
            sample = None

        for result in preds:    
            
            t_id = result['tracking_id']
            if t_id not in tracks.keys():
                tracks[t_id] = []

            if num_lidar_pts:
                if sample_token not in pointcloud_cache:
                    if sample:
                        pointcloud_path = Path(f"/workspace/CenterPoint/work_dirs/PCs_npy_vox/{sub_path}") / f"{sample_token}.npy"
                        pointcloud = np.load(str(pointcloud_path))
                        sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                    else:
                        sd_record = nusc.get('sample_data', result['lidar_token'])
                        pointcloud_path = os.path.join(nusc.dataroot, sd_record['filename'])
                        pointcloud = LidarPointCloud.from_file(pointcloud_path)
                        pointcloud = cp.asarray(pointcloud.points.T)
                        pointcloud = farthest_point_sampling(pointcloud, num_samples=15000).get()
                    
                pointcloud_cache[sample_token] = pointcloud

                pointcloud = pointcloud_cache[sample_token]
                cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                
                box = EvalBoxes.deserialize({result['sample_token']: [result]}, DetectionBox)
                box = list(box.boxes.values())
                box_in_sensor = boxes_to_sensor(box[0], pose_record, cs_record)[0]
                points = sum(points_in_box(box_in_sensor, pointcloud[:, :3].T))

                result['num_lidar_pts'] = points

            tracks[t_id].append(
                result)
                # (box, result['TP'], sample_token))
    
    length = [len(t) for t in tracks.values()]

    print(f"Extracted up to {len(length)} tracks.")

    # length_thresh = 2
    # tracks = {k: t for k, t in tracks.items() if len(t) > length_thresh}
    # print(f"Removed Tracks with length smaller than {length_thresh}")
    # print(f"Updates to {len(length)} number of tracks.")
    
    # length = [len(t) for t in tracks.values()]
    print(f"Max length of tracks is {max(length)} and Minimum is {min(length)} and Mean is {np.mean(length)}")

    return tracks


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path, cls=None):
    with open(file_path, 'w') as f:
        json.dump(data, f, cls=cls, indent=2)


def fix_immortal_detections(file_path):
    data = load_json(file_path)
    keys = list(data['results'].keys())
    for k in keys:
        if not data['results'][k]:
            del data['results'][k]
            continue

        for r in data['results'][k]:
            # if isinstance(r['sample_token'], list):
            r['lidar_token'] = k
                # del r['sample_token']

    save_json(data, file_path)


def convert_pc_npy(output_directory, nusc, tokens):

    # output_directory = '/workspace/CenterPoint/work_dirs/PCs_npy/train'

    # nusc = NuScenes(version='v1.0-trainval', dataroot="data/nuScenes", verbose=True)
    # # Save pointclouds as npy
    # tokens = list(detection_results['results'].keys())

    print(f"Export points to {output_directory}")
    for token in tqdm.tqdm(tokens):
        output_file = os.path.join(output_directory, f"{token}.npy")
        if os.path.isfile(output_file):
            continue
        save_point_cloud(token, output_directory, nusc)


def extract_gt_tracks(nusc: NuScenes, sample_tokens: List[str]) -> Dict[str, List[Dict]]:
    gt_tracks = {}

    for sample_token in tqdm.tqdm(sample_tokens):
        sample = nusc.get('sample', sample_token)
        ann_tokens = sample['anns']
        sample_annotations = [nusc.get('sample_annotation', t) for t in ann_tokens]

        for annotation in sample_annotations:
            instance_token = annotation['instance_token']
            if instance_token not in gt_tracks:
                gt_tracks[instance_token] = []

            c = 0
            i = instance_token
            # while len(gt_tracks[i]) >= 5:
            #     if i.endswith(f"_{c}"):
            #         c += 1
            #         i = i[:-1] + f"{c}"
            #     else:
            #         i = i[:-1] + f"_{c}"
                
            #     if i not in gt_tracks.keys():
            #         gt_tracks[i] = []
            
            annotation['TP'] = 1
            annotation['dist_TP'] = [1, 1, 1, 1]
            annotation['tracking_id'] = instance_token
            # box = {
            #     'translation': annotation['translation'],
            #     'size': annotation['size'],
            #     'rotation': annotation['rotation']
            #     }
            pointcloud_path = Path('/workspace/CenterPoint/work_dirs/PCs_npy_vox/train') / f"{sample_token}.npy"
            pointcloud = np.load(str(pointcloud_path))
            
            sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

            annotation['detection_name'] = category_to_detection_name(annotation['category_name'])
            if not annotation['detection_name']:
                continue
            annotation['velocity'] = (0, 0)
            annotation['detection_score'] = -1
            annotation['attribute_name'] = ''


            box = EvalBoxes.deserialize({annotation['sample_token']: [annotation]}, DetectionBox)
            box = list(box.boxes.values())
            box_in_sensor = boxes_to_sensor(box[0], pose_record, cs_record)[0]
            points = sum(points_in_box(box_in_sensor, pointcloud[:, :3].T))

            annotation['num_lidar_pts'] = points

            gt_tracks[i].append(annotation)

    gt_tracks = {k: t for k, t in gt_tracks.items() if t != []}

    # set_trace()
    length = [len(t) for t in gt_tracks.values()]

    print(f"Extracted up to {len(length)} tracks.")
    print(f"Max length of GT tracks is {max(length)} and Minimum is {min(length)} and Mean is {np.mean(length)}")

    return gt_tracks


def get_scene_from_sample(nusc, sample_token):
    # Traverse the sample back to the scene's first sample to find the corresponding scene
    for scene in nusc.scene:
        if scene['first_sample_token'] == sample_token or scene['last_sample_token'] == sample_token:
            return scene['name']
        cur_token = scene['first_sample_token']
        while cur_token != '':
            if cur_token == sample_token:
                return scene['name']
            cur_token = nusc.get('sample', cur_token)['next']
    
    return None


def padd_scene_tracks(nusc, scene_track_map, tracks):
    dummy = {'translation': [-500, -500, -500], 'size': [-500, -500, -500], 'rotation': [-500, -500, -500, -500],
             'sample_token': 'dummy', 'TP': -500, 'num_lidar_pts': -500, 'tracking_id': 'dummy', 'dist_TP': [0, 0, 0, 0]}
    max_track_len = 41
    
    print("Padding scene tracks")
    for scene in tqdm.tqdm(nusc.scene):
        if scene['name'] not in scene_track_map.keys():
            continue
        cur_token = scene['first_sample_token']
        idx = 0
        while cur_token != '':
            for t in scene_track_map[scene['name']]:
                # if len(tracks[t]) < 2:
                #     continue
                
                if len(tracks[t]) == idx:
                    if tracks[t][idx-1] == dummy:
                        tracks[t].append(dummy)
                    elif tracks[t][idx-1]['sample_token'] != cur_token:
                        tracks[t] = tracks[t][:idx-1] + [dummy] + tracks[t][idx-1:]
                elif tracks[t][idx]['sample_token'] != cur_token:
                    tracks[t] = tracks[t][:idx] + [dummy] + tracks[t][idx:]
            cur_token = nusc.get('sample', cur_token)['next']
            idx += 1
        
    for t in tracks.keys():
        if len(tracks[t]) < max_track_len:
            track_len_diff = max_track_len - len(tracks[t])
            for _ in range(track_len_diff):
                tracks[t].append(dummy)
    

    lengths = [len(lst) for lst in tracks.values()]
    print(f"Minimum length of tracks is {min(lengths)} and Maximum is {max(lengths)}")
    # set_trace()
    return tracks


def get_scene_from_lidar(nusc, lidar_token):
    # Get the sample_data object from the lidar token
    sample_data = nusc.get('sample_data', lidar_token)
    
    # Retrieve the sample token associated with this lidar data
    sample_token = sample_data['sample_token']
    
    # Use the previous function to find the scene name from the sample token
    return get_scene_from_sample(nusc, sample_token)


def create_scene_track_mapping(nusc, scene_names, tracks):
    scene_mapping = {n: [] for n in scene_names}
    
    print("Create Scene mapping")
    for k, v in tqdm.tqdm(tracks.items()):
        first_detection = list(v)[0]
        if 'sample_token' in first_detection.keys():
            name = get_scene_from_sample(nusc, first_detection['sample_token'])
        else:
            print(1)
            name = get_scene_from_lidar(nusc, first_detection['lidar_token'])

        if name not in scene_names:
            continue
        scene_mapping[name].append(k)

    lengths = [len(lst) for lst in scene_mapping.values()]
    print(f"Minimum number of tracks in scene is {min(lengths)} and Maximum is {max(lengths)}")

    return scene_mapping


def remove_det_by_track_id(detection_results, tracks, len_thresh=2, score_thresh=0.1):
    removeable_ids = [t[0]['tracking_id'] for t in tracks.values() if len(t) <= len_thresh]
    tp = []
    fp = []

    print(f"Found {len(removeable_ids)} tracks to be removed")

    num_dets_original = len([det for t in detection_results['results'].values() for det in t])
    set_trace()

    for k, v in tqdm.tqdm(detection_results['results'].items()):
        dets = []
        for det in v:
            if det['tracking_id'] not in tracks.keys():
                continue
            elif det['tracking_id'] not in removeable_ids and det['tracking_score'] > score_thresh:
                dets.append(det)
            elif det['TP'] == 1:
                tp.append(det)
            else:
                fp.append(det)

        detection_results['results'][k] = dets

    print(min([det['tracking_score'] for t in detection_results['results'].values() for det in t]))
    num_dets_final = len([det for t in detection_results['results'].values() for det in t])
    print(f"Removed {num_dets_original - num_dets_final} Detections with TP: {len(tp)} and FP: {len(fp)}")
    # assert len(tp) + len(fp) == num_dets_original - num_dets_final
    # set_trace()

    return detection_results


def main():
    # Load the JSON file
    hz20 = False

    sample = None # 5, 10
    gt = False
    extract_pcs = False
    val = True
    remove_non_cp = False
    remove_det_by_track_len = False
    # TODO ALWAYS USE results_tp
    immo_results = '/workspace/CenterPoint/work_dirs/immo/results/results_tp.json' #'/workspace/CenterPoint/work_dirs/Center_point_original_nusc_0075_flip/immo_results/results_tp.json' # './work_dirs/ad_mlp_05/aug_t5/nusc_validation_t01/inference_results.json' # '/workspace/CenterPoint/work_dirs/immo/cp_valset/cp_results.json'
    cp_det_file = "/workspace/CenterPoint/work_dirs/5_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/eval_on_seed/2Hz/baseline_SCs_03Mean/infos_train_10sweeps_withvelo_filter_True.json"

    # for 20 Hz
    # TODO for merged_results: run fix, extract pointclouds to npy for NK frames, adjust number of lidar points for NK
    if hz20:
        fix_immortal_detections(immo_results)

    detection_results = load_json(immo_results)
    detection_results = detection_results if 'results' in detection_results.keys() else {'results': detection_results}
    
    # set_trace()
    ## Save pointclouds as npy
    nusc = NuScenes(version='v1.0-trainval', dataroot="data/nuScenes", verbose=True)
    if extract_pcs:
        tokens = detection_results['results'].keys()
        output_directory='/workspace/CenterPoint/work_dirs/PCs_npy_vox/val' if val else '/workspace/CenterPoint/work_dirs/PCs_npy_vox/train'
        convert_pc_npy(output_directory=output_directory, nusc=nusc, tokens=tokens)


    # remove empty preds
    if remove_non_cp:
        ks = list(detection_results['results'].keys())
        org_keys = load_json(cp_det_file)['results'].keys()
        print(f"Number of frames BEFORE removal: {len(ks)}")
        for k in ks:
            if k not in org_keys:
                del detection_results['results'][k]    
            elif detection_results['results'][k] == [] or detection_results['results'][k] == [[]]:
                del detection_results['results'][k]
        
        ks = list(detection_results['results'].keys())
        print(f"Number of frames AFTER removal: {len(ks)}")

        num_det_no_dummy = len([v for values in detection_results['results'].values() for v in values if v['TP'] != -500])
        print(f"Number of non dummy items in dets is {num_det_no_dummy}")

    ## Create GT tracks
    if gt:
        tokens = list(detection_results['results'].keys())
        gt_tracks = extract_gt_tracks(nusc, tokens)
        save_json(gt_tracks, immo_results.replace('results_tp', 'gt_track_info'), cls=NpEncoder)
    else:
        p_tracks = create_tracks(detection_results, nusc, num_lidar_pts = True if not remove_det_by_track_len else False, sub_path='val') if val else create_tracks(detection_results, nusc, num_lidar_pts = True if not remove_det_by_track_len else False)
        save_json(p_tracks, immo_results.replace('results_tp', 'track_info'), cls=NpEncoder)

    # Remove Detections by track id / track lengths
    if remove_det_by_track_len:
            path = '/workspace/CenterPoint/work_dirs/immo/cp_valset/track_info_tp.json'
            p_tracks = load_json(path)
            detection_results = remove_det_by_track_id(detection_results, p_tracks, len_thresh=4)
            save_json(detection_results, immo_results.replace('.json', '_filtered.json'))

    ## scene to track mapping
    if gt:
        immo_results = immo_results.replace('results_tp', 'gt_track_info')
    else:
        immo_results = immo_results.replace('results_tp', 'track_info')
    
    if sample:
        scene_names = torch.load(f"/workspace/CenterPoint/work_dirs/{sample}_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/train_indices.pth")
    else:
        scene_names = create_splits_scenes()['val'] if val else create_splits_scenes()['train']
    tracks = load_json(immo_results)

    # set_trace()
    scene_mapping = create_scene_track_mapping(nusc, scene_names, tracks)
    save_json(scene_mapping, immo_results.replace('track_info', 'scene2trackname'))

    scene_file = immo_results.replace('track_info', 'scene2trackname')
    map_scene_track = load_json(scene_file)
    tracks = load_json(immo_results)
    padded_tracks = padd_scene_tracks(nusc, map_scene_track, tracks)
    save_json(padded_tracks, immo_results.replace('track_info', 'track_info_padded_tracks'))

if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception, set_trace
    with launch_ipdb_on_exception():
        main()