from pathlib import Path
import os
import json
from typing import Dict, List
import cupy as cp  # Use CuPy for GPU processing
import open3d as o3d
import tqdm
import torch
import numpy as np
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix, points_in_box
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import boxes_to_sensor

# Set CUDA device to GPU 0
cp.cuda.Device(0).use()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
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
    
    mask = cp.isclose(points[:, None, :3], samples[None, :, :], atol=1e-5).all(-1)
    matching_indices = cp.where(mask.any(1))[0]

    if len(matching_indices) > num_samples:
        matching_indices = cp.random.choice(matching_indices, num_samples, replace=False)

    return points[matching_indices, :]


def save_point_cloud(token, output_path, nusc, max_points=5000):
    """
    Loads the point cloud for a given sample token and saves it as an .npy file.

    Args:
    - token (str): The sample token to load.
    - output_path (str): Directory to save the .npy file.
    """
    # Get the sample data for the LIDAR_TOP sensor
    sample = nusc.get('sample', token)
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    
    # Get the path to the LIDAR_TOP point cloud data file (in .pcd.bin format)
    point_cloud_path = os.path.join(nusc.dataroot, lidar_data['filename'])

    # Load the point cloud
    point_cloud = LidarPointCloud.from_file(point_cloud_path)

    # Convert the point cloud to a CuPy array
    points = cp.asarray(point_cloud.points.T)  # Shape (N, 4) if you want to include intensity
    if len(points) > max_points:
        points = farthest_point_sampling(points, max_points)
    else:
        print(f"Frame {token} has less than {max_points} points with only {len(points)} points")

    # Define the output file path
    output_file = os.path.join(output_path, f"{token}.npy")

    # Save the point cloud as an .npy file (convert back to NumPy array for saving)
    cp.save(output_file, points)
    # print(f"Saved point cloud to {output_file}")


def create_tracks(detection_results, nusc):

    tracks = {} # box, tp_label, sample_token

    for sample_token, preds in tqdm.tqdm(detection_results['results'].items()):
        try:
            sample = nusc.get('sample', sample_token)
        except:
            sample = None

        for result in preds:    
            
            t_id = result['tracking_id']
            if t_id not in tracks.keys():
                tracks[t_id] = []
            
            # box = result['translation'] + result['size'] + result['rotation']
            # box = torch.Tensor(box)

            c = 0
            i = t_id
            # while len(tracks[i]) >= 5:
            #     if i.endswith(f"_{c}"):
            #         c += 1
            #         i = i[:-1] + f"{c}"
            #     else:
            #         i = i[:-1] + f"_{c}"
                
            #     if i not in tracks.keys():
            #         tracks[i] = []
            
            if sample:
                pointcloud_path = Path('/workspace/CenterPoint/work_dirs/PCs_npy/train') / f"{sample_token}.npy"
                pointcloud = np.load(str(pointcloud_path))

                sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                
                box = EvalBoxes.deserialize({result['sample_token']: [result]}, DetectionBox)
                box = list(box.boxes.values())
                box_in_sensor = boxes_to_sensor(box[0], pose_record, cs_record)[0]
                points = sum(points_in_box(box_in_sensor, pointcloud[:, :3].T))

                result['num_lidar_pts'] = points

            tracks[i].append(
                result)
                # (box, result['TP'], sample_token))
    
    length = [len(t) for t in tracks.values()]

    print(f"Extracted up to {len(length)} tracks.")
    print(f"Max length of tracks is {max(length)} and Minimum is {min(length)} and Mean is {np.mean(length)}")

    # print("DO NK-TRACK REMOVAL")

    # keys = list(tracks.keys())
    # for k in keys:
    #     samples = False
    #     for det in tracks[k]:
    #         try:
    #             sample = nusc.get('sample', det['sample_token'])
    #             samples = True
    #             break
    #         except:
    #             continue

    #     if not samples:
    #         del tracks[k]
    
    # length = [len(t) for t in tracks.values()]
    # print(f"Extracted up to {len(length)} tracks.")
    # print(f"Max length of tracks is {max(length)} and Minimum is {min(length)} and Mean is {np.mean(length)}")

    # set_trace()
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

        for i in range(len(data['results'][k])):
            if isinstance(r['sample_token'], list):
                data['results'][k][i]['lidar_token'] = k
                del data['results'][k][i]['sample_token']

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

    for sample_token in sample_tokens:
        sample = nusc.get('sample', sample_token)
        ann_tokens = sample['anns']
        sample_annotations = [nusc.get('sample_annotation', t) for t in ann_tokens]

        for annotation in sample_annotations:
            instance_token = annotation['instance_token']
            if instance_token not in gt_tracks:
                gt_tracks[instance_token] = []

            c = 0
            i = instance_token
            while len(gt_tracks[i]) >= 5:
                if i.endswith(f"_{c}"):
                    c += 1
                    i = i[:-1] + f"{c}"
                else:
                    i = i[:-1] + f"_{c}"
                
                if i not in gt_tracks.keys():
                    gt_tracks[i] = []
            
            annotation['TP'] = 1
            # box = {
            #     'translation': annotation['translation'],
            #     'size': annotation['size'],
            #     'rotation': annotation['rotation']
            #     }
            pointcloud_path = Path('/workspace/CenterPoint/work_dirs/PCs_npy/train') / f"{sample_token}.npy"
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
             'sample_token': 'dummy', 'TP': -500, 'num_lidar_pts': -500}
    max_track_len = 41

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
    
    for k, v in tracks.items():
        first_detection = list(v)[0]
        if 'sample_token' in first_detection.keys():
            name = get_scene_from_sample(nusc, first_detection['sample_token'])
            scene_mapping[name].append(k)
        else:
            name = get_scene_from_lidar(nusc, first_detection['lidar_token'])
            scene_mapping[name].append(k)

    return scene_mapping


def main():
    # Load the JSON file
    input_file = '/workspace/CenterPoint/work_dirs/immo/cp_5_seed_2hz/results_tp.json' # immo/results_max_pass_ctrl/results_tp.json'
    # output_file = input_file.replace('.json', '_tp.json')
    # output_dir = input_file.replace('results.json', '')

    # for 20 Hz
    # TODO for merged_results: run fix, extract pointclouds to npy for NK frames, adjust number of lidar points for NK
    # fix_immortal_detections(input_file)
    
    nusc = NuScenes(version='v1.0-trainval', dataroot="data/nuScenes", verbose=True)

    # detection_results = load_json(input_file)
    # p_tracks = create_tracks(detection_results, nusc)
    # save_json(p_tracks, input_file.replace('results', 'track_info'), cls=NpEncoder)

    # Save pointclouds as npy
    # tokens = list(detection_results['results'].keys())

    # gt_tracks = extract_gt_tracks(nusc, tokens)
    # save_json(gt_tracks, input_file.replace('results', 'gt_track_info'), cls=NpEncoder)

    ## scene to track mapping
    # input_file = '/workspace/CenterPoint/work_dirs/immo/cp_5_seed_2hz/track_info_tp.json'
    # scene_names = torch.load('/workspace/CenterPoint/work_dirs/5_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/train_indices.pth')
    # tracks = load_json(input_file)
    # scene_mapping = create_scene_track_mapping(nusc, scene_names, tracks)
    # save_json(scene_mapping, input_file.replace('track_info', 'scene2trackname'))


    input_file = '/workspace/CenterPoint/work_dirs/immo/cp_5_seed_2hz/scene2trackname_tp.json'
    map_scene_track = load_json(input_file)
    tracks = load_json('/workspace/CenterPoint/work_dirs/immo/cp_5_seed_2hz/track_info_tp.json')
    padded_tracks = padd_scene_tracks(nusc, map_scene_track, tracks)
    save_json(padded_tracks, '/workspace/CenterPoint/work_dirs/immo/cp_5_seed_2hz/track_info_tp_padded_tracks.json')

if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception, set_trace
    with launch_ipdb_on_exception():
        main()