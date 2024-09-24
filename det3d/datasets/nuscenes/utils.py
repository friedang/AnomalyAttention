import json
from typing import Dict, List
from typing import Callable

import numpy as np
import torch
import tqdm

def create_tracks(detection_results):

    tracks = {} # box, tp_label, sample_token

    for sample_token, preds in tqdm.tqdm(detection_results['results'].items()):
        for result in preds:    
            
            if result['tracking_id'] not in tracks.keys():
                tracks[result['tracking_id']] = []
            
            box = result['translation'] + result['size'] + result['rotation']
            box = torch.Tensor(box)

            tracks[result['tracking_id']].append(
                (box, result['TP'], sample_token))

    length = [len(t) for t in tracks.values()]
    set_trace()

    print(f"Max length of tracks is {max(length)} and Minimum is {min(length)} and Mean is {np.mean(length)}")


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    # Load the JSON file
    input_file = '/workspace/CenterPoint/work_dirs/immo/results_max_pass_ctrl/results_tp.json'
    # output_file = input_file.replace('.json', '_tp.json')
    # output_dir = input_file.replace('results.json', '')
    detection_results = load_json(input_file)

    # Initialize nuScenes
    create_tracks(detection_results)


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception, set_trace
    with launch_ipdb_on_exception():
        main()