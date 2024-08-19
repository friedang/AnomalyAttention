from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory

# Initialize NuScenes dataset
nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuScenes/', verbose=True)

# Use the detection configuration
config = config_factory('detection_cvpr_2019')

# Initialize the evaluator
evaluator = DetectionEval(nusc, config=config, result_path='work_dirs/10_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/infos_val_10sweeps_withvelo_filter_True.json', eval_set='val', output_dir='path_to_output')

# Run the evaluation
metrics_summary = evaluator.main()

import json

# Load the detailed evaluation results from the output directory
summary_results_path = 'work_dirs/10_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/metrics_summary.json'
with open(summary_results_path, 'r') as f:
    summary_results = json.load(f)

detailed_results_path = 'work_dirs/10_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/metrics_details.json'
with open(detailed_results_path, 'r') as f:
    detailed_results = json.load(f)

from ipdb import launch_ipdb_on_exception
with launch_ipdb_on_exception():
    # Example: Extracting precision and recall for the car class
    car_pr_curve = detailed_results['detailed_results']['car']['tp_errors']

    # The PR data will have thresholds and corresponding precision and recall
    recalls = car_pr_curve['recall']  # List of recall values
    precisions = car_pr_curve['precision']  # List of precision values

    import matplotlib.pyplot as plt

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    # plt.plot(recalls, precisions, label='Car')
    plt.plot(recalls, precisions, marker='o', linestyle='-', color='b')
    plt.title('Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(f"{detailed_results_path.replace('metrics_details.json', 'car')}.png")

