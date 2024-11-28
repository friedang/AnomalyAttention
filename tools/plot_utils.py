import matplotlib.pyplot as plt
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from ipdb import set_trace
import seaborn as sns
import numpy as np
from det3d.datasets.nuscenes.utils import load_json

def plot_bar():
    # Daten für die letzten AP-Werte pro Klasse
    ap_values_last = [0.77776, 0.40686, 0.55114, 0.19430, 0.10124, 0.76500, 0.38754, 
                      0.20544, 0.49153, 0.46540]
    classes = ["car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian", 
               "motorcycle", "bicycle", "traffic_cone", "barrier"]

    # Balkendiagramm für die letzten AP-Werte erstellen
    plt.figure(figsize=(12, 6))
    plt.barh(classes, ap_values_last, color='lightcoral')
    plt.xlabel('AP')
    plt.title('Average Precision (AP) per Class')
    plt.gca().invert_yaxis()
    plt.show()


def plot_pr():
    # Initialize NuScenes dataset
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuScenes/', verbose=True)

    # Use the detection configuration
    config = config_factory('detection_cvpr_2019')

    # Initialize the evaluator
    evaluator = DetectionEval(nusc, config=config, result_path='work_dirs/10_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/infos_val_10sweeps_withvelo_filter_True.json', eval_set='val', output_dir='path_to_output')

    # Run the evaluation
    metrics_summary = evaluator.main()

    import json
    detailed_results_path = 'work_dirs/10_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/metrics_details.json'
    with open(detailed_results_path, 'r') as f:
        detailed_results = json.load(f)

    # Example: Extracting precision and recall for the car class
    car_pr_curve = detailed_results['detailed_results']['car']['tp_errors']

    # The PR data will have thresholds and corresponding precision and recall
    set_trace()
    recalls = car_pr_curve['recall']  # List of recall values
    precisions = car_pr_curve['precision']  # List of precision values

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


def track_length_hist():
    # TODO the same for inference results
    tracks = load_json('/workspace/CenterPoint/work_dirs/Center_point_original_nusc_0075_flip/immo_results/track_info.json')
    data = list(tracks.values()) if not 'results' in tracks.keys() else list(tracks['results'].values())
    # data = [t for t in data if len(t) <= 10]
    lengths = [len(t) for t in data]

    tp_1_counts = np.zeros((max(lengths),))
    tp_0_counts = np.zeros((max(lengths),))

    # Process lists of varying lengths
    for i, sublist in enumerate(data):
        tp_1_counts[lengths[i]-1] += sum(entry['TP'] == 1 for entry in sublist)    
        tp_0_counts[lengths[i]-1] += sum(entry['TP'] == 0 for entry in sublist)

    # Lengths of the lists
    list_lengths = np.arange(1, max(lengths)+1)

    # Plot the data
    plt.figure(figsize=(12, 6))
    bar_width = 0.35

    # Bar plots for TP=1 and TP=0 counts
    plt.bar(list_lengths - bar_width / 2, tp_1_counts, bar_width, label='TP=1', color='g')
    plt.bar(list_lengths + bar_width / 2, tp_0_counts, bar_width, label='TP=0', color='r')

    # Save the plot
    plt.savefig('/workspace/CenterPoint/work_dirs/track_length_hist.png')


def main():
    # track_length_hist()
    # return
    data = load_json('./work_dirs/immo/cp_valset/results_tp_filtered.json')
    data = list(data.values()) if not 'results' in data.keys() else list(data['results'].values())
    data = [dic for res in data for dic in res]

    # Separate detection scores based on TP values

    scores_tp_1 = [entry['tracking_score'] for entry in data if entry['TP'] == 1 and entry['sample_token'] != 'dummy']
    scores_tp_0 = [entry['tracking_score'] for entry in data if entry['TP'] == 0 and entry['sample_token'] != 'dummy']

    plt.figure(figsize=(10, 6))
    sns.histplot(scores_tp_1, kde=False, color='g', label='TP = 1', bins=20, alpha=0.6)
    sns.histplot(scores_tp_0, kde=False, color='r', label='TP = 0', bins=20, alpha=0.6)

    # Customize plot
    plt.title('Histogram of Tracking Scores by TP Value')
    plt.xlabel('Tracking Score')
    plt.ylabel('Frequency')
    plt.legend()

    # Save the plot
    plt.savefig('/workspace/CenterPoint/work_dirs/tracking_score_histogram.png')


# import json


# # Daten für die Klassenverteilung
# dist = {'car': 19053, 'truck': 3926, 'bus': 680, 'trailer': 1575, 'vehicle.construction': 835, 'pedestrian': 9075, 'motorcycle': 509, 'bicycle': 634, 'trafficcone': 4462, 'barrier': 6598}

# labels = list(dist.keys())
# sizes = list(dist.values())
# # labels = ['Car', 'Pedestrian', 'Barrier', 'Traffic Cone', 'Truck', 'Trailer', 'Motorcycle', 'Construction', 'Bus', 'Bicycle']
# # sizes = [490000, 210000, 152000, 98000, 88000, 24000, 12000, 14000, 15000, 12000]

# # Tortendiagramm erstellen
# plt.figure(figsize=(10, 7))
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
# plt.title('Class Distribution')
# plt.savefig("/workspace/CenterPoint/class_dist_5_1.png")
    

if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception, set_trace
    with launch_ipdb_on_exception():
        main()