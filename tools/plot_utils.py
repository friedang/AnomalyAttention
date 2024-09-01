import matplotlib.pyplot as plt
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory

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

    # Load the detailed evaluation results from the output directory
    summary_results_path = 'work_dirs/10_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/metrics_summary.json'
    with open(summary_results_path, 'r') as f:
        summary_results = json.load(f)

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

# Daten für die Klassenverteilung
dist = {'car': 19053, 'truck': 3926, 'bus': 680, 'trailer': 1575, 'vehicle.construction': 835, 'pedestrian': 9075, 'motorcycle': 509, 'bicycle': 634, 'trafficcone': 4462, 'barrier': 6598}

labels = list(dist.keys())
sizes = list(dist.values())
# labels = ['Car', 'Pedestrian', 'Barrier', 'Traffic Cone', 'Truck', 'Trailer', 'Motorcycle', 'Construction', 'Bus', 'Bicycle']
# sizes = [490000, 210000, 152000, 98000, 88000, 24000, 12000, 14000, 15000, 12000]

# Tortendiagramm erstellen
plt.figure(figsize=(10, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Class Distribution')
plt.savefig("/workspace/CenterPoint/class_dist_5_1.png")
