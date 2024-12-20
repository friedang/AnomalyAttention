import matplotlib.pyplot as plt
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from ipdb import set_trace
import seaborn as sns
import torch
import numpy as np
from det3d.datasets.nuscenes.utils import load_json
from sklearn.metrics import precision_recall_curve, confusion_matrix, PrecisionRecallDisplay
import os


def plot_pr_confidence(precision, recall, thresholds, workdir, name):
    # Precision-Recall curve
    plt.figure()
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.title(f"Precision-Recall Curve {name}")
    plt.savefig(os.path.join(workdir, f"precision_recall_curve_{name}.png"))

    # Precision-Confidence plot
    plt.figure()
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Precision")
    plt.title(f"Precision vs. Confidence {name}")
    plt.savefig(os.path.join(workdir, f"precision_confidence_curve_{name}.png"))

    # Recall-Confidence plot
    plt.figure()
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Recall")
    plt.title(f"Recall vs. Confidence {name}")
    plt.savefig(os.path.join(workdir, f"recall_confidence_curve_{name}.png"))
    plt.close()


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, workdir, name):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create the plot
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix {name}")
    plt.colorbar()

    # Add labels for the axes
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Anomaly', 'Nomaly'])
    plt.yticks(tick_marks, ['Anomaly', 'Nomaly'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Add the counts in the matrix cells
    thresh = cm.max() / 2.  # Define a threshold for text color contrast
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    # Save the confusion matrix plot
    plt.savefig(os.path.join(workdir, f"confusion_matrix_{name}.png"))
    plt.close()  # Close the figure after saving to avoid display in notebooks


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


def track_length_bar(data=None, work_dir=None, filter=None):
    if data:
        labels, lengths = data  # labels are correct/false predictions in this case when it comes from test_ad.py
        data = [[{'TP': l}] for l in labels]
    else:
        tracks = load_json('/workspace/CenterPoint/work_dirs/Center_point_original_nusc_0075_flip/immo_results/track_info.json')
        data = list(tracks.values()) if not 'results' in tracks.keys() else list(tracks['results'].values())
        lengths = [len(t) for t in data]

        data = load_json('/workspace/CenterPoint/work_dirs/ad_pc_mlp_05/total_seed_org01th/plot_dir/data_for_plots.json')
        all_labels = np.array(data['gt_labels'])
        all_predictions = np.array(data['predictions'])
        all_lengths = data['track_lengths']
        matches_array = (all_labels == all_predictions).astype(int)
        anomaly_match_length = ([], [])
        nomaly_match_length = ([], [])
        for i, m in enumerate(matches_array):
            if all_labels[i] == 0:
                anomaly_match_length[0].append(m)
                anomaly_match_length[1].append(all_lengths[i])
            elif all_labels[i] == 1:
                nomaly_match_length[0].append(m)
                nomaly_match_length[1].append(all_lengths[i])

        labels, lengths = nomaly_match_length  # labels are correct/false predictions in this case when it comes from test_ad.py
        data = [[{'TP': l}] for l in labels]
        filter = 'anomaly'

    tp_1_counts = np.zeros((max(lengths),))
    tp_0_counts = np.zeros((max(lengths),))

    # Process lists of varying lengths
    # tp_1_len_counts = []
    # tp_0_len_counts = []
    for i, sublist in enumerate(data):
        # tp_1_len_counts.extend([lengths[i] for _ in range(sum(entry['TP'] == 1 for entry in sublist))])
        # tp_0_len_counts.extend([lengths[i] for _ in range(sum(entry['TP'] == 0 for entry in sublist))])
        tp_1_counts[lengths[i]-1] += sum(entry['TP'] == 1 for entry in sublist)    
        tp_0_counts[lengths[i]-1] += sum(entry['TP'] == 0 for entry in sublist)

    correct_rate = tp_1_counts / (tp_1_counts + tp_0_counts)
    false_rate = np.ones_like(correct_rate) - correct_rate

    # Lengths of the lists
    list_lengths = np.arange(1, max(lengths)+1)

    import pandas as pd
    # Prepare data for Seaborn
    df = pd.DataFrame({
        'Track Length': np.concatenate([list_lengths, list_lengths]),
        'Frequency': np.concatenate([tp_1_counts, tp_0_counts]),
        'TP': ['Correct'] * len(tp_1_counts) + ['False'] * len(tp_0_counts)
    })

    # # Plot using Seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='Track Length', y='Frequency', hue='TP', palette={'Correct': 'g', 'False': 'r'})


    label = ['TNR', 'FNR'] if filter == 'anomaly' else ['TPR', 'FPR']

    tpr_values = [f"{rate:.2f}" if not np.isnan(rate) else "N/A" for rate in correct_rate]
    fpr_values = [f"{rate:.2f}" if not np.isnan(rate) else "N/A" for rate in false_rate]
    table_data = {"Track Length": list_lengths, f"{label[0]}": tpr_values, f"{label[1]}": fpr_values}

    # Create the table
    table = plt.table(
        cellText=[list(table_data.values())[i] for i in range(len(table_data))],
        rowLabels=list(table_data.keys()),
        colLabels=None,
        cellLoc="center",
        loc="bottom",
        bbox=[0, -0.3, 1, 0.15],  # Adjust position and size
    )
    plt.subplots_adjust(bottom=0.3)


    # valid_patches = ax.patches[:len(correct_rate) * 2]  # Ensure we only use patches corresponding to actual bars

    # for i, p in enumerate(valid_patches):
    #     track_length_idx = i // 2  # Get the index of the track length (each track length has 2 bars)
    #     if i % 2 == 0:  # "Correct" bar
    #         ax.annotate(f"{label[0]}: {correct_rate[track_length_idx]:.2f}", #, {label[1]}: {false_rate[track_length_idx]:.2f}", 
    #                     (p.get_x() + p.get_width() / 2., p.get_height()), 
    #                     ha='center', va='center', 
    #                     xytext=(0, 8), textcoords='offset points', 
    #                     fontsize=12, color='black')


    # Plot the data
    # bar_width = 0.35
    # Bar plots for TP=1 and TP=0 counts
    # plt.bar(list_lengths - bar_width / 2, tp_1_counts, bar_width, label='Correct', color='g')
    # plt.bar(list_lengths + bar_width / 2, tp_0_counts, bar_width, label='False', color='r')

    # sns.histplot(tp_1_len_counts, kde=False, color='g', label='Correct', bins=42) # , alpha=0.6)
    # sns.histplot(tp_0_len_counts, kde=False, color='r', label='False', bins=42) # , alpha=0.6)

    title = 'Predictions by Track Length' if not filter else f"Predictions by Track Length {filter}"
    plt.title(title)
    plt.xlabel('Track Length')
    plt.ylabel('Frequency')
    plt.legend()

    file_name = 'track_length_hist.png' if not filter else f"track_length_hist_{filter}.png"
    # Save the plot
    out = work_dir + '/' + file_name if work_dir else '/workspace/CenterPoint/work_dirs/' + file_name
    plt.savefig(out)
    plt.close()

def track_score_bar(data=None, work_dir=None, filter=None):
    if data:
        labels, lengths = data  # labels are correct/false predictions in this case when it comes from test_ad.py
        data = [[{'TP': l}] for l in labels]
    else:
        tracks = load_json('/workspace/CenterPoint/work_dirs/Center_point_original_nusc_0075_flip/immo_results/track_info.json')
        data = list(tracks.values()) if not 'results' in tracks.keys() else list(tracks['results'].values())
        lengths = [len(t) for t in data]

        data = load_json('/workspace/CenterPoint/work_dirs/ad_pc_mlp_05/total_seed_org01th/plot_dir/data_for_plots.json')
        all_labels = np.array(data['gt_labels'])
        all_predictions = np.array(data['predictions'])
        all_lengths = data['track_lengths']
        all_scores = data['scores']
        matches_array = (all_labels == all_predictions).astype(int)
        anomaly_match_length = ([], [])
        nomaly_match_length = ([], [])
        for i, m in enumerate(matches_array):
            if all_labels[i] == 0:
                anomaly_match_length[0].append(m)
                anomaly_match_length[1].append(all_lengths[i])
            elif all_labels[i] == 1:
                nomaly_match_length[0].append(m)
                nomaly_match_length[1].append(all_lengths[i])

        labels, lengths = nomaly_match_length  # labels are correct/false predictions in this case when it comes from test_ad.py
        data = [[{'TP': l}] for l in labels]
        filter = 'anomaly'

    scores_tp_0 = [s for i, s in enumerate(all_scores) if all_labels[i] == 0]
    scores_tp_1 = [s for i, s in enumerate(all_scores) if all_labels[i] == 1]

    num_bins = 20  # Adjust the number of bins as needed
    bins = np.linspace(0, 1, num_bins + 1)  # Create bin edges from 0 to 1
    bin_labels = [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    bin_indices = np.digitize(all_scores, bins, right=False) - 1  # Assign scores to bins

    frequency = np.bincount(bin_indices, minlength=num_bins)

    tp_1_counts = np.zeros((max(num_bins),))
    tp_0_counts = np.zeros((max(num_bins),))

    for i, sublist in zip(bin_indices, matches_array):
        tp_1_counts[i] += sum(entry['TP'] == 1 for entry in sublist)    
        tp_0_counts[i] += sum(entry['TP'] == 0 for entry in sublist)

    correct_rate = tp_1_counts / (tp_1_counts + tp_0_counts)
    false_rate = np.ones_like(correct_rate) - correct_rate

    # Lengths of the lists
    list_lengths = np.arange(1, max(lengths)+1)

    import pandas as pd
    # Prepare data for Seaborn
    df = pd.DataFrame({
        'Track Length': np.concatenate([list_lengths, list_lengths]),
        'Frequency': np.concatenate([tp_1_counts, tp_0_counts]),
        'TP': ['Correct'] * len(tp_1_counts) + ['False'] * len(tp_0_counts)
    })

    # # Plot using Seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='Track Length', y='Frequency', hue='TP', palette={'Correct': 'g', 'False': 'r'})


    label = ['TNR', 'FNR'] if filter == 'anomaly' else ['TPR', 'FPR']

    tpr_values = [f"{rate:.2f}" if not np.isnan(rate) else "N/A" for rate in correct_rate]
    fpr_values = [f"{rate:.2f}" if not np.isnan(rate) else "N/A" for rate in false_rate]
    table_data = {"Track Length": list_lengths, f"{label[0]}": tpr_values, f"{label[1]}": fpr_values}

    # Create the table
    table = plt.table(
        cellText=[list(table_data.values())[i] for i in range(len(table_data))],
        rowLabels=list(table_data.keys()),
        colLabels=None,
        cellLoc="center",
        loc="bottom",
        bbox=[0, -0.3, 1, 0.15],  # Adjust position and size
    )
    plt.subplots_adjust(bottom=0.3)

    title = 'Predictions by Track Length' if not filter else f"Predictions by Track Length {filter}"
    plt.title(title)
    plt.xlabel('Track Length')
    plt.ylabel('Frequency')
    plt.legend()

    file_name = 'track_length_hist.png' if not filter else f"track_length_hist_{filter}.png"
    # Save the plot
    out = work_dir + '/' + file_name if work_dir else '/workspace/CenterPoint/work_dirs/' + file_name
    plt.savefig(out)

def plot_ad_by_track_score(data=None, work_dir=None, filter=None):
    if data:
        all_labels, all_scores = data
        
        if isinstance(all_scores[0], torch.Tensor):
            scores_tp_0 = [s.item() for i, s in enumerate(all_scores) if all_labels[i] == 0]
            scores_tp_1 = [s.item() for i, s in enumerate(all_scores) if all_labels[i] == 1]
        else:
            scores_tp_0 = [s for i, s in enumerate(all_scores) if all_labels[i] == 0]
            scores_tp_1 = [s for i, s in enumerate(all_scores) if all_labels[i] == 1]

    else:
        data = load_json('./work_dirs/immo/cp_valset/results_tp_filtered.json')
        data = list(data.values()) if not 'results' in data.keys() else list(data['results'].values())
        data = [dic for res in data for dic in res]

        # Separate detection scores based on TP values
        scores_tp_1 = [entry['tracking_score'] for entry in data if entry['TP'] == 1 and entry['sample_token'] != 'dummy']
        scores_tp_0 = [entry['tracking_score'] for entry in data if entry['TP'] == 0 and entry['sample_token'] != 'dummy']

        data = load_json('/workspace/CenterPoint/work_dirs/ad_pc_mlp_05/total_seed_org01th/plot_dir/data_for_plots.json')
        all_labels = np.array(data['gt_labels'])
        all_predictions = np.array(data['predictions'])
        all_scores = data['scores']
        matches_array = (all_labels == all_predictions).astype(int)
        all_labels = matches_array # nomaly_match_length[0]
        # anomaly_match_length = ([], [])
        # nomaly_match_length = ([], [])
        # for i, m in enumerate(matches_array):
        #     if all_labels[i] == 0:
        #         anomaly_match_length[0].append(m)
        #         anomaly_match_length[1].append(all_scores[i])
        #     elif all_labels[i] == 1:
        #         nomaly_match_length[0].append(m)
        #         nomaly_match_length[1].append(all_scores[i])

        # all_labels, all_scores = anomaly_match_length  # labels are correct/false predictions in this case when it comes from test_ad.py
        scores_tp_0 = [s for i, s in enumerate(all_scores) if all_labels[i] == 0]
        scores_tp_1 = [s for i, s in enumerate(all_scores) if all_labels[i] == 1]
        # filter = 'anomaly'


    # num_bins = 20  # Adjust the number of bins as needed
    # bins = np.linspace(0, 1, num_bins + 1)  # Create bin edges from 0 to 1
    # bin_labels = [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    # bin_indices = np.digitize(all_scores, bins, right=False) - 1  # Assign scores to bins

    # frequency = np.bincount(bin_indices, minlength=num_bins)

    # tp_1_counts = np.zeros((num_bins,))
    # tp_0_counts = np.zeros((num_bins,))

    # for i, m in zip(bin_indices, matches_array):
    #     if m == 1:
    #         tp_1_counts[i] += 1
    #     else:
    #         tp_0_counts[i] += 1

    # correct_rate = tp_1_counts / (tp_1_counts + tp_0_counts)
    # false_rate = np.ones_like(correct_rate) - correct_rate

    # label = ['TNR', 'FNR'] if filter == 'anomaly' else ['TPR', 'FPR']

    # tpr_values = [f"{rate:.2f}" if not np.isnan(rate) else "N/A" for rate in correct_rate]
    # fpr_values = [f"{rate:.2f}" if not np.isnan(rate) else "N/A" for rate in false_rate]
    # table_data = {"ScoreBin": bin_labels, f"{label[0]}": tpr_values, f"{label[1]}": fpr_values}

    # # Create the table
    # table = plt.table(
    #     cellText=[list(table_data.values())[i] for i in range(len(table_data))],
    #     rowLabels=list(table_data.keys()),
    #     colLabels=None,
    #     cellLoc="center",
    #     loc="bottom",
    #     bbox=[0, -0.3, 1, 0.15],  # Adjust position and size
    # )
    plt.subplots_adjust(bottom=0.3)

    plt.figure(figsize=(10, 6))
    sns.histplot(scores_tp_1, kde=False, color='g', label='Correct', bins=20, alpha=0.6)
    sns.histplot(scores_tp_0, kde=False, color='r', label='False', bins=20, alpha=0.6)

    # Customize plot
    title = 'Histogram of Tracking Scores by TP Value' if not filter else f"Histogram of Tracking Scores by TP Value {filter}"
    plt.title(title)
    plt.xlabel('Tracking Score')
    plt.ylabel('Frequency')
    plt.legend()

    # Save the plot
    file_name = 'tracking_score_histogram.png' if not filter else f"tracking_score_histogram_{filter}.png"
    out = work_dir + '/' + file_name if work_dir else '/workspace/CenterPoint/work_dirs/' + file_name
    plt.savefig(out)
    plt.close()


def main():
    # track_length_bar()
    # plot_ad_by_track_score()
    return



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