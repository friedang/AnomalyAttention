from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import precision_recall_curve
import logging
import wandb

from tools.train_ad import setup_logging, process_point_clouds
from tools.plot_utils import track_length_bar, plot_confusion_matrix, plot_pr_confidence, plot_ad_by_track_score

from det3d.datasets.nuscenes.nuscenes_track_dataset import SceneDataset
from det3d.datasets.nuscenes.utils import NpEncoder, save_json, load_json
from det3d.models.classifier.mlp import TrackMLPClassifier
from det3d.models.classifier.lstm import LSTMClassifier
from det3d.models.classifier.transformer import TrackTransformerClassifier
from det3d.models.classifier.pc_mlp import PCTrackMLPClassifier
from det3d.models.classifier.track2pc_mlp import Track2PCTrackMLPClassifier
from det3d.models.classifier.anomaly_attention import AnomalyAttention
from det3d.models.classifier.voxel_mlp import VoxelTrackMLPClassifier


# Define sigmoid function for converting logits to probabilities
sigmoid = nn.Sigmoid()

LENGTH_THRESH=10

# total seed
SCORE_THRESH=0.2
CLASS_CONF_THRESH = {'car': 0,'bus': 0.5,'trailer': 0.5,'truck': 0.3,'pedestrian': 0,'bicycle': 0.4,
                     'motorcycle': 0.3,'construction_vehicle': 0.5, 'barrier': 0.3, 'traffic_cone': 0.01}

# seed 5
# SCORE_THRESH=0.5
# LENGTH_THRESH=10 total seed # > 10 seed10
# CLASS_CONF_THRESH = {'car': 0, 'bus': 0.4, 'trailer': 0.3, 'truck': 0.4, 'pedestrian': 0, 'bicycle': 0.4,
#                      'motorcycle': 0.4, 'construction_vehicle': 0.3, 'barrier': 0.1, 'traffic_cone': 0.01}

# seed 10
# CLASS_CONF_THRESH = {'car': 0,'bus': 0.3,'trailer': 0.3,'truck': 0.3,'pedestrian': 0,'bicycle': 0.5,
#                      'motorcycle': 0.4,'construction_vehicle': 0.4, 'barrier': 0.1, 'traffic_cone': 0.1}

def log_model_info(model):
    # Log the model architecture
    logging.info("Model Architecture:\n%s", model)
    
    # Calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    logging.info("Number of parameters: %d", num_params)
    
    # Calculate the model size in MB
    model_size_mb = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)
    logging.info("Model size: %.2f MB", model_size_mb)


def inference(model, dataloader, device, threshold, workdir, val=False, args=None):
    model.eval()
    output_dict = {}
    all_track_scores = []
    all_lengths = []
    all_labels = []
    all_predictions = []
    all_confidences = []
    all_class_names = []

    # Iterate over batches
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            tp, meta, inputs = batch

            if torch.all(tp == -500):
                continue

            tokens, detection_ids, scene_data = meta
            scene_data = scene_data[0] if len(scene_data) < len(tokens) else scene_data
            tokens = [t[0] for t in tokens]
            detection_ids = [t[0] for t in detection_ids]
            inputs = inputs.to(device)

            if args.pc or args.track_pc or args.voxel:
                pc_batch = process_point_clouds(meta, args, val=True) if args.val else process_point_clouds(meta, args)
                pc_batch = torch.tensor(np.stack(pc_batch, axis=0)).float().to(device)
                inputs = (inputs, pc_batch)

            if not any([[t for t in detection_ids if 'car' in t or 'ped' in t]]):
                outputs = model(inputs)
                confidences = sigmoid(outputs).squeeze(2).permute(1, 0).cpu().numpy()
                # Apply threshold to get TP labels (0 or 1)
                if not threshold:
                    detection_names = [meta['detection_name'][0] for meta in scene_data if meta['detection_name'][0] != 'dummy']
                    predicted_labels = (confidences >= CLASS_CONF_THRESH[detection_names[0]]).astype(int) 
                else:
                    predicted_labels = (confidences >= float(threshold)).astype(int)
            else:
                confidences = torch.ones_like(tp)[0, :, :]
                predicted_labels = torch.ones_like(tp)[0, :, :]

            valid_idx = torch.zeros(tp.shape)
            length = len([t for t in detection_ids if 'dummy' not in t])

            # Filter out 'dummy' tokens or 'dummy' ids
            for i, (t, det_id, conf, pred, meta_data) in enumerate(zip(tokens, detection_ids, confidences, predicted_labels, scene_data)):
                
                if 'dummy' not in det_id and not 'gt' in det_id:  # Filter based on TP and id
                    key = t if isinstance(t, str) else t[0]
                    if key not in output_dict.keys():
                        output_dict[key] = []

                    output = {"confidence": float(conf), "TP": int(pred),
                                "id": det_id, "tracking_id": det_id.replace(key, '').replace('_', ''),
                                "sample_token": key, "track_length": length}
                    
                    if length > LENGTH_THRESH: # TODO add arg  # length > 10 and length <= 36 seed5  # length > 10 total seed # > 10 seed10
                        output['TP'] = 1

                    if isinstance(meta_data, dict):
                        output.update({k: v for k, v in meta_data.items() if k not in output.keys()})
                        if output['tracking_score'] > SCORE_THRESH: # TODO add arg                  # 0.5 for seed5 # 0.2 for total seed # 0.4 seed10
                            output['TP'] = 1
                        if 'car' in output['detection_name'] or 'ped' in output['detection_name']:
                            output['TP'] = 1
                            output_dict[key].append(output)
                            continue

                    output_dict[key].append(output)                      

                    all_confidences.extend(conf)
                    all_predictions.extend(pred)
                    all_lengths.append(length)
                    if 'tracking_score' in output.keys():
                        all_track_scores.append(output['tracking_score'])
                        all_class_names.append(output['detection_name'])
                    valid_idx[0, i, 0] = 1

            # Filter out -500 (padding) labels
            valid_tp = tp[valid_idx == 1].cpu().numpy()
            all_labels.extend(valid_tp)

    # Save the results to a json file
    if args.local_rank == 0 or args.world_size == 1:
        fp = len([v for values in output_dict.values() for v in values if v['TP'] == 0])
        logging.info(f"Number of TP is {len([v for values in output_dict.values() for v in values if v['TP'] == 1])}")
        logging.info(f"Number of Dets detected as AD/FP is {fp}")
        if fp == 0:
            logging.info(f"No anomalies were detected for the threshold {threshold}")    

    save_json(output_dict, os.path.join(workdir, 'inference_results.json'), cls=NpEncoder)

    if (args.local_rank == 0 or args.world_size == 1):
        plot_data = {'gt_labels': all_labels,
                     'predictions': all_predictions,
                     'scores': all_track_scores,
                     'confidences': all_confidences,
                     'track_lengths': all_lengths,
                     'class_names': all_class_names
                     }
        
        plot_dir = os.path.join(workdir, 'plot_dir')
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        
        save_json(plot_data, os.path.join(plot_dir, 'data_for_plots.json'), cls=NpEncoder)

        # plot_data = load_json(os.path.join(plot_dir, 'data_for_plots.json'))
        all_labels = plot_data['gt_labels']
        all_predictions = plot_data['predictions']
        all_track_scores = plot_data['scores']
        all_confidences = plot_data['confidences']
        all_lengths = plot_data['track_lengths']
        all_class_names = plot_data['class_names']

        # Precision-Recall curve
        pr_stuff_to_plot = [(all_labels, all_confidences, all_predictions, 'all_classes')]
        all_class_names = [c[0] for c in all_class_names]

        for c in list(set(all_class_names)):
            labels = [l for i, l in enumerate(all_labels) if all_class_names[i] == c]
            confs = [l for i, l in enumerate(all_confidences) if all_class_names[i] == c]
            preds = [l for i, l in enumerate(all_predictions) if all_class_names[i] == c]
            pr_stuff_to_plot.append((labels, confs, preds, c))

        for stuff in pr_stuff_to_plot:
            labels, confs, preds, name = stuff
            
            precision, recall, thresholds = precision_recall_curve(labels, confs)
            # Plot precision-recall and confidence plots
            plot_pr_confidence(precision, recall, thresholds, plot_dir, name)
            # Plot confusion matrix
            plot_confusion_matrix(labels, preds, plot_dir, name)

        # Final TP/FP evaluation
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        matches_array = (all_labels == all_predictions).astype(int)
        matches = np.sum(matches_array)
        mismatches = np.sum(all_labels != all_predictions)

        anomaly_match_length = ([], [])
        nomaly_match_length = ([], [])
        anomaly_track_scores = []
        nomaly_track_scores = []
        for i, m in enumerate(matches_array):
            if all_labels[i] == 0:
                anomaly_match_length[0].append(m)
                anomaly_match_length[1].append(all_lengths[i])
                if all_track_scores:
                    anomaly_track_scores.append(all_track_scores[i])
            elif all_labels[i] == 1:
                nomaly_match_length[0].append(m)
                nomaly_match_length[1].append(all_lengths[i])
                if all_track_scores:
                    nomaly_track_scores.append(all_track_scores[i])

        track_length_bar(anomaly_match_length, plot_dir, filter='anomaly')
        track_length_bar(nomaly_match_length, plot_dir, filter='nomaly')
        track_length_bar((matches_array, all_lengths), plot_dir)

        if all_track_scores:
            plot_ad_by_track_score((anomaly_match_length[0], anomaly_track_scores), plot_dir, filter='anomaly')
            plot_ad_by_track_score((nomaly_match_length[0], nomaly_track_scores), plot_dir,  filter='nomaly')
            plot_ad_by_track_score((matches_array, all_track_scores), plot_dir)

        # Calculate percentages
        total_predictions = len(all_predictions)
        correct_preds_percentage = (matches / total_predictions) * 100
        correct_tp_percentage = (np.sum((all_labels == 1) & (all_predictions == 1)) / np.sum(all_labels == 1)) * 100
        correct_tn_percentage = (np.sum((all_labels == 0) & (all_predictions == 0)) / np.sum(all_labels == 0)) * 100

        # Log the values
        # wandb.log({
        #     "Correct preds": matches,
        #     "Wrong preds": mismatches,
        #     "Accuracy": correct_preds_percentage,
        #     "TPR (nomalies / label 1)": correct_tp_percentage,
        #     "TNR (anomalies / label 0)": correct_tn_percentage
        # })

        # Print and log information
        # logging.info(f"WandB Run Name: {wandb.run.name}")
        logging.info(f"Total matches (correct predictions): {matches}")
        logging.info(f"Total mismatches (incorrect predictions): {mismatches}")
        logging.info(f"Correct predictions (Accuracy): {correct_preds_percentage:.2f}%")
        logging.info(f"Correct predictions (Relativ_Accuracy): {(correct_tp_percentage + correct_tn_percentage) / 2:.2f}%")
        logging.info(f"Nomalies found (TPR / nomalies / label 1): {correct_tp_percentage:.2f}%")
        logging.info(f"Anomalies found (TNR / anomalies / label 0): {correct_tn_percentage:.2f}%")
        logging.info(f"FPR: {100 - correct_tp_percentage:.2f}%")
        logging.info(f"FNR: {100 - correct_tn_percentage:.2f}%")
        logging.info(f"TP: {np.sum((all_labels == 1) & (all_predictions == 1))}")
        logging.info(f"TN: {np.sum((all_labels == 0) & (all_predictions == 0))}")
        logging.info(f"FP: {np.sum((all_labels == 0) & (all_predictions == 1))}")
        logging.info(f"FN: {np.sum((all_labels == 1) & (all_predictions == 0))}")

        print(f"Total number of predictions: {total_predictions}")
        print(f"Total matches (correct predictions): {matches}")
        print(f"Total mismatches (incorrect predictions): {mismatches}")
        print(f"Correct predictions (Total_Accuracy): {correct_preds_percentage:.2f}%")
        print(f"Correct predictions (Relativ_Accuracy): {(correct_tp_percentage + correct_tn_percentage) / 2:.2f}%")
        print(f"Nomalies found (TPR / nomalies / label 1): {correct_tp_percentage:.2f}%")
        print(f"Anomalies found (TNR / anomalies / label 0): {correct_tn_percentage:.2f}%")
        print(f"FPR: {100 - correct_tp_percentage:.2f}%")
        print(f"FNR: {100 - correct_tn_percentage:.2f}%")
        print(f"TP: {np.sum((all_labels == 1) & (all_predictions == 1))}")
        print(f"TN: {np.sum((all_labels == 0) & (all_predictions == 0))}")
        print(f"FP: {np.sum((all_labels == 0) & (all_predictions == 1))}")
        print(f"FN: {np.sum((all_labels == 1) & (all_predictions == 0))}")

def test(rank, world_size, args):
    # Initialize distributed processing group for multi-GPU inference
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl", init_method='env://', rank=rank, world_size=world_size)

    # Load dataset
    if args.load_chunks_from:
        dataset = SceneDataset(load_chunks_from=args.load_chunks_from, inference=True)
    else:
        dataset = SceneDataset(scenes_info_path=args.scenes_info, track_info=args.track_info, inference=True, chunk_size=41, max_track_len=41)
        shutil.copy(args.scenes_info.replace('scene2trackname.json', 'inference_chunks.pt'), args.workdir)
    
    print(f"Chunk size is {len(dataset.chunks[0][2])}")

    label_arrays = [tp[0][tp[0] != -500].cpu().numpy() for tp in dataset.chunks if tp[0][tp[0] != -500].nelement() != 0]
    data = []
    for l in label_arrays:  
        data.extend(l)
    data = np.array(data)
    num_1 = len([i for i in data if i == 1])
    num_0 = len([i for i in data if i == 0])
    logging.info(f"# FP samples: {num_0} - # TP samples: {num_1}")

    

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler)

    # Load model
    device = torch.device(f"cuda:{rank}" if (not args.cpu) and torch.cuda.is_available() else 'cpu')
    if args.track_pc:
        model = Track2PCTrackMLPClassifier(input_size=12, hidden_size=128, num_layers=3)
    elif args.pc:
        model = PCTrackMLPClassifier()
    elif args.voxel:
            model = VoxelTrackMLPClassifier(input_size=12, hidden_size=128, num_layers=3)
    elif args.attention:
            model = AnomalyAttention(
                hidden_size=256,
                num_layers=6,
                num_heads=8,
                mlp_feature_dim=256
            )
    else:
        model = TrackMLPClassifier()

    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device)['model_state_dict'], strict=False)
    log_model_info(model)

    model.to(device)
    model = DDP(model, device_ids=[rank]) if world_size > 1 else model    

    # Logging setup
    # if rank == 0 or args.world_size == 1:
    #     if args.run_name:
    #         wandb.init(project="EvalTrackMLPClassifier", name=args.run_name)
    #     else:
    #         wandb.init(project="EvalTrackMLPClassifier")
    #     # wandb.config.update(args)
    #     wandb.watch(model)

    # Output directory
    if not os.path.exists(args.workdir):
        os.makedirs(args.workdir)

    # Perform inference
    inference(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=args.confidence_threshold,
        workdir=args.workdir,
        val=args.val,
        args=args
    )

    if world_size > 1:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes_info', type=str, help="Path to scenes info")
    parser.add_argument('--track_info', type=str, help="Path to track info")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference")
    parser.add_argument('--confidence_threshold', default=None, help="Threshold for Nomaly")
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument('--load_chunks_from', type=str, help="Path to the chunk data")
    parser.add_argument('--workdir', type=str, required=True, help="Directory to save output json and plots")
    parser.add_argument('--val', action="store_true", help="Run evaluation with validation labels")
    parser.add_argument('--cpu', action="store_true", help="Use CPU for inference")
    parser.add_argument('--world_size', type=int, default=1, help="Number of GPUs for distributed inference")
    parser.add_argument('--attention', action="store_true", help="Use AnomalyAttention")
    parser.add_argument('--pc', action="store_true", help="Use PointNet")
    parser.add_argument('--voxel', action="store_true", help="Use VoxelNet")
    parser.add_argument('--track_pc', action="store_true", help="Use point clouds")
    parser.add_argument('--run_name', type=str, default='', help="wandb run name")
    parser.add_argument("--local-rank", type=int, default=0)

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    args = parser.parse_args()

    if not os.path.isdir(args.workdir):
        os.mkdir(args.workdir)

    if args.local_rank == 0 or args.world_size == 1:
        setup_logging(os.path.join(args.workdir, 'testing.log'))


    from ipdb import launch_ipdb_on_exception, set_trace
    with launch_ipdb_on_exception():
        if args.world_size > 1:
            print('WARNING MULTI GPU INFERENCE NOT DEBUGGED YET')
            # Use torch.multiprocessing.spawn to launch multi-GPU inference
            torch.multiprocessing.spawn(test, args=(args.world_size, args), nprocs=args.world_size, join=True)
        else:
            test(0, args.world_size, args)
