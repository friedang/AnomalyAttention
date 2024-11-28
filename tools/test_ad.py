from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from tqdm import tqdm
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix, PrecisionRecallDisplay
import logging
import wandb

from tools.train_ad import setup_logging, process_point_clouds

from det3d.datasets.nuscenes.nuscenes_track_dataset import SceneDataset
from det3d.datasets.nuscenes.utils import NpEncoder, save_json
from det3d.models.classifier.mlp import TrackMLPClassifier
from det3d.models.classifier.lstm import LSTMClassifier
from det3d.models.classifier.transformer import TrackTransformerClassifier
from det3d.models.classifier.pc_mlp import PCTrackMLPClassifier
from det3d.models.classifier.track2pc_mlp import Track2PCTrackMLPClassifier
from det3d.models.classifier.voxel_at import VoxelTrackMLPClassifier


# Define sigmoid function for converting logits to probabilities
sigmoid = nn.Sigmoid()


def log_model_info(model):
    # Log the model architecture
    logging.info("Model Architecture:\n%s", model)
    
    # Calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    logging.info("Number of parameters: %d", num_params)
    
    # Calculate the model size in MB
    model_size_mb = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)
    logging.info("Model size: %.2f MB", model_size_mb)

# Function to plot precision-recall curve and confidence plots
def plot_pr_confidence(precision, recall, thresholds, workdir):
    # Precision-Recall curve
    plt.figure()
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(workdir, 'precision_recall_curve.png'))

    # Precision-Confidence plot
    plt.figure()
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Precision")
    plt.title("Precision vs. Confidence")
    plt.savefig(os.path.join(workdir, 'precision_confidence_curve.png'))

    # Recall-Confidence plot
    plt.figure()
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Recall")
    plt.title("Recall vs. Confidence")
    plt.savefig(os.path.join(workdir, 'recall_confidence_curve.png'))

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, workdir):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create the plot
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Add labels for the axes
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0', '1'])
    plt.yticks(tick_marks, ['0', '1'])
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
    plt.savefig(os.path.join(workdir, 'confusion_matrix.png'))
    plt.close()  # Close the figure after saving to avoid display in notebooks

def inference(model, dataloader, device, threshold, workdir, validate=False, args=None):
    model.eval()
    output_dict = {}
    all_labels = []
    all_predictions = []
    all_confidences = []
    inf = True if not validate else False

    # Iterate over batches
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            tp, meta, inputs = batch

            if torch.all(tp == -500):
                continue

            if inf:
                tokens, detection_ids, scene_data = meta
            else:
                if len(meta) == 2:
                    tokens, detection_ids = meta
                else:
                    tokens, detection_ids, _ = meta
                scene_data = [{}] * len(tokens)
            tokens = [t[0] for t in tokens]
            detection_ids = [t[0] for t in detection_ids]
            inputs = inputs.to(device)

            if args.pc or args.track_pc or args.voxel:
                pc_batch = process_point_clouds(meta, args) if args.eval else process_point_clouds(meta, args, inf=True)
                pc_batch = torch.tensor(np.stack(pc_batch, axis=0)).float().to(device)
                inputs = (inputs, pc_batch)

            outputs = model(inputs)
            confidences = sigmoid(outputs).squeeze(2).permute(1, 0).cpu().numpy()

            # Apply threshold to get TP labels (0 or 1)
            predicted_labels = (confidences >= threshold).astype(int)

            valid_idx = torch.zeros(tp.shape)

            # Filter out 'dummy' tokens or 'dummy' ids
            for i, (t, det_id, conf, pred, meta_data) in enumerate(zip(tokens, detection_ids, confidences, predicted_labels, scene_data)):
                if 'dummy' not in det_id and not 'gt' in det_id:  # Filter based on TP and id
                    # Convert token and detection_id to string for JSON compatibility
                    key = t if isinstance(t, str) else t[0]
                    if key not in output_dict.keys():
                        output_dict[key] = []

                    output = {"confidence": float(conf), "TP": int(pred),
                                "id": det_id, "tracking_id": det_id.replace(key, '').replace('_', ''),
                                "sample_token": key,}
                    if isinstance(meta_data, dict):
                        output.update({k: v for k, v in meta_data.items() if k not in output.keys()})

                    output_dict[key].append(output)
                    all_confidences.extend(conf)    # python3 ./tools/merge_seed_pseudo_gt.py
                    all_predictions.extend(pred)
                    valid_idx[0, i, 0] = 1

            # Collect true labels if in validation mode
            if validate:
                # Filter out -500 (padding) labels
                valid_tp = tp[valid_idx == 1].cpu().numpy()
                all_labels.extend(valid_tp)

    # Save the results to a json file
    if args.local_rank == 0 or args.world_size == 1:
        fp = len([v for values in output_dict.values() for v in values if v['TP'] == 0])
        logging.info(f"Number of TP is {len([v for values in output_dict.values() for v in values if v['TP'] == 1])}")
        logging.info(f"Number of Dets detected as AD/FP is {fp}")
        assert fp != 0, f"No anomalies were detected for the threshold {threshold}"
    

    save_json(output_dict, os.path.join(workdir, 'inference_results.json'), cls=NpEncoder)

    if (args.local_rank == 0 or args.world_size == 1) and validate:
        # Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(all_labels, all_confidences)

        # Plot precision-recall and confidence plots
        plot_pr_confidence(precision, recall, thresholds, workdir)

        # Plot confusion matrix
        plot_confusion_matrix(all_labels, all_predictions, workdir)

        # Final TP/FP evaluation
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        matches = np.sum(all_labels == all_predictions)
        mismatches = np.sum(all_labels != all_predictions)

        # Calculate percentages
        total_predictions = len(all_predictions)
        correct_preds_percentage = (matches / total_predictions) * 100
        correct_tp_percentage = (np.sum((all_labels == 1) & (all_predictions == 1)) / np.sum(all_labels == 1)) * 100
        correct_tn_percentage = (np.sum((all_labels == 0) & (all_predictions == 0)) / np.sum(all_labels == 0)) * 100

        # Log the values
        wandb.log({
            "Correct preds": matches,
            "Wrong preds": mismatches,
            "Correct preds %": correct_preds_percentage,
            "Correct TP % (label 1)": correct_tp_percentage,
            "Correct TN % (label 0)": correct_tn_percentage
        })

        # Print and log information
        logging.info(f"Total matches (correct predictions): {matches}")
        logging.info(f"Total mismatches (incorrect predictions): {mismatches}")
        logging.info(f"Correct predictions percentage: {correct_preds_percentage:.2f}%")
        logging.info(f"Correct TP (label 1) percentage: {correct_tp_percentage:.2f}%")
        logging.info(f"Correct TN (label 0) percentage: {correct_tn_percentage:.2f}%")

        print(f"Total number of predictions: {total_predictions}")
        print(f"Total matches (correct predictions): {matches}")
        print(f"Total mismatches (incorrect predictions): {mismatches}")
        print(f"Correct predictions percentage: {correct_preds_percentage:.2f}%")
        print(f"Correct TP (label 1) percentage: {correct_tp_percentage:.2f}%")
        print(f"Correct TN (label 0) percentage: {correct_tn_percentage:.2f}%")

def test(rank, world_size, args):
    # Initialize distributed processing group for multi-GPU inference
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl", init_method='env://', rank=rank, world_size=world_size)

    # Load dataset
    inf = True if not args.eval else False
    if args.load_chunks_from:
        dataset = SceneDataset(load_chunks_from=args.load_chunks_from, inference=inf)
    else:
        dataset = SceneDataset(scenes_info_path=args.scenes_info, track_info=args.track_info, inference=inf)
    
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
        model = PCTrackMLPClassifier() # (input_size=12, hidden_size=128, num_layers=3)
    elif args.voxel:
            model = VoxelTrackMLPClassifier() # (input_size=12, hidden_size=128, num_layers=3)
    else:
        model = TrackMLPClassifier() # (input_size=12, hidden_size=128, num_layers=3)

    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device)['model_state_dict'], strict=False)
    log_model_info(model)

    model.to(device)
    model = DDP(model, device_ids=[rank]) if world_size > 1 else model    

    # Logging setup
    if rank == 0 or args.world_size == 1:
        if args.run_name:
            wandb.init(project="EvalTrackMLPClassifier", name=args.run_name)
        else:
            wandb.init(project="EvalTrackMLPClassifier")
        # wandb.config.update(args)
        wandb.watch(model)

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
        validate=args.eval,
        args=args
    )

    if world_size > 1:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes_info', type=str, help="Path to scenes info")
    parser.add_argument('--track_info', type=str, help="Path to track info")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference")
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help="Threshold for TP (default: 0.5)")
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument('--load_chunks_from', type=str, help="Path to the chunk data")
    parser.add_argument('--workdir', type=str, required=True, help="Directory to save output json and plots")
    parser.add_argument('--eval', action="store_true", help="Run evaluation with validation labels")
    parser.add_argument('--cpu', action="store_true", help="Use CPU for inference")
    parser.add_argument('--world_size', type=int, default=1, help="Number of GPUs for distributed inference")
    parser.add_argument('--pc', action="store_true", help="Use point clouds")
    parser.add_argument('--voxel', action="store_true", help="Use point clouds")
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
