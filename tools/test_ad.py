import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix, PrecisionRecallDisplay
import logging

from det3d.datasets.nuscenes.nuscenes_track_dataset import SceneDataset
from det3d.models.classifier.mlp import TrackMLPClassifier

# Define sigmoid function for converting logits to probabilities
sigmoid = nn.Sigmoid()

# Set up logging
logging.basicConfig(filename='inference.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_json(output_dict, filename):
    with open(filename, 'w') as f:
        json.dump(output_dict, f, indent=4)

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
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0', '1'])
    plt.yticks(tick_marks, ['0', '1'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(os.path.join(workdir, 'confusion_matrix.png'))

# Inference function
def inference(model, dataloader, device, threshold, workdir, validate=False):
    model.eval()
    output_dict = {}
    all_labels = []
    all_predictions = []
    all_confidences = []

    # Iterate over batches
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            tp, meta, inputs = batch
            tokens, detection_ids = meta
            tokens = [t[0] for t in tokens]
            detection_ids = [t[0] for t in detection_ids]
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            confidences = sigmoid(outputs).squeeze(2).permute(1, 0).cpu().numpy()

            # Apply threshold to get TP labels (0 or 1)
            predicted_labels = (confidences >= threshold).astype(int)

            # Filter out 'dummy' tokens or 'dummy' ids
            for _, (t, det_id, conf, pred) in enumerate(zip(tokens, detection_ids, confidences, predicted_labels)):
                if 'dummy' not in det_id:  # Filter based on TP and id
                    # Convert token and detection_id to string for JSON compatibility
                    key = str(t)
                    if key not in output_dict.keys():
                        output_dict[key] = []    
                    
                    output_dict[key].append(
                        {"confidence": float(conf), "TP": int(pred), 
                            "id": det_id, "tracking_id": det_id.replace(t, '').replace('_', '')})
                    all_confidences.append(conf)
                    all_predictions.append(pred)

            # Collect true labels if in validation mode
            if validate:
                # Filter out -500 (padding) labels
                valid_tp = tp[tp != -500].cpu().numpy()
                all_labels.extend(valid_tp)

    # Save the results to a json file
    logging.info(f"Number of TP is {len([v for values in output_dict.values() for v in values if v['TP'] == 1])}")
    logging.info(f"Number of FP is {len([v for values in output_dict.values() for v in values if v['TP'] != 1])}")
    save_json(output_dict, os.path.join(workdir, 'inference_results.json'))

    if validate:
        # Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(all_labels, all_confidences)

        # Plot precision-recall and confidence plots
        plot_pr_confidence(precision, recall, thresholds, workdir)

        # Plot confusion matrix
        plot_confusion_matrix(all_labels, all_predictions, workdir)

        # Final TP/FP evaluation
        set_trace()
        all_labels = np.array(all_labels)
        # all_predictions = np.array(all_predictions)
        # matches = np.sum(all_labels == all_predictions)
        # mismatches = np.sum(all_labels != all_predictions)

        # logging.info(f"Total matches (correct predictions): {matches}")
        # logging.info(f"Total mismatches (incorrect predictions): {mismatches}")

        # print(f"Total matches (correct predictions): {matches}")
        # print(f"Total mismatches (incorrect predictions): {mismatches}")

def test(rank, world_size, args):
    # Initialize distributed processing group for multi-GPU inference
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Load dataset
    dataset = SceneDataset(load_chunks_from=args.load_chunks_from)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    device = torch.device(f"cuda:{rank}" if (not args.cpu) and torch.cuda.is_available() else 'cpu')
    model = TrackMLPClassifier(input_size=11, hidden_size=128, num_layers=3).to(device)
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device)['model_state_dict'], strict=False)

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
    )

    if world_size > 1:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference")
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help="Threshold for TP (default: 0.5)")
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument('--load_chunks_from', type=str, required=True, help="Path to the chunk data")
    parser.add_argument('--workdir', type=str, required=True, help="Directory to save output json and plots")
    parser.add_argument('--eval', action="store_true", help="Run evaluation with validation labels")
    parser.add_argument('--cpu', action="store_true", help="Use CPU for inference")
    parser.add_argument('--world_size', type=int, default=1, help="Number of GPUs for distributed inference")

    args = parser.parse_args()

    from ipdb import launch_ipdb_on_exception, set_trace
    with launch_ipdb_on_exception():
        if args.world_size > 1:
            # Use torch.multiprocessing.spawn to launch multi-GPU inference
            torch.multiprocessing.spawn(test, args=(args.world_size, args), nprocs=args.world_size, join=True)
        else:
            test(0, args.world_size, args)
