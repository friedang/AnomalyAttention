from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from tqdm import tqdm
import wandb
from sklearn.utils.class_weight import compute_class_weight

import argparse
import os
import shutil
import logging

from det3d.datasets.nuscenes.nuscenes_track_dataset import SceneDataset
from det3d.models.classifier.mlp import TrackMLPClassifier
from det3d.models.classifier.pc_mlp import PCTrackMLPClassifier
from det3d.models.classifier.track2pc_mlp import Track2PCTrackMLPClassifier
from det3d.models.classifier.transformer import TrackTransformerClassifier
from det3d.models.classifier.anomaly_attention import AnomalyAttention
from det3d.models.classifier.voxel_mlp import VoxelTrackMLPClassifier


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification.

        Args:
            alpha (float): Balancing factor for the positive class. Default is 1.0.
            gamma (float): Focusing parameter. Default is 2.0.
            reduction (str): Specifies the reduction to apply to the output: 
                             'none' | 'mean' | 'sum'. Default is 'mean'.
        """
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.tensor([1, 1])  # Default to equal weighting
        else:
            self.alpha = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha)  # Pass computed class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for the focal loss.

        Args:
            inputs (Tensor): Predictions from the model (logits).
            targets (Tensor): Ground truth labels (0 or 1).

        Returns:
            Tensor: Computed focal loss.
        """
        # Apply sigmoid to the inputs to get probabilities
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets.float())
        
        # Get the probabilities
        pt = torch.exp(-bce_loss)  # Probability of the true class
        self.alpha = self.alpha.to(targets.device)
        
        alpha_t = self.alpha[targets.long()] if self.alpha.shape < targets.shape else self.alpha  # Apply class-specific weights
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def setup_logging(log_file):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level

    # File handler to log to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Stream handler to log to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def log_model_info(model):
    # Log the model architecture
    logging.info("Model Architecture:\n%s", model)
    
    # Calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    logging.info("Number of parameters: %d", num_params)
    
    # Calculate the model size in MB
    model_size_mb = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)
    logging.info("Model size: %.2f MB", model_size_mb)


def save_gradients(grad):
    # Visualize or save the gradient values to check them later
    grad_mean = grad.mean().item()
    if grad_mean != 0:
        print("Gradient:", grad_mean)
    return grad

# Define BCEWithLogitsLoss with masking support
def masked_loss(predictions, labels, class_weights=None):
    
    loss_fn = FocalLoss(alpha=class_weights, gamma=2.0) if class_weights is not None else nn.BCEWithLogitsLoss() # 
    # loss_fn = nn.BCEWithLogitsLoss() # 
    loss = loss_fn(predictions, labels)
    # loss = loss * mask  # Apply mask to ignore padding
    loss = loss.mean() if class_weights is not None else loss.sum()
    return loss

def compute_accuracy(outputs, labels, threshold=0.5):
    sigmoid = nn.Sigmoid()
    confidences = sigmoid(outputs).cpu().numpy()
    predictions = (confidences >= threshold).astype(int)
    matches = np.sum(labels.cpu().numpy() == predictions)
    total_predictions = len(predictions)
    correct_preds_percentage = (matches / total_predictions) * 100
    return correct_preds_percentage

# Validation step
def validate(model, dataloader, device, args=None, pc=True, tuning=False):
    model.eval()
    running_val_loss = 0.0
    running_val_acc = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            tp, meta, inputs = batch
            tp, inputs = tp.to(device), inputs.to(device)

            # Skip if all labels are padding
            # if tuning and any([inputs[0, i, -1] for i in range(inputs.shape[1]) if inputs[0, i, -1] in [1, 5]]):
            #     continue
            if torch.all(tp == -500):
                continue

            # Process point clouds if necessary
            if pc:
                pc_batch = process_point_clouds(meta, args)                    
                pc_batch = torch.tensor(np.stack(pc_batch, axis=0)).float().to(device)
                inputs = (inputs, pc_batch)

            # Mask where TP is not padding
            mask = (tp != -500).float()

            # Forward pass
            outputs = model(inputs)

            # Mask labels and predictions
            masked_outputs = outputs[mask.bool()] 
            masked_labels = tp[mask.bool()] 

            # Compute loss
            loss = masked_loss(masked_outputs, masked_labels)
            running_val_loss += loss.item()

            # Compute accuracy
            accuracy = compute_accuracy(masked_outputs, masked_labels)
            running_val_acc += accuracy.item() * masked_labels.size(0)
            total_samples += masked_labels.size(0)

    # Average loss and accuracy
    avg_loss = running_val_loss / len(dataloader)
    avg_acc = running_val_acc / total_samples if total_samples > 0 else 0
    return avg_loss, avg_acc


def process_point_clouds(meta, args=None, val=False):
    tokens = meta[0]
    token_len = len(tokens[0])
    pc_batch = []

    for i in range(token_len):
        dummy_status = True
        for t in tokens:
            if t[i] != "dummy":
                if val:
                    pointcloud_path = Path('/workspace/CenterPoint/work_dirs/PCs_npy_vox/val') / f"{t[i]}.npy" # if args and args.voxel else Path('/workspace/CenterPoint/work_dirs/PCs_npy/val') / f"{t[i]}.npy"
                else:
                    pointcloud_path = Path('/workspace/CenterPoint/work_dirs/PCs_npy_vox/train') / f"{t[i]}.npy" # if args and args.voxel else Path('/workspace/CenterPoint/work_dirs/PCs_npy/train') / f"{t[i]}.npy"
                pc_batch.append(np.load(str(pointcloud_path)).T)
                dummy_status = False
                break
        if dummy_status:
            pc_batch.append([])

    min_points = min([pc.shape[1] for pc in pc_batch if isinstance(pc, np.ndarray)])
    pc_batch = [pc if isinstance(pc, np.ndarray) else np.zeros((5, min_points)) for pc in pc_batch]
    pc_batch = [pc[:, np.random.permutation(min_points)] for pc in pc_batch]
    
    return pc_batch


def train(rank, world_size, args):
    if world_size > 1:
        # Initialize the process group for distributed training
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Load the full dataset
    if args.load_chunks_from:
        train_dataset = SceneDataset(load_chunks_from=args.load_chunks_from + 'train_chunks.pt') # TODO use pathlib
        val_dataset = SceneDataset(load_chunks_from=args.load_chunks_from + 'val_chunks.pt') # TODO use pathlib
    else:
        if args.gt:
            train_dataset = SceneDataset(scenes_info_path=args.scenes_info, track_info=args.track_info,
                                        sample_ratio=args.sample_ratio,
                                        gt_scenes_info_path=args.scenes_info.replace('scene2', 'gt_scene2'), gt_track_info=args.track_info.replace('track_info', 'gt_track_info'))
            val_dataset = SceneDataset(load_chunks_from=args.scenes_info.replace('scene2trackname.json', 'val_chunks.pt')) # TODO use pathlib
        else:
            train_dataset = SceneDataset(scenes_info_path=args.scenes_info, track_info=args.track_info, sample_ratio=args.sample_ratio)
            val_dataset = SceneDataset(load_chunks_from=args.scenes_info.replace('scene2trackname.json', 'val_chunks.pt')) # TODO use pathlib

        shutil.copy(args.scenes_info.replace('scene2trackname.json', 'val_chunks.pt'), args.workdir)
        shutil.copy(args.scenes_info.replace('scene2trackname.json', 'train_chunks.pt'), args.workdir)

    assert train_dataset.chunks != val_dataset.chunks

    logging.info(f"Training on {len(train_dataset)} samples - Validation on {len(val_dataset)} samples.")

    label_arrays = [tp[0][tp[0] != -500].cpu().numpy() for tp in train_dataset.chunks if tp[0][tp[0] != -500].nelement() != 0]
    
    # creating class weights
    logging.info('FOR TRAINING')
    data = []
    for l in label_arrays:  
        data.extend(l)
    data = np.array(data)
    num_1 = len([i for i in data if i == 1])
    num_0 = len([i for i in data if i == 0])
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=data)
    logging.info(f"# FP samples: {num_0} - # TP samples: {num_1} - Class weights are {class_weights}")
    # class_weights = [class_weights[1], class_weights[0]]

    logging.info('FOR VALIDATION')
    label_arrays = [tp[0][tp[0] != -500].cpu().numpy() for tp in val_dataset.chunks if tp[0][tp[0] != -500].nelement() != 0]
    data = []
    for l in label_arrays:
        data.extend(l)
    data = np.array(data)
    num_1 = len([i for i in data if i == 1])
    num_0 = len([i for i in data if i == 0])
    not_used_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=data)
    logging.info(f"# FP samples: {num_0} - # TP samples: {num_1} - Class weights are {not_used_weights}")


    # Set up samplers for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if world_size > 1 else None

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler)

    # Instantiate the model and move to device
    if args.track_pc:
        model = Track2PCTrackMLPClassifier(input_size=12, hidden_size=128, num_layers=3)
    elif args.pc:
        model = PCTrackMLPClassifier() # input_size=12, hidden_size=128, num_layers=3)
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

    device = torch.device(f'cuda:{rank}' if (not args.cpu) and torch.cuda.is_available() else 'cpu')
    model.to(device)

    if world_size > 1:
        # Wrap the model for distributed training
        model = DDP(model, device_ids=[rank])

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # , weight_decay=0.01)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0003)

    # Option to resume from checkpoint
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    # Logging setup
    if rank == 0 or args.world_size == 1:
        if args.run_name:
            wandb.init(project="TrackMLPClassifier", name=args.run_name)
        else:
            wandb.init(project="TrackMLPClassifier")
        # wandb.config.update(args)
        wandb.watch(model)
    
    log_model_info(model)
    logging.info(f"WandB Run Name: {wandb.run.name}")

    # steps_per_epoch = int(np.ceil(len(train_dataset.chunks) / args.batch_size))
    # total_steps = steps_per_epoch * args.epochs
    # lr_max = 0.1
    # moms = [0.85, 0.95]
    # div_factor = 25
    # pct_start = 0.3
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, verbose=True) # torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if args.world_size > 1:
            dist.barrier()
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Rank {rank}, Epoch {epoch + 1}"):
            optimizer.zero_grad()

            # Move batch data to device
            tp, meta, inputs = batch
            tp, inputs = tp.to(device), inputs.to(device)

            if torch.all(tp == -500):
                continue

            if args.pc or args.track_pc or args.voxel:
                pc_batch = process_point_clouds(meta, args)                    
                pc_batch = torch.tensor(np.stack(pc_batch, axis=0)).float().to(device)
                inputs = (inputs, pc_batch)

            # Forward pass
            outputs = model(inputs)

            # Mask where TP is not padding (-500)
            mask = (tp != -500).float()
            masked_outputs = outputs[mask.bool()] # outputs * mask
            masked_labels = tp[mask.bool()] # tp * mask

            # Compute loss
            weights = class_weights
            detection_ids = meta[1]
            detection_ids = [t[0] for t in detection_ids]
            
            if len(meta) == 3 and not any(['gt' in i for i in detection_ids]):
                weights = torch.zeros_like(tp)
                dist_tps = meta[2].to(device)
                B, N, _ = weights.shape
                
                for b in range(B):
                    for n in range(N):
                        num_tp = len([t for t in dist_tps[b][n] if t == 1])
                        if num_tp == 0:
                            weights[b][n] = class_weights[0]
                        elif num_tp == 1:
                            weights[b][n] = class_weights[1]
                        elif num_tp == 2:
                            weights[b][n] = class_weights[1] * 1.5
                        elif num_tp == 3:
                            weights[b][n] = class_weights[1] * 2
                        else:
                            weights[b][n] = class_weights[1] * 2.5 # 1.45 # * 1.25  # * num_tp

                weights = weights.to(device)
                weights = weights[mask.bool()]
            elif any(['gt' in i for i in detection_ids]):
                weights = weights * 0.5
            
            # last run was with gt, max norm 5, and weights max 1.45
            loss = masked_loss(masked_outputs, masked_labels, weights)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=30, norm_type=2)
            optimizer.step()

            running_loss += loss.item()

        # scheduler.step(loss)

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         param.register_hook(save_gradients)

        if args.world_size > 1:
            dist.barrier()
        if rank == 0 or args.world_size == 1:  # Only the master node logs
            if (epoch + 1) % 5 == 0:
                # Save checkpoint
                checkpoint_path = os.path.join(args.workdir, f"epoch_{epoch + 1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)

                # Validation step
                pc = True if (args.track_pc or args.voxel) else False
                val_loss, val_acc = validate(model, val_loader, device, pc=pc) # , class_weights)

                avg_train_loss = running_loss / len(train_loader)
                logging.info(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Val Accuracy: {val_acc}, Val Loss: {val_loss}")
                print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Val Accuracy: {val_acc}, Val Loss: {val_loss}")
                wandb.log({"Train Loss": avg_train_loss, "Val Accuracy": val_acc, "Val Loss": val_loss})
            else:
                avg_train_loss = running_loss / len(train_loader)
                logging.info(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}")
                print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}")
                wandb.log({"Train Loss": avg_train_loss})

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    "TRAINING AERGS"
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=24, help="Batch size per GPU") # 22 for non freezed
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--world_size', type=int, default=1, help="Number of GPUs for distributed training")
    parser.add_argument('--scenes_info', type=str, help="Path to scenes info")
    parser.add_argument('--run_name', type=str, default='', help="wandb run name")
    parser.add_argument('--track_info', type=str, help="Path to track info")
    parser.add_argument('--gt', action="store_true", help="Use GT track info")
    parser.add_argument('--pc', action="store_true", help="Use point clouds")
    parser.add_argument('--track_pc', action="store_true", help="Use point clouds")
    parser.add_argument('--attention', action="store_true", help="Use AnomalyAttention")
    parser.add_argument('--voxel', action="store_true", help="Use VoxelNetAD")
    parser.add_argument('--cpu', action="store_true", help="Use CPU")
    parser.add_argument('--load_chunks_from', type=str, help="Use GT track info")
    parser.add_argument('--workdir', type=str, required=True, help="Directory to workdir")
    parser.add_argument('--resume_from', type=str, help="Checkpoint resume from file")
    parser.add_argument('--sample_ratio', type=float, default=0.85, help="Ratio for training/validation")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--single_class", action="store_true")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if not os.path.isdir(args.workdir):
            os.mkdir(args.workdir)

    if args.local_rank == 0 or args.world_size == 1:
            setup_logging(os.path.join(args.workdir, 'training.log'))

    world_size = args.world_size
    from ipdb import launch_ipdb_on_exception, set_trace
    with launch_ipdb_on_exception():
        if world_size > 1:
            # Use torch.multiprocessing.spawn to launch distributed training
            torch.multiprocessing.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
        else:
            if args.single_class:
                workdir = args.workdir
                for i in ['car', 'bus', 'trailer', 'truck', 'pedestrian', 'bicycle', 'motorcycle', 'construction_vehicle', 'barrier', 'traffic_cone']:
                    args.workdir = os.path.join(workdir, f"class_{i}")
                    if not os.path.isdir(args.workdir):
                        os.mkdir(args.workdir)
                    args.single_class = i
                    
                    train(0, world_size, args)
            else:
                train(0, world_size, args)

