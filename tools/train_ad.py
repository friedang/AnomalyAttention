from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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
            self.alpha = torch.tensor([0.75, 0.25])  # Default to equal weighting
        else:
            self.alpha = torch.tensor(alpha)  # Pass computed class weights
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
        alpha_t = self.alpha[targets.long()]  # Apply class-specific weights
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def save_gradients(grad):
    # Visualize or save the gradient values to check them later
    grad_mean = grad.mean().item()
    if grad_mean != 0:
        print("Gradient:", grad_mean)
    return grad

# Define BCEWithLogitsLoss with masking support
def masked_bce_loss(predictions, labels, class_weights):
    # set_trace()
    loss_fn = FocalLoss(alpha=class_weights, gamma=2.0) #  nn.BCEWithLogitsLoss() # 
    loss = loss_fn(predictions, labels)
    # loss = loss * mask  # Apply mask to ignore padding
    return loss.mean()

# Validation step
def validate(model, dataloader, device, class_weights):
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            tp, meta, inputs = batch
            tp, inputs = tp.to(device), inputs.to(device)

            if args.pc or args.track_pc:
                tokens, _, _ = meta
                token_len = len(tokens[0])
                pc_batch = []
                for i in range(token_len):
                    dummy_status = True
                    for t in tokens:
                        if t[i] != "dummy":
                            pointcloud_path = Path('/workspace/CenterPoint/work_dirs/PCs_npy/train') / f"{t[i]}.npy"
                            pc_batch.append(np.load(str(pointcloud_path)).reshape(4, 5000))
                            dummy_status = False
                            break
                    if dummy_status:
                        pc_batch.append(np.zeros((4, 5000)))
                    
                pc_batch = torch.tensor(np.stack(pc_batch, axis=0)).float().to(device)
                
                # tokens, _, _ = meta
                # token_len = len(tokens[0])
                # pc_batch = []
                # for i in range(token_len):
                #     pc_data = []
                #     for t in tokens:
                #         if t[i] == "dummy":
                #             pc_data.append(np.zeros((4, 5000)))
                #         else:
                #             pointcloud_path = Path('/workspace/CenterPoint/work_dirs/PCs_npy/train') / f"{t[i]}.npy"
                #             pc_data.append(np.load(str(pointcloud_path)).reshape(4, 5000))
                    
                #     pc_batch.append(np.concatenate(pc_data, axis=1))
                
                # pc_batch = torch.tensor(np.stack(pc_batch, axis=0)).float().to(device)
                
                inputs = (inputs, pc_batch)

            # Mask where TP is not padding (-500)
            mask = (tp != -500).float()

            # Forward pass
            outputs = model(inputs)

            # Mask labels and predictions
            masked_outputs = outputs[mask.bool()] 
            masked_labels = tp[mask.bool()] 

            # Compute loss
            loss = masked_bce_loss(masked_outputs, masked_labels, class_weights) # , mask.unsqueeze(-1))
            running_val_loss += loss.item()
    
    return running_val_loss / len(dataloader)

# Training step
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

    # assert train_dataset.chunks != val_dataset.chunks
    print(f"Training on {len(train_dataset)} samples - Validation on {len(val_dataset)} samples.")
    label_arrays = [tp[0][tp[0] != -500].cpu().numpy() for tp in train_dataset.chunks]
    
    # creating class weights
    print('FOR TRAINING')
    data = []
    for l in label_arrays:
        data.extend(l)
    data = np.array(data)
    num_1 = len([i for i in data if i == 1])
    num_0 = len([i for i in data if i == 0])
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=data)
    print(f"# FP samples: {num_0} - # TP samples: {num_1} - Class weights are {class_weights}")
    class_weights = [class_weights[1], class_weights[0]]

    print('FOR VALIDATION')
    label_arrays = [tp[0][tp[0] != -500].cpu().numpy() for tp in val_dataset.chunks]
    data = []
    for l in label_arrays:
        data.extend(l)
    data = np.array(data)
    num_1 = len([i for i in data if i == 1])
    num_0 = len([i for i in data if i == 0])
    print(f"# FP samples: {num_0} - # TP samples: {num_1} - Class weights are {class_weights}")

    # Set up samplers for distributed training
    # set_trace()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if world_size > 1 else None

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    # Instantiate the model and move to device
    if args.track_pc:
        model = Track2PCTrackMLPClassifier(input_size=11, hidden_size=128, num_layers=3)
    elif args.pc:
        model = PCTrackMLPClassifier(input_size=11, hidden_size=128, num_layers=3)
    else:
        model = TrackMLPClassifier(input_size=11, hidden_size=128, num_layers=3)

    device = torch.device(f'cuda:{rank}' if (not args.cpu) and torch.cuda.is_available() else 'cpu')
    model.to(device)

    if world_size > 1:
        # Wrap the model for distributed training
        model = DDP(model, device_ids=[rank])

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Option to resume from checkpoint
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    # Logging setup
    if rank == 0:
        if args.run_name:
            wandb.init(project="TrackMLPClassifier", name=args.run_name)
        else:
            wandb.init(project="TrackMLPClassifier")
        # wandb.config.update(args)
        wandb.watch(model)
        logging.basicConfig(filename=os.path.join(args.workdir, 'training.log'), level=logging.INFO)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Rank {rank}, Epoch {epoch + 1}"):
            optimizer.zero_grad()

            # Move batch data to device
            tp, meta, inputs = batch
            tp, inputs = tp.to(device), inputs.to(device)

            if args.pc or args.track_pc:
                tokens, _, _ = meta
                token_len = len(tokens[0])
                pc_batch = []

                for i in range(token_len):
                    dummy_status = True
                    for t in tokens:
                        if t[i] != "dummy":
                            pointcloud_path = Path('/workspace/CenterPoint/work_dirs/PCs_npy/train') / f"{t[i]}.npy"
                            pc_batch.append(np.load(str(pointcloud_path)).reshape(4, 5000))
                            dummy_status = False
                            break
                    if dummy_status:
                        pc_batch.append(np.zeros((4, 5000)))
                    
                pc_batch = torch.tensor(np.stack(pc_batch, axis=0)).float().to(device)

                # for i in range(token_len):
                #     pc_data = []
                #     for t in tokens:
                #         if t[i] == "dummy":
                #             pc_data.append(np.zeros((4, 5000)))
                #         else:
                #             pointcloud_path = Path('/workspace/CenterPoint/work_dirs/PCs_npy/train') / f"{t[i]}.npy"
                #             pc_data.append(np.load(str(pointcloud_path)).reshape(4, 5000))
                    
                #     pc_batch.append(np.concatenate(pc_data, axis=1))
                
                # pc_batch = torch.tensor(np.stack(pc_batch, axis=0)).float().to(device)
                
                inputs = (inputs, pc_batch)

            # Mask where TP is not padding (-500)
            mask = (tp != -500).float()

            # Forward pass
            outputs = model(inputs)

            # Mask labels and predictions
            masked_outputs = outputs[mask.bool()] # outputs * mask
            masked_labels = tp[mask.bool()] # tp * mask

            # Compute loss
            loss = masked_bce_loss(masked_outputs, masked_labels, class_weights)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         param.register_hook(save_gradients)


        if rank == 0:  # Only the master node logs
            if (epoch + 1) % 10 == 0:
                # Validation step
                val_loss = validate(model, val_loader, device, class_weights)

                avg_train_loss = running_loss / len(train_loader)
                logging.info(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Val Loss: {val_loss}")
                print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Val Loss: {val_loss}")
                wandb.log({"Train Loss": avg_train_loss, "Val Loss": val_loss})

                # Save checkpoint
                checkpoint_path = os.path.join(args.workdir, f"epoch_{epoch + 1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
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
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size per GPU")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--world_size', type=int, default=1, help="Number of GPUs for distributed training")
    parser.add_argument('--scenes_info', type=str, help="Path to scenes info")
    parser.add_argument('--run_name', type=str, default='', help="wandb run name")
    parser.add_argument('--track_info', type=str, help="Path to track info")
    parser.add_argument('--gt', action="store_true", help="Use GT track info")
    parser.add_argument('--pc', action="store_true", help="Use point clouds")
    parser.add_argument('--track_pc', action="store_true", help="Use point clouds")
    parser.add_argument('--cpu', action="store_true", help="Use CPU")
    parser.add_argument('--load_chunks_from', type=str, help="Use GT track info")
    parser.add_argument('--workdir', type=str, required=True, help="Directory to workdir")
    parser.add_argument('--resume-from', type=str, help="Checkpoint resume from file")
    parser.add_argument('--sample_ratio', type=float, default=0.85, help="Ratio for training/validation")

    args = parser.parse_args()

    if not os.path.isdir(args.workdir):
        os.mkdir(args.workdir)

    world_size = args.world_size
    from ipdb import launch_ipdb_on_exception, set_trace
    with launch_ipdb_on_exception():
        if world_size > 1:
            # Use torch.multiprocessing.spawn to launch distributed training
            torch.multiprocessing.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
        else:
            train(0, world_size, args)
