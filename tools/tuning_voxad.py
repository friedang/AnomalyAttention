from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from optuna.visualization import *
from ray import init
init()

from tqdm import tqdm
import wandb
from sklearn.utils.class_weight import compute_class_weight

import argparse
import os
import shutil
import logging

from det3d.datasets.nuscenes.nuscenes_track_dataset import SceneDataset
from det3d.models.classifier.voxel_at import VoxelTrackMLPClassifier

from tools.train_ad import validate, masked_loss, process_point_clouds


def objective(trial):
    # Parameter space for Optuna
    lr = trial.suggest_float("lr", 1e-6, 1e-3)
    # max_norm = trial.suggest_int("max_norm", 6, 30, step=6)
    weights_type = trial.suggest_categorical("weights_type", ["small", "medium", "big"])
    # weights_gt = trial.suggest_float("weights_gt", 0.5, 1)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3)
    scheduler_type = trial.suggest_categorical("scheduler", ["step", "plateau", "none"])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])

    # print("TODO have num_heads for tuning on 8")
    num_heads = trial.suggest_int("num_heads", 8, 16, step=8)
    hidden_size = trial.suggest_int("hidden_size", 128, 512, step=64)
    num_layers = trial.suggest_int("num_layers", 3, 6)
    mlp_feature_dim = trial.suggest_int("mlp_feature_dim", 128, 512, step=64)
    # print("TODO Get max batch size by starting training on max model size")

    # Load dataset
    load_chunks_from = '/workspace/CenterPoint/work_dirs/ad_pc_mlp_05/voxel_at_org01th_chunk42_exCarPed/'
    train_dataset = SceneDataset(load_chunks_from=load_chunks_from + 'train_chunks.pt') # TODO use pathlib
    val_dataset = SceneDataset(load_chunks_from=load_chunks_from + 'val_chunks.pt') # TODO use pathlib
    train_loader = DataLoader(train_dataset, batch_size=18)
    val_loader = DataLoader(val_dataset, batch_size=18)
    print(f"Chunk size is {len(train_dataset.chunks[0][2])}")

    # Model setup
    model = VoxelTrackMLPClassifier(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_feature_dim=mlp_feature_dim
    ).cuda()

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,) # weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)
    else:
        scheduler = None

    # Compute class weights
    label_arrays = [tp[0][tp[0] != -500].cpu().numpy() for tp in train_dataset.chunks]
    data = []
    for l in label_arrays:
        data.extend(l)
    data = np.array(data)
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=data)

    print(f"Trial {trial.number}: Hyperparameters: {trial.params}")

    # Training loop
    best_value = 110
    no_improve_count = 0
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        patience = 4
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()

            # Move batch data to device
            tp, meta, inputs = batch
            tp, inputs = tp.cuda(), inputs.cuda()
            detection_ids = meta[1]
            detection_ids = [t[0] for t in detection_ids]
            
            # if any([inputs[0, i, -1] for i in range(inputs.shape[1]) if inputs[0, i, -1] in [1, 5]]):
            #     continue

            if torch.all(tp == -500):
                continue

            pc_batch = process_point_clouds(meta)                    
            pc_batch = torch.tensor(np.stack(pc_batch, axis=0)).float().cuda()
            inputs = (inputs, pc_batch)

            # Forward pass
            outputs = model(inputs)

            # Mask where TP is not padding (-500)
            mask = (tp != -500).float()
            masked_outputs = outputs[mask.bool()] # outputs * mask
            masked_labels = tp[mask.bool()] # tp * mask

            # Compute loss
            weights = class_weights
            if weights_type == 'small':
                w = [1.15, 1.3, 1.45]
            elif weights_type == 'medium':
                w = [1.3, 1.6, 1.9]
            else:
                w = [1.5, 2, 2.5]
            
            if len(meta) == 3 and not any(['gt' in i for i in detection_ids]):
                weights = torch.zeros_like(tp)
                dist_tps = meta[2].cuda()
                B, N, _ = weights.shape
                
                for b in range(B):
                    for n in range(N):
                        num_tp = len([t for t in dist_tps[b][n] if t == 1])
                        if num_tp == 0:
                            weights[b][n] = class_weights[0]
                        elif num_tp == 1:
                            weights[b][n] = class_weights[1]
                        elif num_tp == 2:
                            weights[b][n] = class_weights[1] * w[0]
                        elif num_tp == 3:
                            weights[b][n] = class_weights[1] * w[1]
                        else:
                            weights[b][n] = class_weights[1] * w[2] # * 1.25  # * num_tp

                weights = weights.cuda()
                weights = weights[mask.bool()]
            
            # elif any(['gt' in i for i in detection_ids]):
            #     weights = weights * weights_gt
                # weights = weights[mask.bool()].to(tp.device)
                    
            loss = masked_loss(masked_outputs, masked_labels, weights)
            loss.backward()
            # clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=2)  #  max_norm=max_norm, norm_type=2)
            optimizer.step()
            running_loss += loss.item()

        # Validation
        # if (epoch + 1) % 1 == 0:
        val_loss, val_acc = validate(model, val_loader, device=tp.device, tuning=True)
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Val Accuracy: {val_acc}, Val Loss: {val_loss}")
        trial.report(val_loss, epoch)

        if scheduler_type == "step":
            scheduler.step()
        elif scheduler_type == "plateau":
            scheduler.step(val_loss)

        # Prune if the trial is not promising
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if val_loss < best_value and epoch > 3:
            best_value = val_loss
            no_improve_count = 0  # Reset counter on improvement
        elif epoch > 3:
            no_improve_count += 1
        
        # Stop trial if no improvement for `patience` epochs
        if no_improve_count >= patience:
            print(f"Trial {trial.number} stopped early due to no improvement.")
            raise optuna.TrialPruned()

    return val_acc


if __name__ == "__main__":
    n_startup_trials = 20
    n_jobs=1
    n_warmup_steps=4
    study_name = 'Voxel_AT'
    plot_results = True


    sampler = TPESampler(n_startup_trials=n_startup_trials)
    # Pruners automatically stop unpromising trials at the early stage of training
    # Do not prune before several trials were performed. In each trial, do at least n_warmup_steps for a fair chance.
    pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)
    

    # *******************
    # CREATE OPTUNA STUDY
    # *******************

    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True,
                                sampler=sampler,
                                pruner=pruner,
                                direction="minimize")
    study.optimize(objective, n_trials=n_startup_trials, n_jobs=n_jobs)

    # Save best parameters
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("  Best trial: ", best_trial)
    print("  Best value: ", best_value)
    print("  Best params: ")
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))

    if plot_results:
        plot_optimization_history(study).show(renderer="browser")
        plot_param_importances(study).show(renderer="browser")
        plot_intermediate_values(study).show(renderer="browser")
        plot_contour(study).show(renderer="browser")