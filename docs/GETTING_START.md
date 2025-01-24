# Getting Started with AnomalyAttention


## Fully-Supervised Test Case


### 1. Generate Object Detections on nuScenes (Original CenterPoint)

Refer to [nuScenes](NUSC.md) for detailed instructions. Start by training CenterPoint on the nuScenes training dataset and then run inference on this dataset.

The resulting detections will be saved as a JSON file in the nuScenes detection format, named something like `infos_train_10sweeps_withvelo_filter_True.json`. You will need this file as input for the next step.

---

### 2. Generate Object Tracks (ImmortalTracker)

Refer to [ImmortalTracker](https://github.com/esdolo/ImmortalTracker) for detailed instructions.


Modify the `score_threshold` to `0.1` and `min_hits_to_birth` to `0` in the [immortal.yaml](https://github.com/esdolo/ImmortalTracker/blob/main/configs/nu_configs/cp_plus.yaml) configuration file, or use our forked version of ImmortalTracker. If using the forked version, you can also use our `run_all.sh` script after updating paths and settings.

---

### 3. Preprocess Tracks

ImmortalTracker's output will be a file named `results.json`, which is in the nuScenes detection format but includes a `tracking-id` field for each detection. Using these IDs, you can create a dataset of tracks.


After updating paths and settings in the preprocessing scripts, run the following command:


```bash
bash tools/track_to_traindata.sh
```

---

### 4. Train AnomalyAttention

Using the generated track structure files track_info_padded_tracks.json and scene2trackname.json, train the model by running:

```bash
python3 tools/train_ad.py --attention
    --scenes_info PATH_FOR_scene2trackname.json
    --track_info PATH_FOR_track_info_padded_tracks.json
    --workdir WORKDIR_PATH
    --batch_size 24
    --epochs 200
```

---

### 5. Evaluate on nuScene's detection benchmark

Repeat steps 1 to 3 for the nuScenes validation dataset to generate the required data structure for CenterPoint detections on it.

After updating paths, run the following command:
```bash
bash tools/val_ad.sh
```

---

### 6. Inference

Repeat steps 1 to 3 for the inference dataset, then run:

```bash
python3 tools/test_ad.py --attention
    --model_checkpoint PATH_TO_MODEL_WEIGHTS_FILE_epoch_X.pt
    --workdir WORKDIR_PATH
    --scenes_info PATH_FOR_scene2trackname.json
    --track_info PATH_FOR_track_info_padded_tracks.json
    --confidence_threshold YOUR_CONFIDENCE_THRESHOLD
```

---
---
---


## Semi-Supervised Test Case

### A. Train on Seed Dataset
For semi-supervised training, you can either re-use our sampled scene indices by uncommenting the path for `load_indices` or specify a seed size using `sample_ratio` (e.g., 0.05 for 5% seed size) in the CenterPoint configuration file `configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py`. This step generates a seed and pseudo-indices file (non-seed-sampled indices), both of which can later be used for `load_indices`.

Next, follow steps 1 to 4 from the **Fully-Supervised Test Case** to train all modules on this seed dataset. Once this is completed, AnomalyAttention will be trained and ready for inference on CenterPoint's detections.

### B. Generate Pseudo Labels by CenterPoint
In the configuration file `configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py`, set the path for `load_indices` to your `pseudo_indices.pth` file. Then, run inference with CenterPoint, which was trained on your seed dataset from step A:
```bash
python ./tools/dist_test.py CONFIG_PATH --testset \
    --work_dir WORKDIR_PATH \
    --checkpoint PATH_TO_MODEL_WEIGHTS_FILE_epoch_X
```

This step generates a `prediction.pkl` file containing all detections, which will be used in the next step.

---

### C. Create Seed+Pseudo Training Dataset

Replace the ground truth annotations in nuScenes' training information file (`./data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl`) with the pseudo-detections from the `prediction.pkl` file generated in step B. Run the following command:
```bash
python3 ./tools/merge_seed_pseudo_gt.py --nusc_data_info_path PATH_TO_NUSC_TRAIN_INF0 
    --pseudo_path  PATH_TO_PREDICTION_PKL_FILE
    --workdir PATH_TO_WORKDIR
```

---

### D. Retrain CenterPoint

To retrain CenterPoint from scratch on the seed+pseudo training dataset, set the merged pickle file path (from step C) for `train_anno` in the config file and start training:
```bash
python3 tools/train.py CONFIG_PATH
```


For our 5% seed dataset test case, you can the following:
```bash
python3 tools/train.py ./configs/nusc/voxelnet/pseudo95_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py
```

---

### E. Evaluation & Inference

For evaluation and inference, refer to steps 5 and 6 in the **Fully-Supervised Test Case**.