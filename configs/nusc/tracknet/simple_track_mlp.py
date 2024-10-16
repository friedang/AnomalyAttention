import itertools
import logging

# Define the classes for binary classification (TP/FP)
tasks = [
    dict(num_class=1, class_names=["TP"]),  # Positive class (TP)
    dict(num_class=1, class_names=["FP"]),  # Negative class (FP)
]

class_names = ["TP", "FP"]

# Model settings
model = dict(
    type="MLPClassifier",  # Custom MLP network for binary classification
    num_features=TODO,  # Specify the number of input features excluding velocity and scores
    hidden_layers=[128, 64],  # Example hidden layer configuration
    activation='ReLU',  # Activation function
    output_dim=1,  # Output dimension (binary classification)
    use_batch_norm=True,  # Use batch normalization
)

# Dataset settings
dataset_type = "NuScenesTrackDataset"  # Custom dataset handling 3D tracking data
data_root = "data/tracking"  # Path to your dataset
max_track_length = 5

train_pipeline = [
    dict(type="LoadTrackingData", max_track_length=max_track_length),  # Load tracking data
    dict(type="NormalizeData"),  # Optional: Normalize translation, size, etc.
    dict(type="PadTracks", max_length=max_track_length),  # Pad the tracks to max length
    dict(type="FormatForMLP"),  # Prepare the data for MLP input
    dict(type="AssignTPFPLabel", label_key='TP'),  # Assign the TP/FP labels
    dict(type="Reformat"),  # Final format for the model
]

test_pipeline = train_pipeline.copy()  # Test pipeline follows the same format

train_cfg = dict(
    loss=dict(type="BCELoss"),  # Binary cross-entropy loss
)

test_cfg = dict()

# Optimizer and learning rate settings
optimizer = dict(
    type="adam",
    lr=0.001,
    weight_decay=0.01,
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    type="StepLR",  # Placeholder: Adjust as necessary
    step_size=5,
    gamma=0.1,
)

# Data loading settings
data = dict(
    samples_per_gpu=32,  # Batch size per GPU
    workers_per_gpu=4,  # Number of workers
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_file='/workspace/CenterPoint/work_dirs/immo/cp_5_seed_2hz/results.json',
        pipeline=train_pipeline,
    ), 
    val=dict(
        type=dataset_type,
        root_path=data_root,
        ann_file="data/tracking/val.pkl",  # Validation annotation file (TODO: Path)
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        ann_file="data/tracking/test.pkl",  # Test annotation file (TODO: Path)
        pipeline=test_pipeline,
    ),
)

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=10,
    hooks=[dict(type="TextLoggerHook")],
)

# Runtime settings
total_epochs = 20
device_ids = range(2)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
workflow = [('train', 1)] #, ('val', 1)]

evaluation = dict(interval=1, metric=['accuracy', 'precision', 'recall'])  # Track binary classification metrics
