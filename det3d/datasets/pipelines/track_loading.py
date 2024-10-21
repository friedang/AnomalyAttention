import numpy as np
from ..registry import PIPELINES

@PIPELINES.register_module
class LoadTrackFromFile(object):
    def __init__(self, dataset="NuScenesTrackDataset", **kwargs):
        self.type = dataset

    def __call__(self, res, info):
        # Assuming `res` contains the raw track data
        track_data = info['track_data']  # Replace this with the actual track loading logic
        res['track'] = track_data
        return res, info

@PIPELINES.register_module
class TrackPadding(object):
    def __init__(self, max_length=5):
        self.max_length = max_length

    def __call__(self, res, info):
        track = res['track']
        
        features = []
        for detection in track:
            translation = detection['translation']  # 3D position (x, y, z)
            size = detection['size']  # Dimensions (length, width, height)
            rotation = detection['rotation'][:3]  # Quaternion (ignore last term)
            
            # Combine translation, size, and rotation as feature vector
            feature = translation + size + rotation
            features.append(feature)

        # Pad features to max_length
        features = np.array(features)
        length = len(features)
        padded_features = np.zeros((self.max_length, features.shape[1]))
        padded_features[:length] = features

        # Create a mask for valid entries
        mask = np.zeros(self.max_length)
        mask[:length] = 1
        
        res['padded_track'] = padded_features
        res['mask'] = mask

        return res, info

@PIPELINES.register_module
class ReformatForMLP(object):
    def __init__(self):
        pass

    def __call__(self, res, info):
        # Extract padded track and mask for MLP input
        res['mlp_input'] = {
            'padded_track': res['padded_track'],
            'mask': res['mask']
        }

        # Optional: Extract the label if needed
        res['label'] = info['TP']

        return res, info
