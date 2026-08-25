import yaml
import os

_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config.yaml')
if not os.path.exists(_config_path):
    _config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')
if not os.path.exists(_config_path):
    _config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
if not os.path.exists(_config_path):
    _config_path = './config.yaml'

with open(_config_path, 'r') as _f:
    config = yaml.safe_load(_f)

import os
import torch
from torch.utils.data import Dataset
import numpy as np

class CharadesSpatialDataset(Dataset):
    def __init__(self, annotation_path, video_features_dir, max_seq_len=8, is_train=True):
        """
        Args:
            annotation_path (str): Path to charades_sta_train.txt or charades_sta_test.txt
            video_features_dir (str): Directory containing pre-extracted video features (.npy or .pt)
            max_seq_len (int): Maximum temporal sequence length for padding/truncating
            is_train (bool): Whether in training mode
        """
        self.video_features_dir = video_features_dir
        self.max_seq_len = max_seq_len
        self.is_train = is_train
        
        # Load video lengths from official Charades CSVs
        import csv
        self.video_lengths = {}
        for csv_path in [
            config['paths']['charades_v1_train'],
            config['paths']['charades_v1_test']
        ]:
            if os.path.exists(csv_path):
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.video_lengths[row['id']] = float(row['length'])
        
        self.samples = []
        self._load_annotations(annotation_path)

    def _load_annotations(self, annotation_path):
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
            
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: video_id start_time end_time##text_query
                parts = line.split('##')
                if len(parts) != 2:
                    continue
                
                meta, query = parts[0], parts[1]
                meta_parts = meta.split()
                if len(meta_parts) != 3:
                    continue
                    
                video_id = meta_parts[0]
                start_time = float(meta_parts[1])
                end_time = float(meta_parts[2])
                
                # Check if feature exists and we have length metadata
                feature_path = os.path.join(self.video_features_dir, f"{video_id}.pt")
                if os.path.exists(feature_path) and video_id in self.video_lengths:
                    self.samples.append({
                        'video_id': video_id,
                        'query': query,
                        'start_time': start_time,
                        'end_time': end_time,
                        'feature_path': feature_path
                    })

        print(f"Loaded {len(self.samples)} valid samples from {annotation_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video features
        data = torch.load(sample['feature_path'], map_location='cpu', weights_only=False)
        features = data['patches'].float() # [T, P, D] e.g. [8, 49, 768]
        
        seq_len = features.shape[0]
        num_patches = features.shape[1]
        feature_dim = features.shape[2]
        
        
        vid_len = self.video_lengths[sample['video_id']]
        
        # Map timestamps to frame indices correctly based on actual video duration
        # Assuming features are uniformly sampled across the video length
        start_idx = min(max(int((sample['start_time'] / max(vid_len, 0.1)) * (seq_len - 1)), 0), seq_len - 1)
        end_idx = min(max(int((sample['end_time'] / max(vid_len, 0.1)) * (seq_len - 1)), 0), seq_len - 1)
        
        # Pad or truncate features
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            pad = torch.zeros((pad_len, num_patches, feature_dim), dtype=features.dtype)
            features_padded = torch.cat([features, pad], dim=0)
            mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        else:
            # Truncate
            features_padded = features[:self.max_seq_len]
            mask = torch.ones(self.max_seq_len)
            # Adjust indices if they exceed max_seq_len
            start_idx = min(start_idx, self.max_seq_len - 1)
            end_idx = min(end_idx, self.max_seq_len - 1)
            
        return {
            'video_id': sample['video_id'],
            'query': sample['query'],
            'video_features': features_padded,
            'attention_mask': mask,
            'start_idx': torch.tensor(start_idx, dtype=torch.long),
            'end_idx': torch.tensor(end_idx, dtype=torch.long),
            'start_time': torch.tensor(sample['start_time'], dtype=torch.float),
            'end_time': torch.tensor(sample['end_time'], dtype=torch.float),
            'video_duration': torch.tensor(vid_len, dtype=torch.float)
        }
