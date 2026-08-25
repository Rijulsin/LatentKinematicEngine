import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np

class ActivityNetDataset(Dataset):
    def __init__(self, annotation_paths, video_features_dir, max_seq_len=128, is_train=False, pad_to_768=False):
        """
        Args:
            annotation_paths (list of str): Paths to val_1.json, val_2.json, etc.
            video_features_dir (str): Directory containing pre-extracted video features (.npz)
            max_seq_len (int): Maximum temporal sequence length for padding/truncating
            is_train (bool): Whether in training mode
            pad_to_768 (bool): Whether to zero-pad 512d features to 768d for zero-shot testing
        """
        self.video_features_dir = video_features_dir
        self.max_seq_len = max_seq_len
        self.is_train = is_train
        self.pad_to_768 = pad_to_768
        
        self.samples = []
        
        for ann_path in annotation_paths:
            self._load_annotations(ann_path)

    def _load_annotations(self, annotation_path):
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
            
        with open(annotation_path, 'r') as f:
            data = json.load(f)
            
        valid_count = 0
        for vid_key, vid_info in data.items():
            video_id = vid_key[2:] if vid_key.startswith('v_') else vid_key
            duration = vid_info.get('duration', 0.0)
            timestamps = vid_info.get('timestamps', [])
            sentences = vid_info.get('sentences', [])
            
            if not timestamps or not sentences or len(timestamps) != len(sentences):
                continue
                
            # Check if feature exists
            feature_path = os.path.join(self.video_features_dir, f"{video_id}.npz")
            if not os.path.exists(feature_path):
                feature_path_v = os.path.join(self.video_features_dir, f"{vid_key}.npz")
                if os.path.exists(feature_path_v):
                    feature_path = feature_path_v
                else:
                    continue
                    
            for (start_time, end_time), query in zip(timestamps, sentences):
                self.samples.append({
                    'video_id': video_id,
                    'query': query,
                    'start_time': start_time,
                    'end_time': end_time,
                    'video_duration': duration,
                    'feature_path': feature_path
                })
                valid_count += 1

        print(f"Loaded {valid_count} valid samples from {annotation_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video features
        data = np.load(sample['feature_path'])
        features = torch.from_numpy(data['features']).float() # Shape: (T, D)
        
        seq_len = features.shape[0]
        feature_dim = features.shape[-1]
        
        # Pad feature dim from 512 to 768 to match the Charades checkpoint expectation
        if self.pad_to_768 and feature_dim < 768:
            feat_pad = torch.zeros((seq_len, 768 - feature_dim), dtype=features.dtype)
            features = torch.cat([features, feat_pad], dim=-1)
            feature_dim = 768
            
        vid_len = sample['video_duration']
        
        # Pad or truncate features
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            pad = torch.zeros((pad_len, feature_dim), dtype=features.dtype)
            features_padded = torch.cat([features, pad], dim=0)
            mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        else:
            # Truncate
            features_padded = features[:self.max_seq_len]
            mask = torch.ones(self.max_seq_len)
            
        return {
            'video_id': sample['video_id'],
            'query': sample['query'],
            'start_time': sample['start_time'],
            'end_time': sample['end_time'],
            'video_duration': vid_len,
            'video_features': features_padded,
            'attention_mask': mask
        }
