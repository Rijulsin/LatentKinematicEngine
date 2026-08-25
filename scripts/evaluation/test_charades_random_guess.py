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
import random
import numpy as np

def compute_iou(pred_start, pred_end, gt_start, gt_end):
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union == 0:
        return 0.0
    return intersection / union

def evaluate_random_guessing(test_txt_path, num_trials=5):
    # Read all lines
    with open(test_txt_path, 'r') as f:
        lines = f.readlines()
        
    print(f"Loaded {len(lines)} test samples.")
    
    # We don't have the exact video length for every video loaded in this simple script,
    # but we know the average Charades video is roughly 30 seconds.
    # To be extremely accurate, we can extract the video lengths if needed, 
    # but since Charades ground truths are given in seconds, we can just use 
    # a conservative uniform random guess between 0 and 30 seconds.
    # Better yet, let's use the actual dataset loader to get exact video lengths!
    pass

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.data_loader_sta import CharadesSTADataset

def main():
    TEST_TXT = config['paths']['test_txt']
    FEAT_DIR = config['paths']['feat_dir_32f']
    
    dataset = CharadesSTADataset(TEST_TXT, FEAT_DIR, max_seq_len=64, is_train=False)
    
    num_samples = len(dataset)
    trials_iou5 = []
    
    for trial in range(5):
        hits_05 = 0
        for i in range(num_samples):
            sample = dataset[i]
            gt_start = sample['start_time'].item()
            gt_end = sample['end_time'].item()
            vid_len = sample['video_duration'].item()
            
            # Pure random guessing
            pred_start = random.uniform(0, vid_len)
            pred_end = random.uniform(pred_start, vid_len)
            
            # Enforce a minimum length just to be fair
            pred_end = max(pred_start + 0.5, pred_end)
            
            iou = compute_iou(pred_start, pred_end, gt_start, gt_end)
            if iou >= 0.5:
                hits_05 += 1
                
        trials_iou5.append((hits_05 / num_samples) * 100)
        
    avg_r1_5 = sum(trials_iou5) / len(trials_iou5)
    print("===========================================")
    print("PURE RANDOM GUESSING RESULTS (Charades-STA)")
    print("===========================================")
    for i, t in enumerate(trials_iou5):
        print(f"Trial {i+1}: R@1 (IoU=0.5) = {t:.2f}%")
    print(f"\nAverage R@1 (IoU=0.5): {avg_r1_5:.2f}%")
    print("===========================================")

if __name__ == '__main__':
    main()
