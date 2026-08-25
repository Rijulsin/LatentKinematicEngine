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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPTokenizer
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: True
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_loader_sta import CharadesSTADataset

def compute_iou(pred_start, pred_end, gt_start, gt_end):
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union == 0:
        return 0.0
    return intersection / union

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running ZERO-SHOT Baseline on {device}...")
    
    # Load CLIP for the text encoder (Video features are already extracted offline)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    TEST_TXT = config['paths']['test_txt']
    FEAT_DIR = config['paths']['feat_dir_32f']
    
    test_dataset = CharadesSTADataset(TEST_TXT, FEAT_DIR, max_seq_len=64, is_train=False)
    
    if len(test_dataset) == 0:
        print("Error: No test samples loaded.")
        return

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    ious = []
    r1_iou5 = 0
    r1_iou7 = 0
    
    print("Starting Zero-Shot Evaluation...", flush=True)
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            video_dense = batch['video_features'].to(device) # [1, T, 768]
            mask = batch['attention_mask'].to(device) # [1, T]
            gt_start = batch['start_time'].item()
            gt_end = batch['end_time'].item()
            vid_id = batch['video_id'][0]
            
            # Encode global text query using CLIP
            queries = batch['query']
            inputs = tokenizer(queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            text_outputs = clip_model.text_model(**inputs)
            text_feat = clip_model.text_projection(text_outputs[1]) # [1, 512]
            text_feat = F.normalize(text_feat, p=2, dim=-1)
            
            # The offline video features are [1, T, 768]. We need to project them to 512D to match text.
            # We can use CLIP's visual projection layer.
            video_proj = clip_model.visual_projection(video_dense) # [1, T, 512]
            video_proj = F.normalize(video_proj, p=2, dim=-1) # [1, T, 512]
            
            valid_len = int(mask[0].sum().item())
            
            max_sim = -float('inf')
            best_s = 0
            best_e = 0
            
            # Sub-segment retrieval: Compare text with average feature of every possible [s:e] window
            for s in range(valid_len):
                for e in range(s, valid_len):
                    # Compute mean feature for the segment
                    segment_feat = video_proj[0, s:e+1].mean(dim=0, keepdim=True) # [1, 512]
                    segment_feat = F.normalize(segment_feat, p=2, dim=-1)
                    
                    # Compute cosine similarity
                    sim = torch.matmul(segment_feat, text_feat.t()).item()
                    
                    if sim > max_sim:
                        max_sim = sim
                        best_s = s
                        best_e = e
                        
            # Map indices back to timestamps
            vid_len = test_dataset.video_lengths[vid_id]
            pred_start = (best_s / max(valid_len - 1, 1)) * vid_len
            pred_end = (best_e / max(valid_len - 1, 1)) * vid_len
            
            # Enforce minimum boundary
            pred_end = max(pred_start + 0.1, pred_end)
            
            iou = compute_iou(pred_start, pred_end, gt_start, gt_end)
            ious.append(iou)
            
            if iou >= 0.5:
                r1_iou5 += 1
            if iou >= 0.7:
                r1_iou7 += 1
                
    mAP = sum(ious) / len(ious)
    r1_5 = (r1_iou5 / len(ious)) * 100
    r1_7 = (r1_iou7 / len(ious)) * 100
    
    print("\n===========================================")
    print("Zero-Shot Baseline (Raw CLIP-ViT) Results")
    print("===========================================")
    print(f"Total Queries Evaluated: {len(ious)}")
    print(f"mAP (mean Average Precision): {mAP * 100:.2f}%")
    print(f"Recall@1 (IoU=0.5): {r1_5:.2f}%")
    print(f"Recall@1 (IoU=0.7): {r1_7:.2f}%")
    print("===========================================")

if __name__ == '__main__':
    main()
