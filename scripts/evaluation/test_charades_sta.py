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
from src.model_sta import QueryAddressedGroundingMemory

def compute_iou(pred_start, pred_end, gt_start, gt_end):
    """
    Computes Intersection over Union (IoU) between predicted and ground truth temporal segments.
    """
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union == 0:
        return 0.0
    return intersection / union

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading CLIP Model...", flush=True)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    TEST_TXT = config['paths']['test_txt']
    FEAT_DIR = config['paths']['feat_dir_32f']
    
    print("Loading Charades-STA Test Dataset...", flush=True)
    test_dataset = CharadesSTADataset(TEST_TXT, FEAT_DIR, max_seq_len=64, is_train=False)
    
    if len(test_dataset) == 0:
        print("Error: No test samples loaded.")
        return

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Test Grounding Samples: {len(test_dataset)}", flush=True)
    
    model = QueryAddressedGroundingMemory(d_video=768, d_text=512, d_model=512, num_layers=4).to(device)
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "best_charades_sta_model.pt"
    print(f"Loading weights from {ckpt_path}...", flush=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    ious = []
    r1_iou5 = 0
    r1_iou7 = 0
    
    print("Starting Evaluation...", flush=True)
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            video_dense = batch['video_features'].to(device)
            mask = batch['attention_mask'].to(device)
            gt_s_sec = batch['start_time'].numpy()
            gt_e_sec = batch['end_time'].numpy()
            vid_dur = batch['video_duration'].numpy()
            
            # Encode query
            queries = batch['query']
            inputs = tokenizer(queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            text_outputs = clip_model.text_model(**inputs)
            text_feat = clip_model.text_projection(text_outputs[1])
            text_feat = F.normalize(text_feat, p=2, dim=-1)
            
            # Forward pass
            start_logits, end_logits, _ = model(video_dense, mask, text_feat)
            
            start_probs = torch.softmax(start_logits, dim=-1)
            end_probs = torch.softmax(end_logits, dim=-1)
            
            batch_size = video_dense.shape[0]
            for i in range(batch_size):
                max_prob = -1
                best_s = 0
                best_e = 0
                
                valid_len = int(mask[i].sum().item())
                
                for s in range(valid_len):
                    for e in range(s, valid_len):
                        prob = start_probs[i, s].item() * end_probs[i, e].item()
                        if prob > max_prob:
                            max_prob = prob
                            best_s = s
                            best_e = e
                            
                pred_start = (best_s / max(valid_len - 1, 1)) * vid_dur[i]
                pred_end = (best_e / max(valid_len - 1, 1)) * vid_dur[i]
                
                iou = compute_iou(pred_start, pred_end, gt_s_sec[i], gt_e_sec[i])
                ious.append(iou)
                
                if iou >= 0.5:
                    r1_iou5 += 1
                if iou >= 0.7:
                    r1_iou7 += 1
                
    mAP = sum(ious) / len(ious)
    r1_5 = (r1_iou5 / len(ious)) * 100
    r1_7 = (r1_iou7 / len(ious)) * 100
    
    print("\n===========================================")
    print("Charades-STA Video Moment Retrieval Results")
    print("===========================================")
    print(f"Total Queries Evaluated: {len(ious)}")
    print(f"mAP (mean Average Precision): {mAP * 100:.2f}%")
    print(f"Recall@1 (IoU=0.5): {r1_5:.2f}%")
    print(f"Recall@1 (IoU=0.7): {r1_7:.2f}%")
    print("===========================================")

if __name__ == '__main__':
    main()
