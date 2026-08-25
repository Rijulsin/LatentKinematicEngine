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
from torch.utils.data import DataLoader
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: True
from transformers import CLIPModel, CLIPTokenizer
import spacy

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.data_loader_sta import CharadesSTADataset
from src.model_lke_assoc import LatentKinematicGroundingMemory
import torch.nn.functional as F

def temporal_iou(pred_s_sec, pred_e_sec, gt_s_sec, gt_e_sec):
    intersection = max(0.0, min(pred_e_sec, gt_e_sec) - max(pred_s_sec, gt_s_sec))
    union = (pred_e_sec - pred_s_sec) + (gt_e_sec - gt_s_sec) - intersection
    if union <= 0: return 0.0
    return intersection / union

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    nlp = spacy.load("en_core_web_sm")
    clip_model.eval()

    TEST_TXT = config['paths']['test_txt']
    FEAT_DIR = config['paths']['feat_dir_32f']
    
    test_dataset = CharadesSTADataset(TEST_TXT, FEAT_DIR, max_seq_len=64, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = LatentKinematicGroundingMemory(d_video=768, d_text=512, d_model=512, num_layers=4).to(device)
    model.load_state_dict(torch.load(config['paths']['model_lke'], map_location=device, weights_only=True))
    model.eval()

    cont_iou_05_hits = 0
    cont_total_iou = 0.0
    
    torch.manual_seed(42) # Ensure reproducibility of the random noise
    
    with torch.no_grad():
        for batch in test_dataloader:
            video_dense = batch['video_features'].to(device)
            mask = batch['attention_mask'].to(device)
            
            gt_s_sec = batch['start_time'].numpy()
            gt_e_sec = batch['end_time'].numpy()
            vid_dur = batch['video_duration'].numpy()
            
            batch_size = len(batch['query'])
            
            # REPLACE REAL TEXT FEATURES WITH RANDOM NOISE EMBEDDINGS
            # The original CLIP text features have dimension 512
            n_feat = F.normalize(torch.randn(batch_size, 512, device=device), p=2, dim=-1)
            v_feat = F.normalize(torch.randn(batch_size, 512, device=device), p=2, dim=-1)
            
            start_logits, end_logits, _ = model(video_dense, mask, n_feat, v_feat)
            
            start_probs = F.softmax(start_logits, dim=-1)
            end_probs = F.softmax(end_logits, dim=-1)
            
            for i in range(batch_size):
                valid_len = int(mask[i].sum().item())
                best_score = -1
                pred_s, pred_e = 0, 0
                for s in range(valid_len):
                    for e in range(s, valid_len):
                        score = start_probs[i, s].item() * end_probs[i, e].item()
                        if score > best_score:
                            best_score = score
                            pred_s = s
                            pred_e = e
                            
                p_s_sec = (pred_s / max(valid_len - 1, 1)) * vid_dur[i]
                p_e_sec = (pred_e / max(valid_len - 1, 1)) * vid_dur[i]
                p_e_sec = max(p_s_sec + 0.01, p_e_sec)
                
                cont_iou = temporal_iou(p_s_sec, p_e_sec, gt_s_sec[i], gt_e_sec[i])
                cont_total_iou += cont_iou
                if cont_iou >= 0.5:
                    cont_iou_05_hits += 1
                
    cont_mIoU = (cont_total_iou / len(test_dataset)) * 100
    cont_r_05 = (cont_iou_05_hits / len(test_dataset)) * 100
    print(f"RANDOM NOISE EMBEDDINGS: Continuous mIoU: {cont_mIoU:.2f}% | Continuous Recall@1@0.5: {cont_r_05:.2f}%")

if __name__ == '__main__':
    main()
