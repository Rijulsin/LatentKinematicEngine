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
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_loader_spatial import CharadesSpatialDataset
from src.model_spatial_lke import SpatialLatentKinematicGroundingMemory
import torch.nn.functional as F

def extract_noun_verb(query, nlp):
    doc = nlp(query)
    nouns = " ".join([token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'PRON', 'ADJ']])
    verbs = " ".join([token.text for token in doc if token.pos_ in ['VERB', 'ADV']])
    if not nouns.strip(): nouns = query
    if not verbs.strip(): verbs = query
    return nouns, verbs

def compute_iou(pred_s, pred_e, gt_s, gt_e):
    intersection = max(0, min(pred_e, gt_e) - max(pred_s, gt_s) + 1)
    union = (pred_e - pred_s + 1) + (gt_e - gt_s + 1) - intersection
    if union <= 0:
        return 0.0
    return intersection / union

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading CLIP Model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    nlp = spacy.load("en_core_web_sm")
    clip_model.eval()

    TEST_TXT = config['paths']['test_txt']
    FEAT_DIR = config['paths']['feat_dir_patches_32f']
    
    print("Loading Charades-STA Test Dataset...")
    test_dataset = CharadesSpatialDataset(TEST_TXT, FEAT_DIR, max_seq_len=32, is_train=False)
    
    if len(test_dataset) == 0:
        print("Error: No test samples loaded.")
        return

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Test Grounding Samples: {len(test_dataset)}")
    
    model = SpatialLatentKinematicGroundingMemory(d_video=768, d_text=512, d_model=512, num_layers=4).to(device)
    model.load_state_dict(torch.load("best_charades_sta_spatial_lke_model.pt", map_location=device, weights_only=True))
    model.eval()

    iou_05_hits = 0
    iou_07_hits = 0
    total_iou = 0.0
    
    print("Starting LKE Evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            video_dense = batch['video_features'].to(device)
            mask = batch['attention_mask'].to(device)
            gt_s = batch['start_idx'].item()
            gt_e = batch['end_idx'].item()
            
            n_query, v_query = extract_noun_verb(batch['query'][0], nlp)
            
            n_inputs = tokenizer([n_query], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            v_inputs = tokenizer([v_query], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            
            n_outputs = clip_model.text_model(**n_inputs)
            n_feat = F.normalize(clip_model.text_projection(n_outputs[1]), p=2, dim=-1)
            
            v_outputs = clip_model.text_model(**v_inputs)
            v_feat = F.normalize(clip_model.text_projection(v_outputs[1]), p=2, dim=-1)
            
            start_logits, end_logits, _ = model(video_dense, mask, n_feat, v_feat)
            
            start_probs = F.softmax(start_logits, dim=-1)[0]
            end_probs = F.softmax(end_logits, dim=-1)[0]
            
            valid_len = int(mask[0].sum().item())
            
            best_score = -1
            pred_s, pred_e = 0, 0
            for s in range(valid_len):
                for e in range(s, valid_len):
                    score = start_probs[s].item() * end_probs[e].item()
                    if score > best_score:
                        best_score = score
                        pred_s = s
                        pred_e = e
                        
            iou = compute_iou(pred_s, pred_e, gt_s, gt_e)
            total_iou += iou
            if iou >= 0.5:
                iou_05_hits += 1
            if iou >= 0.7:
                iou_07_hits += 1
                
    mIoU = (total_iou / len(test_dataset)) * 100
    r_05 = (iou_05_hits / len(test_dataset)) * 100
    r_07 = (iou_07_hits / len(test_dataset)) * 100
    
    print("\n===========================================")
    print("LKE Video Moment Retrieval Results")
    print("===========================================")
    print(f"Total Queries Evaluated: {len(test_dataset)}")
    print(f"mIoU (mean Intersection over Union): {mIoU:.2f}%")
    print(f"Recall@1 (IoU=0.5): {r_05:.2f}%")
    print(f"Recall@1 (IoU=0.7): {r_07:.2f}%")
    print("===========================================\n")

if __name__ == '__main__':
    main()
