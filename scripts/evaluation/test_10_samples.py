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
import csv
import torch
import torch.nn.functional as F
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: True
from transformers import CLIPModel, CLIPTokenizer
import spacy

import sys
# Add parent directory to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.model_lke_assoc import LatentKinematicGroundingMemory

def calculate_iou(pred_s, pred_e, gt_s, gt_e):
    intersection = max(0, min(pred_e, gt_e) - max(pred_s, gt_s))
    union = max(pred_e, gt_e) - min(pred_s, gt_s)
    if union == 0: return 0.0
    return intersection / union

def extract_noun_verb(query, nlp):
    doc = nlp(query)
    nouns = " ".join([token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'PRON', 'ADJ']])
    verbs = " ".join([token.text for token in doc if token.pos_ in ['VERB', 'ADV']])
    if not nouns.strip(): nouns = query
    if not verbs.strip(): verbs = query
    return nouns, verbs

def get_samples():
    samples = []
    # 10 Test Samples
    with open(config['paths']['test_txt'], "r") as f:
        lines = f.readlines()[10:20]
        for line in lines:
            parts = line.strip().split("##")
            meta = parts[0].split()
            samples.append({'id': meta[0], 'start': float(meta[1]), 'end': float(meta[2]), 'query': parts[1], 'split': 'TEST'})
            
    # 10 Train Samples
    with open(config['paths']['train_txt'], "r") as f:
        lines = f.readlines()[10:20]
        for line in lines:
            parts = line.strip().split("##")
            meta = parts[0].split()
            samples.append({'id': meta[0], 'start': float(meta[1]), 'end': float(meta[2]), 'query': parts[1], 'split': 'TRAIN'})
            
    return samples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n[1/4] Loading CLIP Model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    
    print("[2/4] Loading spaCy NLP...")
    nlp = spacy.load("en_core_web_sm")
    
    print("[3/4] Loading Trained LKE 1D Model...")
    lke_model = LatentKinematicGroundingMemory(d_video=768, d_text=512, d_model=512, num_layers=4).to(device)
    model_path = config['paths']['model_lke_assoc']
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    lke_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    lke_model.eval()
    
    print("[4/4] Loading Charades Video Metadata...")
    video_lengths = {}
    for csv_path in [
        config['paths']['charades_v1_train'],
        config['paths']['charades_v1_test']
    ]:
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    video_lengths[row['id']] = float(row['length'])

    FEAT_DIR = config['paths']['feat_dir_32f']
    
    samples = get_samples()
    
    print("\n" + "="*80)
    print(" 20-SAMPLE GROUNDING EXPERIMENT (1D ASSOCIATIVE MEMORY)")
    print("="*80)
    
    for idx, sample in enumerate(samples):
        video_id = sample['id']
        query = sample['query']
        gt_s = sample['start']
        gt_e = sample['end']
        split = sample['split']
        
        print(f"\nSample {idx+1}/20 [{split}] | Video: {video_id} | Query: '{query}'")
        print(f"Ground Truth : {gt_s:.2f}s - {gt_e:.2f}s")
        
        feat_path = os.path.join(FEAT_DIR, f"{video_id}.pt")
        if not os.path.exists(feat_path):
            print(f"  -> Error: Feature file '{feat_path}' not found. Skipping.")
            continue
        
        try:
            # Load Features
            feat_data = torch.load(feat_path, map_location='cpu', weights_only=False)
            features = feat_data['dense'].float()
            seq_len = features.shape[0]
            
            # Pad
            max_seq_len = 64
            if seq_len < max_seq_len:
                pad_len = max_seq_len - seq_len
                pad = torch.zeros((pad_len, features.shape[-1]), dtype=features.dtype)
                features_padded = torch.cat([features, pad], dim=0)
                mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
            else:
                features_padded = features[:max_seq_len]
                mask = torch.ones(max_seq_len)
                
            video_dense = features_padded.unsqueeze(0).to(device)
            attn_mask = mask.unsqueeze(0).to(device)
            
            # Text Processing
            n_query, v_query = extract_noun_verb(query, nlp)
            n_inputs = tokenizer([n_query], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            v_inputs = tokenizer([v_query], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            
            with torch.no_grad():
                n_outputs = clip_model.text_model(**n_inputs)
                n_feat = F.normalize(clip_model.text_projection(n_outputs[1]), p=2, dim=-1)
                
                v_outputs = clip_model.text_model(**v_inputs)
                v_feat = F.normalize(clip_model.text_projection(v_outputs[1]), p=2, dim=-1)
                
                start_logits, end_logits, _ = lke_model(video_dense, attn_mask, n_feat, v_feat)
                
                start_probs = F.softmax(start_logits, dim=-1)[0]
                end_probs = F.softmax(end_logits, dim=-1)[0]
                
                valid_len = int(attn_mask[0].sum().item())
                
                best_score = -1
                pred_s, pred_e = 0, 0
                for s in range(valid_len):
                    for e in range(s, valid_len):
                        score = start_probs[s].item() * end_probs[e].item()
                        if score > best_score:
                            best_score = score
                            pred_s = s
                            pred_e = e
                            
            vid_dur = video_lengths[video_id]
            pred_s_sec = (pred_s / max(valid_len - 1, 1)) * vid_dur
            pred_e_sec = (pred_e / max(valid_len - 1, 1)) * vid_dur
            pred_e_sec = max(pred_s_sec + 0.01, pred_e_sec)
            
            iou = calculate_iou(pred_s_sec, pred_e_sec, gt_s, gt_e)
            status = "SUCCESS (IoU > 0.3)" if iou >= 0.3 else "FAILURE (IoU < 0.3)"
            
            print(f"Prediction   : {pred_s_sec:.2f}s - {pred_e_sec:.2f}s")
            print(f"IoU          : {iou:.3f} [{status}]")
            
        except Exception as e:
            print(f"  -> Error during inference: {e}")

if __name__ == "__main__":
    main()
