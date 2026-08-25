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
import sys
import torch
from torch.utils.data import DataLoader
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: True
from transformers import CLIPModel, CLIPTokenizer
import spacy
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_loader_anet import ActivityNetDataset
from src.model_lke_assoc import LatentKinematicGroundingMemory
import torch.nn.functional as F

def extract_noun_verb(query, nlp):
    doc = nlp(query)
    nouns = " ".join([token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'PRON', 'ADJ']])
    verbs = " ".join([token.text for token in doc if token.pos_ in ['VERB', 'ADV']])
    if not nouns.strip(): nouns = query
    if not verbs.strip(): verbs = query
    return nouns, verbs

def temporal_iou(pred_s_sec, pred_e_sec, gt_s_sec, gt_e_sec):
    intersection = max(0.0, min(pred_e_sec, gt_e_sec) - max(pred_s_sec, gt_s_sec))
    union = (pred_e_sec - pred_s_sec) + (gt_e_sec - gt_s_sec) - intersection
    if union <= 0:
        return 0.0
    return intersection / union

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading CLIP Model (ViT-B/32)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    nlp = spacy.load("en_core_web_sm")
    clip_model.eval()

    # Paths for ActivityNet Zero-Shot
    VAL1_JSON = config['paths']['anet_val1_json']
    VAL2_JSON = config['paths']['anet_val2_json']
    FEAT_DIR = config['paths']['feat_dir_anet']
    
    print("Loading ActivityNet Validation Dataset...")
    test_dataset = ActivityNetDataset(
        annotation_paths=[VAL1_JSON, VAL2_JSON], 
        video_features_dir=FEAT_DIR, 
        max_seq_len=128, 
        is_train=False,
        pad_to_768=True
    )
    
    if len(test_dataset) == 0:
        print("Error: No test samples loaded.")
        return

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Zero-Shot Grounding Samples: {len(test_dataset)}")
    
    # Initialize the model exactly as it was trained on Charades (d_video=768)
    model = LatentKinematicGroundingMemory(d_video=768, d_text=512, d_model=512, num_layers=4).to(device)
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else config['paths'].get('model_lke_assoc_opt', './checkpoints/best_charades_sta_lke_assoc_opt_model.pt')
    
    print(f"Loading pre-trained Charades weights from {ckpt_path}...", flush=True)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}. Cannot perform zero-shot evaluation.")
        return
        
    model.eval()

    cont_iou_05_hits = 0
    cont_iou_07_hits = 0
    cont_total_iou = 0.0
    
    print("Starting LKE Zero-Shot Evaluation on ActivityNet...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating ActivityNet"):
            video_dense = batch['video_features'].to(device) # Padded to [B, T, 768]
            mask = batch['attention_mask'].to(device)
            
            gt_s_sec = batch['start_time'].numpy()
            gt_e_sec = batch['end_time'].numpy()
            vid_dur = batch['video_duration'].numpy()
            
            n_queries, v_queries = [], []
            for query in batch['query']:
                n, v = extract_noun_verb(query, nlp)
                n_queries.append(n)
                v_queries.append(v)
            
            n_inputs = tokenizer(n_queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            v_inputs = tokenizer(v_queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            
            n_feat = F.normalize(clip_model.text_projection(clip_model.text_model(**n_inputs)[1]), p=2, dim=-1)
            v_feat = F.normalize(clip_model.text_projection(clip_model.text_model(**v_inputs)[1]), p=2, dim=-1)
            
            start_logits, end_logits, _ = model(video_dense, mask, n_feat, v_feat)
            
            start_probs = F.softmax(start_logits, dim=-1)
            end_probs = F.softmax(end_logits, dim=-1)
            
            batch_size = video_dense.shape[0]
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
                            
                pred_s_sec = (pred_s / max(valid_len - 1, 1)) * vid_dur[i]
                pred_e_sec = (pred_e / max(valid_len - 1, 1)) * vid_dur[i]
                
                # Ensure bounds
                pred_e_sec = max(pred_s_sec + 0.01, pred_e_sec)
                
                cont_iou = temporal_iou(pred_s_sec, pred_e_sec, gt_s_sec[i], gt_e_sec[i])
                cont_total_iou += cont_iou
                if cont_iou >= 0.5:
                    cont_iou_05_hits += 1
                if cont_iou >= 0.7:
                    cont_iou_07_hits += 1
                
    cont_mIoU = (cont_total_iou / len(test_dataset)) * 100
    cont_r_05 = (cont_iou_05_hits / len(test_dataset)) * 100
    cont_r_07 = (cont_iou_07_hits / len(test_dataset)) * 100
    
    print("\n===========================================")
    print("LKE Zero-Shot ActivityNet Retrieval Results")
    print("===========================================")
    print(f"Total Queries Evaluated: {len(test_dataset)}")
    print(f"Continuous mIoU: {cont_mIoU:.2f}%")
    print(f"Recall@1 (IoU=0.5): {cont_r_05:.2f}%")
    print(f"Recall@1 (IoU=0.7): {cont_r_07:.2f}%")
    print("===========================================\n")

if __name__ == '__main__':
    main()
