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
import random

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.data_loader_sta import CharadesSTADataset
from src.model_lke_assoc import LatentKinematicGroundingMemory
import torch.nn.functional as F
from tqdm import tqdm

def temporal_iou(pred_s_sec, pred_e_sec, gt_s_sec, gt_e_sec):
    intersection = max(0.0, min(pred_e_sec, gt_e_sec) - max(pred_s_sec, gt_s_sec))
    union = (pred_e_sec - pred_s_sec) + (gt_e_sec - gt_s_sec) - intersection
    if union <= 0: return 0.0
    return intersection / union

def extract_noun_verb(query, nlp):
    doc = nlp(query)
    nouns = " ".join([token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'PRON', 'ADJ']])
    verbs = " ".join([token.text for token in doc if token.pos_ in ['VERB', 'ADV']])
    if not nouns.strip(): nouns = query
    if not verbs.strip(): verbs = query
    return nouns, verbs

def evaluate_batch(model, clip_model, tokenizer, video_dense, mask, noun_queries, verb_queries, gt_s_sec, gt_e_sec, vid_dur, device):
    n_inputs = tokenizer(noun_queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    v_inputs = tokenizer(verb_queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    
    with torch.no_grad():
        n_feat = F.normalize(clip_model.text_projection(clip_model.text_model(**n_inputs)[1]), p=2, dim=-1)
        v_feat = F.normalize(clip_model.text_projection(clip_model.text_model(**v_inputs)[1]), p=2, dim=-1)
        
        start_logits, end_logits, _ = model(video_dense, mask, n_feat, v_feat)
        
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)
        
        batch_size = video_dense.shape[0]
        hits_05 = 0
        
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
            if cont_iou >= 0.5:
                hits_05 += 1
                
        return hits_05

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading CLIP & SpaCy...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    nlp = spacy.load("en_core_web_sm")
    clip_model.eval()

    TEST_TXT = config['paths']['test_txt']
    FEAT_DIR = config['paths']['feat_dir_32f']
    
    test_dataset = CharadesSTADataset(TEST_TXT, FEAT_DIR, max_seq_len=64, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = LatentKinematicGroundingMemory(d_video=768, d_text=512, d_model=512, num_layers=4).to(device)
    model.load_state_dict(torch.load(config['paths']['model_lke_assoc'], map_location=device, weights_only=True))
    model.eval()

    all_queries = [sample['query'] for sample in test_dataset.samples]
    
    all_nouns = []
    all_verbs = []
    for q in all_queries:
        n, v = extract_noun_verb(q, nlp)
        all_nouns.append(n)
        all_verbs.append(v)
        
    random.seed(42)
    shuffled_nouns = all_nouns.copy()
    shuffled_verbs = all_verbs.copy()
    random.shuffle(shuffled_nouns)
    random.shuffle(shuffled_verbs)
    
    hits_normal = 0
    hits_scrambled_noun = 0
    hits_scrambled_verb = 0
    
    query_idx = 0
    
    print("Starting Gate Ablation Study...", flush=True)
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Ablating Gates"):
            video_dense = batch['video_features'].to(device)
            mask = batch['attention_mask'].to(device)
            
            gt_s_sec = batch['start_time'].numpy()
            gt_e_sec = batch['end_time'].numpy()
            vid_dur = batch['video_duration'].numpy()
            
            batch_size = len(batch['query'])
            
            # Condition 1: Normal
            n_normal = all_nouns[query_idx : query_idx + batch_size]
            v_normal = all_verbs[query_idx : query_idx + batch_size]
            hits_normal += evaluate_batch(model, clip_model, tokenizer, video_dense, mask, n_normal, v_normal, gt_s_sec, gt_e_sec, vid_dur, device)
            
            # Condition 2: Scrambled Noun, Correct Verb
            n_scrambled = shuffled_nouns[query_idx : query_idx + batch_size]
            hits_scrambled_noun += evaluate_batch(model, clip_model, tokenizer, video_dense, mask, n_scrambled, v_normal, gt_s_sec, gt_e_sec, vid_dur, device)
            
            # Condition 3: Correct Noun, Scrambled Verb
            v_scrambled = shuffled_verbs[query_idx : query_idx + batch_size]
            hits_scrambled_verb += evaluate_batch(model, clip_model, tokenizer, video_dense, mask, n_normal, v_scrambled, gt_s_sec, gt_e_sec, vid_dur, device)
            
            query_idx += batch_size
            
    n_samples = len(test_dataset)
    print("\n===========================================")
    print("Associative Gate Ablation Study Results")
    print("===========================================")
    print(f"1. Normal (Correct Noun + Correct Verb) R@1: {(hits_normal / n_samples) * 100:.2f}%")
    print(f"2. Scrambled Noun, Correct Verb R@1:         {(hits_scrambled_noun / n_samples) * 100:.2f}%")
    print(f"3. Correct Noun, Scrambled Verb R@1:         {(hits_scrambled_verb / n_samples) * 100:.2f}%")
    print("===========================================")

if __name__ == '__main__':
    main()
