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
from flask import Flask, request, jsonify, render_template

import sys
# Add parent directory to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.model_lke_assoc import LatentKinematicGroundingMemory

app = Flask(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading CLIP Model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
nlp = spacy.load("en_core_web_sm")
clip_model.eval()

print("Loading LKE 1D Model...")
lke_model = LatentKinematicGroundingMemory(d_video=768, d_text=512, d_model=512, num_layers=4).to(device)
lke_model.load_state_dict(torch.load(config['paths']['model_lke_assoc'], map_location=device, weights_only=True))
lke_model.eval()

print("Loading Video Durations...")
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

def extract_noun_verb(query):
    doc = nlp(query)
    nouns = " ".join([token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'PRON', 'ADJ']])
    verbs = " ".join([token.text for token in doc if token.pos_ in ['VERB', 'ADV']])
    if not nouns.strip(): nouns = query
    if not verbs.strip(): verbs = query
    return nouns, verbs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/retrieve', methods=['POST'])
def retrieve():
    data = request.json
    query = data.get('query', '')
    video_id = data.get('video_id', '')
    
    if not query or not video_id:
        return jsonify({'error': 'Please provide both query and video ID.'}), 400
        
    if video_id not in video_lengths:
        return jsonify({'error': f'Video ID {video_id} not found in dataset.'}), 404
        
    feat_path = os.path.join(FEAT_DIR, f"{video_id}.pt")
    if not os.path.exists(feat_path):
        return jsonify({'error': f'Features for video {video_id} not found.'}), 404
        
    try:
        # 1. Load Video Features
        feat_data = torch.load(feat_path, map_location='cpu', weights_only=False)
        features = feat_data['dense'].float()
        seq_len = features.shape[0]
        
        # Pad to max_seq_len=64 (like training)
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
        
        # 2. Extract Noun/Verb & CLIP Features
        n_query, v_query = extract_noun_verb(query)
        n_inputs = tokenizer([n_query], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        v_inputs = tokenizer([v_query], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        
        with torch.no_grad():
            n_outputs = clip_model.text_model(**n_inputs)
            n_feat = F.normalize(clip_model.text_projection(n_outputs[1]), p=2, dim=-1)
            
            v_outputs = clip_model.text_model(**v_inputs)
            v_feat = F.normalize(clip_model.text_projection(v_outputs[1]), p=2, dim=-1)
            
            # 3. Model Inference
            start_logits, end_logits, _ = lke_model(video_dense, attn_mask, n_feat, v_feat)
            
            start_probs = F.softmax(start_logits, dim=-1)[0]
            end_probs = F.softmax(end_logits, dim=-1)[0]
            
            valid_len = int(attn_mask[0].sum().item())
            
            # Find best joint probability (start <= end)
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
        
        # 4. Map back to seconds
        pred_s_sec = (pred_s / max(valid_len - 1, 1)) * vid_dur
        pred_e_sec = (pred_e / max(valid_len - 1, 1)) * vid_dur
        pred_e_sec = max(pred_s_sec + 0.01, pred_e_sec)
        
        # Project confidence from 1D Gaussian peak (0.04) to 0-100 scale
        projected_confidence = min((best_score / 0.04) * 100, 100.0)
        
        return jsonify({
            'success': True,
            'start_time': round(pred_s_sec, 2),
            'end_time': round(pred_e_sec, 2),
            'video_duration': round(vid_dur, 2),
            'confidence': round(projected_confidence, 2),
            'noun_extracted': n_query,
            'verb_extracted': v_query
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting LKE Web Server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
