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
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPTokenizer
import spacy
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_loader_anet import ActivityNetDataset
from src.model_lke_assoc import LatentKinematicGroundingMemory, BoundaryLoss
import torch.nn.functional as F

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: True

CHECKPOINT_DIR = config['paths']['save_dir']
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'best_anet_lke_assoc_model.pt')

def extract_noun_verb(query, nlp):
    doc = nlp(query)
    nouns = " ".join([token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'PRON', 'ADJ']])
    verbs = " ".join([token.text for token in doc if token.pos_ in ['VERB', 'ADV']])
    if not nouns.strip(): nouns = query
    if not verbs.strip(): verbs = query
    return nouns, verbs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading CLIP Model (ViT-B/32) & spaCy...")
    # We use ViT-B/32 text encoder since ActivityNet features are ViT-B/32 visual embeddings
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    nlp = spacy.load("en_core_web_sm")
    clip_model.eval()

    # Paths for ActivityNet
    TRAIN_JSON = config['paths']['anet_train_json']
    VAL1_JSON = config['paths']['anet_val1_json']
    VAL2_JSON = config['paths']['anet_val2_json']
    FEAT_DIR = config['paths']['feat_dir_anet']
    
    print("Loading ActivityNet Datasets...")
    train_dataset = ActivityNetDataset([TRAIN_JSON], FEAT_DIR, max_seq_len=128, is_train=True)
    val_dataset = ActivityNetDataset([VAL1_JSON, VAL2_JSON], FEAT_DIR, max_seq_len=128, is_train=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")

    # NOTE: d_video is set to 512 because ActivityNet features from UniVTG are ViT-B/32.
    # d_model and num_layers can be updated based on your hyperparameter sweep results!
    model = LatentKinematicGroundingMemory(d_video=512, d_text=512, d_model=512, num_layers=4).to(device)
    
    criterion = BoundaryLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    best_val_loss = float('inf')
    num_epochs = 20
    
    print("\n--- STARTING ACTIVITYNET PRE-TRAINING ---")
    
    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            # Features are [B, T, 512] for ActivityNet natively
            video_dense = batch['video_features'].to(device)
            mask = batch['attention_mask'].to(device)
            
            # Map continuous timestamps to index bounds for training
            vid_dur = batch['video_duration'].to(device)
            start_sec = batch['start_time'].to(device)
            end_sec = batch['end_time'].to(device)
            
            seq_lens = mask.sum(dim=1)
            start_idx = torch.clamp(((start_sec / vid_dur) * seq_lens).long(), min=0, max=127)
            end_idx = torch.clamp(((end_sec / vid_dur) * seq_lens).long(), min=0, max=127)

            noun_queries, verb_queries = [], []
            for query in batch['query']:
                n, v = extract_noun_verb(query, nlp)
                noun_queries.append(n)
                verb_queries.append(v)

            n_inputs = tokenizer(noun_queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            v_inputs = tokenizer(verb_queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)

            with torch.no_grad():
                n_feat = F.normalize(clip_model.text_projection(clip_model.text_model(**n_inputs)[1]), p=2, dim=-1)
                v_feat = F.normalize(clip_model.text_projection(clip_model.text_model(**v_inputs)[1]), p=2, dim=-1)

            optimizer.zero_grad()
            # The features are naturally 512d, so they plug directly into the 512d model
            start_logits, end_logits, _ = model(video_dense, mask, n_feat, v_feat)
            
            loss = criterion(start_logits, end_logits, start_idx, end_idx, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                video_dense = batch['video_features'].to(device)
                mask = batch['attention_mask'].to(device)
                
                vid_dur = batch['video_duration'].to(device)
                start_sec = batch['start_time'].to(device)
                end_sec = batch['end_time'].to(device)
                
                seq_lens = mask.sum(dim=1)
                start_idx = torch.clamp(((start_sec / vid_dur) * seq_lens).long(), min=0, max=127)
                end_idx = torch.clamp(((end_sec / vid_dur) * seq_lens).long(), min=0, max=127)

                noun_queries, verb_queries = [], []
                for query in batch['query']:
                    n, v = extract_noun_verb(query, nlp)
                    noun_queries.append(n)
                    verb_queries.append(v)

                n_inputs = tokenizer(noun_queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
                v_inputs = tokenizer(verb_queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)

                n_feat = F.normalize(clip_model.text_projection(clip_model.text_model(**n_inputs)[1]), p=2, dim=-1)
                v_feat = F.normalize(clip_model.text_projection(clip_model.text_model(**v_inputs)[1]), p=2, dim=-1)

                start_logits, end_logits, _ = model(video_dense, mask, n_feat, v_feat)
                loss = criterion(start_logits, end_logits, start_idx, end_idx, mask)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch:02d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"--> Best Validation Loss Improved! Saving checkpoint to {CHECKPOINT_PATH}")
            torch.save(model.state_dict(), CHECKPOINT_PATH)

if __name__ == '__main__':
    main()
