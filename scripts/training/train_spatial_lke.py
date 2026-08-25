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
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPTokenizer
import spacy

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: True

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_loader_spatial import CharadesSpatialDataset
from src.model_spatial_lke import SpatialLatentKinematicGroundingMemory
from src.model_lke_assoc import BoundaryLoss
import torch.nn.functional as F

def extract_noun_verb(query, nlp):
    doc = nlp(query)
    nouns = " ".join([token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'PRON', 'ADJ']])
    verbs = " ".join([token.text for token in doc if token.pos_ in ['VERB', 'ADV']])
    
    #Fallback to original query if parsing fails or misses crucial parts
    if not nouns.strip(): nouns = query
    if not verbs.strip(): verbs = query
    return nouns, verbs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading CLIP Model...", flush=True)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    print("Loading spaCy for Kinematic Disentanglement...", flush=True)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
        
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval()

    TRAIN_TXT = config['paths']['train_txt']
    FEAT_DIR = config['paths']['feat_dir_patches_32f']
    
    print("Loading Charades-STA Dataset...", flush=True)
    train_dataset = CharadesSpatialDataset(TRAIN_TXT, FEAT_DIR, max_seq_len=32, is_train=True)
    
    if len(train_dataset) == 0:
        print("Error: No training samples loaded. Check paths and features.")
        return

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=2)
    print(f"Train Grounding Samples: {len(train_dataset)}", flush=True)
    
    print("\n--- INITIALIZING SPATIAL LATENT KINEMATIC ENGINE ---", flush=True)
    model = SpatialLatentKinematicGroundingMemory(d_video=768, d_text=512, d_model=512, num_layers=4).to(device)
    criterion = BoundaryLoss().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    EPOCHS = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_dataloader))
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} - Batch {batch_idx}/{len(train_dataloader)} loaded...", flush=True)
            video_dense = batch['video_features'].to(device)
            mask = batch['attention_mask'].to(device)
            start_idx = batch['start_idx'].to(device)
            end_idx = batch['end_idx'].to(device)
            
            # 1. Independent Prompting
            if not hasattr(model, 'spacy_cache'):
                model.spacy_cache = {}
                
            noun_queries = []
            verb_queries = []
            for query in batch['query']:
                if query not in model.spacy_cache:
                    model.spacy_cache[query] = extract_noun_verb(query, nlp)
                n, v = model.spacy_cache[query]
                noun_queries.append(n)
                verb_queries.append(v)
            
            #Encode nouns and verbs independently
            n_inputs = tokenizer(noun_queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            v_inputs = tokenizer(verb_queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            
            with torch.no_grad():
                n_outputs = clip_model.text_model(**n_inputs)
                n_feat = clip_model.text_projection(n_outputs[1])
                n_feat = F.normalize(n_feat, p=2, dim=-1)
                
                v_outputs = clip_model.text_model(**v_inputs)
                v_feat = clip_model.text_projection(v_outputs[1])
                v_feat = F.normalize(v_feat, p=2, dim=-1)
                
            optimizer.zero_grad()
            
            # 2. Forward pass through LKE
            start_logits, end_logits = model(video_dense, mask, n_feat, v_feat)
            
            loss = criterion(start_logits, end_logits, start_idx, end_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch}/{EPOCHS} LKE Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.1f}s", flush=True)
        torch.save(model.state_dict(), "best_charades_sta_spatial_lke_model.pt")
        
    print("\nLKE Training Complete.")

if __name__ == '__main__':
    main()
