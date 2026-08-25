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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_loader_sta import CharadesSTADataset
from src.model_lke_assoc import LatentKinematicGroundingMemory, BoundaryLoss
import torch.nn.functional as F

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: True

def extract_noun_verb(query, nlp):
    doc = nlp(query)
    nouns = " ".join([token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'PRON', 'ADJ']])
    verbs = " ".join([token.text for token in doc if token.pos_ in ['VERB', 'ADV']])
    if not nouns.strip(): nouns = query
    if not verbs.strip(): verbs = query
    return nouns, verbs

def train_one_epoch(model, dataloader, criterion, clip_model, tokenizer, nlp, device, optimizer, scheduler=None):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        video_dense = batch['video_features'].to(device)
        mask = batch['attention_mask'].to(device)
        start_idx = batch['start_idx'].to(device)
        end_idx = batch['end_idx'].to(device)

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
        start_logits, end_logits, pooled = model(video_dense, mask, n_feat, v_feat)
        loss = criterion(start_logits, end_logits, start_idx, end_idx, pooled=pooled, mask=mask)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        running_loss += loss.item()

    return running_loss / len(dataloader)

def eval_one_epoch(model, dataloader, criterion, clip_model, tokenizer, nlp, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            video_dense = batch['video_features'].to(device)
            mask = batch['attention_mask'].to(device)
            start_idx = batch['start_idx'].to(device)
            end_idx = batch['end_idx'].to(device)

            noun_queries, verb_queries = [], []
            for query in batch['query']:
                n, v = extract_noun_verb(query, nlp)
                noun_queries.append(n)
                verb_queries.append(v)

            n_inputs = tokenizer(noun_queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            v_inputs = tokenizer(verb_queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)

            n_feat = F.normalize(clip_model.text_projection(clip_model.text_model(**n_inputs)[1]), p=2, dim=-1)
            v_feat = F.normalize(clip_model.text_projection(clip_model.text_model(**v_inputs)[1]), p=2, dim=-1)

            start_logits, end_logits, pooled = model(video_dense, mask, n_feat, v_feat)
            loss = criterion(start_logits, end_logits, start_idx, end_idx, pooled=pooled, mask=mask)
            running_loss += loss.item()

    return running_loss / len(dataloader)

def run_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading CLIP Model & spaCy...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    nlp = spacy.load("en_core_web_sm")
    clip_model.eval()

    TRAIN_TXT = config['paths']['train_txt']
    TEST_TXT = config['paths']['test_txt']
    FEAT_DIR = config['paths']['feat_dir_32f']
    
    print("Loading datasets...")
    train_dataset = CharadesSTADataset(TRAIN_TXT, FEAT_DIR, max_seq_len=32, is_train=True)
    val_dataset = CharadesSTADataset(TEST_TXT, FEAT_DIR, max_seq_len=32, is_train=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Hyperparameter Grid
    num_layers_list = [2, 4, 6]
    d_model_list = [256, 512, 1024]
    learning_rate = 1e-5
    epochs_per_config = 15 # Train for fewer epochs just to check early convergence

    best_overall_val_loss = float('inf')
    best_overall_config = None

    print("\n--- STARTING LKE HYPERPARAMETER SWEEP ---")
    
    for num_layers in num_layers_list:
        for d_model in d_model_list:
            config_name = f"Layers: {num_layers}, d_model: {d_model}"
            print(f"\nEvaluating Configuration: [{config_name}]")
            
            model = LatentKinematicGroundingMemory(d_video=768, d_text=512, d_model=d_model, num_layers=num_layers).to(device)
            criterion = BoundaryLoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_config * len(train_dataloader))
            
            best_val_loss = float('inf')
            
            for epoch in range(1, epochs_per_config + 1):
                train_loss = train_one_epoch(model, train_dataloader, criterion, clip_model, tokenizer, nlp, device, optimizer, scheduler)
                val_loss = eval_one_epoch(model, val_dataloader, criterion, clip_model, tokenizer, nlp, device)
                
                print(f"  Epoch {epoch:02d}/{epochs_per_config} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    
            print(f"Result for [{config_name}] -> Best Val Loss: {best_val_loss:.4f}")
            
            if best_val_loss < best_overall_val_loss:
                best_overall_val_loss = best_val_loss
                best_overall_config = config_name
                
    print("\n=============================================")
    print("SWEEP COMPLETE")
    print(f"Best Configuration: {best_overall_config}")
    print(f"Best Val Loss: {best_overall_val_loss:.4f}")
    print("=============================================\n")

if __name__ == '__main__':
    run_sweep()
