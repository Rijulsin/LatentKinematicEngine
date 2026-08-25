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
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: True

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_loader_sta import CharadesSTADataset
from src.model_sta import QueryAddressedGroundingMemory
from src.model_lke_assoc import BoundaryLoss
import torch.nn.functional as F

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading CLIP Model for dynamic text queries...", flush=True)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Freeze CLIP
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval()

    TRAIN_TXT = config['paths']['train_txt']
    FEAT_DIR = config['paths']['feat_dir_32f']
    
    print("Loading Charades-STA Dataset...", flush=True)
    train_dataset = CharadesSTADataset(TRAIN_TXT, FEAT_DIR, max_seq_len=64, is_train=True)
    
    if len(train_dataset) == 0:
        print("Error: No training samples loaded. Check paths and features.")
        return

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=2)
    
    print(f"Train Grounding Samples: {len(train_dataset)}", flush=True)
    
    print("\n--- INITIALIZING GROUNDING MODEL ---", flush=True)
    model = QueryAddressedGroundingMemory(d_video=768, d_text=512, d_model=512, num_layers=4).to(device)
    criterion = BoundaryLoss().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    EPOCHS = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_dataloader))
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            video_dense = batch['video_features'].to(device)
            mask = batch['attention_mask'].to(device)
            start_idx = batch['start_idx'].to(device)
            end_idx = batch['end_idx'].to(device)
            
            # Encode Text Queries on the fly
            queries = batch['query']
            inputs = tokenizer(queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            
            with torch.no_grad():
                text_outputs = clip_model.text_model(**inputs)
                text_feat = clip_model.text_projection(text_outputs[1])
                text_feat = F.normalize(text_feat, p=2, dim=-1)
                
            optimizer.zero_grad()
            
            # Predict boundaries
            start_logits, end_logits, pooled = model(video_dense, mask, text_feat)
            
            # Compute loss
            loss = criterion(start_logits, end_logits, start_idx, end_idx, pooled=pooled, mask=mask)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch}/{EPOCHS} Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.1f}s", flush=True)
        # Save dynamically at end of epoch
        torch.save(model.state_dict(), "best_charades_sta_model.pt")
        
    print("\nTraining Complete.")

if __name__ == '__main__':
    main()
