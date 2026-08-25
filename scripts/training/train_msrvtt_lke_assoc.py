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
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import sys
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.model_lke_assoc import LatentKinematicGroundingMemory

class MSRVTT_LKE_Dataset(Dataset):
    def __init__(self, split="train", num_frames=64, 
                 video_dir=config['paths']['msrvtt_video_dir'], 
                 text_dir=config['paths']['msrvtt_text_dir']):
        self.samples = []
        self.num_frames = num_frames
        self.split = split
        
        if split == "train":
            start_idx, end_idx = 0, 6512
        elif split == "val":
            start_idx, end_idx = 6513, 7009
        else:
            start_idx, end_idx = 7010, 9999
            
        for i in range(start_idx, end_idx + 1):
            vid_name = f"video{i}"
            vf = os.path.join(video_dir, f"{vid_name}.pt")
            tf = os.path.join(text_dir, f"{vid_name}.pt")
            if os.path.exists(vf) and os.path.exists(tf):
                self.samples.append({
                    "video_path": vf,
                    "text_path": tf
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        video_dict = torch.load(sample["video_path"], map_location="cpu", weights_only=False)
        sparse_frames = video_dict["sparse"].float() 
        frame_indices = np.linspace(0, sparse_frames.shape[0] - 1, self.num_frames, dtype=int)
        video_dense = sparse_frames[frame_indices]
        
        text_list = torch.load(sample["text_path"], map_location="cpu", weights_only=False)
        if self.split == "train":
            chosen_text = random.choice(text_list)
        else:
            chosen_text = text_list[0]
            
        t_global = chosen_text["global"].float()
        t_nouns = chosen_text["nouns"].float()
        t_verbs = chosen_text["verbs"].float()
        
        return video_dense, t_global, t_nouns, t_verbs, idx

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 STARTING MSR-VTT LKE-ASSOC ON {str(device).upper()}", flush=True)

    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 5e-5

    print("Loading Datasets...", flush=True)
    train_dataset = MSRVTT_LKE_Dataset(split="train")
    val_dataset = MSRVTT_LKE_Dataset(split="val")
    test_dataset = MSRVTT_LKE_Dataset(split="test")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

    # Initialize model
    model = LatentKinematicGroundingMemory(d_video=768, d_text=512, d_model=512, num_layers=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_dataloader))
    
    # We will use a fixed temperature for InfoNCE retrieval
    temperature = 0.07

    print(f"Total Samples: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}", flush=True)
    
    best_val_loss = float('inf')
    ckpt_path = os.path.join(config['paths']['save_dir'], 'best_msrvtt_lke_assoc_model.pt')
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start_time = time.time()
        running_loss = 0.0

        for batch_idx, (video_dense, t_global, t_nouns, t_verbs, _) in enumerate(train_dataloader):
            video_dense = video_dense.to(device)
            t_global = t_global.to(device)
            t_nouns = t_nouns.to(device)
            t_verbs = t_verbs.to(device)
            
            mask = torch.ones((video_dense.shape[0], video_dense.shape[1]), device=device)

            optimizer.zero_grad()
            
            # Forward pass
            _, _, pooled = model(video_dense, mask, t_nouns, t_verbs)
            
            # Bidirectional InfoNCE Loss
            logits = torch.matmul(pooled, t_global.t()) / temperature
            labels = torch.arange(logits.shape[0], device=logits.device)
            
            loss_v = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.t(), labels)
            loss = (loss_v + loss_t) / 2.0
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"   [Epoch {epoch} / Batch {batch_idx+1}] Loss: {loss.item():.4f}", flush=True)

        epoch_loss = running_loss / len(train_dataloader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for video_dense, t_global, t_nouns, t_verbs, _ in val_dataloader:
                video_dense = video_dense.to(device)
                t_global = t_global.to(device)
                t_nouns = t_nouns.to(device)
                t_verbs = t_verbs.to(device)
                mask = torch.ones((video_dense.shape[0], video_dense.shape[1]), device=device)

                _, _, pooled = model(video_dense, mask, t_nouns, t_verbs)
                
                logits = torch.matmul(pooled, t_global.t()) / temperature
                labels = torch.arange(logits.shape[0], device=logits.device)
                loss_v = F.cross_entropy(logits, labels)
                loss_t = F.cross_entropy(logits.t(), labels)
                loss = (loss_v + loss_t) / 2.0
                
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        elapsed_time = time.time() - start_time
        
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            improved = " <-- BEST"

        print(f" Epoch [{epoch:02d}/{EPOCHS:02d}] -> Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed_time:.2f}s{improved}", flush=True)

    # Evaluate the best model
    print("\n--- Running Final Evaluation on Test Set ---", flush=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    all_v_emb = []
    all_t_emb = []
    
    with torch.no_grad():
        for video_dense, t_global, t_nouns, t_verbs, _ in test_dataloader:
            video_dense = video_dense.to(device)
            t_global = t_global.to(device)
            t_nouns = t_nouns.to(device)
            t_verbs = t_verbs.to(device)
            mask = torch.ones((video_dense.shape[0], video_dense.shape[1]), device=device)
            
            _, _, pooled = model(video_dense, mask, t_nouns, t_verbs)
            
            all_v_emb.append(pooled)
            all_t_emb.append(t_global)
            
    all_v_emb = torch.cat(all_v_emb, dim=0)
    all_t_emb = torch.cat(all_t_emb, dim=0)
    
    sim = torch.matmul(all_v_emb, all_t_emb.t())
    
    N = sim.shape[0]
    recall_1 = recall_5 = recall_10 = 0
    
    for i in range(N):
        sorted_indices = torch.argsort(sim[i], descending=True)
        if sorted_indices[0].item() == i:
            recall_1 += 1
        if i in sorted_indices[:5].tolist():
            recall_5 += 1
        if i in sorted_indices[:10].tolist():
            recall_10 += 1
            
    print("\n=========================================================")
    print("  MSR-VTT LKE-ASSOC RESULTS (ZERO-SHOT PROXY)  ")
    print("=========================================================")
    print(f" 🎯 Recall@1 : {(recall_1 / N) * 100:.2f}% ({recall_1}/{N})")
    print(f" 🎯 Recall@5 : {(recall_5 / N) * 100:.2f}% ({recall_5}/{N})")
    print(f" 🎯 Recall@10: {(recall_10 / N) * 100:.2f}% ({recall_10}/{N})")
    print("=========================================================\n", flush=True)

if __name__ == "__main__":
    run_training()
