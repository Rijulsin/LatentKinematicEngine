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
from torch.utils.data import DataLoader, random_split
from transformers import CLIPModel, CLIPTokenizer
import spacy

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: True

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_loader_sta import CharadesSTADataset
from src.model_lke_assoc import LatentKinematicGroundingMemory, BoundaryLoss
import torch.nn.functional as F

CHECKPOINT_DIR = config['paths']['save_dir']
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'best_charades_sta_lke_assoc_64f_model.pt')

def extract_noun_verb(query, nlp):
    doc = nlp(query)
    nouns = " ".join([token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'PRON', 'ADJ']])
    verbs = " ".join([token.text for token in doc if token.pos_ in ['VERB', 'ADV']])
    if not nouns.strip(): nouns = query
    if not verbs.strip(): verbs = query
    return nouns, verbs

def run_epoch(model, dataloader, criterion, clip_model, tokenizer, nlp, device,
              optimizer=None, scheduler=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

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

        if train:
            optimizer.zero_grad()
            start_logits, end_logits, pooled = model(video_dense, mask, n_feat, v_feat)
            loss = criterion(start_logits, end_logits, start_idx, end_idx, pooled=pooled, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        else:
            with torch.no_grad():
                start_logits, end_logits, pooled = model(video_dense, mask, n_feat, v_feat)
                loss = criterion(start_logits, end_logits, start_idx, end_idx, pooled=pooled, mask=mask)

        running_loss += loss.item()

    return running_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Loading CLIP Model...", flush=True)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval()

    print("Loading spaCy for Kinematic Disentanglement...", flush=True)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    TRAIN_TXT = config['paths']['train_txt']
    FEAT_DIR = config['paths']['feat_dir_64f']

    print("Loading Charades-STA Dataset (64-frame dense features)...", flush=True)
    full_dataset = CharadesSTADataset(TRAIN_TXT, FEAT_DIR, max_seq_len=128, is_train=True)

    if len(full_dataset) == 0:
        print("Error: No training samples loaded. Check paths and features.")
        return

    # 90/10 train/val split with fixed seed for reproducibility
    val_size = int(0.10 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}", flush=True)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=2)

    print("\n--- INITIALIZING LATENT KINEMATIC ENGINE ---", flush=True)
    model = LatentKinematicGroundingMemory(d_video=768, d_text=512, d_model=512, num_layers=4).to(device)
    criterion = BoundaryLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    EPOCHS = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_dataloader))

    best_val_loss = float('inf')
    print(f"Checkpoint will be saved to: {CHECKPOINT_PATH}", flush=True)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss = run_epoch(model, train_dataloader, criterion, clip_model, tokenizer, nlp, device,
                               optimizer=optimizer, scheduler=scheduler, train=True)
        val_loss = run_epoch(model, val_dataloader, criterion, clip_model, tokenizer, nlp, device, train=False)

        elapsed = time.time() - start_time
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            improved = "  <-- BEST SAVED"

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {elapsed:.1f}s{improved}", flush=True)

    print(f"\nTraining Complete. Best Val Loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {CHECKPOINT_PATH}")


if __name__ == '__main__':
    main()
