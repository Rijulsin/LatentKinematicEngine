import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAdapter(nn.Module):
    """
    Temporal transformer that models contextual dependencies between frames.
    Adapted from the best baseline.
    """
    def __init__(self, d_model: int = 768, nhead: int = 8, num_layers: int = 4, max_frames: int = 256):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_frames, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_norm = nn.LayerNorm(d_model)
        
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        x = self.input_norm(x)
        x += self.pos_embedding[:, :T, :]
        
        if mask is not None:
            # mask: [B, T] (1 for valid, 0 for pad). Transformer expects padding_mask where True means ignore.
            padding_mask = ~(mask.bool())
            x = self.transformer(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x)
        return x # [B, T, D]

class QueryAddressedGroundingMemory(nn.Module):
    """
    Takes inspiration from QueryAddressedAssociativeMemory in src/model_episodic_memory.py.
    Instead of retrieving pre-chunked events, it aligns continuous video frames 
    with a text query to natively predict start/end boundaries at the frame level.
    """
    def __init__(self, d_video=768, d_text=512, d_model=512, num_layers=4):
        super().__init__()
        self.temporal_adapter = TemporalAdapter(d_model=d_video, num_layers=num_layers)
        
        # Project video and text to shared semantic space
        self.v_proj = nn.Linear(d_video, d_model)
        self.t_proj = nn.Linear(d_text, d_model)
        
        # Query-Addressed Grounding Heads
        # Takes the fused (video * query) representation and predicts logits
        self.start_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        self.end_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, video_features, mask, text_query):
        """
        video_features: [B, T, 768] (Dense video frames)
        mask: [B, T] (1 for valid, 0 for pad)
        text_query: [B, 512] (Global CLIP text embedding)
        """
        # 1. Temporal Contextualization (Episodic Frame Memory)
        v_context = self.temporal_adapter(video_features, mask) # [B, T, 768]
        
        # 2. Projection to Shared Space
        v_shared = self.v_proj(v_context) # [B, T, d_model]
        t_shared = self.t_proj(text_query).unsqueeze(1) # [B, 1, d_model]
        
        # 3. Query-Addressed Fusion 
        # (Element-wise multiplication aligns the video stream with the text query's semantic intent)
        v_fused = v_shared * t_shared # [B, T, d_model]
        
        # 4. Predict Temporal Boundaries
        start_logits = self.start_head(v_fused).squeeze(-1) # [B, T]
        end_logits = self.end_head(v_fused).squeeze(-1) # [B, T]
        
        # Mask out padded regions by setting logits to -1e4 (prevent NaN in KL)
        start_logits = start_logits.masked_fill(~mask.bool(), -1e4)
        end_logits = end_logits.masked_fill(~mask.bool(), -1e4)
        
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            pooled = (v_fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            pooled = v_fused.mean(dim=1)
        pooled = F.normalize(pooled, p=2, dim=-1, eps=1e-8)
        
        return start_logits, end_logits, pooled

class BoundaryLoss(nn.Module):
    """
    Standard Cross-Entropy Loss over the temporal sequence dimension 
    for exact start and end frame prediction.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, start_logits, end_logits, start_idx, end_idx):
        loss_start = F.cross_entropy(start_logits, start_idx)
        loss_end = F.cross_entropy(end_logits, end_idx)
        return loss_start + loss_end
