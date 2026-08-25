import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialLatentKinematicGroundingMemory(nn.Module):
    def __init__(self, d_video=768, d_text=512, d_model=512, num_layers=4):
        super().__init__()
        
        self.v_proj = nn.Linear(d_video, d_model)
        
        self.noun_proj = nn.Linear(d_text, d_model)
        self.verb_proj = nn.Linear(d_text, d_model)
        
        self.noun_scale = nn.Parameter(torch.ones([]) * 14.28) # ~ln(1/0.07) like CLIP
        self.verb_scale = nn.Parameter(torch.ones([]) * 14.28)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.temporal_adapter = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pos_embedding = nn.Parameter(torch.zeros(1, 256, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        self.start_head = nn.Linear(d_model, 1)
        self.end_head = nn.Linear(d_model, 1)
        
    def forward(self, video_patches, attention_mask, noun_feat, verb_feat):
        """
        video_patches: [B, T, P, d_video]  (e.g., [B, 8, 49, 768])
        attention_mask: [B, T]
        noun_feat: [B, d_text]
        verb_feat: [B, d_text]
        """
        B, T, P, _ = video_patches.shape
        
        # 1. Spatial Noun Filtering (2D Associative Memory)
        v_features = self.v_proj(video_patches) # [B, T, P, d_model]
        n_query = self.noun_proj(noun_feat).unsqueeze(1).unsqueeze(1) # [B, 1, 1, d_model]
        
        # L2 Normalize
        v_features_norm = F.normalize(v_features, p=2, dim=-1)
        n_query_norm = F.normalize(n_query, p=2, dim=-1)
        
        # Cosine Similarity -> Sigmoid Gate
        noun_sim = (v_features_norm * n_query_norm).sum(dim=-1) # [B, T, P]
        noun_gate = torch.sigmoid(noun_sim * self.noun_scale) # [B, T, P]
        
        # Apply spatial mask to cut out background pixels
        filtered_patches = v_features * noun_gate.unsqueeze(-1) # [B, T, P, d_model]
        
        # Pool the isolated object patches back into a 1D sequence
        # We use mean pooling across the 49 patches
        pooled_video = filtered_patches.mean(dim=2) # [B, T, d_model]
        
        # Add Positional Embeddings before Temporal Transformer
        pooled_video = pooled_video + self.pos_embedding[:, :T, :]
        
        # 2. Kinematic Modeling
        src_key_padding_mask = (attention_mask == 0)
        memory = self.temporal_adapter(pooled_video, src_key_padding_mask=src_key_padding_mask) # [B, T, d_model]
        
        # 3. Temporal Verb Anchoring (1D Associative Memory)
        v_query = self.verb_proj(verb_feat).unsqueeze(1) # [B, 1, d_model]
        
        memory_norm = F.normalize(memory, p=2, dim=-1)
        v_query_norm = F.normalize(v_query, p=2, dim=-1)
        
        verb_sim = (memory_norm * v_query_norm).sum(dim=-1) # [B, T]
        verb_gate = torch.sigmoid(verb_sim * self.verb_scale) # [B, T]
        
        final_memory = memory * verb_gate.unsqueeze(-1) # [B, T, d_model]
        
        # 4. Action Boundary Prediction
        start_logits = self.start_head(final_memory).squeeze(-1) # [B, T]
        end_logits = self.end_head(final_memory).squeeze(-1) # [B, T]
        
        # Mask out padding
        start_logits = start_logits.masked_fill(attention_mask == 0, float('-inf'))
        end_logits = end_logits.masked_fill(attention_mask == 0, float('-inf'))
        
        return start_logits, end_logits
