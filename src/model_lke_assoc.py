import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAdapter(nn.Module):
    """
    Temporal transformer that models contextual dependencies between frames.
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
            padding_mask = ~(mask.bool())
            x = self.transformer(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x)
        return x

class LatentKinematicGroundingMemory(nn.Module):
    """
    Implements the Latent Kinematic Engine with explicit Noun/Verb disentanglement.
    1. Spatial Filtering: Noun embedding filters the raw video features to isolate the objects.
    2. Kinematic Modeling: Temporal Adapter processes the object-isolated sequence.
    3. Action Grounding: Verb embedding aligns with the kinematic output to predict boundaries.
    """
    def __init__(self, d_video=768, d_text=512, d_model=512, num_layers=4):
        super().__init__()
        import math
        
        # 1. Noun Spatial Projection
        self.noun_proj = nn.Linear(d_text, d_video)
        
        # 2. Temporal Kinematic Adapter
        self.temporal_adapter = TemporalAdapter(d_model=d_video, num_layers=num_layers)
        
        # 3. Final Shared Space Projection
        self.v_proj = nn.Linear(d_video, d_model)
        self.verb_proj = nn.Linear(d_text, d_model)
        
        # Learnable temperatures for associative memory
        self.noun_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.verb_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        
        # 4. Boundary Grounding Heads
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
        
    def forward(self, video_features, mask, noun_feat, verb_feat):
        """
        video_features: [B, T, 768]
        noun_feat: [B, 512] (Disentangled Noun Embedding)
        verb_feat: [B, 512] (Disentangled Verb Embedding)
        """
        # STEP 1: Spatial Object Filtering (Associative Memory)
        # Noun filters out irrelevant frames where target object is not present
        n_query = self.noun_proj(noun_feat).unsqueeze(1) # [B, 1, 768]
        
        v_norm = F.normalize(video_features, p=2, dim=-1)
        n_norm = F.normalize(n_query, p=2, dim=-1)
        n_scale = self.noun_scale.exp().clamp(max=100)
        
        noun_attn = torch.bmm(v_norm, n_norm.transpose(1, 2)) * n_scale # [B, T, 1]
        noun_attn = torch.sigmoid(noun_attn)
        video_object_space = video_features * noun_attn # [B, T, 768]
        
        # STEP 2: Temporal Kinematic Modeling
        # Now the transformer only models the physics of the isolated objects
        v_context = self.temporal_adapter(video_object_space, mask) # [B, T, 768]
        
        # STEP 3: Kinematic Action Grounding (Associative Memory)
        v_shared = self.v_proj(v_context) # [B, T, d_model]
        v_action = self.verb_proj(verb_feat).unsqueeze(1) # [B, 1, d_model]
        
        vs_norm = F.normalize(v_shared, p=2, dim=-1)
        va_norm = F.normalize(v_action, p=2, dim=-1)
        v_scale = self.verb_scale.exp().clamp(max=100)
        
        verb_attn = torch.bmm(vs_norm, va_norm.transpose(1, 2)) * v_scale # [B, T, 1]
        verb_attn = torch.sigmoid(verb_attn)
        v_fused = v_shared * verb_attn # [B, T, d_model]
        
        # STEP 4: Boundary Prediction
        start_logits = self.start_head(v_fused).squeeze(-1) # [B, T]
        end_logits = self.end_head(v_fused).squeeze(-1) # [B, T]
        
        if mask is not None:
            # Use -1e4 instead of -inf to prevent NaN/Inf in KL Divergence
            start_logits = start_logits.masked_fill(~mask.bool(), -1e4)
            end_logits = end_logits.masked_fill(~mask.bool(), -1e4)
        
        # Pooled video-text representation for contrastive loss
        # Mask-aware mean pool over valid frames
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            pooled = (v_fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            pooled = v_fused.mean(dim=1) # [B, d_model]
        pooled = F.normalize(pooled, p=2, dim=-1, eps=1e-8)
        
        return start_logits, end_logits, pooled

class BoundaryLoss(nn.Module):
    """
    Advanced Boundary Loss:
    1. KL-Divergence with 1D Gaussian smoothed labels (replaces strict CE)
    2. Soft Generalized IoU (GIoU) Loss (prevents zero-gradients on non-overlapping windows)
    3. In-batch Contrastive Loss
    """
    def __init__(self, lambda_iou=0.5, lambda_contrast=0.3, temperature=0.07, sigma=2.0):
        super().__init__()
        self.lambda_iou = lambda_iou
        self.lambda_contrast = lambda_contrast
        self.temperature = temperature
        self.sigma = sigma
        
    def get_gaussian_labels(self, target_indices, seq_len, mask=None):
        """Generates a 1D Gaussian distribution centered at the target index."""
        B = target_indices.shape[0]
        positions = torch.arange(seq_len, dtype=torch.float, device=target_indices.device)
        positions = positions.unsqueeze(0).expand(B, -1) # [B, T]
        
        target = target_indices.unsqueeze(1).float() # [B, 1]
        
        #Unnormalized Gaussian
        gaussian = torch.exp(-((positions - target) ** 2) / (2 * self.sigma ** 2))
        
        if mask is not None:
            #Prevent Gaussian from bleeding into masked/padded frames
            gaussian = gaussian * mask.float()
            
        #Normalize to sum to 1 (probability distribution)
        #Calmp to avoid division by 0
        return gaussian / gaussian.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
    def kl_divergence_loss(self, logits, target_indices, mask=None):
        """Computes KL divergence between predicted log_softmax and gaussian target."""
        T = logits.shape[1]
        log_preds = F.log_softmax(logits, dim=-1)
        targets = self.get_gaussian_labels(target_indices, T, mask=mask)
        
        #Divide by T to normalize to same scale as CE loss (~4.0 range)
        #F.kl_div with batch mean divides by B but not T
        return F.kl_div(log_preds, targets, reduction='batchmean') / T

    def soft_giou_loss(self, start_logits, end_logits, start_idx, end_idx):
        """Differentiable Generalized IoU (GIoU) loss via expected position."""
        T = start_logits.shape[1]
        positions = torch.arange(T, dtype=torch.float, device=start_logits.device) # [T]
        
        start_probs = F.softmax(start_logits, dim=-1) # [B, T]
        end_probs = F.softmax(end_logits, dim=-1)     # [B, T]
        
        #start and end positions
        pred_s = (start_probs * positions).sum(dim=-1) # [B]
        pred_e = (end_probs * positions).sum(dim=-1)   # [B]
        pred_e = torch.maximum(pred_e, pred_s + 1.0)
        
        gt_s = start_idx.float()
        gt_e = end_idx.float()
        
        #Intersection
        inter_s = torch.maximum(pred_s, gt_s)
        inter_e = torch.minimum(pred_e, gt_e)
        intersection = (inter_e - inter_s).clamp(min=0)
        
        #Union
        union_s = torch.minimum(pred_s, gt_s)
        union_e = torch.maximum(pred_e, gt_e)
        union = (union_e - union_s).clamp(min=1e-6)
        
        iou = intersection / union #IOU
        
        #Enclosing window (the smallest window containing both pred and gt)
        enclose_s = torch.minimum(pred_s, gt_s)
        enclose_e = torch.maximum(pred_e, gt_e)
        enclose_area = (enclose_e - enclose_s).clamp(min=1e-6)
        
        # GIoU = IoU - (Enclosing_Area - Union_Area) / Enclosing_Area
        giou = iou - ((enclose_area - union) / enclose_area)
        
        return (1.0 - giou).mean()
        
    def contrastive_loss(self, pooled):
        """In-batch InfoNCE."""
        B = pooled.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=pooled.device)
        
        sim = torch.matmul(pooled, pooled.t()) / self.temperature
        labels = torch.arange(B, device=pooled.device)
        return F.cross_entropy(sim, labels)
        
    def forward(self, start_logits, end_logits, start_idx, end_idx, pooled=None, mask=None):
        #KL-Divergence with Gaussian smoothed targets
        loss_start = self.kl_divergence_loss(start_logits, start_idx, mask=mask)
        loss_end = self.kl_divergence_loss(end_logits, end_idx, mask=mask)
        loss_kl = loss_start + loss_end
        
        #Soft GIoU loss
        loss_giou = self.soft_giou_loss(start_logits, end_logits, start_idx, end_idx)
        
        #Contrastive loss
        loss_contrast = self.contrastive_loss(pooled) if pooled is not None else 0.0
        
        total = loss_kl + self.lambda_iou * loss_giou + self.lambda_contrast * loss_contrast
        return total
