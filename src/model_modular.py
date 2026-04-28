"""
Modular OOD Detection Framework (Final Clean Version)
Supports flexible feature selection, fusion, and loss combinations.
Optimized for Automatic Mixed Precision (AMP) with @autocast decorators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import math
import clip
import json
import os
import numpy as np
from torch.cuda.amp import autocast


# ============================================================================
# PART 1: BASE CLASSES
# ============================================================================

class BaseSelector(ABC, nn.Module):
    """Abstract base class for feature selectors."""
    
    def __init__(self, input_dim: int, num_select: int, cfg: Dict = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_select = num_select
        self.cfg = cfg or {}
    
    @abstractmethod
    def forward(self, local_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
        Returns:
            selected_feats: (B, N, D) or (B, K, D)
            aux_loss: dict
            mask: (B, N, 1) binary-like mask for mixup
        """
        pass


class BaseFuser(ABC, nn.Module):
    """Abstract base class for feature fusers."""
    
    def __init__(self, input_dim: int, cfg: Dict = None):
        super().__init__()
        self.input_dim = input_dim
        self.cfg = cfg or {}
    
    @abstractmethod
    def forward(self, selected_feats: torch.Tensor, 
                text_feats: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass


# ============================================================================
# PART 2: FEATURE SELECTORS (Precision Managed)
# ============================================================================

class IdentitySelector(BaseSelector):
    """No-op selector."""
    
    def __init__(self, input_dim: int, num_select: int, cfg: Dict = None):
        super().__init__(input_dim, num_select, cfg)
    
    def forward(self, local_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        B, N, D = local_feats.shape
        # Return all ones mask
        mask = torch.ones(B, N, 1, device=local_feats.device, dtype=local_feats.dtype)
        return local_feats, {}, mask


class MultiHeadMLPSelector(BaseSelector):
    """
    Multi-head MLP-based feature selector with STE.
    Runs scorer in mixed precision, but selection logic in FP32.
    """
    
    def __init__(self, input_dim: int, num_select: int, num_heads: int = 8, cfg: Dict = None):
        super().__init__(input_dim, num_select, cfg)
        self.num_heads = num_heads
        
        mlp_hidden_ratio = cfg.get('mlp_hidden_ratio', 0.25) if cfg else 0.25
        mlp_hidden_dim = int(input_dim * mlp_hidden_ratio)
        
        self.scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, 1)
            ) for _ in range(num_heads)
        ])
    
    def forward(self, local_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        B, N, D = local_feats.shape
        dtype = local_feats.dtype
        device = local_feats.device
        
        # 1. Convert to FP32 for scorer operations (linear layers expect FP32 params)
        local_feats_fp32 = local_feats.float()
        head_scores_list = [scorer(local_feats_fp32) for scorer in self.scorers]
        scores = torch.cat(head_scores_list, dim=-1)  # (B, N, H)
        
        # 2. Selection Logic (Force FP32 for stability with TopK & Indices)
        with autocast(enabled=False):
            scores_fp32 = scores.float()
            final_mask = torch.zeros(B, N, 1, device=device, dtype=torch.float32)
            k_per_head = max(1, self.num_select // self.num_heads)
            
            for h in range(self.num_heads):
                scores_h = scores_fp32[:, :, h]
                _, topk_indices_h = torch.topk(scores_h, k=k_per_head, dim=1)
                
                # Vectorized scatter for efficiency
                # Create a src tensor of ones: (B, k, 1)
                src = torch.ones(B, k_per_head, 1, device=device, dtype=torch.float32)
                # Expand indices: (B, k, 1)
                indices = topk_indices_h.unsqueeze(-1)
                
                final_mask.scatter_(1, indices, src)
                
            final_mask = torch.clamp(final_mask, 0.0, 1.0)
            
            # 3. STE (Straight-Through Estimator)
            scores_max = scores_fp32.max(dim=-1)[0].unsqueeze(-1)
            ste_mask = (final_mask - scores_max).detach() + scores_max
            
            # Diversity Loss Calculation (FP32)
            diversity_loss = self._compute_diversity_loss_fp32(scores_fp32)

        # 4. Apply mask (Back to original dtype)
        ste_mask = ste_mask.to(dtype)
        selected_feats = local_feats * ste_mask
        
        return selected_feats, {'diversity': diversity_loss.to(dtype)}, ste_mask
    
    @staticmethod
    def _compute_diversity_loss_fp32(head_scores: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss in FP32."""
        B, N, num_heads = head_scores.shape
        # Normalize along patch dim
        head_scores_norm = F.normalize(head_scores, dim=1, eps=1e-6)
        # Gram matrix: (B, H, H)
        gram_matrix = torch.bmm(head_scores_norm.transpose(1, 2), head_scores_norm)
        # Identity
        I = torch.eye(num_heads, device=head_scores.device).unsqueeze(0).expand(B, -1, -1)
        return (gram_matrix - I).abs().mean()


class SparseSlotAttentionSelector(BaseSelector):
    """
    Slot Attention-based feature selector.
    Critically optimized: Logic runs in FP32 to prevent NaN during softmax/exp.
    """
    
    def __init__(self, input_dim: int, num_select: int, num_slots: int = 8, cfg: Dict = None):
        super().__init__(input_dim, num_select, cfg)
        self.num_slots = num_slots
        
        ffn_ratio = cfg.get('slot_ffn_ratio', 4.0) if cfg else 4.0
        ffn_hidden_dim = int(input_dim * ffn_ratio)
        
        self.slots = nn.Parameter(torch.empty(num_slots, input_dim))
        nn.init.orthogonal_(self.slots.data, gain=1.0)
        self.slots.data = self.slots.data.unsqueeze(0)
        
        self.norm_slots = nn.LayerNorm(input_dim)
        self.norm_input = nn.LayerNorm(input_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        self.temperature = cfg.get('selector_temperature', 0.1) if cfg else 0.1

    @autocast(enabled=False)
    def _forward_fp32(self, local_feats: torch.Tensor, slots: torch.Tensor):
        """Internal FP32 logic for numerical stability."""
        B, N, D = local_feats.shape
        device = local_feats.device
        
        # 1. Cosine Similarity Logits
        # 这里实际上就是 Cross-Attention 的 Score 计算过程
        x_norm = F.normalize(local_feats, dim=-1, eps=1e-6)
        s_norm = F.normalize(slots, dim=-1, eps=1e-6)
        
        # Scale by 10 for better gradient flow
        attn_logits = torch.einsum('bkd,bnd->bkn', s_norm, x_norm) * 10.0

        # 2. Top-K Masking
        patches_per_slot = self.cfg.get('patches_per_slot_attn', 
                                      max(1, self.num_select // self.num_slots))
        patches_per_slot = min(patches_per_slot, N)
        
        topk_val, _ = torch.topk(attn_logits, k=patches_per_slot, dim=-1)
        threshold = topk_val[:, :, -1].unsqueeze(-1) # (B, S, 1)
        
        # 硬截断掩码
        mask_hard = (attn_logits >= threshold).float()

        # 3. STE (Straight-Through Estimator)
        mask_soft = torch.sigmoid(attn_logits / self.temperature)
        mask = (mask_hard - mask_soft).detach() + mask_soft

        # 4. Masked Softmax (Key for stability)
        neg_inf = -1e4 
        masked_logits = attn_logits * mask + neg_inf * (1.0 - mask)
        attn = F.softmax(masked_logits, dim=-1)

        # 5. Weighted Sum (Aggregation)
        # 这就是真正的“提取”步骤：Slot 根据 Attention 权重聚合 Image Patch
        slot_feats = torch.einsum('bkn,bnd->bkd', attn, local_feats)
        
        # 6. Global Mask (Union over slots) for Mixup
        img_space_mask = mask.max(dim=1)[0].unsqueeze(-1) # (B, N, 1)
        
        # 7. Orthogonality Loss
        # 约束 Attention Map 正交，保证不同 Slot 关注不同区域
        gram = torch.bmm(attn, attn.transpose(1, 2))
        I = torch.eye(self.num_slots, device=device).unsqueeze(0).expand(B, -1, -1)
        ortho_loss = (gram - I).abs().mean()
        
        return slot_feats, ortho_loss, img_space_mask

    def forward(self, local_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        B, N, D = local_feats.shape
        dtype = local_feats.dtype
        
        # 归一化输入，有助于训练稳定
        local_feats_norm = self.norm_input(local_feats)
        
        # Expand slots
        slots_expanded = self.slots.expand(B, -1, -1)
        slots_norm = self.norm_slots(slots_expanded)
        
        # --- 核心逻辑 ---
        # 1. 提取特征 (Run heavy lifting in FP32)
        slot_feats_fp32, ortho_loss, img_space_mask_fp32 = self._forward_fp32(
            local_feats_norm.float(), slots_norm.float()
        )
        
        # 2. 转回原始精度
        slot_feats = slot_feats_fp32.to(dtype)
        
        # 3. 特征加工 (FFN)
        # 原来的 MHA 是多余的，这里加一个 FFN 做非线性变换即可
        # 类似于 Transformer Block 里的 FeedForward
        slot_feats = slot_feats + self.ff(slot_feats)
        
        # 返回提取出的特征 (B, Slots, D)，而不是 Queries
        return slot_feats, {'orthogonality': ortho_loss.to(dtype)}, img_space_mask_fp32.to(dtype)


# ============================================================================
# PART 3: FEATURE FUSERS
# ============================================================================

class MeanPoolFuser(BaseFuser):
    def forward(self, selected_feats: torch.Tensor, 
                global_feat: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        return selected_feats.mean(dim=1)


class QueryGuidedAttentionFuser(BaseFuser):
    def __init__(self, input_dim: int, num_heads: int = 4, cfg: Dict = None):
        super().__init__(input_dim, cfg)
        self.mha = nn.MultiheadAttention(input_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, selected_feats: torch.Tensor, 
                text_feats: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if text_feats is None:
            return selected_feats.mean(dim=1)
        
        B = selected_feats.shape[0]
        dtype = selected_feats.dtype
        
        # Determine query
        # text_feats can be (Num_Classes, D) or (B, D)
        if text_feats.dim() == 2 and text_feats.shape[0] != B:
            if self.training and labels is not None:
                query = text_feats[labels].unsqueeze(1) # (B, 1, D)
            else:
                generic_query = text_feats.mean(dim=0, keepdim=True)
                query = generic_query.expand(B, 1, -1)
        else:
            query = text_feats.unsqueeze(1)
            
        query = query.to(dtype)
        
        attn_out, _ = self.mha(query, selected_feats, selected_feats)
        final_feats = self.norm(query + attn_out).squeeze(1)
        return final_feats


# class SelfAttentionFuser(BaseFuser):
#     """Fixed: Added correct arguments to forward matching BaseFuser."""
#     def __init__(self, input_dim: int, num_heads: int = 4, cfg: Dict = None):
#         super().__init__(input_dim, cfg)
#         self.encoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_dim, nhead=num_heads, dim_feedforward=4*input_dim,
#             batch_first=True, activation='gelu'
#         )
#         self.norm = nn.LayerNorm(input_dim)
#     
#     def forward(self, selected_feats: torch.Tensor, 
#                 text_feats: Optional[torch.Tensor] = None, 
#                 labels: Optional[torch.Tensor] = None) -> torch.Tensor:
#         
#         attended = self.encoder_layer(selected_feats)
#         attended = self.norm(attended)
#         return attended.mean(dim=1)


class SelfAttentionFuser(BaseFuser):
    def __init__(self, input_dim, num_heads=8, cfg=None):
        super().__init__(input_dim, cfg)
        
        fuser_ffn_ratio = cfg.get('fuser_ffn_ratio', 8.0) if cfg else 8.0
        
        num_layers = cfg.get('fuser_layers', 2) if cfg else 2
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads,
            dim_feedforward=int(input_dim * fuser_ffn_ratio),
            batch_first=True, activation='gelu',
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(input_dim)
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.GELU(),
            nn.Linear(input_dim*2, input_dim)
        )
        # 依然保持零初始化，保证初始阶段不破坏原特征
        # nn.init.zeros_(self.proj[-1].weight)
        nn.init.normal_(self.proj[-1].weight, std=1e-4)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, selected_feats, global_feat, text_feats=None, labels=None):
        """
        selected_feats: [B, N, D] (Patches)
        global_feat:    [B, D]    (CLIP Original CLS)
        """
        B = selected_feats.shape[0]
        dtype = selected_feats.dtype
        
        # 【修改点 2】拼接：把 Global Feat 变成序列的第一个 Token
        # [B, D] -> [B, 1, D]
        global_token = global_feat.unsqueeze(1)
        
        # 拼接: [B, 1+N, D]
        # 这样 Transformer 里的 Self-Attention 就会计算 Global 与 Patches 的交互
        x = torch.cat((global_token, selected_feats), dim=1)
        
        # 交互：转换为 FP32 以匹配编码器参数 dtype
        x_fp32 = x.float()
        x_fp32 = self.encoder(x_fp32)
        x_fp32 = self.norm(x_fp32)
        
        # 转回原始精度
        x = x_fp32.to(dtype)
        
        # 取出第一个 Token (也就是被 Refine 过的 Global Token)
        x_aggregated = x[:, 0, :] 
        
        # 投影：转换为 FP32 计算，再转回原始精度
        x_out_fp32 = self.proj(x_aggregated.float())
        x_out = x_out_fp32.to(dtype)
        
        return x


class SimpleAdapterFuser(BaseFuser):
    def __init__(self, input_dim, num_heads=8, cfg=None):
        super().__init__(input_dim, cfg)
        
        adapter_ratio = cfg.get('adapter_ratio', 0.5) if cfg else 0.5
        hidden_dim = int(input_dim * adapter_ratio)
        
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim, bias=False),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(input_dim)
        
        # 零初始化输出层，保证初始阶段不破坏原特征
        nn.init.zeros_(self.adapter[-2].weight)
        # nn.init.zeros_(self.adapter[-2].bias)

    def forward(self, selected_feats, global_feat, text_feats=None, labels=None):
        """
        selected_feats: [B, N, D] (Patches) - not used in this simple adapter
        global_feat:    [B, D]    (CLIP Original CLS)
        """
        dtype = global_feat.dtype
        
        # 简单的残差adapter: global_feat + adapter(global_feat)
        global_feat_fp32 = global_feat.float()
        adapter_out = self.adapter(global_feat_fp32)
        x = self.norm(adapter_out)
        
        # 转回原始精度
        x = x.to(dtype)
        
        return x


class SharedAdapterFuser(BaseFuser):
    """
    Shared Adapter that processes both CLS token and patch tokens with the same adapter.
    The adapter is shared, so CLS and patches are in the same feature space.
    Returns adapted CLS and adapted patches for separate loss computation.
    """
    def __init__(self, input_dim, num_heads=8, cfg=None):
        super().__init__(input_dim, cfg)
        
        adapter_ratio = cfg.get('adapter_ratio', 0.5) if cfg else 0.5
        hidden_dim = int(input_dim * adapter_ratio)
        
        # Shared adapter for both CLS and patches
        self.shared_adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim, bias=False),
            nn.GELU()
        )
        
        # Separate normalization layers for CLS and patches
        self.cls_norm = nn.LayerNorm(input_dim)
        self.patch_norm = nn.LayerNorm(input_dim)
        
        # 零初始化输出层，保证初始阶段不破坏原特征
        nn.init.zeros_(self.shared_adapter[-2].weight)
    
    def forward(self, selected_feats, global_feat, text_feats=None, labels=None):
        """
        selected_feats: [B, N, D] (Patches)
        global_feat:    [B, D]    (CLIP Original CLS)
        
        Returns:
            adapted_cls: [B, D] - Adapted CLS token
            adapted_patches: [B, N, D] - Adapted patch tokens
        """
        dtype = global_feat.dtype
        
        # Process CLS token with shared adapter
        global_feat_fp32 = global_feat.float()
        cls_adapter_out = self.shared_adapter(global_feat_fp32)
        adapted_cls = self.cls_norm(cls_adapter_out)
        adapted_cls = adapted_cls.to(dtype)
        
        # Process patch tokens with the SAME shared adapter
        selected_feats_fp32 = selected_feats.float()
        B, N, D = selected_feats_fp32.shape
        selected_feats_flat = selected_feats_fp32.reshape(B * N, D)
        
        # Use the same adapter
        patch_adapter_out = self.shared_adapter(selected_feats_flat)
        patch_adapter_out = patch_adapter_out.reshape(B, N, D)
        adapted_patches = self.patch_norm(patch_adapter_out)
        adapted_patches = adapted_patches.to(dtype)
        
        return adapted_cls, adapted_patches


class CrossAttentionFuser(BaseFuser):
    def __init__(self, input_dim, num_heads=8, cfg=None):
        super().__init__(input_dim, cfg)
        
        fuser_ffn_ratio = cfg.get('fuser_ffn_ratio', 4.0) if cfg else 4.0
        
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(input_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, int(input_dim * fuser_ffn_ratio)),
            nn.GELU(),
            nn.Linear(int(input_dim * fuser_ffn_ratio), input_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)
        
        # 零初始化投影层，保证初始阶段不破坏原特征
        nn.init.zeros_(self.ffn[-1].weight)
        # nn.init.normal_(self.ffn[-1].weight, std=1e-4)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, selected_feats, global_feat, text_feats=None, labels=None):
        """
        selected_feats: [B, N, D] (Patches)
        global_feat:    [B, D]    (CLIP Original CLS)
        """
        dtype = selected_feats.dtype
        
        # Q: CLS, K/V: patches
        q = global_feat.unsqueeze(1).float()  # [B, 1, D]
        k = v = selected_feats.float()        # [B, N, D]
        
        # Cross-attention: CLS 作为 query，patches 作为 key 和 value
        attn_out, _ = self.attn(q, k, v)
        x = self.norm1(q + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # 转回原始精度并squeeze
        x = x.squeeze(1).to(dtype)
        
        return x



# ============================================================================
# PART 4: LOSS FUNCTIONS (FP32 Enforced)
# ============================================================================

def compute_diversity_loss(selector_type: str, selector) -> torch.Tensor:
    if hasattr(selector, 'aux_loss') and 'diversity' in selector.aux_loss:
        return selector.aux_loss['diversity']
    return torch.tensor(0.0, device=next(selector.parameters()).device)

@autocast(enabled=False)
def compute_semantic_exclusion_loss(final_feat: torch.Tensor, 
                                   pos_text_feat: torch.Tensor,
                                   neg_text_feats: torch.Tensor,
                                   margin: float = 0.1, 
                                   logit_scale: float = 100.0) -> torch.Tensor:
    """
    Robust Ranking Loss in FP32.
    """
    # 1. Cast inputs to Float
    final_feat = final_feat.float()
    pos_text_feat = pos_text_feat.float()
    neg_text_feats = neg_text_feats.float()

    # 2. Normalize
    final_feat = F.normalize(final_feat, dim=-1, eps=1e-8)
    pos_text_feat = F.normalize(pos_text_feat, dim=-1, eps=1e-8)
    neg_text_feats = F.normalize(neg_text_feats, dim=-1, eps=1e-8)

    # 3. Similarities & Scaling
    pos_sim = (final_feat * pos_text_feat).sum(dim=1) * logit_scale
    neg_sims = torch.einsum("bd,bnd->bn", final_feat, neg_text_feats) * logit_scale

    # 4. Stable LogSumExp
    neg_score = torch.logsumexp(neg_sims, dim=1)
    
    # 5. Margin
    scaled_margin = margin
    
    # 6. Loss
    loss = F.relu(neg_score - pos_sim + scaled_margin)
    
    return loss.mean()

@autocast(enabled=False)
def compute_redundancy_loss(selected_feats: torch.Tensor) -> torch.Tensor:
    """Robust Redundancy Loss in FP32."""
    selected_feats = selected_feats.float()
    B, K, D = selected_feats.shape
    
    selected_feats_norm = F.normalize(selected_feats, dim=-1, eps=1e-8)
    gram = torch.bmm(selected_feats_norm, selected_feats_norm.transpose(1, 2))
    
    diag_mask = torch.eye(K, device=selected_feats.device).unsqueeze(0).expand(B, -1, -1)
    off_diag = gram * (1 - diag_mask)
    
    return off_diag.abs().mean()

@autocast(enabled=False)
def compute_mixup_invariance_loss(final_feat: torch.Tensor,
                                 local_feats: torch.Tensor,
                                 bg_mask: torch.Tensor,
                                 text_feats: torch.Tensor,
                                 labels: Optional[torch.Tensor] = None,
                                 logit_scale: float = 100.0,
                                 alpha: float = 0.2) -> torch.Tensor:
    """Robust Mixup Loss in FP32."""
    final_feat = final_feat.float()
    local_feats = local_feats.float()
    bg_mask = bg_mask.float()
    text_feats = text_feats.float()
    
    B = final_feat.shape[0]
    
    # Foreground Feature
    bg_mask_sum = bg_mask.sum(1) + 1e-8
    fg_feat = (local_feats * bg_mask).sum(1) / bg_mask_sum
    
    # Background Feature
    bg_weights = (1.0 - bg_mask)
    bg_weights_sum = bg_weights.sum(1) + 1e-8
    bg_feat_pooled = (local_feats * bg_weights).sum(1) / bg_weights_sum
    
    # Causal Intervention: Detach Background
    bg_feat_pooled = bg_feat_pooled.detach()
    
    # Shuffle & Mix
    shuffle_idx = torch.randperm(B, device=final_feat.device)
    shuffled_bg = bg_feat_pooled[shuffle_idx]
    mixed_feat = fg_feat + alpha * shuffled_bg
    
    # Logits
    fg_feat_norm = F.normalize(fg_feat, dim=-1, eps=1e-8)
    mixed_feat_norm = F.normalize(mixed_feat, dim=-1, eps=1e-8)
    
    orig_logits = logit_scale * fg_feat_norm @ text_feats.T
    mixed_logits = logit_scale * mixed_feat_norm @ text_feats.T
    
    # Stable KL Divergence
    # Shift logits for stability
    orig_logits = orig_logits - orig_logits.max(dim=1, keepdim=True)[0].detach()
    mixed_logits = mixed_logits - mixed_logits.max(dim=1, keepdim=True)[0].detach()
    
    orig_probs = F.softmax(orig_logits, dim=1)
    mixed_log_probs = F.log_softmax(mixed_logits, dim=1)
    
    loss = F.kl_div(mixed_log_probs, orig_probs.detach(), reduction='batchmean')
    return loss

@autocast(enabled=False)
def compute_intra_class_consistency_loss(final_feats: torch.Tensor,
                                       labels: torch.Tensor,
                                       temperature: float = 0.1) -> torch.Tensor:
    """
    Intra-class Consistency Loss in FP32.
    Encourages samples from the same class to have similar final features.
    """
    # 1. Cast inputs to Float
    final_feats = final_feats.float()
    labels = labels.float()
    
    B, D = final_feats.shape
    
    # 2. Normalize features
    final_feats_norm = F.normalize(final_feats, dim=-1, eps=1e-8)
    
    # 3. Compute pairwise similarity matrix
    # sim_matrix[i, j] = similarity between sample i and sample j
    sim_matrix = torch.mm(final_feats_norm, final_feats_norm.t())  # [B, B]
    
    # 4. Create class mask: same_class_mask[i, j] = 1 if labels[i] == labels[j]
    labels_expanded_1 = labels.unsqueeze(1)  # [B, 1]
    labels_expanded_2 = labels.unsqueeze(0)  # [1, B]
    same_class_mask = (labels_expanded_1 == labels_expanded_2).float()  # [B, B]
    
    # Remove diagonal (self-similarity)
    diag_mask = torch.eye(B, device=final_feats.device).unsqueeze(0)
    same_class_mask = same_class_mask * (1 - diag_mask)
    
    # 5. Compute loss: maximize similarity for same-class pairs
    # Use temperature scaling
    scaled_sim = sim_matrix / temperature
    
    # Apply mask to only consider same-class pairs
    masked_sim = scaled_sim * same_class_mask
    
    # Count number of same-class pairs (excluding diagonal)
    num_pairs = same_class_mask.sum()
    
    if num_pairs > 0:
        # No same-class pairs in batch, return zero loss
        return torch.tensor(0.0, device=final_feats.device)
    
    # Loss: negative mean similarity (we want to maximize similarity)
    loss = -masked_sim.sum() / num_pairs
    
    return loss


@autocast(enabled=False)
def compute_locoop_ood_loss(local_feats: torch.Tensor,
                           text_feats: torch.Tensor,
                           labels: torch.Tensor,
                           top_k: int = 200,
                           logit_scale: float = 100.0) -> torch.Tensor:
    """
    LoCoOp-style OOD regularization loss.
    Identifies OOD patches by checking if their top-k predictions don't contain the ground truth label,
    then maximizes the entropy of these OOD patches.
    
    Args:
        local_feats: [B, N, D] - Local patch features
        text_feats: [C, D] - Text features for all classes
        labels: [B] - Ground truth labels
        top_k: Number of top predictions to check
        logit_scale: Temperature scaling for logits
    
    Returns:
        OOD regularization loss (negative entropy of OOD patches)
    """
    # Cast to FP32 for stability
    local_feats = local_feats.float()
    text_feats = text_feats.float()
    
    B, N, D = local_feats.shape
    C = text_feats.shape[0]
    
    # Normalize features
    local_feats_norm = F.normalize(local_feats, dim=-1, eps=1e-8)
    text_feats_norm = F.normalize(text_feats, dim=-1, eps=1e-8)
    
    # Compute logits for all patches: [B, N, C]
    # Each patch gets logits for all classes
    patch_logits = logit_scale * torch.einsum('bnd,cd->bnc', local_feats_norm, text_feats_norm)
    
    # Reshape to [B*N, C] for easier processing
    patch_logits_flat = patch_logits.reshape(B * N, C)
    
    # Repeat labels for each patch: [B*N]
    labels_repeated = labels.unsqueeze(1).expand(-1, N).reshape(B * N)
    
    # Get top-k predictions for each patch
    pred_topk = torch.topk(patch_logits_flat, k=top_k, dim=1)[1]  # [B*N, top_k]
    
    # Check if ground truth label is in top-k predictions
    # contains_label[i] = True if labels[i] is in pred_topk[i]
    contains_label = pred_topk.eq(labels_repeated.unsqueeze(1)).any(dim=1)  # [B*N]
    
    # Select OOD patches (those where ground truth is NOT in top-k)
    ood_mask = ~contains_label  # [B*N]
    
    # Get probabilities for OOD patches
    patch_probs = F.softmax(patch_logits_flat, dim=-1)  # [B*N, C]
    ood_probs = patch_probs[ood_mask]  # [num_ood_patches, C]
    
    # If no OOD patches found, return zero loss
    if ood_probs.shape[0] == 0:
        return torch.tensor(0.0, device=local_feats.device)
    
    # Compute entropy of OOD patches
    # Entropy = -sum(p * log(p))
    entropy = -torch.sum(ood_probs * torch.log(ood_probs + 1e-8), dim=1)  # [num_ood_patches]
    
    # Return negative mean entropy (we want to maximize entropy)
    return -entropy.mean()


@autocast(enabled=False)
def compute_locoop_cls_loss(cls_feats: torch.Tensor,
                          text_feats: torch.Tensor,
                          labels: torch.Tensor,
                          top_k: int = 200,
                          logit_scale: float = 100.0) -> torch.Tensor:
    """
    LoCoOp-style OOD regularization loss for CLS token.
    Similar to patch-based loss but applied to CLS token only.
    
    Args:
        cls_feats: [B, D] - CLS token features
        text_feats: [C, D] - Text features for all classes
        labels: [B] - Ground truth labels
        top_k: Number of top predictions to check
        logit_scale: Temperature scaling for logits
    
    Returns:
        OOD regularization loss for CLS token
    """
    # Cast to FP32 for stability
    cls_feats = cls_feats.float()
    text_feats = text_feats.float()
    
    B, D = cls_feats.shape
    C = text_feats.shape[0]
    
    # Normalize features
    cls_feats_norm = F.normalize(cls_feats, dim=-1, eps=1e-8)
    text_feats_norm = F.normalize(text_feats, dim=-1, eps=1e-8)
    
    # Compute logits for CLS tokens: [B, C]
    cls_logits = logit_scale * torch.einsum('bd,cd->bc', cls_feats_norm, text_feats_norm)
    
    # Get top-k predictions for each sample
    pred_topk = torch.topk(cls_logits, k=top_k, dim=1)[1]  # [B, top_k]
    
    # Check if ground truth label is in top-k predictions
    # contains_label[i] = True if labels[i] is in pred_topk[i]
    contains_label = pred_topk.eq(labels.unsqueeze(1)).any(dim=1)  # [B]
    
    # Select OOD samples (those where ground truth is NOT in top-k)
    ood_mask = ~contains_label  # [B]
    
    # Get probabilities for OOD samples
    cls_probs = F.softmax(cls_logits, dim=-1)  # [B, C]
    ood_probs = cls_probs[ood_mask]  # [num_ood_samples, C]
    
    # If no OOD samples found, return zero loss
    if ood_probs.shape[0] == 0:
        return torch.tensor(0.0, device=cls_feats.device)
    
    # Compute entropy of OOD samples
    # Entropy = -sum(p * log(p))
    entropy = -torch.sum(ood_probs * torch.log(ood_probs + 1e-8), dim=1)  # [num_ood_samples]
    
    # Return negative mean entropy (we want to maximize entropy)
    return -entropy.mean()


@autocast(enabled=False)
def compute_locoop_dual_loss(cls_feats: torch.Tensor,
                            patch_feats: torch.Tensor,
                            text_feats: torch.Tensor,
                            labels: torch.Tensor,
                            top_k: int = 200,
                            logit_scale: float = 100.0,
                            lambda_cls: float = 0.5,
                            lambda_patch: float = 0.5) -> torch.Tensor:
    """
    Combined LoCoOp OOD regularization loss for both CLS and patch tokens.
    
    Args:
        cls_feats: [B, D] - CLS token features
        patch_feats: [B, N, D] - Patch token features
        text_feats: [C, D] - Text features for all classes
        labels: [B] - Ground truth labels
        top_k: Number of top predictions to check
        logit_scale: Temperature scaling for logits
        lambda_cls: Weight for CLS loss
        lambda_patch: Weight for patch loss
    
    Returns:
        Combined OOD regularization loss
    """
    # Compute CLS loss
    cls_loss = compute_locoop_cls_loss(
        cls_feats, text_feats, labels, top_k=top_k, logit_scale=logit_scale
    )
    
    # Compute patch loss
    patch_loss = compute_locoop_ood_loss(
        patch_feats, text_feats, labels, top_k=top_k, logit_scale=logit_scale
    )
    
    # Combine losses
    total_loss = lambda_cls * cls_loss + lambda_patch * patch_loss
    
    return total_loss


# ============================================================================
# PART 5: MAIN MODEL - CustomCLIP
# ============================================================================

class ModularCustomCLIP(nn.Module):
    def __init__(self, cfg: Dict, classnames: list, clip_model, class_negatives: Dict = None):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.num_classes = len(classnames)
        self.device = cfg.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.lab2idx = {c: i for i, c in enumerate(self.classnames)}
        self.ood_text_feats = {}
        self.has_ood_map = False
        self.class_negatives = class_negatives or {}
        
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        # Freeze backbone
        for p in self.image_encoder.parameters(): p.requires_grad = False
        for p in self.text_encoder.parameters(): p.requires_grad = False
        
        try: self.feat_dim = self.image_encoder.output_dim
        except: self.feat_dim = clip_model.ln_final.weight.shape[0]
        
        self._build_components()
        self._cache_text_features()
        self._cache_negative_text_features()
        print(f"✓ ModularCustomCLIP initialized. Dtype: {self.dtype}")

    def _build_components(self):
        # Selector disabled - use all patches directly
        # s_type = self.cfg.get('selector_type', 'mlp')
        # num_sel = self.cfg.get('num_select', 49)
        # 
        # if s_type == 'mlp':
        #     self.selector = MultiHeadMLPSelector(self.feat_dim, num_sel, self.cfg.get('num_heads_selector', 8), self.cfg)
        # elif s_type == 'slot':
        #     self.selector = SparseSlotAttentionSelector(self.feat_dim, num_sel, self.cfg.get('num_slots', 8), self.cfg)
        # else:
        #     self.selector = IdentitySelector(self.feat_dim, num_sel, self.cfg)
        # # self.selector = self.selector.to(self.dtype)
        
        f_type = self.cfg.get('fuser_type', 'mean')
        if f_type == 'query_attn':
            self.fuser = QueryGuidedAttentionFuser(self.feat_dim, self.cfg.get('num_heads_fuser', 8), self.cfg)
        elif f_type == 'self_attn':
            self.fuser = SelfAttentionFuser(self.feat_dim, self.cfg.get('num_heads_fuser', 8), self.cfg)
        elif f_type == 'cross_attn':
            self.fuser = CrossAttentionFuser(self.feat_dim, self.cfg.get('num_heads_fuser', 8), self.cfg)
        elif f_type == 'simple_adapter':
            self.fuser = SimpleAdapterFuser(self.feat_dim, self.cfg.get('num_heads_fuser', 8), self.cfg)
        elif f_type == 'shared_adapter':
            self.fuser = SharedAdapterFuser(self.feat_dim, self.cfg.get('num_heads_fuser', 8), self.cfg)
        else:
            self.fuser = MeanPoolFuser(self.feat_dim)
        # self.fuser = self.fuser.to(self.dtype)

    def _cache_text_features(self):
        templates = self.cfg.get('templates', ["a photo of a {}"])
        if isinstance(templates, str): templates = [templates]
        
        # Check if text features path is provided
        text_features_path = self.cfg.get('text_features_path', '')
        if text_features_path and os.path.exists(text_features_path):
            # Load pre-computed text features
            # Supports two formats:
            # 1. .pt file: direct PyTorch tensor [num_classes, feat_dim]
            # 2. .npy file: NumPy array [num_classes, feat_dim]
            
            file_ext = os.path.splitext(text_features_path)[1].lower()
            
            if file_ext == '.pt':
                loaded_data = torch.load(text_features_path)
                # Check if it's a dict or direct tensor
                if isinstance(loaded_data, dict):
                    loaded_features = loaded_data['text_features']
                else:
                    loaded_features = loaded_data
            elif file_ext == '.npy':
                loaded_features = np.load(text_features_path)
                loaded_features = torch.from_numpy(loaded_features).float()
            else:
                raise ValueError(f"Unsupported file format: {file_ext}. Expected .pt or .npy")
            
            # Check shape matches number of classes
            num_classes_expected = len(self.classnames)
            if loaded_features.shape[0] != num_classes_expected:
                raise ValueError(
                    f"Text features shape mismatch: expected {num_classes_expected} classes, "
                    f"but got {loaded_features.shape[0]} classes. "
                    f"Features shape: {loaded_features.shape}"
                )
            
            self.text_features = loaded_features.to(self.device).type(self.dtype)
            self.register_buffer('_text_features', self.text_features)
            return
        
        # Otherwise, compute text features from scratch
        text_features_list = []
        with torch.no_grad():
            for classname in self.classnames:
                classname = classname.replace('_', ' ')
                texts = [t.format(classname) for t in templates]
                texts = clip.tokenize(texts).to(self.device)
                text_embeddings = self.text_encoder.encode_text(texts)
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
                text_feat = text_embeddings.mean(dim=0)
                text_feat = text_feat / text_feat.norm()
                text_features_list.append(text_feat)
        
        self.text_features = torch.stack(text_features_list, dim=0).to(self.device).type(self.dtype)
        self.register_buffer('_text_features', self.text_features)
    
    def _cache_negative_text_features(self):
        """
        Cache negative text features for all classes in class_negatives.
        Maps each class to its negative class features for efficient retrieval during training.
        """
        if not self.class_negatives:
            self.negative_text_features = {}
            self.label_to_neg_features = {}
            return
            
        self.negative_text_features = {}
        templates = self.cfg.get('templates', ["a photo of a {}"])
        if isinstance(templates, str): templates = [templates]
        
        with torch.no_grad():
            for classname, neg_classnames in self.class_negatives.items():
                if classname not in self.lab2idx:
                    continue  # Skip if class not in our dataset
                    
                neg_feats_list = []
                for neg_classname in neg_classnames:
                    # Format negative class name with templates
                    neg_classname = neg_classname.replace('_', ' ')
                    texts = [t.format(neg_classname) for t in templates]
                    
                    # Tokenize and encode
                    tokens = clip.tokenize(texts).to(self.device)
                    text_embeddings = self.text_encoder.encode_text(tokens)
                    
                    # Normalize and average
                    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
                    neg_feat = text_embeddings.mean(dim=0)
                    neg_feat = neg_feat / neg_feat.norm()
                    
                    neg_feats_list.append(neg_feat)
                
                # Stack and store for this class
                if neg_feats_list:
                    self.negative_text_features[classname] = torch.stack(neg_feats_list, dim=0).to(self.device).type(self.dtype)
        
        # Create label-to-negative-features mapping for efficient lookup during forward pass
        self.label_to_neg_features = {}
        for classname, feats in self.negative_text_features.items():
            if classname in self.lab2idx:
                label = self.lab2idx[classname]
                self.label_to_neg_features[label] = feats
        
        print(f"✓ Cached negative text features for {len(self.negative_text_features)} classes")

    def set_ood_classmap(self, ood_map: Dict[str, list], templates: Optional[list] = None):
        # ... (Same as before, skipped for brevity)
        pass

    def _compute_llm_negatives_loss_wrapper(self, final_feats, neg_text_tokens, pos_text_feats):
        # Wrapper to pass logit scale
        scale_val = self.logit_scale.exp().item()
        return compute_semantic_exclusion_loss(
            final_feats, pos_text_feats, neg_text_tokens,
            margin=self.cfg.get('margin', 0.1),
            logit_scale=scale_val
        )

    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image into global and local features."""
        image_features, local_features = self.image_encoder(image.type(self.dtype))
        B = image_features.shape[0]
        local_features = local_features.reshape(B, -1, self.feat_dim)
        return image_features, local_features
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text into features."""
        return self.text_encoder.encode_text(text)

    def forward(self, image: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                negative_text_tokens: Optional[torch.Tensor] = None) -> Dict:

        B = image.shape[0]

        # 1. Encode Image (FP16)
        image_features, local_features = self.encode_image(image)
        
        # image_features = F.normalize(image_features, dim=-1)
        # local_features = F.normalize(local_features, dim=-1)
        
        # 2. Text Features
        text_feats = self._text_features.type(self.dtype)
        if labels is not None:
            pos_text_feat = text_feats[labels]
        else:
            pos_text_feat = text_feats.mean(dim=0, keepdim=True).expand(B, -1)
            
        # 3. Fusion (No selector, use all patches directly)
        # Check if using SharedAdapterFuser
        if isinstance(self.fuser, SharedAdapterFuser):
            # Shared adapter returns adapted CLS and adapted patches separately
            # Both use the SAME adapter, so they're in the same feature space
            adapted_cls, adapted_patches = self.fuser(local_features, global_feat=image_features)
            adapted_patches_for_loss = self.cfg.get('residual_coef', 0.2) * adapted_patches + local_features
            # For final classification, use adapted CLS (or combine with patches)
            # Here we use adapted CLS for simplicity
            # final_feats = adapted_cls
            global_features = image_features + self.cfg.get('residual_coef', 0.2) * adapted_cls
            final_feats = global_features / global_features.norm(dim=-1, keepdim=True)
            
            # Store adapted features for OOD loss computation
            # adapted_cls_for_loss = final_feats
            # adapted_patches_for_loss = local_features
        else:
            # Standard fuser returns single feature
            final_feats = self.fuser(local_features, global_feat=image_features)
            global_features = image_features + final_feats
            final_feats = global_features / global_features.norm(dim=-1, keepdim=True)
            
            # Use original features for OOD loss
            adapted_cls_for_loss = image_features
            adapted_patches_for_loss = local_features
        
        # final_feats = F.normalize(global_features, dim=-1)
        # final_feats = F.normalize(final_feats, dim=-1)

        # 4. Logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * final_feats @ text_feats.T
        
        # 5. Losses
        aux_losses = {}
        
        # B. LLM Negatives - now uses cached negative features
        if negative_text_tokens is None and self.cfg.get('use_llm_negatives', False) and labels is not None:
            scale_val = logit_scale.item()
            
            # 收集 Batch 中每个样本的负特征
            valid_indices = []
            raw_neg_list = []
            
            for i, label in enumerate(labels):
                lbl = label.item()
                if hasattr(self, 'label_to_neg_features') and lbl in self.label_to_neg_features:
                    neg_feats = self.label_to_neg_features[lbl]
                    raw_neg_list.append(neg_feats)
                    valid_indices.append(i)
            
            if valid_indices:
                # 直接拼接，无需padding（因为每个类的负样本数量相同）
                neg_feats_batch = torch.stack(raw_neg_list, dim=0)
                
                # 提取对应的特征和正样本
                valid_final_feats = final_feats[valid_indices]
                valid_pos_text_feat = pos_text_feat[valid_indices]
                
                llm_loss = compute_semantic_exclusion_loss(
                    valid_final_feats, valid_pos_text_feat, neg_feats_batch,
                    margin=self.cfg.get('margin', 0.1),
                    logit_scale=scale_val
                )
                
                aux_losses['llm_negatives'] = self.cfg.get('lambda_llm_negatives', 0.05) * llm_loss
            
        # C. Semantic Exclusion (All other classes)
        if self.cfg.get('use_semantic_exclusion', False) and labels is not None:
            mask = torch.ones(B, self.num_classes, dtype=torch.bool, device=self.device)
            mask[torch.arange(B), labels] = False
            neg_text_feats = text_feats.unsqueeze(0).expand(B, -1, -1)[mask].reshape(B, -1, self.feat_dim)
            
            scale_val = logit_scale.item()
            sem_loss = compute_semantic_exclusion_loss(
                final_feats, pos_text_feat, neg_text_feats,
                margin=self.cfg.get('margin', 0.1),
                logit_scale=scale_val
            )
            aux_losses['semantic_exclusion'] = self.cfg.get('lambda_llm_negatives', 0.5) * sem_loss
            
        # D. Mixup (Disabled - requires bg_mask from selector)
        # if self.cfg.get('use_mixup_invariance', False):
        #     scale_val = logit_scale.item()
        #     mixup_loss = compute_mixup_invariance_loss(
        #         final_feats, local_features, bg_mask,
        #         text_feats=text_feats,
        #         labels=labels,
        #         logit_scale=scale_val,
        #         alpha=self.cfg.get('mixup_alpha', 0.2)
        #     )
        #     aux_losses['mixup_invariance'] = self.cfg.get('lambda_mixup', 0.1) * mixup_loss
        
        # E. Intra-class Consistency
        if self.cfg.get('use_intra_class_consistency', False) and labels is not None:
            scale_val = logit_scale.item()
            intra_loss = compute_intra_class_consistency_loss(
                final_feats, labels,
                temperature=self.cfg.get('intra_class_temp', 0.1)
            )
            aux_losses['intra_class_consistency'] = self.cfg.get('lambda_intra_class', 0.1) * intra_loss
        
        # F. LoCoOp-style OOD regularization
        if self.cfg.get('use_locoop_ood', False) and labels is not None:
            scale_val = logit_scale.item()
            top_k = self.cfg.get('locoop_topk', 200)
            
            # Check if using shared adapter (separate CLS and patch losses)
            if isinstance(self.fuser, SharedAdapterFuser):
                # Shared adapter returns adapted CLS and adapted patches separately
                # Both use SAME adapter, so they're in the same feature space
                
                # CLS token: used for classification only, NO OOD regularization
                # (all training samples are ID, CLS should match ground truth)
                
                # Patch tokens: apply OOD regularization
                # (patches have foreground/ID and background/OOD distinction)
                
                # Apply OOD regularization ONLY to patch tokens
                ood_loss = compute_locoop_ood_loss(
                    adapted_patches_for_loss,  # Only patches
                    text_feats,
                    labels,
                    top_k=top_k,
                    logit_scale=scale_val
                )
                aux_losses['locoop_ood'] = self.cfg.get('lambda_locoop_ood', 0.1) * ood_loss
            else:
                # Use standard patch-based loss
                ood_loss = compute_locoop_ood_loss(
                    adapted_patches_for_loss,  # Use adapted patches or selected features
                    text_feats,
                    labels,
                    top_k=top_k,
                    logit_scale=scale_val
                )
                aux_losses['locoop_ood'] = self.cfg.get('lambda_locoop_ood', 0.1) * ood_loss
        # import pdb
        # pdb.set_trace()
        return {
            'logits': logits,
            'aux_losses': aux_losses,
            'selected_feats': adapted_patches_for_loss,
            'final_feats': final_feats,
            'global_features': global_features,
            'local_features': local_features,
            'global_feature': image_features,
        }


# ============================================================================
# Helper function to build model
# ============================================================================

def build_modular_model(cfg: Dict, classnames: list, clip_model, class_negatives: Dict = None):
    """Factory function to create modular model."""
    return ModularCustomCLIP(cfg, classnames, clip_model, class_negatives=class_negatives)


if __name__ == '__main__':
    # Quick test
    import clip as clip_module
    
    cfg = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'selector_type': 'mlp',  # or 'slot'
        'fuser_type': 'query_attn',  # or 'mean', 'self_attn'
        'num_select': 49,
        'num_heads_selector': 4,
        'num_heads_fuser': 4,
        'templates': ["a photo of a"],
        'use_semantic_exclusion': True,
    }
    
    classnames = ["dog", "cat", "bird"]
    clip_model, _ = clip_module.load("ViT-B/16", device=cfg['device'])
    
    model = build_modular_model(cfg, classnames, clip_model)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224).to(cfg['device'])
    labels = torch.tensor([0, 1]).to(cfg['device'])
    
    output = model(x, labels)
    print(f"✓ Forward pass successful")
    print(f"  logits shape: {output['logits'].shape}")
    print(f"  aux_losses: {output['aux_losses'].keys()}")