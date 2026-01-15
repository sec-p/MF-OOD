"""
Hybrid CLIP Model combining Multi-modal and Pure Visual approaches
- Multi-modal branch: Uses fuser to fuse features, computes similarity with text features
- Pure visual branch: Uses cls+selected patch mean, computes similarity with prototypes
- Combines both scores for OOD detection
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import clip

from .model_modular import (
    BaseSelector, BaseFuser,
    IdentitySelector, MultiHeadMLPSelector, SparseSlotAttentionSelector,
    MeanPoolFuser, QueryGuidedAttentionFuser, SelfAttentionFuser
)
from .model_prototype import PrototypeCLIP


class HybridCLIP(nn.Module):
    """
    Hybrid CLIP Model combining multi-modal and pure visual approaches.
    
    Multi-modal branch:
    - Selector selects features
    - Fuser fuses selected features with global feature
    - Computes similarity with text features
    
    Pure visual branch:
    - Uses cls token + mean of selected patches
    - Computes similarity with visual prototypes
    
    Both branches can be combined for OOD detection.
    """
    
    def __init__(self, cfg: Dict, classnames: list, clip_model, class_negatives: Dict = None):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.num_classes = len(classnames)
        self.device = cfg.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.class_negatives = class_negatives or {}
        
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        # Freeze backbone
        for p in self.image_encoder.parameters(): p.requires_grad = False
        for p in self.text_encoder.parameters(): p.requires_grad = False
        
        # Get feature dimensions from ViT
        try:
            self.raw_feat_dim = self.image_encoder.width
        except:
            self.raw_feat_dim = clip_model.ln_final.weight.shape[0]
        
        try:
            self.proj_feat_dim = self.image_encoder.output_dim
        except:
            self.proj_feat_dim = clip_model.ln_final.weight.shape[0]
        
        self.feat_dim = self.proj_feat_dim
        
        # Build components
        self._build_selector()
        self._build_fuser()
        
        # Initialize prototypes for pure visual branch (768D)
        self.prototypes = nn.Parameter(
            torch.empty(self.num_classes, self.raw_feat_dim, dtype=self.dtype),
            requires_grad=False
        )
        
        # Add projection layer for visual branch (768D -> 512D)
        # This allows visual features to be used with text features (512D) for local scores
        self.visual_proj = nn.Linear(self.raw_feat_dim, self.proj_feat_dim)
        # Initialize to identity (no-op initially)
        nn.init.eye_(self.visual_proj.weight)
        nn.init.zeros_(self.visual_proj.bias)
        self.visual_proj = self.visual_proj.to(self.dtype)
        
        # Cache text features for multi-modal branch (512D)
        self._cache_text_features()
        self._cache_negative_text_features()
        
        # OOD score combination weight
        self.multimodal_weight = cfg.get('multimodal_weight', 0.5)
        self.visual_weight = cfg.get('visual_weight', 0.5)
        
        # Add attribute for detection_util.py to identify HybridCLIP
        self.ood_score_combined = None  # Placeholder, computed in forward
        
        print(f"✓ HybridCLIP initialized. Dtype: {self.dtype}")
        print(f"  Multi-modal weight: {self.multimodal_weight}, Visual weight: {self.visual_weight}")
    
    def _build_selector(self):
        """Build selector component for feature selection."""
        s_type = self.cfg.get('selector_type', 'mlp')
        num_sel = self.cfg.get('num_select', 64)
        
        if s_type == 'mlp':
            self.selector = MultiHeadMLPSelector(
                self.proj_feat_dim, num_sel, 
                self.cfg.get('num_heads_selector', 8), self.cfg
            )
        elif s_type == 'slot':
            self.selector = SparseSlotAttentionSelector(
                self.proj_feat_dim, num_sel, 
                self.cfg.get('num_slots', 8), self.cfg
            )
        else:
            self.selector = IdentitySelector(self.proj_feat_dim, num_sel, self.cfg)
        
        self.selector = self.selector.to(self.dtype)
    
    def _build_fuser(self):
        """Build fuser component for multi-modal branch (works in projected space - 512D)."""
        f_type = self.cfg.get('fuser_type', 'mean')
        
        if f_type == 'query_attn':
            self.fuser = QueryGuidedAttentionFuser(
                self.proj_feat_dim,  # Use projected dimension (512)
                self.cfg.get('num_heads_fuser', 8), self.cfg
            )
        elif f_type == 'self_attn':
            self.fuser = SelfAttentionFuser(
                self.proj_feat_dim,  # Use projected dimension (512)
                self.cfg.get('num_heads_fuser', 8), self.cfg
            )
        else:
            self.fuser = MeanPoolFuser(self.proj_feat_dim)  # Use projected dimension (512)
        
        self.fuser = self.fuser.to(self.dtype)
    
    def _cache_text_features(self):
        """Cache text features for multi-modal branch."""
        templates = self.cfg.get('templates', ["a photo of a {}"])
        if isinstance(templates, str): 
            templates = [templates]
        
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
        """Cache negative text features for multi-modal branch."""
        if not self.class_negatives:
            self.negative_text_features = {}
            self.label_to_neg_features = {}
            return
        
        self.negative_text_features = {}
        templates = self.cfg.get('templates', ["a photo of a {}"])
        if isinstance(templates, str): 
            templates = [templates]
        
        with torch.no_grad():
            for classname, neg_classnames in self.class_negatives.items():
                neg_feats_list = []
                for neg_classname in neg_classnames:
                    neg_classname = neg_classname.replace('_', ' ')
                    texts = [t.format(neg_classname) for t in templates]
                    tokens = clip.tokenize(texts).to(self.device)
                    text_embeddings = self.text_encoder.encode_text(tokens)
                    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
                    neg_feat = text_embeddings.mean(dim=0)
                    neg_feat = neg_feat / neg_feat.norm()
                    neg_feats_list.append(neg_feat)
                
                if neg_feats_list:
                    self.negative_text_features[classname] = torch.stack(
                        neg_feats_list, dim=0
                    ).to(self.device).type(self.dtype)
        
        self.label_to_neg_features = {}
        for classname, feats in self.negative_text_features.items():
            if classname in self.classnames:
                label = self.classnames.index(classname)
                self.label_to_neg_features[label] = feats
        
        print(f"✓ Cached negative text features for {len(self.negative_text_features)} classes")
    
    def set_prototypes(self, prototypes: torch.Tensor):
        """Set the prototypes for pure visual branch."""
        assert prototypes.shape == (self.num_classes, self.raw_feat_dim), \
            f"Expected prototypes shape {(self.num_classes, self.raw_feat_dim)}, got {prototypes.shape}"
        self.prototypes.data = prototypes.to(self.device).type(self.dtype)
    
    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode image into global and local features."""
        cls_token_raw, patch_tokens_raw, cls_token_proj, patch_tokens_proj = self.image_encoder(image.type(self.dtype))
        B = cls_token_raw.shape[0]
        
        if len(patch_tokens_raw.shape) == 4:
            H, W, C = patch_tokens_raw.shape[1], patch_tokens_raw.shape[2], patch_tokens_raw.shape[3]
            local_features_raw = patch_tokens_raw.reshape(B, H * W, C)
        else:
            local_features_raw = patch_tokens_raw
        
        if len(patch_tokens_proj.shape) == 4:
            H, W, C = patch_tokens_proj.shape[1], patch_tokens_proj.shape[2], patch_tokens_proj.shape[3]
            local_features_proj = patch_tokens_proj.reshape(B, H * W, C)
        else:
            local_features_proj = patch_tokens_proj
        
        return cls_token_raw, local_features_raw, cls_token_proj, local_features_proj
    
    def forward(self, image: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                negative_text_tokens: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass that computes both multi-modal and pure visual scores.
        
        Returns:
            dict containing:
                - logits_multimodal: logits from multi-modal branch
                - logits_visual: logits from pure visual branch
                - logits_combined: combined logits
                - aux_losses: auxiliary losses
                - final_feats_multimodal: fused features from multi-modal branch
                - final_feats_visual: global features from pure visual branch
                - ood_score_multimodal: OOD score from multi-modal branch
                - ood_score_visual: OOD score from pure visual branch
                - ood_score_combined: combined OOD score
        """
        B = image.shape[0]
        
        # 1. Encode Image
        cls_token_raw, patch_tokens_raw, cls_token_proj, patch_tokens_proj = self.encode_image(image)
        
        # 2. Use selector on projected features (512D)
        selected_feats_proj, sel_aux_loss, bg_mask = self.selector(patch_tokens_proj)
        
        # 3. Apply mask to raw features (768D) for visual branch
        selected_feats_raw = patch_tokens_raw * bg_mask
        
        # 4. Text Features (512D)
        text_feats = self._text_features.type(self.dtype)
        if labels is not None:
            pos_text_feat = text_feats[labels]
        else:
            pos_text_feat = text_feats.mean(dim=0, keepdim=True).expand(B, -1)
        
        # 5. Multi-modal branch: Fuser fusion in projected space (512D)
        # Fuser takes: selected_feats_proj (B, N, 512) + cls_token_proj (B, 512)
        final_feats_multimodal = self.fuser(selected_feats_proj, global_feat=cls_token_proj)
        global_features_multimodal = cls_token_proj + final_feats_multimodal
        
        # Normalize (already in projected space, no need to project)
        global_features_multimodal = global_features_multimodal / global_features_multimodal.norm(dim=-1, keepdim=True)
        
        # Compute multi-modal logits (both in 512D space)
        logit_scale = self.logit_scale.exp()
        logits_multimodal = logit_scale * global_features_multimodal @ text_feats.T
        
        # 6. Pure visual branch: cls + selected patch mean in original space (768D)
        selected_patch_mean = selected_feats_raw.mean(dim=1)
        global_features_visual = cls_token_raw + selected_patch_mean
        global_features_visual = global_features_visual / global_features_visual.norm(dim=-1, keepdim=True)
        
        # Project visual features to 512D space for local scores
        global_features_visual_proj = self.visual_proj(global_features_visual)
        global_features_visual_proj = global_features_visual_proj / global_features_visual_proj.norm(dim=-1, keepdim=True)
        
        # Normalize prototypes (768D)
        prototypes_norm = self.prototypes / self.prototypes.norm(dim=-1, keepdim=True)
        
        # Compute visual logits (both in 768D space)
        logits_visual = logit_scale * global_features_visual @ prototypes_norm.T
        
        # 7. Combine logits
        logits_combined = self.multimodal_weight * logits_multimodal + self.visual_weight * logits_visual
        
        # 8. Compute OOD scores
        # Multi-modal OOD score: max softmax probability
        ood_score_multimodal = F.softmax(logits_multimodal, dim=1).max(dim=1)[0]
        
        # Visual OOD score: max softmax probability
        ood_score_visual = F.softmax(logits_visual, dim=1).max(dim=1)[0]
        
        # Combined OOD score
        ood_score_combined = self.multimodal_weight * ood_score_multimodal + self.visual_weight * ood_score_visual
        
        # 9. Compute auxiliary losses
        aux_losses = sel_aux_loss.copy()
        
        # Add semantic exclusion loss if enabled
        if self.cfg.get('use_semantic_exclusion', False) and labels is not None:
            mask = torch.ones(B, self.num_classes, dtype=torch.bool, device=self.device)
            mask[torch.arange(B), labels] = False
            neg_text_feats = text_feats.unsqueeze(0).expand(B, -1, -1)[mask].reshape(B, -1, self.proj_feat_dim)  # Use proj_feat_dim (512)
            
            scale_val = logit_scale.item()
            sem_loss = self._compute_semantic_exclusion_loss(
                global_features_multimodal, pos_text_feat, neg_text_feats,
                margin=self.cfg.get('margin', 0.1),
                logit_scale=scale_val
            )
            aux_losses['semantic_exclusion'] = self.cfg.get('lambda_semantic_exclusion', 0.1) * sem_loss
        
        return {
            'logits_multimodal': logits_multimodal,
            'logits_visual': logits_visual,
            'logits_combined': logits_combined,
            'aux_losses': aux_losses,
            'final_feats_multimodal': global_features_multimodal,  # Already in 512D space
            'final_feats_visual': global_features_visual,  # In 768D space
            'ood_score_multimodal': ood_score_multimodal,
            'ood_score_visual': ood_score_visual,
            'ood_score_combined': ood_score_combined,
            'selected_feats': selected_feats_raw,
            'selected_feats_proj': selected_feats_proj,  # For multi-modal branch local scores (512D)
            'global_features': global_features_multimodal,
            'local_features': patch_tokens_raw,  # For visual branch local scores (768D)
            'local_features_proj': patch_tokens_proj,  # For multi-modal branch local scores (512D)
            'prototypes': prototypes_norm,  # For visual branch local scores (768D)
            'bg_mask': bg_mask
        }
    
    @staticmethod
    def _compute_semantic_exclusion_loss(final_feat: torch.Tensor, 
                                       pos_text_feat: torch.Tensor,
                                       neg_text_feats: torch.Tensor,
                                       margin: float = 0.1, 
                                       logit_scale: float = 100.0) -> torch.Tensor:
        """Compute semantic exclusion loss."""
        final_feat = final_feat.float()
        pos_text_feat = pos_text_feat.float()
        neg_text_feats = neg_text_feats.float()
        
        final_feat = F.normalize(final_feat, dim=-1, eps=1e-8)
        pos_text_feat = F.normalize(pos_text_feat, dim=-1, eps=1e-8)
        neg_text_feats = F.normalize(neg_text_feats, dim=-1, eps=1e-8)
        
        pos_sim = (final_feat * pos_text_feat).sum(dim=1) * logit_scale
        neg_sims = torch.einsum("bd,bnd->bn", final_feat, neg_text_feats) * logit_scale
        
        neg_score = torch.logsumexp(neg_sims, dim=1)
        scaled_margin = margin * logit_scale
        
        loss = F.relu(neg_score - pos_sim + scaled_margin)
        return loss.mean()
    
    def init_prototypes_from_features(self, train_features: torch.Tensor, train_labels: torch.Tensor):
        """Initialize prototypes from training features."""
        with torch.no_grad():
            prototypes = torch.zeros(self.num_classes, self.raw_feat_dim, device=self.device)
            for cls in range(self.num_classes):
                cls_mask = (train_labels == cls)
                if cls_mask.sum() > 0:
                    cls_feats = train_features[cls_mask]
                    prototypes[cls] = cls_feats.mean(dim=0)
            
            prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
            self.set_prototypes(prototypes)
            print(f"✓ Prototypes initialized from {train_features.shape[0]} training samples")
    
    def init_prototypes_from_images(self, train_loader):
        """Initialize prototypes from training images."""
        all_features = []
        all_labels = []
        
        self.eval()
        with torch.no_grad():
            for batch in train_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self(images)
                features = outputs['final_feats_visual']
                
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        self.init_prototypes_from_features(all_features, all_labels)
    
    def save_selector_weights(self, save_path: str):
        """Save selector weights to a file."""
        selector_state = {
            'selector_type': self.cfg.get('selector_type', 'identity'),
            'selector_state_dict': self.selector.state_dict(),
            'cfg': self.cfg
        }
        torch.save(selector_state, save_path)
        print(f"✓ Selector weights saved to {save_path}")
    
    def load_selector_weights(self, load_path: str):
        """Load selector weights from a file."""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Selector weights file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        
        if 'selector_state_dict' in checkpoint:
            selector_state_dict = checkpoint['selector_state_dict']
            saved_selector_type = checkpoint.get('selector_type', 'identity')
        elif 'state_dict' in checkpoint:
            full_state_dict = checkpoint['state_dict']
            selector_state_dict = {}
            selector_prefix = 'selector.'
            for key in full_state_dict:
                if key.startswith(selector_prefix):
                    new_key = key[len(selector_prefix):]
                    selector_state_dict[new_key] = full_state_dict[key]
            saved_selector_type = checkpoint.get('config', {}).get('selector_type', 'identity')
        else:
            raise ValueError(f"Unknown checkpoint format.")
        
        current_selector_type = self.cfg.get('selector_type', 'mlp')
        if saved_selector_type != current_selector_type:
            print(f"Warning: Selector type mismatch. Saved: {saved_selector_type}, Current: {current_selector_type}")
        
        self.selector.load_state_dict(selector_state_dict)
        print(f"✓ Selector weights loaded from {load_path}")
        print(f"  Selector type: {saved_selector_type}")
