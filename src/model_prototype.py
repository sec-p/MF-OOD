"""
Prototype-based CLIP Model for Training-Free OOD Detection
Uses raw ViT features (cls token and patch tokens) without projection to shared space.
Classifies using visual prototypes initialized from training data.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import clip


class PrototypeCLIP(nn.Module):
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
        # For ViT-B/16:
        # - raw_feat_dim = 768 (width, before projection)
        # - proj_feat_dim = 512 (output_dim, after projection)
        try: 
            self.raw_feat_dim = self.image_encoder.width
        except: 
            try:
                self.raw_feat_dim = self.image_encoder.output_dim
            except:
                self.raw_feat_dim = clip_model.ln_final.weight.shape[0]
        
        try:
            self.proj_feat_dim = self.image_encoder.output_dim
        except:
            self.proj_feat_dim = clip_model.ln_final.weight.shape[0]
        
        # Use raw_feat_dim for prototypes (visual prototypes are in raw feature space)
        self.feat_dim = self.raw_feat_dim
        
        # Create prototype classifier - this will be our "text" features
        # These are initialized later from training data
        self.prototypes = nn.Parameter(torch.empty(self.num_classes, self.feat_dim, dtype=self.dtype), requires_grad=False)
        
        # Build selector component for feature selection
        self._build_selector()
        
        # Cache text features for comparison if needed
        self._cache_text_features()
        print(f"✓ PrototypeCLIP initialized. Dtype: {self.dtype}, Feat Dim: {self.feat_dim}")
    
    def _build_selector(self):
        """Build selector component for feature selection."""
        # Import here to avoid circular import
        from .model_modular import IdentitySelector, MultiHeadMLPSelector, SparseSlotAttentionSelector

        s_type = self.cfg.get('selector_type', 'mlp')
        num_sel = self.cfg.get('num_select', 64)
        
        # Use projected feature dimension for selector (512 for ViT-B/16)
        if s_type == 'mlp':
            self.selector = MultiHeadMLPSelector(self.proj_feat_dim, num_sel, self.cfg.get('num_heads_selector', 8), self.cfg)
        elif s_type == 'slot':
            self.selector = SparseSlotAttentionSelector(self.proj_feat_dim, num_sel, self.cfg.get('num_slots', 8), self.cfg)
        else:
            self.selector = IdentitySelector(self.proj_feat_dim, num_sel, self.cfg)
        
        # Convert selector to correct dtype
        self.selector = self.selector.to(self.dtype)
    
    def _cache_text_features(self):
        """Cache original text features for reference."""
        templates = self.cfg.get('templates', ["a photo of a {}"])
        if isinstance(templates, str): templates = [templates]
        
        text_features_list = []
        with torch.no_grad():
            for classname in self.classnames:
                classname = classname.replace('_', ' ').strip()
                texts = [t.format(classname) for t in templates]
                texts = clip.tokenize(texts).to(self.device)
                text_embeddings = self.text_encoder.encode_text(texts)
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
                text_feat = text_embeddings.mean(dim=0)
                text_feat = text_feat / text_feat.norm()
                text_features_list.append(text_feat)
        
        self.text_features = torch.stack(text_features_list, dim=0).to(self.device).type(self.dtype)
        self.register_buffer('_text_features', self.text_features)
    
    def set_prototypes(self, prototypes: torch.Tensor):
        """Set the prototypes for classification.
        prototypes: (num_classes, feat_dim) - should be in projected feature space (512 for ViT-B/16)
        """
        assert prototypes.shape == (self.num_classes, self.feat_dim), \
            f"Expected prototypes shape {(self.num_classes, self.feat_dim)}, got {prototypes.shape}"
        self.prototypes.data = prototypes.to(self.device).type(self.dtype)
    
    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode image into global and local features (raw ViT features).
        Returns both raw (before projection) and projected (after projection) features.
        """
        # Get raw and projected features from ViT
        # The image_encoder.visual should have return_both_features=True
        cls_token_raw, patch_tokens_raw, cls_token_proj, patch_tokens_proj = self.image_encoder(image.type(self.dtype))
        B = cls_token_raw.shape[0]
        
        # patch_tokens shape: (B, H, W, C) where C = self.feat_dim (raw) or output_dim (proj)
        # Reshape to (B, H*W, C)
        if len(patch_tokens_raw.shape) == 4:
            H, W, C = patch_tokens_raw.shape[1], patch_tokens_raw.shape[2], patch_tokens_raw.shape[3]
            local_features_raw = patch_tokens_raw.reshape(B, H * W, C)
        else:
            # Already reshaped
            local_features_raw = patch_tokens_raw
        
        if len(patch_tokens_proj.shape) == 4:
            H, W, C = patch_tokens_proj.shape[1], patch_tokens_proj.shape[2], patch_tokens_proj.shape[3]
            local_features_proj = patch_tokens_proj.reshape(B, H * W, C)
        else:
            # Already reshaped
            local_features_proj = patch_tokens_proj
        
        return cls_token_raw, local_features_raw, cls_token_proj, local_features_proj
    
    def forward(self, image: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                negative_text_tokens: Optional[torch.Tensor] = None) -> Dict:
        
        B = image.shape[0]
        
        # 1. Encode Image (FP16) - get raw and projected ViT features
        cls_token_raw, patch_tokens_raw, cls_token_proj, patch_tokens_proj = self.encode_image(image)
        
        # 2. Use selector on projected features to get mask
        selected_feats_proj, sel_aux_loss, bg_mask = self.selector(patch_tokens_proj)
        
        # 3. Apply mask to raw features to get selected raw features
        # bg_mask shape: (B, N, 1) or (B, K, 1) depending on selector
        selected_feats_raw = patch_tokens_raw * bg_mask
        
        # 4. Calculate mean of selected raw patches
        selected_patch_mean = selected_feats_raw.mean(dim=1)
        
        # 5. Use cls_token_raw + selected_patch_mean as global feature for OOD detection
        # This is in raw feature space (768 for ViT-B/16)
        global_features = cls_token_raw + selected_patch_mean
        
        # 6. Normalize input features
        global_features = global_features / global_features.norm(dim=-1, keepdim=True)
        
        # 7. Ensure prototypes are normalized
        prototypes_norm = self.prototypes / self.prototypes.norm(dim=-1, keepdim=True)
        
        # 8. Logits - same as CLIP's similarity calculation
        # Both global_features and prototypes are in raw feature space (768 for ViT-B/16)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * global_features @ prototypes_norm.T
        
        return {
            'logits': logits,
            'aux_losses': sel_aux_loss,
            'final_feats': global_features,
            'global_features': global_features,
            'local_features': patch_tokens_raw,
            'selected_feats': selected_feats_raw,
            'bg_mask': bg_mask
        }
    
    def init_prototypes_from_features(self, train_features: torch.Tensor, train_labels: torch.Tensor):
        """Initialize prototypes from training features.
        train_features: (N, feat_dim) - should be in raw feature space (768 for ViT-B/16)
        train_labels: (N,)
        """
        with torch.no_grad():
            prototypes = torch.zeros(self.num_classes, self.feat_dim, device=self.device)
            for cls in range(self.num_classes):
                cls_mask = (train_labels == cls)
                if cls_mask.sum() > 0:
                    cls_feats = train_features[cls_mask]
                    # Take the mean of all features for this class as prototype
                    prototypes[cls] = cls_feats.mean(dim=0)
            
            # Normalize prototypes
            prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
            self.set_prototypes(prototypes)
            print(f"✓ Prototypes initialized from {train_features.shape[0]} training samples")
    
    def init_prototypes_from_images(self, train_loader):
        """Initialize prototypes from training images.
        This is a convenience method that extracts features and initializes prototypes.
        """
        all_features = []
        all_labels = []
        
        self.eval()
        with torch.no_grad():
            for batch in train_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Extract features using the model's forward pass (includes selector)
                # The global_features are already in raw feature space (768 for ViT-B/16)
                outputs = self(images)
                features = outputs['global_features']
                
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        self.init_prototypes_from_features(all_features, all_labels)
    
    def save_selector_weights(self, save_path: str):
        """Save selector weights to a file.
        
        Args:
            save_path: Path to save the selector weights.
        """
        selector_state = {
            'selector_type': self.cfg.get('selector_type', 'identity'),
            'selector_state_dict': self.selector.state_dict(),
            'cfg': self.cfg
        }
        torch.save(selector_state, save_path)
        print(f"✓ Selector weights saved to {save_path}")
    
    def load_selector_weights(self, load_path: str):
        """Load selector weights from a file.
        
        Args:
            load_path: Path to load the selector weights from.
                     Can be either:
                     1. A file saved by save_selector_weights (contains selector_state_dict)
                     2. A checkpoint saved by train_eval.py (contains full state_dict)
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Selector weights file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        
        # Check if this is a full checkpoint or just selector weights
        if 'selector_state_dict' in checkpoint:
            # Format 1: Saved by save_selector_weights
            selector_state_dict = checkpoint['selector_state_dict']
            saved_selector_type = checkpoint.get('selector_type', 'identity')
        elif 'state_dict' in checkpoint:
            # Format 2: Full checkpoint from train_eval.py
            # Extract selector parameters from full state_dict
            full_state_dict = checkpoint['state_dict']
            selector_state_dict = {}
            
            # Get selector parameter prefix
            selector_prefix = 'selector.'
            
            # Extract all selector parameters
            for key in full_state_dict:
                if key.startswith(selector_prefix):
                    # Remove the 'selector.' prefix to match selector's state_dict keys
                    new_key = key[len(selector_prefix):]
                    selector_state_dict[new_key] = full_state_dict[key]
            
            saved_selector_type = checkpoint.get('config', {}).get('selector_type', 'identity')
        else:
            raise ValueError(f"Unknown checkpoint format. Expected 'selector_state_dict' or 'state_dict' in checkpoint.")
        
        # Verify selector type matches
        current_selector_type = self.cfg.get('selector_type', 'mlp')
        if saved_selector_type != current_selector_type:
            print(f"Warning: Selector type mismatch. Saved: {saved_selector_type}, Current: {current_selector_type}")
        
        # Load weights
        self.selector.load_state_dict(selector_state_dict)
        print(f"✓ Selector weights loaded from {load_path}")
        print(f"  Selector type: {saved_selector_type}")
        print(f"  Number of parameters loaded: {len(selector_state_dict)}")