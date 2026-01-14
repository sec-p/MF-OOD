"""
Prototype-based CLIP Model for Training-Free OOD Detection
Uses raw ViT features (cls token and patch tokens) without projection to shared space.
Classifies using visual prototypes initialized from training data.
"""

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
        
        # Get feature dimension from ViT (before projection)
        # For ViT-B/16, this is 768 (width), not 512 (embed_dim)
        try: 
            self.feat_dim = self.image_encoder.width
        except: 
            # Fallback to output_dim if width not available
            try:
                self.feat_dim = self.image_encoder.output_dim
            except:
                self.feat_dim = clip_model.ln_final.weight.shape[0]
        
        # Create prototype classifier - this will be our "text" features
        # These are initialized later from training data
        self.prototypes = nn.Parameter(torch.empty(self.num_classes, self.feat_dim), requires_grad=False)
        
        # Cache text features for comparison if needed
        self._cache_text_features()
        print(f"✓ PrototypeCLIP initialized. Dtype: {self.dtype}, Feat Dim: {self.feat_dim}")
    
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
        prototypes: (num_classes, feat_dim)
        """
        assert prototypes.shape == (self.num_classes, self.feat_dim), \
            f"Expected prototypes shape {(self.num_classes, self.feat_dim)}, got {prototypes.shape}"
        self.prototypes.data = prototypes.to(self.device).type(self.dtype)
    
    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image into global and local features (raw ViT features)."""
        # Get raw features from ViT (cls token and patch tokens)
        # The image_encoder.visual should have return_raw_features=True
        cls_token, patch_tokens = self.image_encoder(image.type(self.dtype))
        B = cls_token.shape[0]
        
        # patch_tokens shape: (B, H, W, C) where C = self.feat_dim
        # Reshape to (B, H*W, C)
        if len(patch_tokens.shape) == 4:
            H, W, C = patch_tokens.shape[1], patch_tokens.shape[2], patch_tokens.shape[3]
            local_features = patch_tokens.reshape(B, H * W, C)
        else:
            # Already reshaped
            local_features = patch_tokens
        
        return cls_token, local_features
    
    def forward(self, image: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                negative_text_tokens: Optional[torch.Tensor] = None) -> Dict:
        
        B = image.shape[0]
        
        # 1. Encode Image (FP16) - get raw ViT features
        cls_token, patch_tokens = self.encode_image(image)
        
        # 2. Calculate patch mean (for prototype initialization)
        patch_mean = patch_tokens.mean(dim=1)
        
        # 3. Use patch mean as global feature for OOD detection
        # This allows using patch token pooling in OOD score calculation
        global_features = cls_token + patch_mean
        
        # 4. Normalize input features
        global_features = global_features / global_features.norm(dim=-1, keepdim=True)
        
        # 5. Ensure prototypes are normalized
        prototypes_norm = self.prototypes / self.prototypes.norm(dim=-1, keepdim=True)
        
        # 6. Logits - same as CLIP's similarity calculation
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * global_features @ prototypes_norm.T
        
        return {
            'logits': logits,
            'aux_losses': {},
            'final_feats': global_features,
            'global_features': global_features,  # Changed to patch mean
            'local_features': patch_tokens
        }
    
    def init_prototypes_from_features(self, train_features: torch.Tensor, train_labels: torch.Tensor):
        """Initialize prototypes from training features.
        train_features: (N, feat_dim)
        train_labels: (N,)
        """
        with torch.no_grad():
            prototypes = torch.zeros(self.num_classes, self.feat_dim, device=self.device)
            for cls in range(self.num_classes):
                cls_mask = (train_labels == cls)
                if cls_mask.sum() > 0:
                    cls_feats = train_features[cls_mask]
                    # Take the mean of all patch token features for this class as prototype
                    # Note: train_features should be patch means, not cls_token + patch_mean
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
                
                # Extract patch mean features (not cls_token + patch_mean)
                cls_token, patch_tokens = self.encode_image(images)
                patch_mean = patch_tokens.mean(dim=1)
                features = patch_mean / patch_mean.norm(dim=-1, keepdim=True)
                
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        self.init_prototypes_from_features(all_features, all_labels)