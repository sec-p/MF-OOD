"""
Unified Training and Evaluation Script with Per-Epoch OOD Testing
Trains modular models with automatic ImageNet validation and OOD evaluation on each epoch.
Optimized for Linux servers with checkpoint savings for learnable parameters only.
Now supports Automatic Mixed Precision (AMP) for stability and speed.
"""

imagenet_templates = [
    'a photo of a {}.'
]

import os
import sys
import json
import argparse
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy.stats import entropy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import clip
from src.model_modular import build_modular_model
from utils.common import get_test_labels, setup_seed
from utils.train_eval_util import (
    generate_fewshot_dataset_by_class, 
    set_train_loader, 
    set_val_loader, 
    set_ood_loader_ImageNet
)
from utils.train_eval_util import set_model_clip
from utils.detection_util import (
    stable_cumsum, 
    fpr_and_fdr_at_recall, 
    get_measures as detection_get_measures,
    get_ood_scores_clip, 
    get_ood_scores_dual_stream,
    get_and_print_results, 
    print_measures
)
from utils.file_ops import setup_log as setup_file_log

class TrainEvalOrchestrator:
    """Orchestrates training and evaluation with per-epoch OOD testing."""
    
    def __init__(self, method: str, epochs: int, lr: float, 
                 batch_size: int, seed: int, device: torch.device,
                 selector_type: str = None, fuser_type: str = None, id_dataset: str = 'ImageNet',
                 root_path: str = '/data/datasets', shots: int = 16, lambda_llm_negatives: float = 0.1,
                 lambda_mixup: float = 0.1, margin: float = 0.2, num_select: int = 16,
                 backbone: str = 'ViT-L/16', class_negatives_path: str = '', use_full_data: bool = False,
                 num_ood_sumple: int = -1,
                 # Advanced settings
                 selector_temperature: float = 1.0,
                 patches_per_slot_attn: int = 16,
                 # OOD score parameters
                 score_type: str = 'GL-MCM',
                 temperature: float = 1.0,
                 lambda_local: float = 1.0,
                 # Stage 2 parameters
                 use_weighted_pool: bool = False,
                 adapter_hidden_dim: int = None,
                 use_visual_prototypes: bool = False):
        
        self.method = method
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        
        # Model components
        self.selector_type = selector_type
        self.fuser_type = fuser_type
        
        # Dataset settings
        self.id_dataset = id_dataset
        self.root_path = root_path
        self.shots = shots
        self.use_full_data = use_full_data
        self.num_ood_sumple = num_ood_sumple
        
        # Loss function coefficients (hyperparameters)
        self.lambda_llm_negatives = lambda_llm_negatives
        self.lambda_mixup = lambda_mixup
        self.margin = margin
        
        # OOD score parameters
        self.score_type = score_type
        self.temperature = temperature
        self.lambda_local = lambda_local
        
        # Stage 2 parameters
        self.use_weighted_pool = use_weighted_pool
        self.adapter_hidden_dim = adapter_hidden_dim
        self.use_visual_prototypes = use_visual_prototypes
        
        # Model settings
        self.num_select = num_select
        self.backbone = backbone
        self.class_negatives_path = class_negatives_path
        
        # Advanced settings
        self.selector_temperature = selector_temperature
        self.patches_per_slot_attn = patches_per_slot_attn
        
        # Setup random seeds
        self._setup_seed(seed)
        
        # Classnames will be set in setup_data
        self.classnames = []
        
        # Setup logging
        self.log_dir = self._setup_logging()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None # For AMP
        self.train_loader = None
        self.test_loader = None
        self.ood_loaders = {}
        self.classnames = []
        
    def _setup_seed(self, seed: int):
        """Setup random seeds for reproducibility using common utility."""
        setup_seed(seed)

    def _setup_logging(self) -> str:
        """Setup logging directory and configure logging using file_ops utility."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f'/data/ICML2026/GL_MCM_FA/logs/{self.method}_{self.selector_type}_{self.seed}_{timestamp}'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(f'{log_dir}/checkpoints', exist_ok=True)
        
        # Create a simple args object for setup_file_log
        class LogArgs:
            def __init__(self, log_directory, name):
                self.log_directory = log_directory
                self.name = name
        
        log_args = LogArgs(log_dir, f'{self.method}_{self.seed}')
        self.logger = setup_file_log(log_args)
        return log_dir
    
    def setup_data(self):
        """Setup all data loaders (train, ID test, OOD tests) using GL-MCM style."""
        self.logger.debug('Setting up data...')
        
        # Create args-like object for train_eval_util functions
        class Args:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        
        # Setup args for data loaders
        data_args = Args(
            root_dir=self.root_path,
            batch_size=self.batch_size,
            seed=self.seed,
            shots=self.shots if not self.use_full_data else 0,
            in_dataset=self.id_dataset,
            num_ood_sumple=self.num_ood_sumple if hasattr(self, 'num_ood_sumple') else -1,  # Use instance variable if available
            gpu=0  # Default GPU
        )
        
        # Use GL-MCM style data loaders from train_eval_util
        train_transform = self._get_train_transform()
        
        # Get preprocess from CLIP model (same as eval_ood_detection.py)
        _, preprocess = set_model_clip(self.backbone)
        
        # Setup training data loader
        self.train_loader, full_train_dataset = set_train_loader(data_args, transform=train_transform)
        
        # Setup test data loader using CLIP's preprocess
        self.test_loader = set_val_loader(data_args, preprocess=preprocess)
        
        # Get classnames using GL-MCM standard method
        self.classnames = get_test_labels(data_args)
        
        # Load class negatives
        self.class_negatives = self._load_class_negatives()
        if self.class_negatives:
            self.logger.debug(f"Loaded class negatives for {len(self.class_negatives)} classes")
        else:
            self.logger.debug("No class negatives loaded or file not found.")
        
        # Setup OOD data loaders using GL-MCM style
        OOD_DATASETS = ['iNaturalist', 'SUN', 'places365', 'Texture']  # Use GL-MCM naming convention
        for ood_dataset in OOD_DATASETS:
            try:
                ood_loader = set_ood_loader_ImageNet(data_args, ood_dataset, preprocess, root=self.root_path)
                self.ood_loaders[ood_dataset] = ood_loader
                self.logger.debug(f'  ✓ Loaded OOD dataset: {ood_dataset}')
            except Exception as e:
                self.logger.debug(f'  ⚠ Failed to load OOD dataset {ood_dataset}: {e}')
        
        # Log dataset sizes
        train_size = len(self.train_loader.dataset)
        test_size = len(self.test_loader.dataset)
        self.logger.debug(f'  ✓ Training samples: {train_size}')
        self.logger.debug(f'  ✓ ID test samples: {test_size}')
    
    def _load_class_negatives(self) -> Dict:
        """Load class negatives from file."""
        class_negatives = {}
        neg_path = self.class_negatives_path
        if neg_path and os.path.exists(neg_path):
            with open(neg_path, 'r') as f:
                class_negatives = json.load(f)
            return class_negatives
        
        # Fallback to default locations
        fallback_paths = [
            os.path.join(os.path.dirname(__file__), 'configs', 'class_negatives.json'),
            os.path.join(project_root, 'configs', 'class_negatives.json')
        ]
        
        for fallback in fallback_paths:
            if os.path.exists(fallback):
                with open(fallback, 'r') as f:
                    return json.load(f)
                    
        return {}
    
    def _get_train_transform(self):
        """Get training transforms."""
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1),
                                        interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])
    
    def _get_test_transform(self):
        """Get test transforms."""
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])
    
    def setup_model(self):
        """Initialize model, optimizer, and scheduler using GL-MCM style."""
        self.logger.debug('Setting up model...')
        
        # Load CLIP using GL-MCM style
        clip_model, preprocess = set_model_clip(self.backbone)
        clip_model = clip_model.to(self.device)
        self.preprocess = preprocess
        
        # Create config dict
        cfg = {
            'device': self.device,
            'selector_type': self.selector_type,
            'num_select': self.num_select,
            'fuser_type': self.fuser_type,
            'lambda_llm_negatives': self.lambda_llm_negatives,
            'lambda_mixup': self.lambda_mixup,
            'margin': self.margin,
            'selector_temperature': self.selector_temperature,
            'patches_per_slot_attn': self.patches_per_slot_attn,
            'templates': imagenet_templates,
            'use_visual_prototypes': self.use_visual_prototypes,
            
            # Feature flags
            'use_redundancy_loss': True,
            'use_llm_negatives': False if self.lambda_llm_negatives > 0 else False,
            'use_semantic_exclusion': True,
            'use_mixup_invariance': True if self.lambda_mixup > 0 else False,
            
            # Loss weights
            'lambda_redundancy': 0.1,
            
            # Stage 2 parameters
            'use_weighted_pool': self.use_weighted_pool,
            'adapter_hidden_dim': self.adapter_hidden_dim,
        }
        
        self.logger.debug(f'Config: {json.dumps(cfg, default=str, indent=2)}')
        
        # Build modular model
        self.model = build_modular_model(cfg, self.classnames, clip_model, class_negatives=self.class_negatives)
        self.model = self.model.to(self.device)
        
        # Setup optimizer (only for trainable parameters)
        trainable_params = self._get_trainable_params()
        
        if not trainable_params:
            self.logger.debug("WARNING: No trainable parameters found! Check your model configuration.")
        
        self.optimizer = torch.optim.SGD(
            trainable_params,
            lr=self.lr,
            momentum=0.9,
            weight_decay=1e-5
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        # Setup GradScaler for AMP
        self.scaler = GradScaler()
        
        self.logger.debug(f'  ✓ Model: {self.method}')
        self.logger.debug(f'  ✓ Trainable parameters: {sum(p.numel() for p in trainable_params)}')
    
    def _get_trainable_params(self):
        """Get only trainable parameters."""
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with AMP (Automatic Mixed Precision)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Tqdm bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]', ncols=100)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Prepare Data
            images, labels = images.to(self.device), labels.to(self.device)
            negative_text_tokens = None  # No longer needed since we use cached negative features
            
            self.optimizer.zero_grad()
            
            # Forward with Autocast (Mixed Precision)
            with autocast():
                output_dict = self.model(images, labels=labels, negative_text_tokens=negative_text_tokens)
                
                logits = output_dict['logits']
                aux_losses = output_dict['aux_losses']
                
                # Main Loss
                ce_loss = F.cross_entropy(logits, labels)
                
                # Aux Losses
                total_aux_loss = 0.0
                for k, v in aux_losses.items():
                    if v.requires_grad:
                        total_aux_loss += v
                
                loss = ce_loss + total_aux_loss
                # loss = ce_loss

            # Backward with Scaler
            self.scaler.scale(loss).backward()

            # Gradient Clipping (must unscale first)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer Step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics & Logging
            total_loss += loss.item()
            with torch.no_grad():
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            # Prepare loss info for display
            loss_info = {
                'Loss': f"{loss.item():.4f}",
                'CE': f"{ce_loss.item():.4f}",
                'Acc': f"{100.*correct/total:.2f}%"
            }
            
            # Add auxiliary losses to display
            for k, v in aux_losses.items():
                if v.requires_grad:
                    loss_info[k] = f"{v.item():.4f}"
            
            # Update progress bar with detailed loss info
            pbar.set_postfix(loss_info)
            
            # Log detailed loss info occasionally (reduced frequency)
            if batch_idx % 100 == 0:
                loss_str = f"CE: {ce_loss.item():.4f}"
                for k, v in aux_losses.items():
                    loss_str += f", {k}: {v.item():.4f}"
                self.logger.debug(f"  Batch {batch_idx}: {loss_str}")

        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        train_acc = 100.0 * correct / total if total > 0 else 0
        
        return avg_loss, train_acc, True
    
    def evaluate_id(self) -> float:
        """Evaluate on ID (ImageNet) test set."""

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Inference only needs images (labels are for metric calc only)
                with autocast():
                    output_dict = self.model(images, labels=None)  # No labels passed = Inference mode
                    logits = output_dict['logits']
                
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        id_acc = 100.0 * correct / total if total > 0 else 0.0
        return id_acc
    
    def evaluate_ood_dataset(self, dataset_name: str, id_scores: np.ndarray) -> Tuple[float, float]:
        """Evaluate on single OOD dataset using GL-MCM utils."""
        if dataset_name not in self.ood_loaders:
            return 0.0, 0.0
        
        self.model.eval()
        
        # Create args-like object for detection_util functions
        class Args:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        
        args = Args(
            T=self.temperature,
            score=self.score_type,
            lambda_local=self.lambda_local
        )
        
        # Use GL-MCM's get_ood_scores_clip function for consistent scoring
        # Use pre-computed ID scores to avoid redundant computation
        out_score = get_ood_scores_clip(args, self.model, self.ood_loaders[dataset_name], self.classnames)
        
        # Compute metrics using GL-MCM's get_measures
        if len(id_scores) == 0 or len(out_score) == 0:
            return 0.0, 0.0
        
        measures = detection_get_measures(-id_scores, -out_score)
        auroc, aupr, fpr95 = measures
        
        # Convert AUROC to percentage, FPR95 is already in correct decimal form
        return auroc * 100, fpr95
    

    

    
    def get_measures(self, id_scores: np.ndarray, ood_scores: np.ndarray, recall_level=0.95):
        """Calculate AUROC, AUPR and FPR95 using GL-MCM's method from detection utility."""
        try:
            auroc, aupr, fpr = detection_get_measures(id_scores, ood_scores, recall_level)
        except:
            auroc, aupr, fpr = 0.0, 0.0, 0.0
        
        return avg_auroc, aupr, fpr
    
    def train_stage2(self, stage1_checkpoint: str, epochs: int = 10, lr: float = 0.001) -> Dict:
        """
        Train Stage 2: VisualAdapter + VisualClassifier with CE loss.
        Other components (selector, fuser, backbone) are frozen.
        
        Args:
            stage1_checkpoint: Path to stage 1 checkpoint
            epochs: Number of training epochs for stage 2
            lr: Learning rate for stage 2
        
        Returns:
            Dict containing training metrics
        """
        self.logger.debug('\n' + '='*80)
        self.logger.debug('Starting Stage 2 Training: VisualAdapter + VisualClassifier')
        self.logger.debug('='*80 + '\n')
        
        # Load stage 1 checkpoint
        self.logger.debug(f'Loading stage 1 checkpoint from: {stage1_checkpoint}')
        checkpoint = torch.load(stage1_checkpoint, map_location=self.device, weights_only=False)
        
        # Load state dict (only trainable parameters from stage 1)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load stage 1 parameters
        self.model.load_state_dict(state_dict, strict=False)
        self.logger.debug('✓ Stage 1 parameters loaded')
        
        # Check if visual prototype initialization is enabled
        use_visual_prototypes = self.cfg.get('use_visual_prototypes', False)
        visual_prototypes = None
        
        if use_visual_prototypes:
            # Compute class feature centers from training data
            visual_prototypes = self.model.compute_class_feature_centers(self.train_loader)
        
        # Switch to stage 2 training mode and build components with visual prototypes if available
        self.model.set_training_stage(stage=2)
        
        # Re-build visual classifier with visual prototypes if available
        # This is needed because set_training_stage already builds components without visual prototypes
        if visual_prototypes is not None:
            # Re-build only the visual classifier with the computed prototypes
            self.model.visual_classifier = VisualClassifier(
                self.model.feat_dim,
                self.model.num_classes,
                text_prototypes=self.model.text_features,
                visual_prototypes=visual_prototypes,
                cfg=self.cfg
            )
        
        # Setup optimizer for stage 2 (only adapter and classifier)
        stage2_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                stage2_params.append(param)
                self.logger.debug(f'  Trainable: {name}')
        
        self.optimizer = torch.optim.AdamW(
            stage2_params,
            lr=lr,
            weight_decay=1e-5
        )
        
        # Setup scheduler for stage 2
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # NOTE: Removed GradScaler for Stage 2 as it causes FP16 gradient errors
        # The adapter and classifier are small, so FP32 training is efficient
        
        self.logger.debug(f'✓ Stage 2 trainable parameters: {sum(p.numel() for p in stage2_params)}')
        
        # Training loop
        best_acc = 0.0
        results_history = []
        
        # Setup OOD datasets for evaluation
        OOD_DATASETS = ['iNaturalist', 'SUN', 'places365', 'Texture']
        ood_loaders = {}
        
        # Create args-like object for OOD loader
        class OODArgs:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        
        ood_args = OODArgs(
            root_dir=self.root_path,
            batch_size=self.batch_size,
            seed=self.seed,
            shots=0,
            in_dataset=self.id_dataset,
            num_ood_sumple=-1,
            gpu=0
        )
        
        for ood_dataset in OOD_DATASETS:
            try:
                ood_loaders[ood_dataset] = set_ood_loader_ImageNet(
                    ood_args,
                    ood_dataset,
                    self.preprocess,
                    root=self.root_path
                )
                self.logger.debug(f'  ✓ Loaded OOD dataset: {ood_dataset}')
            except Exception as e:
                self.logger.debug(f'  ⚠ Failed to load OOD dataset {ood_dataset}: {e}')
        
        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch_stage2(epoch)
            self.scheduler.step()
            
            # Evaluate on ID test set
            id_acc = self._evaluate_stage2()
            
            # Evaluate OOD performance (every 5 epochs or last epoch)
            ood_results = {}
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                ood_results = self._evaluate_ood_stage2(ood_loaders)
            
            # Log results
            self.logger.debug(f'\n[Epoch {epoch+1}/{epochs}]')
            self.logger.debug(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            self.logger.debug(f'  ID Test Acc: {id_acc:.2f}%')
            
            # Log OOD results if available
            if ood_results:
                self.logger.debug('  OOD Results:')
                for method in ['DS-MCM', 'Visual-GL-MCM', 'Stage1-GL-MCM']:
                    self.logger.debug(f'    {method}:')
                    for ood_name in OOD_DATASETS:
                        if ood_name in ood_loaders:
                            auroc_key = f'{method}_{ood_name}_auroc'
                            fpr95_key = f'{method}_{ood_name}_fpr95'
                            if auroc_key in ood_results:
                                self.logger.debug(f'      {ood_name:15} AUROC: {ood_results[auroc_key]:.2f}%, FPR95: {ood_results[fpr95_key]:.2f}%')
                
                # Log average OOD metrics
                for method in ['DS-MCM', 'Visual-GL-MCM', 'Stage1-GL-MCM']:
                    avg_auroc_key = f'{method}_avg_auroc'
                    avg_fpr95_key = f'{method}_avg_fpr95'
                    if avg_auroc_key in ood_results:
                        self.logger.debug(f'    {method} Avg: AUROC: {ood_results[avg_auroc_key]:.2f}%, FPR95: {ood_results[avg_fpr95_key]:.2f}%')
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                self._save_stage2_checkpoint(epoch, train_loss, train_acc, id_acc, ood_results)
            
            # Track best
            if id_acc > best_acc:
                best_acc = id_acc
                self._save_stage2_checkpoint(999, train_loss, train_acc, id_acc, ood_results)  # 999 for 'best'
                self.logger.debug(f'  ★ New Best ID Acc: {best_acc:.2f}%')
            
            # Update history
            history_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'id_acc': id_acc
            }
            # Add OOD results if available
            if ood_results:
                history_entry.update(ood_results)
            results_history.append(history_entry)
        
        # Save all results
        results_path = os.path.join(self.log_dir, 'stage2_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_history, f, indent=2)
        
        self.logger.debug('\nStage 2 training completed.')
        self.logger.debug(f'Best ID Accuracy: {best_acc:.2f}%')
        
        return {
            'best_id_acc': best_acc,
            'results_history': results_history
        }
    
    def _train_epoch_stage2(self, epoch: int) -> Tuple[float, float]:
        """Train one epoch for stage 2 with CE loss."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Stage 2]', ncols=100)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward without autocast (FP32 training for adapter/classifier)
            # Use stage 2 forward pass
            output_dict = self.model.forward_stage2(images, labels=labels)
            
            logits = output_dict['logits']
            ce_loss = output_dict['ce_loss']
        
            # Backward pass
            ce_loss.backward()
        
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
        
            # Optimizer step
            self.optimizer.step()
            
            # Metrics
            total_loss += ce_loss.item()
            with torch.no_grad():
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{ce_loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        train_acc = 100.0 * correct / total if total > 0 else 0
        
        return avg_loss, train_acc
    
    def _evaluate_stage2(self) -> float:
        """Evaluate stage 2 on ID test set."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                with autocast():
                    output_dict = self.model.forward_stage2(images, labels=None)
                    logits = output_dict['logits']
                
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        id_acc = 100.0 * correct / total if total > 0 else 0.0
        return id_acc
    
    def _evaluate_ood_stage2(self, ood_loaders: Dict[str, DataLoader]) -> Dict:
        """
        Evaluate OOD performance using three methods:
        1. DS-MCM: Dual-Stream Contrastive Inference
        2. Visual-GL-MCM: Pure visual GL-MCM (adapter output + local features with visual prototypes)
        3. Stage1-GL-MCM: Stage 1 GL-MCM (global features)
        
        Args:
            ood_loaders: Dict of OOD dataset loaders
        
        Returns:
            Dict containing OOD metrics for all methods
        """
        self.model.eval()
        results = {}
        
        # Create args-like object for detection_util functions
        class Args:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        
        # Compute ID scores once for all OOD datasets (for Stage1-GL-MCM)
        args_stage1 = Args(
            T=self.temperature,
            score='GL-MCM',
            lambda_local=self.lambda_local
        )
        id_scores_stage1 = get_ood_scores_clip(args_stage1, self.model, self.test_loader, self.classnames)
        
        # Compute ID scores for DS-MCM
        args_ds = Args(
            T=self.temperature,
            score='DS-MCM',
            lambda_local=self.lambda_local,
            fusion_strategy='geometric'
        )
        id_scores_ds = get_ood_scores_dual_stream(args_ds, self.model, self.test_loader, self.classnames)
        
        # Compute ID scores for Visual-GL-MCM
        id_scores_visual = self._compute_visual_gl_mcm_scores(self.test_loader, self.classnames)
        
        # Evaluate each OOD dataset
        for ood_name, ood_loader in ood_loaders.items():
            # 1. DS-MCM: Dual-Stream Contrastive Inference
            out_scores_ds = get_ood_scores_dual_stream(args_ds, self.model, ood_loader, self.classnames)
            measures_ds = detection_get_measures(-id_scores_ds, -out_scores_ds)
            results[f'DS-MCM_{ood_name}_auroc'] = measures_ds[0] * 100
            results[f'DS-MCM_{ood_name}_fpr95'] = measures_ds[2] * 100
            
            # 2. Visual-GL-MCM: Pure visual (adapter output + local features)
            out_scores_visual = self._compute_visual_gl_mcm_scores(ood_loader, self.classnames)
            measures_visual = detection_get_measures(-id_scores_visual, -out_scores_visual)
            results[f'Visual-GL-MCM_{ood_name}_auroc'] = measures_visual[0] * 100
            results[f'Visual-GL-MCM_{ood_name}_fpr95'] = measures_visual[2] * 100
            
            # 3. Stage1-GL-MCM: Stage 1 global features
            out_scores_stage1 = get_ood_scores_clip(args_stage1, self.model, ood_loader, self.classnames)
            measures_stage1 = detection_get_measures(-id_scores_stage1, -out_scores_stage1)
            results[f'Stage1-GL-MCM_{ood_name}_auroc'] = measures_stage1[0] * 100
            results[f'Stage1-GL-MCM_{ood_name}_fpr95'] = measures_stage1[2] * 100
        
        # Compute average OOD metrics for each method
        for method in ['DS-MCM', 'Visual-GL-MCM', 'Stage1-GL-MCM']:
            aurocs = []
            fpr95s = []
            for ood_name in ood_loaders.keys():
                aurocs.append(results[f'{method}_{ood_name}_auroc'])
                fpr95s.append(results[f'{method}_{ood_name}_fpr95'])
            results[f'{method}_avg_auroc'] = np.mean(aurocs) if aurocs else 0.0
            results[f'{method}_avg_fpr95'] = np.mean(fpr95s) if fpr95s else 0.0
        
        return results
    
    def _compute_visual_gl_mcm_scores(self, loader: DataLoader, test_labels: List[str]) -> np.ndarray:
        """
        Compute Visual-GL-MCM scores using adapter output and local features.
        Uses visual prototypes (initialized from text features).
        
        Args:
            loader: Data loader
            test_labels: Class labels for text features
        
        Returns:
            OOD scores array
        """
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)
        _score = []
        
        # Use cached text features (visual prototypes)
        if hasattr(self.model, 'text_features') and self.model.text_features is not None:
            text_features = self.model.text_features
        else:
            tokenizer = clip.tokenize
            with torch.no_grad():
                text_inputs = tokenizer([f"a photo of a {c}" for c in test_labels])
                text_features = self.model.encode_text(text_inputs.cuda()).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
        
        tqdm_object = tqdm(loader, total=len(loader), desc='Visual-GL-MCM', ncols=100, leave=False)
        
        with torch.no_grad():
            with autocast():
                for batch_idx, (images, labels, *id_flag) in enumerate(tqdm_object):
                    images = images.cuda()
                    
                    # Stage 2 forward pass
                    output_dict = self.model.forward_stage2(images, labels=None, return_features=True)
                    
                    # Get adapter output (visual features)
                    adapted_feats = output_dict['adapted_feats']  # [B, D]
                    
                    # Get local features (selected patches)
                    selected_feats = output_dict['selected_feats']  # [B, N, D]
                    
                    # Normalize features
                    adapted_feats = F.normalize(adapted_feats, dim=-1, eps=1e-8)
                    selected_feats = F.normalize(selected_feats, dim=-1, eps=1e-8)
                    
                    # Global score: adapter output
                    output_global = adapted_feats @ text_features.T  # [B, C]
                    smax_global = to_np(F.softmax(output_global / self.temperature, dim=1))
                    mcm_global_score = -np.max(smax_global, axis=1)
                    
                    # Local score: selected patches
                    # Compute similarity: [B, N, D] @ [D, C] -> [B, N, C]
                    sim = selected_feats @ text_features.T
                    
                    # Max-Pooling: Find best matching patch for each class
                    val, _ = sim.max(dim=1)  # [B, C]
                    
                    # Apply temperature and softmax
                    s_local = to_np(F.softmax(val / self.temperature, dim=1))
                    mcm_local_score = -np.max(s_local, axis=1)
                    
                    # Fusion: Global + lambda * Local
                    final_score = mcm_global_score + self.lambda_local * mcm_local_score
                    
                    _score.append(final_score)
        
        return concat(_score)[:len(loader.dataset)].copy()
    
    def _save_stage2_checkpoint(self, epoch: int, train_loss: float, 
                               train_acc: float, id_acc: float, ood_results: Dict = None):
        """Save stage 2 checkpoint."""
        checkpoint_path = os.path.join(self.log_dir, 'checkpoints', f'stage2_epoch_{epoch:03d}.pt')
        
        # Extract only stage 2 trainable parameters
        stage2_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                stage2_state[name] = param.data.clone()
        
        checkpoint = {
            'epoch': epoch,
            'stage': 2,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'id_acc': id_acc,
            'state_dict': stage2_state,
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        
        # Add OOD results if available
        if ood_results is not None:
            checkpoint['ood_results'] = ood_results
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.debug(f'  ✓ Saved checkpoint: {checkpoint_path}')
        return checkpoint_path

    def _compute_auroc(self, id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
        """Compute AUROC (wrapped for backward compatibility)"""
        auroc, _, _ = self.get_measures(id_scores, ood_scores)
        return auroc * 100
    
    def _compute_fpr95(self, id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
        """Compute FPR95 (wrapped for backward compatibility)"""
        _, _, fpr = self.get_measures(id_scores, ood_scores, recall_level=0.95)
        return fpr * 100
    
    def evaluate_epoch(self, epoch: int) -> Dict:
        """Evaluate on ID and all OOD datasets."""
        results = {}
        
        # ID evaluation
        id_acc = self.evaluate_id()
        results['id_accuracy'] = id_acc
        
        # OOD evaluations
        ood_aurocs = []
        ood_fpr95s = []
        
        # Compute ID scores once for all OOD datasets
        class Args:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        
        args = Args(
            T=self.temperature,
            score=self.score_type,
            lambda_local=self.lambda_local
        )
        
        id_scores = get_ood_scores_clip(args, self.model, self.test_loader, self.classnames)
        
        for ood_name in self.ood_loaders.keys():
            auroc, fpr95 = self.evaluate_ood_dataset(ood_name, id_scores)
            results[f'{ood_name}_auroc'] = auroc
            results[f'{ood_name}_fpr95'] = fpr95
            ood_aurocs.append(auroc)
            ood_fpr95s.append(fpr95)
        
        # Averages
        results['avg_ood_auroc'] = np.mean(ood_aurocs) if ood_aurocs else 0.0
        results['avg_ood_fpr95'] = np.mean(ood_fpr95s) if ood_fpr95s else 0.0
        
        return results
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save checkpoint with only trainable parameters."""
        checkpoint_path = os.path.join(self.log_dir, 'checkpoints', f'epoch_{epoch:03d}.pt')
        
        # Extract only trainable parameters
        trainable_state = {}
        for name, param in self.model.named_parameters():
            # Save if requires grad OR if it's a buffer like running_mean
            if param.requires_grad:
                trainable_state[name] = param.data.clone()
        
        checkpoint = {
            'epoch': epoch,
            'method': self.method,
            'seed': self.seed,
            'state_dict': trainable_state,
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(), # Save scaler state
            'metrics': metrics,
            'config': {
                'selector_type': self.selector_type,
                'fuser_type': self.fuser_type,
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def train_with_eval(self):
        """Main training loop."""
        self.logger.debug('\n' + '='*80)
        self.logger.debug(f'Starting training: {self.method} (seed={self.seed})')
        self.logger.debug('='*80 + '\n')
        
        # Setup
        self.setup_data()
        self.setup_model()
        
        best_avg_auroc = 0.0
        results_history = []
        
        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc, _ = self.train_epoch(epoch)
            self.scheduler.step()
            
            eval_results = {}
            
            # Log results
            self.logger.debug(f'\n[Epoch {epoch+1}/{self.epochs}]')
            self.logger.debug(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

            if (epoch+1) % 10 == 0 and epoch+1 > 20:
                # Evaluate (reduced frequency from 5 to 10 epochs)
                eval_results = self.evaluate_epoch(epoch)
                self.logger.debug(f'  ID Accuracy: {eval_results["id_accuracy"]:.2f}%')
                
                for ood_name in self.ood_loaders.keys():
                    auroc = eval_results[f'{ood_name}_auroc']
                    fpr95 = eval_results[f'{ood_name}_fpr95']
                    self.logger.debug(f'  {ood_name:15} AUROC: {auroc:.2f}%, FPR95: {fpr95:.2f}%')
                
                avg_auroc = eval_results["avg_ood_auroc"]
                avg_fpr95 = eval_results["avg_ood_fpr95"]
                self.logger.debug(f'  Avg OOD AUROC: {avg_auroc:.2f}%, Avg OOD FPR95: {avg_fpr95:.2f}%')
            
                # Save checkpoint (Every 10 epochs or best)
                if (epoch % 10 == 0 or epoch == self.epochs - 1) and epoch+1 > 20:
                    self.save_checkpoint(epoch, eval_results)
            
                # Track best
                if avg_auroc > best_avg_auroc:
                    best_avg_auroc = avg_auroc
                    self.save_checkpoint(999, eval_results) # 999 as code for 'best'
                    self.logger.debug(f"  ★ New Best Avg AUROC: {best_avg_auroc:.2f}%")
            
            # Update history
            results_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                **eval_results
            })
                
        # Save all results at once after training completes
        with open(os.path.join(self.log_dir, 'results.json'), 'w') as f:
            json.dump(results_history, f, indent=2)
                
        self.logger.debug('\nTraining completed.')





def main():
    parser = argparse.ArgumentParser(description='Train modular OOD detection with AMP')
    
    # Basic Config
    parser.add_argument('--method', type=str, default='GL_MCM_FA')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.005) 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    # Components
    parser.add_argument('--selector_type', type=str, default='slot', help="'mlp' or 'slot'")
    parser.add_argument('--fuser_type', type=str, default='query_attn')
    
    # Dataset
    parser.add_argument('--id_dataset', type=str, default='ImageNet')
    parser.add_argument('--root_path', type=str, default='/data/datasets')
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--use_full_data', action='store_true')
    parser.add_argument('--num_ood_sumple', type=int, default=-1, help='Number of OOD samples to use (-1 for all)')
    
    # Loss Weights
    parser.add_argument('--lambda_llm_negatives', type=float, default=0.1)
    parser.add_argument('--lambda_mixup', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=0.2)
    
    # Paths
    parser.add_argument('--class_negatives_path', type=str, default='')
    parser.add_argument('--backbone', type=str, default='ViT-B/16')

    # Model parameters
    parser.add_argument('--num_select', type=int, default=16,
                        help='Number of tokens/features to retain in selector (k)')

    # Advanced params
    parser.add_argument('--selector_temperature', type=float, default=1.0)
    parser.add_argument('--patches_per_slot_attn', type=int, default=16)
    
    # OOD score parameters
    parser.add_argument('--score_type', type=str, default='GL-MCM', help='OOD scoring method')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for OOD scoring')
    parser.add_argument('--lambda_local', type=float, default=1.0, help='Weight for local component in GL-MCM')
    
    # Stage 2 training parameters
    parser.add_argument('--train_stage', type=int, default=1, choices=[1, 2], 
                        help='Training stage: 1 (selector+fuser) or 2 (adapter+classifier)')
    parser.add_argument('--stage1_checkpoint', type=str, default=None,
                        help='Path to stage 1 checkpoint (required for stage 2 training)')
    parser.add_argument('--stage2_epochs', type=int, default=10,
                        help='Number of epochs for stage 2 training')
    parser.add_argument('--stage2_lr', type=float, default=0.001,
                        help='Learning rate for stage 2 training')
    parser.add_argument('--use_weighted_pool', action='store_true',
                        help='Use score-weighted pooling in VisualAdapter')
    parser.add_argument('--adapter_hidden_dim', type=int, default=None,
                        help='Hidden dimension for VisualAdapter (default: same as input_dim)')
    parser.add_argument('--use_visual_prototypes', type=eval, default=False,
                        help='Use class feature centers for visual prototype initialization')

    args = parser.parse_args()

    # Create trainer
    trainer = TrainEvalOrchestrator(
        method=args.method,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=torch.device(args.device),
        selector_type=args.selector_type,
        fuser_type=args.fuser_type,
        id_dataset=args.id_dataset,
        root_path=args.root_path,
        shots=args.shots,
        lambda_llm_negatives=args.lambda_llm_negatives,
        lambda_mixup=args.lambda_mixup,
        margin=args.margin,
        backbone=args.backbone,
        class_negatives_path=args.class_negatives_path,
        use_full_data=args.use_full_data,
        num_ood_sumple=args.num_ood_sumple,
        num_select=args.num_select,
        selector_temperature=args.selector_temperature,
        patches_per_slot_attn=args.patches_per_slot_attn,
        score_type=args.score_type,
        temperature=args.temperature,
        lambda_local=args.lambda_local,
        use_weighted_pool=args.use_weighted_pool,
        adapter_hidden_dim=args.adapter_hidden_dim,
        use_visual_prototypes=args.use_visual_prototypes
    )

    # Check which stage to train
    if args.train_stage == 2:
        # Stage 2 training
        if args.stage1_checkpoint is None:
            raise ValueError("--stage1_checkpoint is required for stage 2 training")
        
        # Setup data and model (needed for stage 2)
        trainer.setup_data()
        trainer.setup_model()
        
        # Train stage 2
        stage2_results = trainer.train_stage2(
            stage1_checkpoint=args.stage1_checkpoint,
            epochs=args.stage2_epochs,
            lr=args.stage2_lr
        )
        
        print(f"\nStage 2 Training Complete!")
        print(f"Best ID Accuracy: {stage2_results['best_id_acc']:.2f}%")
    else:
        # Stage 1 training (default)
        trainer.train_with_eval()

if __name__ == '__main__':
    main()
