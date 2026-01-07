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
                 lambda_local: float = 1.0):
        
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
            
            # Feature flags
            'use_redundancy_loss': True,
            'use_llm_negatives': False if self.lambda_llm_negatives > 0 else False,
            'use_semantic_exclusion': True,
            'use_mixup_invariance': True if self.lambda_mixup > 0 else False,
            
            # Loss weights
            'lambda_redundancy': 1.0,
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
        
        return auroc, aupr, fpr
    
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

            if (epoch+1) % 10 == 0 and epoch+1 > 1:
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
        lambda_local=args.lambda_local
    )

    trainer.train_with_eval()


if __name__ == '__main__':
    main()
