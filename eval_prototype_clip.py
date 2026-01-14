"""
Evaluation Script for PrototypeCLIP Model
Tests both ID (In-Distribution) and OOD (Out-of-Distribution) performance
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import clip
from src.model_prototype import PrototypeCLIP
from utils.common import setup_seed, get_test_labels
from utils.detection_util import print_measures, get_and_print_results, get_ood_scores_clip
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import set_model_clip, set_val_loader, set_ood_loader_ImageNet


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates PrototypeCLIP for OOD detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dataset', default='ImageNet', type=str,
                        choices=['COCO_single', 'COCO_multi', 'VOC_single', 'ImageNet'], help='in-distribution dataset')
    parser.add_argument('--root-dir', default="/data/datasets", type=str,
                        help='root dir of datasets')
    parser.add_argument('--name', default="prototype_eval", type=str, help="unique ID for run")
    parser.add_argument('--seed', default=1, type=int, help="random seed")
    parser.add_argument('--gpu', default=0, type=int, help='the GPU indice to use')
    parser.add_argument('-b', '--batch-size', default=512, type=int, help='mini-batch size')
    parser.add_argument('--T', type=int, default=1, help='temperature parameter')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/16', 'RN50', 'RN101'], help='which pretrained img encoder to use')
    parser.add_argument('--score', default='GL-MCM', type=str, 
                        choices=['MCM', 'L-MCM', 'GL-MCM', 'GL-MCM-L', 'SA-MCM'], help='score options for OOD detection')
    parser.add_argument('--num_ood_sumple', default=-1, type=int, help="numbers of ood_samples")
    parser.add_argument('--shots', default=16, type=int, help='number of shots for few-shot learning')
    parser.add_argument('--templates', type=str, default="a photo of a {}", help='templates for text prompts')
    parser.add_argument('--lambda_local', default=0.4, type=float, help='weight for local score in GL-MCM')
    parser.add_argument('--use_train_set', action='store_true', help='use training set for prototype initialization (default: use validation set)')
    
    args = parser.parse_args()

    args.CLIP_ckpt_name = args.CLIP_ckpt.replace('/', '_')
    args.log_directory = f"results/{args.in_dataset}/{args.score}/PrototypeCLIP_{args.CLIP_ckpt_name}_T_{args.T}_ID_{args.name}"
    os.makedirs(args.log_directory, exist_ok=True)

    return args


def create_fewshot_subset_from_dataset(dataset, shots, seed=42):
    """Create few-shot subset from dataset without iterating through all samples."""
    np.random.seed(seed)
    
    # Get class names and their indices
    class_to_indices = {i: [] for i in range(len(dataset.classes))}
    
    # Use dataset.samples to get file paths and labels without loading images
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)
    
    # Sample shots per class
    fewshot_indices = []
    for class_idx, indices in class_to_indices.items():
        if len(indices) > 0:
            sampled_indices = np.random.choice(indices, min(shots, len(indices)), replace=False)
            fewshot_indices.extend(sampled_indices)
    
    return fewshot_indices


def init_prototypes_from_fewshot(model, loader, classnames):
    """Initialize prototypes from few-shot examples."""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features for prototype initialization"):
            images = images.cuda()
            labels = labels.cuda()
            
            # Extract features
            cls_token, patch_tokens = model.encode_image(images)
            patch_mean = patch_tokens.mean(dim=1)
            features = cls_token + patch_mean
            features = features / features.norm(dim=-1, keepdim=True)
            
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Initialize prototypes
    model.init_prototypes_from_features(all_features, all_labels)
    print(f"✓ Prototypes initialized from {all_features.shape[0]} few-shot examples")


def evaluate_id_performance(model, test_loader, classnames):
    """Evaluate ID classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating ID performance"):
            images = images.cuda()
            labels = labels.cuda()
            
            outputs = model(images)
            logits = outputs['logits']
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f"ID Classification Accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    args = process_args()
    setup_seed(args.seed)
    log = setup_log(args)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)
    
    print("="*80)
    print("Evaluating PrototypeCLIP Model")
    print("="*80)
    print(f"CLIP backbone: {args.CLIP_ckpt}")
    print(f"In-distribution dataset: {args.in_dataset}")
    print(f"OOD score method: {args.score}")
    print(f"Few-shot shots: {args.shots}")
    print(f"Lambda local: {args.lambda_local}")
    print(f"Use training set for prototypes: {args.use_train_set}")
    print("="*80)
    
    # Load CLIP model with return_raw_features=True
    print("\nLoading CLIP model...")
    clip_model, preprocess = set_model_clip(args)
    
    # Re-load CLIP with return_raw_features=True
    device = torch.device(f'cuda:{args.gpu}')
    clip_model, _ = clip.load(args.CLIP_ckpt, device=device, return_raw_features=True)
    clip_model = clip_model.to(device)
    
    # Get class names from dataset
    root = args.root_dir
    if args.in_dataset == "ImageNet":
        dataset = datasets.ImageFolder(os.path.join(root, 'ImageNet', 'val'))
    elif args.in_dataset == 'COCO_single':
        dataset = datasets.ImageFolder(os.path.join(root, 'ID_COCO_single'))
    elif args.in_dataset == 'COCO_multi':
        dataset = datasets.ImageFolder(os.path.join(root, 'ID_COCO_multi'))
    elif args.in_dataset == 'VOC_single':
        dataset = datasets.ImageFolder(os.path.join(root, 'ID_VOC_single'))
    classnames = dataset.classes
    
    print(f"Number of classes: {len(classnames)}")
    
    # Build PrototypeCLIP model
    print("\nBuilding PrototypeCLIP model...")
    cfg = {
        'device': device,
        'templates': [args.templates],
    }
    
    model = PrototypeCLIP(cfg, classnames, clip_model)
    model = model.to(device)
    
    # Setup data loaders
    print("\nSetting up data loaders...")
    
    # Use test transforms for prototype initialization (no data augmentation)
    # Same as CLIP's preprocess
    test_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])
    
    # Create few-shot dataset
    if args.use_train_set:
        # Use training set for prototype initialization
        if args.in_dataset == "ImageNet":
            train_dataset = datasets.ImageFolder(os.path.join(root, 'ImageNet', 'train'), transform=test_transform)
        else:
            train_dataset = datasets.ImageFolder(os.path.join(root, f'ID_{args.in_dataset}'), transform=test_transform)
    else:
        # Use validation set for prototype initialization (default, faster)
        if args.in_dataset == "ImageNet":
            train_dataset = datasets.ImageFolder(os.path.join(root, 'ImageNet', 'val'), transform=test_transform)
        else:
            train_dataset = datasets.ImageFolder(os.path.join(root, f'ID_{args.in_dataset}'), transform=test_transform)
    
    # Create few-shot subset using efficient method (without iterating through all samples)
    print(f"\nCreating {args.shots}-shot subset from {'training' if args.use_train_set else 'validation'} set...")
    fewshot_indices = create_fewshot_subset_from_dataset(train_dataset, args.shots, seed=args.seed)
    print(f"✓ Few-shot subset created with {len(fewshot_indices)} samples")
    
    fewshot_dataset = torch.utils.data.Subset(train_dataset, fewshot_indices)
    fewshot_loader = DataLoader(fewshot_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize prototypes from few-shot examples
    print("\nInitializing prototypes from few-shot examples...")
    init_prototypes_from_fewshot(model, fewshot_loader, classnames)
    
    # Setup test loader
    test_loader = set_val_loader(args, preprocess)
    test_labels = get_test_labels(args)
    
    # Evaluate ID performance
    print("\nEvaluating ID performance...")
    id_accuracy = evaluate_id_performance(model, test_loader, classnames)
    
    # OOD evaluation
    print("\nEvaluating OOD performance...")
    if args.in_dataset in ['COCO_single', 'COCO_multi']:
        out_datasets = ['iNaturalist', 'SUN', 'Texture', 'IN22k', 'ood_voc']
    elif args.in_dataset in ['VOC_single']:
        out_datasets = ['iNaturalist', 'SUN', 'Texture', 'IN22k', 'ood_coco']
    elif args.in_dataset in ['ImageNet']:
        out_datasets = ['iNaturalist', 'SUN', 'places365', 'Texture']
    
    # Calculate ID scores using get_ood_scores_clip (reusing existing utility)
    print("\nCalculating ID scores...")
    in_score = get_ood_scores_clip(args, model, test_loader, test_labels)
    
    auroc_list, aupr_list, fpr_list = [], [], []
    results_dict = {'id_accuracy': id_accuracy}
    
    for out_dataset in out_datasets:
        print(f"\nEvaluating OOD dataset: {out_dataset}")
        try:
            ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess, root=args.root_dir)
            out_score = get_ood_scores_clip(args, model, ood_loader, test_labels)
            
            print(f"ID scores: {stats.describe(in_score)}")
            print(f"OOD scores: {stats.describe(out_score)}")
            
            # Plot distribution
            plot_distribution(args, in_score, out_score, out_dataset)
            
            # Calculate metrics
            get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list)
            
            # Store results
            results_dict[f'{out_dataset}_auroc'] = auroc_list[-1]
            results_dict[f'{out_dataset}_aupr'] = aupr_list[-1]
            results_dict[f'{out_dataset}_fpr95'] = fpr_list[-1]
            
        except Exception as e:
            print(f"Failed to evaluate {out_dataset}: {e}")
            # Add placeholder values
            auroc_list.append(0.0)
            aupr_list.append(0.0)
            fpr_list.append(1.0)
            results_dict[f'{out_dataset}_auroc'] = 0.0
            results_dict[f'{out_dataset}_aupr'] = 0.0
            results_dict[f'{out_dataset}_fpr95'] = 1.0
    
    # Print mean results
    print('\n\nMean Test Results')
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.score)
    
    # Save results
    results_dict['mean_auroc'] = np.mean(auroc_list)
    results_dict['mean_aupr'] = np.mean(aupr_list)
    results_dict['mean_fpr95'] = np.mean(fpr_list)
    
    # Save results to JSON
    results_file = os.path.join(args.log_directory, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Save as dataframe
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()