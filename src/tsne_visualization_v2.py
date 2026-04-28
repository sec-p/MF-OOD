import os
import sys
import argparse
import numpy as np
import torch
import random
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.train_eval_util import set_model_clip, set_val_loader, set_ood_loader_ImageNet
from utils.common import setup_seed, get_test_labels
from src.model_modular import build_modular_model


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_locoop_text_features(text_feature_path, class_names_path):
    """Load LoCoOP trained text features."""
    print(f"Loading LoCoOP text features from: {text_feature_path}")
    text_features = torch.load(text_feature_path, map_location='cpu')
    
    print(f"Loading class names from: {class_names_path}")
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    if isinstance(text_features, dict):
        if 'text_features' in text_features:
            text_features = text_features['text_features']
        elif 'text_embeddings' in text_features:
            text_features = text_features['text_embeddings']
        elif 'logit_scale' in text_features:
            text_features = text_features['text_features']
    
    if not torch.is_tensor(text_features):
        text_features = torch.tensor(text_features)
    
    print(f"  Text features shape: {text_features.shape}")
    print(f"  Number of class names: {len(class_names)}")
    return text_features, class_names


def load_imagenet_class_mapping(root_dir):
    """Load ImageNet synset ID to class name mapping."""
    mapping_file = os.path.join(root_dir, 'imagenet', 'classnames.txt')
    synset_to_name = {}
    name_to_synset = {}
    
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    synset_id, class_name = parts
                    synset_to_name[synset_id] = class_name
                    name_to_synset[class_name] = synset_id
        print(f"Loaded class mapping for {len(synset_to_name)} classes")
    else:
        print(f"Warning: Class mapping file not found at {mapping_file}")
    
    return synset_to_name, name_to_synset


def select_id_classes_from_text_features(all_class_names, num_classes=10, seed=42):
    """Select ID classes from the available text feature class names."""
    np.random.seed(seed)
    selected_indices = np.random.choice(len(all_class_names), num_classes, replace=False)
    selected_class_names = [all_class_names[i] for i in selected_indices]
    print(f"\nSelected {num_classes} ID classes:")
    for i, name in enumerate(selected_class_names):
        print(f"  {i+1}. {name}")
    return selected_class_names, selected_indices


def extract_features(clip_model, modular_model, id_dataloader, ood_dataloader, 
                     id_class_indices, all_class_names, text_features_locoop, device):
    """Extract visual features from ID and OOD samples."""
    clip_model.eval()
    if modular_model is not None:
        modular_model.eval()
    
    id_visual_clip = []
    id_visual_modular = []
    id_labels = []
    ood_visual_clip = []
    ood_visual_modular = []
    ood_labels = []
    
    # Process ID samples
    print("\nExtracting ID visual features...")
    class_sample_count = defaultdict(int)
    
    with torch.no_grad():
        for images, labels in tqdm(id_dataloader, desc="ID samples"):
            images = images.to(device)
            
            # Map original labels to our selected ID class indices
            for i in range(len(labels)):
                orig_label = labels[i].item()
                if orig_label in id_class_indices:
                    new_label = id_class_indices.index(orig_label)
                    if class_sample_count[new_label] < args.samples_per_id_class:
                        # Extract CLIP visual features
                        clip_visual = clip_model.encode_image(images[i:i+1])
                        clip_visual = clip_visual / clip_visual.norm(dim=-1, keepdim=True)
                        id_visual_clip.append(clip_visual.cpu().numpy())
                        
                        # Extract modular model visual features
                        if modular_model is not None:
                            modular_visual = modular_model.encode_image(images[i:i+1])
                            modular_visual = modular_visual / modular_visual.norm(dim=-1, keepdim=True)
                            id_visual_modular.append(modular_visual.cpu().numpy())
                        else:
                            id_visual_modular.append(clip_visual.cpu().numpy())
                        
                        id_labels.append(new_label)
                        class_sample_count[new_label] += 1
            
            # Check if we have enough samples per class
            if all(count >= args.samples_per_id_class for count in class_sample_count.values()):
                if len(class_sample_count) == args.num_id_classes:
                    print(f"\nCollected enough samples for all {args.num_id_classes} classes")
                    break
    
    # Process OOD samples
    print("\nExtracting OOD visual features...")
    ood_count = 0
    
    with torch.no_grad():
        for images, _ in tqdm(ood_dataloader, desc="OOD samples"):
            images = images.to(device)
            
            for i in range(len(images)):
                if ood_count >= args.num_ood_samples:
                    break
                
                # Extract CLIP visual features
                clip_visual = clip_model.encode_image(images[i:i+1])
                clip_visual = clip_visual / clip_visual.norm(dim=-1, keepdim=True)
                ood_visual_clip.append(clip_visual.cpu().numpy())
                
                # Extract modular model visual features
                if modular_model is not None:
                    modular_visual = modular_model.encode_image(images[i:i+1])
                    modular_visual = modular_visual / modular_visual.norm(dim=-1, keepdim=True)
                    ood_visual_modular.append(modular_visual.cpu().numpy())
                else:
                    ood_visual_modular.append(clip_visual.cpu().numpy())
                
                ood_labels.append(args.num_id_classes)
                ood_count += 1
            
            if ood_count >= args.num_ood_samples:
                break
    
    # Convert to numpy arrays
    id_visual_clip = np.concatenate(id_visual_clip, axis=0)
    id_visual_modular = np.concatenate(id_visual_modular, axis=0)
    id_labels = np.array(id_labels)
    ood_visual_clip = np.concatenate(ood_visual_clip, axis=0)
    ood_visual_modular = np.concatenate(ood_visual_modular, axis=0)
    ood_labels = np.array(ood_labels)
    
    print(f"\nFeature extraction complete:")
    print(f"  ID samples: {len(id_visual_clip)}")
    print(f"  OOD samples: {len(ood_visual_clip)}")
    print(f"  CLIP visual feature dim: {id_visual_clip.shape[1]}")
    
    return (id_visual_clip, id_visual_modular, id_labels,
            ood_visual_clip, ood_visual_modular, ood_labels)


def run_tsne(features, n_components=2, perplexity=30, n_iter=1000, random_state=42):
    """Run t-SNE dimensionality reduction."""
    print(f"Running t-SNE with {features.shape[0]} samples, {features.shape[1]} dimensions...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1
    )
    features_2d = tsne.fit_transform(features)
    return features_2d


def plot_tsne(visual_2d, text_2d, labels, id_class_names, title, save_path, 
              num_id_classes=10, text_type="Text"):
    """
    Plot t-SNE visualization.
    
    Args:
        visual_2d: All visual features (ID + OOD) in 2D
        text_2d: Text features (class centers) in 2D
        labels: Labels for visual features (0..num_id_classes-1 for ID, num_id_classes for OOD)
        id_class_names: Names of ID classes
        title: Plot title
        save_path: Path to save the plot
        num_id_classes: Number of ID classes
        text_type: Type of text features (for legend)
    """
    plt.figure(figsize=(16, 10))
    
    # Colors for ID classes
    colors = plt.cm.tab20(np.linspace(0, 1, num_id_classes + 1))
    id_colors = colors[:num_id_classes]
    ood_color = colors[num_id_classes]
    
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # Plot ID visual samples
    for i in range(num_id_classes):
        mask = labels == i
        if np.any(mask):
            class_name = id_class_names[i]
            short_name = class_name[:15] if len(class_name) > 15 else class_name
            
            plt.scatter(
                visual_2d[mask, 0], visual_2d[mask, 1],
                c=[id_colors[i]], s=80, alpha=0.7, edgecolors='white', linewidths=0.5,
                zorder=5
            )
            
            legend_elements.append(Line2D(
                [0], [0], marker='o', color='w', 
                markerfacecolor=id_colors[i], markersize=10,
                markeredgecolor='white', markeredgewidth=0.5,
                label=f'{short_name} (Visual)'
            ))
    
    # Plot OOD visual samples
    ood_mask = labels >= num_id_classes
    if np.any(ood_mask):
        plt.scatter(
            visual_2d[ood_mask, 0], visual_2d[ood_mask, 1],
            c=[ood_color], s=120, alpha=0.9, marker='X', edgecolors='black', linewidths=1.5,
            zorder=6
        )
        legend_elements.append(Line2D(
            [0], [0], marker='X', color='w',
            markerfacecolor=ood_color, markersize=12,
            markeredgecolor='black', markeredgewidth=1.5,
            label='OOD Samples (Visual)'
        ))
    
    # Plot text features (class centers)
    if text_2d is not None and len(text_2d) > 0:
        for i, (tx, ty) in enumerate(text_2d):
            plt.scatter(
                tx, ty, c=[id_colors[i % num_id_classes]],
                s=400, marker='*', edgecolors='black', linewidths=2.5,
                zorder=10
            )
            if id_class_names is not None and i < len(id_class_names):
                label = id_class_names[i][:12] if len(id_class_names[i]) > 12 else id_class_names[i]
                plt.annotate(
                    label, (tx + 3, ty + 3), fontsize=8, zorder=11,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6),
                    fontweight='bold'
                )
        
        legend_elements.append(Line2D(
            [0], [0], marker='*', color='w',
            markerfacecolor='gold', markersize=18,
            markeredgecolor='black', markeredgewidth=2,
            label=f'Text Features ({text_type})'
        ))
    
    plt.title(title, fontsize=16, pad=20, fontweight='bold')
    plt.legend(
        handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
        fontsize=10, framealpha=0.95, fancybox=True, shadow=True
    )
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('t-SNE Dimension 1', fontsize=13, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {save_path}")


def plot_comparison(clip_visual_2d, clip_text_2d, clip_locoop_visual_2d, clip_locoop_text_2d,
                    modular_visual_2d, modular_text_2d, labels, id_class_names, save_path):
    """Plot all three configurations in one figure."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    num_id_classes = len(id_class_names)
    colors = plt.cm.tab20(np.linspace(0, 1, num_id_classes + 1))
    id_colors = colors[:num_id_classes]
    ood_color = colors[num_id_classes]
    
    titles = [
        'Original CLIP',
        'CLIP + LoCoOP Text',
        'Your Model + LoCoOP Text'
    ]
    visual_data = [clip_visual_2d, clip_locoop_visual_2d, modular_visual_2d]
    text_data = [clip_text_2d, clip_locoop_text_2d, modular_text_2d]
    text_types = ['CLIP Text', 'LoCoOP Text', 'LoCoOP Text']
    
    for ax_idx, ax in enumerate(axes):
        visual_2d = visual_data[ax_idx]
        text_2d = text_data[ax_idx]
        
        # Plot ID samples
        for i in range(num_id_classes):
            mask = labels == i
            if np.any(mask):
                ax.scatter(
                    visual_2d[mask, 0], visual_2d[mask, 1],
                    c=[id_colors[i]], s=60, alpha=0.7, edgecolors='white', linewidths=0.3
                )
        
        # Plot OOD samples
        ood_mask = labels >= num_id_classes
        if np.any(ood_mask):
            ax.scatter(
                visual_2d[ood_mask, 0], visual_2d[ood_mask, 1],
                c=[ood_color], s=80, alpha=0.9, marker='X', edgecolors='black', linewidths=1
            )
        
        # Plot text features
        if text_2d is not None and len(text_2d) > 0:
            for i, (tx, ty) in enumerate(text_2d):
                ax.scatter(
                    tx, ty, c=[id_colors[i % num_id_classes]],
                    s=300, marker='*', edgecolors='black', linewidths=2
                )
        
        ax.set_title(titles[ax_idx], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('t-SNE 1', fontsize=11)
        ax.set_ylabel('t-SNE 2', fontsize=11)
    
    from matplotlib.lines import Line2D
    legend_elements = []
    for i in range(min(5, num_id_classes)):
        legend_elements.append(Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=id_colors[i], markersize=10,
            label=f'Class {i+1} (Visual)'
        ))
    legend_elements.append(Line2D(
        [0], [0], marker='X', color='w',
        markerfacecolor=ood_color, markersize=12,
        label='OOD (Visual)'
    ))
    legend_elements.append(Line2D(
        [0], [0], marker='*', color='w',
        markerfacecolor='gold', markersize=16,
        label='Text Features'
    ))
    
    fig.legend(
        handles=legend_elements, loc='center right', 
        bbox_to_anchor=(1.02, 0.5), fontsize=11
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comparison plot: {save_path}")


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.log_directory, exist_ok=True)
    print(f"Output directory: {args.log_directory}")
    
    # Load LoCoOP text features and class names
    text_features_locoop, all_class_names = load_locoop_text_features(
        args.text_feature_path, args.class_names_path
    )
    
    # Select ID classes from text feature class names
    id_class_names, id_class_indices = select_id_classes_from_text_features(
        all_class_names, num_classes=args.num_id_classes, seed=args.seed
    )
    
    # Get selected text features
    selected_text_locoop = text_features_locoop[id_class_indices].numpy()
    
    # Load models
    print("\nLoading models...")
    clip_model, _, _ = get_loaders(args, 'clip', None, None)
    clip_model = clip_model.to(device)
    
    # Load your model
    try:
        from src.model_modular import ModularCLIP
        modular_model = ModularCLIP()
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        modular_model.load_state_dict(checkpoint, strict=False)
        modular_model = modular_model.to(device)
        print("✓ Loaded your modular model")
    except Exception as e:
        print(f"Warning: Could not load your model: {e}")
        modular_model = None
    
    # Get CLIP's original text features
    print("\nGetting CLIP's original text features...")
    with torch.no_grad():
        class_templates = ['a photo of a {}.']
        text_tokens = clip.tokenize([template.format(name) for template in class_templates for name in id_class_names]).to(device)
        text_clip = clip_model.encode_text(text_tokens)
        text_clip = text_clip / text_clip.norm(dim=-1, keepdim=True)
        text_clip = text_clip.cpu().numpy()
    print(f"  CLIP text features shape: {text_clip.shape}")
    
    # Load data
    print("\nLoading data...")
    args.in_dataset = 'imagenet'
    args.batch_size = 64
    args.shuffle = True
    args.num_workers = 4
    
    id_loader, ood_loaders, num_classes = get_loaders(
        args, 'clip', None, None
    )
    
    # Select OOD dataset
    ood_dataset_name = args.ood_dataset
    ood_loader = None
    for name, loader in ood_loaders.items():
        if ood_dataset_name in name:
            ood_loader = loader
            break
    
    if ood_loader is None:
        print(f"Warning: OOD dataset {ood_dataset_name} not found, using first available")
        ood_loader = list(ood_loaders.values())[0]
    
    # Extract features
    (id_visual_clip, id_visual_modular, id_labels,
     ood_visual_clip, ood_visual_modular, ood_labels) = extract_features(
        clip_model, modular_model, id_loader, ood_loader,
        id_class_indices, all_class_names, text_features_locoop, device
    )
    
    # Concatenate all visual features (ID + OOD)
    all_visual_clip = np.concatenate([id_visual_clip, ood_visual_clip], axis=0)
    all_visual_modular = np.concatenate([id_visual_modular, ood_visual_modular], axis=0)
    all_labels = np.concatenate([id_labels, ood_labels], axis=0)
    
    print("\n" + "=" * 80)
    print("Running t-SNE...")
    print("=" * 80)
    
    # 1. Original CLIP: CLIP visual + CLIP text
    print("\n1. Original CLIP (CLIP visual + CLIP text)...")
    all_features_clip = np.concatenate([all_visual_clip, text_clip], axis=0)
    features_2d_clip = run_tsne(all_features_clip, perplexity=args.tsne_perplexity, random_state=args.seed)
    visual_2d_clip = features_2d_clip[:len(all_visual_clip)]
    text_2d_clip = features_2d_clip[len(all_visual_clip):]
    
    # 2. CLIP + LoCoOP Text: CLIP visual + LoCoOP text
    print("\n2. CLIP + LoCoOP Text (CLIP visual + LoCoOP text)...")
    all_features_clip_locoop = np.concatenate([all_visual_clip, selected_text_locoop], axis=0)
    features_2d_clip_locoop = run_tsne(all_features_clip_locoop, perplexity=args.tsne_perplexity, random_state=args.seed)
    visual_2d_clip_locoop = features_2d_clip_locoop[:len(all_visual_clip)]
    text_2d_locoop = features_2d_clip_locoop[len(all_visual_clip):]
    
    # 3. Your model + LoCoOP Text: Your model visual + LoCoOP text
    print("\n3. Your Model + LoCoOP Text (Your model visual + LoCoOP text)...")
    all_features_modular = np.concatenate([all_visual_modular, selected_text_locoop], axis=0)
    features_2d_modular = run_tsne(all_features_modular, perplexity=args.tsne_perplexity, random_state=args.seed)
    visual_2d_modular = features_2d_modular[:len(all_visual_modular)]
    text_2d_modular = features_2d_modular[len(all_visual_modular):]
    
    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)
    
    # Plot 1: Original CLIP
    save_path_1 = os.path.join(args.log_directory, 'tsne_1_clip_original.png')
    plot_tsne(
        visual_2d_clip, text_2d_clip, all_labels, id_class_names,
        title='Original CLIP\nCLIP Visual Features + CLIP Text Features',
        save_path=save_path_1,
        num_id_classes=args.num_id_classes,
        text_type='CLIP Original'
    )
    
    # Plot 2: CLIP + LoCoOP Text
    save_path_2 = os.path.join(args.log_directory, 'tsne_2_clip_locoop.png')
    plot_tsne(
        visual_2d_clip_locoop, text_2d_locoop, all_labels, id_class_names,
        title='CLIP Visual + LoCoOP Text Anchors',
        save_path=save_path_2,
        num_id_classes=args.num_id_classes,
        text_type='LoCoOP Trained'
    )
    
    # Plot 3: Your model + LoCoOP Text
    save_path_3 = os.path.join(args.log_directory, 'tsne_3_modular_locoop.png')
    plot_tsne(
        visual_2d_modular, text_2d_modular, all_labels, id_class_names,
        title='Your Model Visual + LoCoOP Text Anchors',
        save_path=save_path_3,
        num_id_classes=args.num_id_classes,
        text_type='LoCoOP Trained'
    )
    
    # Plot 4: Comparison
    save_path_compare = os.path.join(args.log_directory, 'tsne_comparison.png')
    plot_comparison(
        visual_2d_clip, text_2d_clip,
        visual_2d_clip_locoop, text_2d_locoop,
        visual_2d_modular, text_2d_modular,
        all_labels, id_class_names,
        save_path_compare
    )
    
    print("\n" + "=" * 80)
    print("✓ All plots generated successfully!")
    print("=" * 80)
    print(f"Results saved to: {args.log_directory}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='t-SNE Visualization for MF-OOD')
    parser.add_argument('--root-dir', type=str, required=True, help='Root directory for datasets')
    parser.add_argument('--model_path', type=str, required=True, help='Path to your model checkpoint')
    parser.add_argument('--text-feature-path', type=str, required=True, help='Path to LoCoOP text features')
    parser.add_argument('--class-names-path', type=str, required=True, help='Path to LoCoOP class names JSON')
    parser.add_argument('--ood-dataset', type=str, default='iNaturalist', help='OOD dataset name')
    parser.add_argument('--num-id-classes', type=int, default=10, help='Number of ID classes to use')
    parser.add_argument('--samples-per-id-class', type=int, default=30, help='Samples per ID class')
    parser.add_argument('--num-ood-samples', type=int, default=150, help='Number of OOD samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--name', type=str, default='tsne_viz', help='Experiment name')
    parser.add_argument('--tsne-perplexity', type=int, default=30, help='t-SNE perplexity')
    parser.add_argument('--tsne-n-components', type=int, default=2, help='t-SNE components')
    parser.add_argument('--tsne-n-iter', type=int, default=1000, help='t-SNE iterations')
    
    args = parser.parse_args()
    
    args.log_directory = os.path.join(project_root, f"tsne_results/ImageNet/{args.name}")
    
    main(args)
