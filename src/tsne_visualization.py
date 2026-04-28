import os
import sys
import json
import argparse
import warnings
from collections import defaultdict

import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.train_eval_util import set_model_clip, set_ood_loader_ImageNet
from utils.common import setup_seed
from src.model_modular import build_modular_model


def parse_args():
    parser = argparse.ArgumentParser(description='Distance Metrics Evaluation (Full Torch/GPU Version)')

    parser.add_argument('--in_dataset', default='ImageNet', type=str,
                        choices=['ImageNet'], help='in-distribution dataset')
    parser.add_argument('--root-dir', default="./datasets", type=str,
                        help='root dir of datasets')
    parser.add_argument('--name', default="distance_eval_torch", type=str,
                        help="unique ID for run")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--gpu', default=0, type=int, help='the GPU index to use')
    parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--num-workers', default=8, type=int, help='num_workers for dataloader')

    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/16', 'RN50', 'RN101'],
                        help='which pretrained img encoder to use')

    parser.add_argument('--model_path', type=str, required=True,
                        help='path to trained modular model checkpoint')
    parser.add_argument('--text-feature-path', type=str, required=True,
                        help='path to text features (LoCoOP anchors)')
    parser.add_argument('--class-names-path', type=str, required=True,
                        help='path to class names JSON file')

    parser.add_argument('--selector_type', type=str, default='identity',
                        choices=['mlp', 'slot', 'identity'], help='selector type')
    parser.add_argument('--fuser_type', type=str, default='shared_adapter',
                        choices=['shared_adapter', 'mean', 'query_attn', 'self_attn'],
                        help='fuser type')
    parser.add_argument('--num_select', type=int, default=49,
                        help='number of features to select')
    parser.add_argument('--num_heads_selector', type=int, default=8,
                        help='number of heads for selector')
    parser.add_argument('--num_heads_fuser', type=int, default=8,
                        help='number of heads for fuser')
    parser.add_argument('--templates', type=str, default="a photo of a {}",
                        help='template for text prompts')
    parser.add_argument('--residual_coef', type=float, default=0.2,
                        help='residual coefficient')

    parser.add_argument('--ood-datasets', nargs='+',
                        default=['iNaturalist', 'SUN', 'places365', 'Texture'],
                        help='OOD datasets to use')

    parser.add_argument('--use-amp', action='store_true',
                        help='use automatic mixed precision during feature extraction')
    parser.add_argument('--save-features', action='store_true',
                        help='save extracted features to disk')
    parser.add_argument('--metric-device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='device used to compute metrics')

    args = parser.parse_args()
    args.CLIP_ckpt_name = args.CLIP_ckpt.replace('/', '_')
    args.log_directory = os.path.join(project_root, f"distance_results/{args.in_dataset}/{args.name}")
    os.makedirs(args.log_directory, exist_ok=True)
    return args


def l2_normalize_torch(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def load_imagenet_class_mapping(root_dir):
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

    return synset_to_name, name_to_synset


def build_imagenet_label_to_text_index(id_dataset, locoop_class_names, root_dir):
    """
    构建 ImageNet label -> text_feature_index 的映射
    """
    synset_to_name, _ = load_imagenet_class_mapping(root_dir)
    class_name_to_text_idx = {name: i for i, name in enumerate(locoop_class_names)}

    label_to_text_idx = {}
    missing = []

    for synset, label_idx in id_dataset.class_to_idx.items():
        class_name = synset_to_name.get(synset, None)
        if class_name is None:
            missing.append((synset, label_idx, "synset_not_found"))
            continue
        if class_name not in class_name_to_text_idx:
            missing.append((synset, label_idx, f"class_name_not_in_text_features: {class_name}"))
            continue
        label_to_text_idx[label_idx] = class_name_to_text_idx[class_name]

    return label_to_text_idx, missing


def load_external_text_features_torch(text_feature_path, class_names_path):
    file_ext = os.path.splitext(text_feature_path)[1].lower()

    if file_ext == '.pt':
        loaded_data = torch.load(text_feature_path, map_location='cpu')
        if isinstance(loaded_data, dict):
            if 'text_features' in loaded_data:
                text_features = loaded_data['text_features']
            else:
                raise KeyError(f"'text_features' not found in {text_feature_path}")
        else:
            text_features = loaded_data

        if not isinstance(text_features, torch.Tensor):
            text_features = torch.tensor(text_features, dtype=torch.float32)
        else:
            text_features = text_features.float()

    elif file_ext == '.npy':
        import numpy as np
        text_features = torch.from_numpy(np.load(text_feature_path)).float()

    else:
        raise ValueError(f"Unsupported text feature format: {file_ext}")

    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    text_features = l2_normalize_torch(text_features)

    return text_features.cpu(), class_names


def get_text_features_clip_torch(clip_model, classnames, device, template="a photo of a {}"):
    import clip
    tokenizer = clip.tokenize

    all_text_features = []
    batch_size = 256

    with torch.no_grad():
        for i in tqdm(range(0, len(classnames), batch_size), desc="Extracting CLIP text features"):
            batch_classnames = classnames[i:i + batch_size]
            prompts = [template.format(c) for c in batch_classnames]
            text_inputs = tokenizer(prompts).to(device)
            text_features = clip_model.encode_text(text_inputs)
            text_features = l2_normalize_torch(text_features)
            all_text_features.append(text_features.float().cpu())

    return torch.cat(all_text_features, dim=0)


def extract_both_features_torch(model, loader, device, use_amp=False, desc="Extracting features"):
    """
    一次 forward 同时提取:
      - global_feature: 原始 CLIP visual feature
      - final_feats:   modular visual feature

    返回 CPU tensors，减少 GPU 占用
    """
    all_global_features = []
    all_final_features = []
    all_labels = []

    model.eval()
    amp_enabled = use_amp and device.type == 'cuda'

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            if len(batch) < 2:
                raise ValueError("Batch format not supported. Expect at least (images, labels).")

            images, labels = batch[:2]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                res = model(images)
                global_features = res['global_feature']
                final_features = res['final_feats']

            global_features = l2_normalize_torch(global_features).float().cpu()
            final_features = l2_normalize_torch(final_features).float().cpu()
            labels = labels.long().cpu()

            all_global_features.append(global_features)
            all_final_features.append(final_features)
            all_labels.append(labels)

    global_features = torch.cat(all_global_features, dim=0)
    final_features = torch.cat(all_final_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return global_features, final_features, labels


def prepare_dataloaders(args, preprocess):
    from torchvision import datasets
    from torch.utils.data import DataLoader, ConcatDataset

    imagenet_root = os.path.join(args.root_dir, 'imagenet', 'val')
    if not os.path.exists(imagenet_root):
        imagenet_root = os.path.join(args.root_dir, 'imagenet', 'images', 'val')

    if not os.path.exists(imagenet_root):
        raise FileNotFoundError(f"ImageNet val directory not found at: {imagenet_root}")

    id_dataset = datasets.ImageFolder(imagenet_root, transform=preprocess)

    all_ood_datasets = []
    print(f"\nLoading OOD datasets: {args.ood_datasets}")
    for ood_name in args.ood_datasets:
        try:
            ood_dataset = set_ood_loader_ImageNet(
                args, ood_name, preprocess, root=args.root_dir
            ).dataset
            all_ood_datasets.append(ood_dataset)
            print(f"  Loaded {ood_name}: {len(ood_dataset)} samples")
        except Exception as e:
            print(f"  Warning: Failed to load {ood_name}: {e}")

    if len(all_ood_datasets) == 0:
        raise RuntimeError("No OOD datasets loaded successfully.")

    ood_combined = ConcatDataset(all_ood_datasets)
    print(f"Total OOD samples: {len(ood_combined)}")

    kwargs = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': args.num_workers,
        'pin_memory': True
    }

    if args.num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2

    id_loader = DataLoader(id_dataset, **kwargs)
    ood_loader = DataLoader(ood_combined, **kwargs)

    return id_dataset, ood_combined, id_loader, ood_loader


def compute_intra_class_distance_to_text_torch(
    visual_features, labels, text_features, label_to_text_idx, metric_device
):
    """
    类内距离:
      ID样本到其对应类别文本中心的平均余弦距离

    visual_features: (N_id, D) tensor
    labels:          (N_id,) tensor
    text_features:   (C_text, D) tensor
    label_to_text_idx: dict[int -> int]
    """
    # 构建有效 mask 和 text index
    valid_mask_list = []
    text_index_list = []

    labels_cpu = labels.cpu().tolist()
    for y in labels_cpu:
        if y in label_to_text_idx:
            valid_mask_list.append(True)
            text_index_list.append(label_to_text_idx[y])
        else:
            valid_mask_list.append(False)

    valid_mask = torch.tensor(valid_mask_list, dtype=torch.bool)
    if valid_mask.sum().item() == 0:
        return 0.0, torch.empty(0, dtype=torch.float32)

    vf = visual_features[valid_mask].to(metric_device, non_blocking=True)
    lb = labels[valid_mask].to(metric_device, non_blocking=True)
    text_indices = torch.tensor(text_index_list, dtype=torch.long, device=metric_device)
    tf = text_features.to(metric_device, non_blocking=True)[text_indices]

    cosine_sim = (vf * tf).sum(dim=1)
    cosine_dist = 1.0 - cosine_sim

    intra_dist = cosine_dist.mean()

    unique_labels = torch.unique(lb)
    class_intra_dists = []
    for c in unique_labels:
        mask = (lb == c)
        class_intra_dists.append(cosine_dist[mask].mean())
    class_intra_dists = torch.stack(class_intra_dists, dim=0) if len(class_intra_dists) > 0 else torch.empty(0, device=metric_device)

    return float(intra_dist.item()), class_intra_dists.detach().cpu()


def compute_text_visual_alignment_torch(
    visual_features, labels, text_features, label_to_text_idx, metric_device
):
    """
    这里与类内距离定义相同
    """
    return compute_intra_class_distance_to_text_torch(
        visual_features, labels, text_features, label_to_text_idx, metric_device
    )


def build_label_index_tensors(labels, num_total_labels):
    """
    labels: (N,)
    返回:
      label_indices: list[tensor], 每个标签对应的样本索引
    """
    label_indices = []
    for c in range(num_total_labels):
        idx = torch.nonzero(labels == c, as_tuple=False).squeeze(1)
        label_indices.append(idx)
    return label_indices


def compute_inter_class_distance_fast_torch(
    all_features, all_labels, num_id_classes, metric_device
):
    """
    类间距离:
      仅对 ID 样本统计
      每个 ID 样本 到 全部“非本类标签”的样本 的平均余弦距离
      非本类标签包括:
        - 其他 ID 类
        - OOD 标签

    all_features: (N, D), 已归一化
    all_labels:   (N,)
    """
    all_features = all_features.to(metric_device, non_blocking=True)
    all_labels = all_labels.to(metric_device, non_blocking=True)

    id_mask = all_labels < num_id_classes
    if id_mask.sum().item() == 0:
        return 0.0

    total_sum = all_features.sum(dim=0, dtype=torch.float64)  # (D,)
    N_all = all_features.shape[0]

    # OOD 被统一记为 num_id_classes，因此总标签数为 num_id_classes + 1
    num_total_labels = num_id_classes + 1
    label_indices = build_label_index_tensors(all_labels, num_total_labels)

    total_dist_sum = torch.tensor(0.0, dtype=torch.float64, device=metric_device)
    total_pair_count = 0

    for c in range(num_id_classes):
        idx_c = label_indices[c]
        if idx_c.numel() == 0:
            continue

        Xc = all_features[idx_c]                       # (Nc, D)
        Nc = Xc.shape[0]
        if Nc == 0:
            continue

        class_sum = Xc.sum(dim=0, dtype=torch.float64)   # (D,)

        # 每个样本对全体样本相似度和
        sim_to_all = Xc.double() @ total_sum            # (Nc,)
        # 每个样本对本类样本相似度和
        sim_to_same = Xc.double() @ class_sum           # (Nc,)

        sim_to_other = sim_to_all - sim_to_same
        num_other = N_all - Nc
        if num_other <= 0:
            continue

        # Σ(1 - sim) = num_other - Σsim
        dist_to_other = num_other - sim_to_other

        total_dist_sum += dist_to_other.sum()
        total_pair_count += int(Nc * num_other)

    inter_dist = total_dist_sum / max(total_pair_count, 1)
    return float(inter_dist.item())


def compute_all_metrics_torch(
    id_visual_features,
    id_labels,
    all_visual_features,
    all_labels,
    text_features,
    label_to_text_idx,
    num_id_classes,
    metric_device
):
    intra_dist, class_intra_dists = compute_intra_class_distance_to_text_torch(
        visual_features=id_visual_features,
        labels=id_labels,
        text_features=text_features,
        label_to_text_idx=label_to_text_idx,
        metric_device=metric_device
    )

    inter_dist = compute_inter_class_distance_fast_torch(
        all_features=all_visual_features,
        all_labels=all_labels,
        num_id_classes=num_id_classes,
        metric_device=metric_device
    )

    text_alignment, class_alignments = compute_text_visual_alignment_torch(
        visual_features=id_visual_features,
        labels=id_labels,
        text_features=text_features,
        label_to_text_idx=label_to_text_idx,
        metric_device=metric_device
    )

    metrics = {
        'intra_class_distance': float(intra_dist),
        'inter_class_distance': float(inter_dist),
        'text_visual_alignment': float(text_alignment),
        'intra_inter_ratio': float(intra_dist / (inter_dist + 1e-12)),
        'class_intra_dists': class_intra_dists,
        'class_alignments': class_alignments,
    }
    return metrics


def save_metrics(metrics_results, save_path):
    serializable = {}
    for method_name, method_metrics in metrics_results.items():
        serializable[method_name] = {
            'intra_class_distance': float(method_metrics['intra_class_distance']),
            'inter_class_distance': float(method_metrics['inter_class_distance']),
            'intra_inter_ratio': float(method_metrics['intra_inter_ratio']),
            'text_visual_alignment': float(method_metrics['text_visual_alignment']),
            'class_intra_dists': method_metrics['class_intra_dists'].tolist(),
            'class_alignments': method_metrics['class_alignments'].tolist()
        }

    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def print_metrics(name, metrics):
    print(f"\n{name}:")
    print(f"  Intra-class distance : {metrics['intra_class_distance']:.6f}")
    print(f"  Inter-class distance : {metrics['inter_class_distance']:.6f}")
    print(f"  Intra/Inter ratio    : {metrics['intra_inter_ratio']:.6f}  (smaller is better)")
    print(f"  Text-Visual alignment: {metrics['text_visual_alignment']:.6f}")


def print_comparison_table(metrics_clip, metrics_clip_locoop, metrics_modular):
    print("\n" + "=" * 90)
    print("Comparison Table")
    print("=" * 90)
    print(f"{'Method':<20} {'Intra-Dist':<16} {'Inter-Dist':<16} {'Ratio':<16} {'Text-Align':<16}")
    print("-" * 90)
    print(f"{'CLIP Original':<20} "
          f"{metrics_clip['intra_class_distance']:<16.6f} "
          f"{metrics_clip['inter_class_distance']:<16.6f} "
          f"{metrics_clip['intra_inter_ratio']:<16.6f} "
          f"{metrics_clip['text_visual_alignment']:<16.6f}")
    print(f"{'CLIP + LoCoOP':<20} "
          f"{metrics_clip_locoop['intra_class_distance']:<16.6f} "
          f"{metrics_clip_locoop['inter_class_distance']:<16.6f} "
          f"{metrics_clip_locoop['intra_inter_ratio']:<16.6f} "
          f"{metrics_clip_locoop['text_visual_alignment']:<16.6f}")
    print(f"{'Your Model':<20} "
          f"{metrics_modular['intra_class_distance']:<16.6f} "
          f"{metrics_modular['inter_class_distance']:<16.6f} "
          f"{metrics_modular['intra_inter_ratio']:<16.6f} "
          f"{metrics_modular['text_visual_alignment']:<16.6f}")
    print("=" * 90)


def print_improvement_analysis(metrics_clip, metrics_clip_locoop, metrics_modular):
    print("\n" + "=" * 90)
    print("Improvement Analysis")
    print("=" * 90)

    def safe_improve(old, new):
        if abs(old) < 1e-12:
            return 0.0
        return (old - new) / old * 100.0

    intra_improve_1 = safe_improve(metrics_clip['intra_class_distance'],
                                   metrics_clip_locoop['intra_class_distance'])
    text_improve_1 = safe_improve(metrics_clip['text_visual_alignment'],
                                  metrics_clip_locoop['text_visual_alignment'])

    intra_improve_2 = safe_improve(metrics_clip_locoop['intra_class_distance'],
                                   metrics_modular['intra_class_distance'])
    text_improve_2 = safe_improve(metrics_clip_locoop['text_visual_alignment'],
                                  metrics_modular['text_visual_alignment'])

    intra_improve_total = safe_improve(metrics_clip['intra_class_distance'],
                                       metrics_modular['intra_class_distance'])
    text_improve_total = safe_improve(metrics_clip['text_visual_alignment'],
                                      metrics_modular['text_visual_alignment'])

    print("\nCLIP -> CLIP + LoCoOP:")
    print(f"  Intra-class distance : {intra_improve_1:+.2f}% ({'improved' if intra_improve_1 > 0 else 'degraded'})")
    print(f"  Text-Visual alignment: {text_improve_1:+.2f}% ({'improved' if text_improve_1 > 0 else 'degraded'})")

    print("\nCLIP + LoCoOP -> Your Model:")
    print(f"  Intra-class distance : {intra_improve_2:+.2f}% ({'improved' if intra_improve_2 > 0 else 'degraded'})")
    print(f"  Text-Visual alignment: {text_improve_2:+.2f}% ({'improved' if text_improve_2 > 0 else 'degraded'})")

    print("\nCLIP -> Your Model:")
    print(f"  Intra-class distance : {intra_improve_total:+.2f}% ({'improved' if intra_improve_total > 0 else 'degraded'})")
    print(f"  Text-Visual alignment: {text_improve_total:+.2f}% ({'improved' if text_improve_total > 0 else 'degraded'})")
    print("=" * 90)


def main():
    args = parse_args()
    setup_seed(args.seed)

    infer_device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if args.metric_device == 'cuda' and torch.cuda.is_available():
        metric_device = infer_device
    else:
        metric_device = torch.device('cpu')

    print("=" * 90)
    print("Distance Metrics Evaluation (Full Torch/GPU Version)")
    print("=" * 90)
    print(f"Inference device: {infer_device}")
    print(f"Metric device   : {metric_device}")
    print(f"Log directory   : {args.log_directory}")

    # 1. Load CLIP model
    clip_model, preprocess = set_model_clip(args)
    clip_model = clip_model.to(infer_device).eval()

    # 2. Prepare data
    id_dataset, ood_dataset, id_loader, ood_loader = prepare_dataloaders(args, preprocess)

    # 3. Load LoCoOP text features
    text_locoop_cpu, locoop_class_names = load_external_text_features_torch(
        args.text_feature_path, args.class_names_path
    )
    print(f"\nLoaded LoCoOP text features: {tuple(text_locoop_cpu.shape)}")
    print(f"Loaded LoCoOP class names   : {len(locoop_class_names)}")

    # 4. Build ImageNet label -> text feature index
    label_to_text_idx, missing = build_imagenet_label_to_text_index(
        id_dataset, locoop_class_names, args.root_dir
    )

    if len(missing) > 0:
        print(f"\nWarning: {len(missing)} ImageNet classes cannot be mapped to text features.")
        print("First 10 missing examples:")
        for item in missing[:10]:
            print(" ", item)

    num_id_classes = len(id_dataset.classes)
    print(f"\nNumber of ID classes: {num_id_classes}")
    print(f"Number of mapped classes to text features: {len(label_to_text_idx)}")

    # 5. Build modular model
    cfg = {
        'device': infer_device,
        'selector_type': args.selector_type,
        'fuser_type': args.fuser_type,
        'num_select': args.num_select,
        'num_heads_selector': args.num_heads_selector,
        'num_heads_fuser': args.num_heads_fuser,
        'templates': [args.templates],
        'residual_coef': args.residual_coef
    }

    net = build_modular_model(cfg, locoop_class_names, clip_model).to(infer_device)
    checkpoint = torch.load(args.model_path, map_location=infer_device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    # 6. Extract features
    print("\n" + "=" * 90)
    print("Extracting ID features...")
    print("=" * 90)
    id_visual_clip_cpu, id_visual_modular_cpu, id_labels_cpu = extract_both_features_torch(
        net, id_loader, infer_device, use_amp=args.use_amp, desc="ID features"
    )

    print("\n" + "=" * 90)
    print("Extracting OOD features...")
    print("=" * 90)
    ood_visual_clip_cpu, ood_visual_modular_cpu, _ = extract_both_features_torch(
        net, ood_loader, infer_device, use_amp=args.use_amp, desc="OOD features"
    )

    # 7. OOD labels
    ood_labels_cpu = torch.full(
        (ood_visual_clip_cpu.shape[0],),
        fill_value=num_id_classes,
        dtype=torch.long
    )

    # 8. Extract CLIP text features
    print("\n" + "=" * 90)
    print("Extracting CLIP text features...")
    print("=" * 90)
    text_clip_cpu = get_text_features_clip_torch(
        clip_model, locoop_class_names, infer_device, template=args.templates
    )

    # 9. Normalize again for safety
    id_visual_clip_cpu = l2_normalize_torch(id_visual_clip_cpu)
    id_visual_modular_cpu = l2_normalize_torch(id_visual_modular_cpu)
    ood_visual_clip_cpu = l2_normalize_torch(ood_visual_clip_cpu)
    ood_visual_modular_cpu = l2_normalize_torch(ood_visual_modular_cpu)
    text_clip_cpu = l2_normalize_torch(text_clip_cpu)
    text_locoop_cpu = l2_normalize_torch(text_locoop_cpu)

    all_visual_clip_cpu = torch.cat([id_visual_clip_cpu, ood_visual_clip_cpu], dim=0)
    all_visual_modular_cpu = torch.cat([id_visual_modular_cpu, ood_visual_modular_cpu], dim=0)
    all_labels_cpu = torch.cat([id_labels_cpu, ood_labels_cpu], dim=0)

    # 10. Save features if needed
    if args.save_features:
        feat_path = os.path.join(args.log_directory, 'extracted_features.pt')
        torch.save({
            'id_visual_clip': id_visual_clip_cpu,
            'id_visual_modular': id_visual_modular_cpu,
            'ood_visual_clip': ood_visual_clip_cpu,
            'ood_visual_modular': ood_visual_modular_cpu,
            'id_labels': id_labels_cpu,
            'ood_labels': ood_labels_cpu,
            'text_clip': text_clip_cpu,
            'text_locoop': text_locoop_cpu
        }, feat_path)
        print(f"\nSaved features to: {feat_path}")

    # 11. Compute metrics on metric_device
    print("\n" + "=" * 90)
    print("Computing distance metrics...")
    print("=" * 90)

    metrics_clip = compute_all_metrics_torch(
        id_visual_features=id_visual_clip_cpu,
        id_labels=id_labels_cpu,
        all_visual_features=all_visual_clip_cpu,
        all_labels=all_labels_cpu,
        text_features=text_clip_cpu,
        label_to_text_idx=label_to_text_idx,
        num_id_classes=num_id_classes,
        metric_device=metric_device
    )
    print_metrics("1. Original CLIP (CLIP visual + CLIP text)", metrics_clip)

    metrics_clip_locoop = compute_all_metrics_torch(
        id_visual_features=id_visual_clip_cpu,
        id_labels=id_labels_cpu,
        all_visual_features=all_visual_clip_cpu,
        all_labels=all_labels_cpu,
        text_features=text_locoop_cpu,
        label_to_text_idx=label_to_text_idx,
        num_id_classes=num_id_classes,
        metric_device=metric_device
    )
    print_metrics("2. CLIP + LoCoOP Text (CLIP visual + LoCoOP text)", metrics_clip_locoop)

    metrics_modular = compute_all_metrics_torch(
        id_visual_features=id_visual_modular_cpu,
        id_labels=id_labels_cpu,
        all_visual_features=all_visual_modular_cpu,
        all_labels=all_labels_cpu,
        text_features=text_locoop_cpu,
        label_to_text_idx=label_to_text_idx,
        num_id_classes=num_id_classes,
        metric_device=metric_device
    )
    print_metrics("3. Your Model + LoCoOP Text (Your model visual + LoCoOP text)", metrics_modular)

    # 12. Save metrics
    metrics_results = {
        'CLIP_Original': metrics_clip,
        'CLIP_LoCoOP': metrics_clip_locoop,
        'Your_Model': metrics_modular
    }

    metrics_file = os.path.join(args.log_directory, 'metrics_results.json')
    save_metrics(metrics_results, metrics_file)
    print(f"\n✓ Metrics saved to: {metrics_file}")

    # 13. Print comparison
    print_comparison_table(metrics_clip, metrics_clip_locoop, metrics_modular)
    print_improvement_analysis(metrics_clip, metrics_clip_locoop, metrics_modular)

    print("\n✅ Distance metrics evaluation complete!")
    print(f"Results saved to: {args.log_directory}")


if __name__ == '__main__':
    main()