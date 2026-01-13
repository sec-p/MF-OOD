import os
import sys
import argparse
import numpy as np
import torch
from scipy import stats
from torchvision import datasets
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.common import setup_seed, get_test_labels
from utils.detection_util import print_measures, get_and_print_results, get_ood_scores_clip, get_ood_scores_dual_stream
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import set_model_clip, set_val_loader, set_ood_loader_ImageNet
from src.model_modular import build_modular_model


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates GL-MCM Score for CLIP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dataset', default='ImageNet', type=str,
                        choices=['COCO_single', 'COCO_multi', 'VOC_single', 'ImageNet'], help='in-distribution dataset')
    parser.add_argument('--root-dir', default="/data/ICML2026/clip/FA/my_dataset", type=str,
                        help='root dir of datasets')
    parser.add_argument('--name', default="eval_ood",
                        type=str, help="unique ID for the run")
    parser.add_argument('--seed', default=1, type=int, help="random seed")
    parser.add_argument('--gpu', default=0, type=int,
                        help='the GPU indice to use')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        help='mini-batch size')
    parser.add_argument('--T', type=int, default=1,
                        help='temperature parameter')
    parser.add_argument('--model', default='modular', type=str, choices=['CLIP', 'modular'], help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/16', 'RN50', 'RN101'], help='which pretrained img encoder to use')
    parser.add_argument('--score', default='MCM', type=str, choices=['MCM', 'L-MCM', 'GL-MCM','GL-MCM-L','GPT','SA-MCM','AL-MCM','DS-MCM'], help='score options')
    parser.add_argument('--num_ood_sumple', default=-1, type=int, help="numbers of ood_sumples")
    parser.add_argument('--lambda_local', default=0.4, type=float, help='weight for local score')
    
    # Dual-Stream Fusion parameters
    parser.add_argument('--fusion_strategy', default='geometric', type=str, choices=['arithmetic', 'geometric'], 
                        help='Fusion strategy for dual-stream: arithmetic mean or geometric mean')
    
    # Modular model parameters
    parser.add_argument('--model_path', type=str, default=None, help='path to trained modular model checkpoint')
    parser.add_argument('--selector_type', type=str, default='slot', choices=['mlp', 'slot', 'identity'], help='selector type for modular model')
    parser.add_argument('--fuser_type', type=str, default='self_attn', choices=['mean', 'query_attn', 'self_attn'], help='fuser type for modular model')
    parser.add_argument('--num_select', type=int, default=49, help='number of features to select for modular model')
    parser.add_argument('--num_heads_selector', type=int, default=8, help='number of heads for MLP selector')
    parser.add_argument('--num_heads_fuser', type=int, default=8, help='number of heads for attention fuser')
    parser.add_argument('--templates', type=str, default="a photo of a {}", help='templates for text prompts')
    
    args = parser.parse_args()

    args.CLIP_ckpt_name = args.CLIP_ckpt.replace('/', '_')
    args.log_directory = f"results/{args.in_dataset}/{args.score}/{args.model}_{args.CLIP_ckpt_name}_T_{args.T}_ID_{args.name}"
    os.makedirs(args.log_directory, exist_ok=True)

    return args


def main():
    args = process_args()
    setup_seed(args.seed)
    log = setup_log(args)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    # Load base CLIP model
    clip_model, preprocess = set_model_clip(args)
    
    if args.model == 'modular':
        # Get class names from dataset
        root = args.root_dir
        if args.in_dataset == "ImageNet":
            dataset = datasets.ImageFolder(os.path.join(root, 'ImageNet', 'images', 'val'))
        elif args.in_dataset == 'COCO_single':
            dataset = datasets.ImageFolder(os.path.join(root, 'ID_COCO_single'))
        elif args.in_dataset == 'COCO_multi':
            dataset = datasets.ImageFolder(os.path.join(root, 'ID_COCO_multi'))
        elif args.in_dataset == 'VOC_single':
            dataset = datasets.ImageFolder(os.path.join(root, 'ID_VOC_single'))
        classnames = dataset.classes
        
        # Build modular model config
        cfg = {
            'device': torch.device(f'cuda:{args.gpu}'),
            'selector_type': args.selector_type,
            'fuser_type': args.fuser_type,
            'num_select': args.num_select,
            'num_heads_selector': args.num_heads_selector,
            'num_heads_fuser': args.num_heads_fuser,
            'templates': [args.templates],
        }
        
        # Build modular model
        net = build_modular_model(cfg, classnames, clip_model)
        
        # Move entire model to CUDA device first (critical for new layers)
        device = torch.device(f'cuda:{args.gpu}')
        net = net.to(device)
        
        # Load trained parameters if provided (only load trainable parameters)
        if args.model_path and os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
            # Checkpoint structure: {'state_dict': {...}, 'config': {...}, ...}
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            net.load_state_dict(state_dict, strict=False)
            log.info(f"Loaded model checkpoint from {args.model_path}")
    else:
        # Use original CLIP model
        net = clip_model
    
    net.eval()

    if args.in_dataset in ['COCO_single', 'COCO_multi']:
        out_datasets = ['iNaturalist', 'SUN', 'Texture', 'IN22k', 'ood_voc']
    elif args.in_dataset in ['VOC_single']:
        out_datasets = ['iNaturalist', 'SUN', 'Texture', 'IN22k', 'ood_coco']
    elif args.in_dataset in ['ImageNet']:
        out_datasets = ['iNaturalist', 'SUN', 'places365', 'Texture']

    test_loader = set_val_loader(args, preprocess)
    test_labels = get_test_labels(args)
    
    # Choose scoring method based on args.score
    if args.score == 'DS-MCM':
        # Dual-Stream MCM for Stage 2
        in_score = get_ood_scores_dual_stream(args, net, test_loader, test_labels)
    else:
        # Standard MCM methods
        in_score = get_ood_scores_clip(args, net, test_loader, test_labels)
    
    auroc_list, aupr_list, fpr_list = [], [], []
    for out_dataset in out_datasets:
        log.debug(f"Evaluting OOD dataset {out_dataset}")
        ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess, root=args.root_dir)
        
        if args.score == 'DS-MCM':
            # Dual-Stream MCM for Stage 2
            out_score = get_ood_scores_dual_stream(args, net, ood_loader, test_labels)
        else:
            # Standard MCM methods
            out_score = get_ood_scores_clip(args, net, ood_loader, test_labels)
        
        log.debug(f"in scores: {stats.describe(in_score)}")
        log.debug(f"out scores: {stats.describe(out_score)}")
        plot_distribution(args, in_score, out_score, out_dataset)
        get_and_print_results(args, log, in_score, out_score,
                              auroc_list, aupr_list, fpr_list)
    log.debug('\n\nMean Test Results')
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list),
                   np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)


if __name__ == '__main__':
    main()