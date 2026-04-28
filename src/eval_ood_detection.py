import os
import sys
import argparse
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.common import setup_seed, get_test_labels
from utils.detection_util import print_measures, get_and_print_results
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import set_model_clip, set_val_loader, set_ood_loader_ImageNet
from src.model_modular import build_modular_model

# Import HYBRID scoring functions from GL-MCM style
def get_ood_scores_clip_hybrid(args, net, loader, test_labels, visual_prototypes=None, external_text_features=None):
    """
    Calculate OOD scores using HYBRID approach (from GL-MCM).
    Combines multi-modal and visual branches with logits fusion.
    
    Args:
        args: Arguments object with score type and weights
        net: CLIP model with return_both_features=True
        loader: DataLoader for test data
        test_labels: List of class names
        visual_prototypes: (num_classes, 512) - visual prototypes for visual branch
        external_text_features: optional pre-computed text features
    
    Returns:
        scores: OOD scores for all samples
        id_accuracy: ID classification accuracy
    """
    to_np = lambda x: x.data.cpu().float().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    _predictions = []
    _true_labels = []
    
    # Get text features
    num_classes = len(test_labels)
    text_features = None
    external_text_features_only = None
    use_fa_mode = False
    K = getattr(args, 'K', 3)  # Default to 3 if K not provided
    
    if external_text_features is not None:
        # Check if FA mode is enabled
        if getattr(args, 'FA', 0) == 1:
            print(f"FA mode: using concatenated text features (external + original repeated K times)")
            # FA mode: external_text_features is already concatenated: [external, original_repeated]
            # Shape: ((K+1)*num_classes, 512)
            text_features = external_text_features.half().cuda()
            use_fa_mode = True
            # Extract only external features for visual branch
            external_text_features_only = text_features[:num_classes]
        else:
            # Regular mode: use only external features
            text_features = external_text_features.half().cuda()
            external_text_features_only = text_features
    elif hasattr(net, 'text_features') and net.text_features is not None:
        text_features = net.text_features
        external_text_features_only = text_features
    else:
        import clip
        tokenizer = clip.tokenize
        with torch.no_grad():
            text_inputs = tokenizer([f"a photo of a {c}" for c in test_labels])
            text_features = net.encode_text(text_inputs.cuda()).half()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        external_text_features_only = text_features
    
    # Use text features as visual prototypes if not provided
    if visual_prototypes is None:
        # Always use only external features for visual prototypes
        visual_prototypes = external_text_features_only
        visual_prototypes = visual_prototypes / visual_prototypes.norm(dim=-1, keepdim=True)
    
    # Ensure all features are on the same device
    if external_text_features is not None:
        visual_prototypes = visual_prototypes.half().cuda()
    
    # Get weights
    multimodal_weight = getattr(args, 'multimodal_weight', 0.5)
    visual_weight = getattr(args, 'visual_weight', 0.5)
    
    tqdm_object = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch_idx, (images, labels, *id_flag) in enumerate(tqdm_object):
            images = images.cuda()
            labels = labels.long().cuda()
            
            # Get features from modular model
            res = net(images)
            global_features = res['global_features']
            local_features = res['local_features']
            selected_feats = res['selected_feats']
            global_feature = res['global_feature']
            
            # Normalize features
            global_features = global_features / global_features.norm(dim=-1, keepdim=True)
            local_features = local_features / local_features.norm(dim=-1, keepdim=True)
            selected_feats = selected_feats / selected_feats.norm(dim=-1, keepdim=True)
            global_feature = global_feature / global_feature.norm(dim=-1, keepdim=True)
            
            # Multi-modal branch
            # In FA mode, text_features includes both external and repeated original features
            logits_multimodal = global_features @ text_features.T
            output_local_multimodal = local_features @ text_features.T
            
            # Visual branch (only uses external features for prototypes)
            logits_visual = global_features @ visual_prototypes.T
            output_local_visual = local_features @ visual_prototypes.T
            # import pdb
            # pdb.set_trace()
            # Mean-based filtering for visual logits
            mean_visual = logits_visual.mean(dim=1, keepdim=True)
            logits_visual_filtered = torch.where(
                logits_visual > mean_visual,
                mean_visual,
                logits_visual
            )
            
            mean_visual_local = output_local_visual.mean(dim=-1, keepdim=True)
            output_local_visual_filtered = torch.where(
                output_local_visual > mean_visual_local,
                mean_visual_local,
                output_local_visual
            )
            
            # Fuse logits
            # In FA mode, logits_multimodal has shape (batch_size, (K+1)*num_classes), logits_visual has shape (batch_size, num_classes)
            # We need to expand logits_visual to match logits_multimodal shape
            if use_fa_mode:
                # Calculate how many times to repeat visual logits
                repeat_factor = logits_multimodal.shape[1] // logits_visual.shape[1]
                
                # Expand visual logits to match multimodal logits shape
                logits_visual_expanded = logits_visual.repeat_interleave(repeat_factor, dim=1)
                output_local_visual_expanded = output_local_visual.repeat_interleave(repeat_factor, dim=2)
                
                # Fuse logits
                logits = multimodal_weight * logits_multimodal + visual_weight * logits_visual_expanded
                logits_local = multimodal_weight * output_local_multimodal + visual_weight * output_local_visual_expanded
                
                # For softmax calculations, we need to use the expanded visual logits
                logits_visual_for_softmax = logits_visual_expanded
                output_local_visual_for_softmax = output_local_visual_expanded
            else:
                # Regular mode: both have shape (batch_size, num_classes)
                logits = multimodal_weight * logits_multimodal + visual_weight * logits_visual
                logits_local = multimodal_weight * output_local_multimodal + visual_weight * output_local_visual
                
                # For softmax calculations
                logits_visual_for_softmax = logits_visual
                output_local_visual_for_softmax = output_local_visual
            
            # Apply softmax
            smax_multimodal = to_np(torch.softmax(logits_multimodal / args.T, dim=1))
            
            # For visual softmax, we need to use the expanded version in FA mode
            if use_fa_mode:
                # Expand the filtered visual logits to match the expanded shape
                logits_visual_filtered_expanded = logits_visual_filtered.repeat_interleave(repeat_factor, dim=1)
                output_local_visual_filtered_expanded = output_local_visual_filtered.repeat_interleave(repeat_factor, dim=2)
                
                smax_visual = to_np(torch.softmax(logits_visual_filtered_expanded / args.T, dim=1))
                smax_local_visual = to_np(torch.softmax(output_local_visual_filtered_expanded / args.T, dim=-1))
            else:
                smax_visual = to_np(torch.softmax(logits_visual_filtered / args.T, dim=1))
                smax_local_visual = to_np(torch.softmax(output_local_visual_filtered / args.T, dim=-1))
            
            smax = to_np(torch.softmax(logits / args.T, dim=1))
            smax_local_multimodal = to_np(torch.softmax(output_local_multimodal / args.T, dim=-1))
            smax_local = to_np(torch.softmax(logits_local / args.T, dim=-1))
            
            # Calculate predictions for ID accuracy
            # In FA mode, use only the first num_classes features (external)
            if use_fa_mode:
                predictions = np.argmax(logits_multimodal.cpu().numpy()[:, :num_classes], axis=1)
            else:
                predictions = np.argmax(logits.cpu().numpy(), axis=1)
            _predictions.append(predictions)
            _true_labels.append(labels.cpu().numpy())
            
            # Calculate OOD scores based on score type
            if args.score == 'HYBRID':
                mcm_global_multi = -np.max(smax_multimodal, axis=1)
                mcm_global_visual = -np.max(smax_visual, axis=1)
                mcm_local_multi = -np.max(smax_local_multimodal, axis=(1, 2))
                mcm_local_visual = -np.max(smax_local_visual, axis=(1, 2))
                
                mcm_global = -np.max(smax, axis=1)
                mcm_local = -np.max(smax_local, axis=(1, 2))
                
                ood_scores = mcm_global + args.lambda_local * mcm_local
                
            elif args.score == 'HYBRID-MULTI':
                mcm_global = -np.max(smax_multimodal, axis=1)
                mcm_local = -np.max(smax_local_multimodal, axis=(1, 2))
                ood_scores = mcm_global + args.lambda_local * mcm_local
                
            elif args.score == 'HYBRID-VISUAL':
                mcm_global = -np.max(smax_visual, axis=1)
                mcm_local = -np.max(smax_local_visual, axis=(1, 2))
                ood_scores = mcm_global + args.lambda_local * mcm_local
                
            elif args.score == 'HYBRID-MCM':
                mcm_multi = -np.max(smax_multimodal, axis=1)
                mcm_visual = -np.max(smax_visual, axis=1)
                ood_scores = multimodal_weight * mcm_multi + visual_weight * mcm_visual
                
            else:
                raise NotImplementedError(f"Unknown score type: {args.score}")
            
            _score.append(ood_scores)
    
    # Calculate ID accuracy
    all_predictions = concat(_predictions)
    all_true_labels = concat(_true_labels)
    id_accuracy = np.mean(all_predictions == all_true_labels) * 100
    
    return concat(_score)[:len(loader.dataset)].copy(), id_accuracy


def get_ood_scores_clip_with_accuracy(args, net, loader, test_labels, external_text_features=None):
    """
    Wrapper for get_ood_scores_clip that also returns ID accuracy.
    """
    from utils.detection_util import get_ood_scores_clip
    
    to_np = lambda x: x.data.cpu().float().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    _predictions = []
    _true_labels = []
    
    # Get text features
    num_classes = len(test_labels)
    text_features = None
    use_fa_mode = False
    
    if external_text_features is not None:
        # Check if FA mode is enabled
        if getattr(args, 'FA', 0) == 1:
            print(f"FA mode: using concatenated text features (external + original repeated K times)")
            # FA mode: external_text_features is already concatenated: [external, original_repeated]
            # Shape: ((K+1)*num_classes, 512)
            text_features = external_text_features.half().cuda()
            use_fa_mode = True
        else:
            # Regular mode: use only external features
            text_features = external_text_features.half().cuda()
    elif hasattr(net, 'text_features') and net.text_features is not None:
        text_features = net.text_features
    else:
        import clip
        tokenizer = clip.tokenize
        with torch.no_grad():
            text_inputs = tokenizer([f"a photo of a {c}" for c in test_labels])
            text_features = net.encode_text(text_inputs.cuda()).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
    
    tqdm_object = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch_idx, (images, labels, *id_flag) in enumerate(tqdm_object):
            bz = images.size(0)
            labels = labels.long().cuda()
            images = images.cuda()
            
            res = net(images)
            global_features = res['global_features']
            local_features = res['local_features']
            selected_feats = res['selected_feats']
            
            global_features = global_features / global_features.norm(dim=-1, keepdim=True)
            local_features = local_features / local_features.norm(dim=-1, keepdim=True)
            selected_feats = selected_feats / selected_feats.norm(dim=-1, keepdim=True)
            
            # In FA mode, text_features already includes both external and repeated original features
            output_global = global_features @ text_features.T
            output_local = local_features @ text_features.T
            output_selected = selected_feats @ text_features.T
            
            smax_global = to_np(torch.softmax(output_global / args.T, dim=1))
            smax_local = to_np(torch.softmax(output_local / args.T, dim=-1))
            smax_selected = to_np(torch.softmax(output_selected / args.T, dim=-1))
            
            # Calculate predictions for ID accuracy
            # In FA mode, use only the first num_classes features (external)
            if use_fa_mode:
                predictions = np.argmax(output_global.cpu().numpy()[:, :num_classes], axis=1)
            else:
                predictions = np.argmax(output_global.cpu().numpy(), axis=1)
            _predictions.append(predictions)
            _true_labels.append(labels.cpu().numpy())
            
            if args.score == 'MCM':
                _score.append(-np.max(smax_global, axis=1)) 
            elif args.score == 'L-MCM':
                mcm_local_score = -np.max(smax_local, axis=(1, 2))
                _score.append(mcm_local_score) 
            elif args.score == 'GL-MCM':
                mcm_global_score = -np.max(smax_global, axis=1)
                mcm_local_score = -np.max(smax_local, axis=(1, 2))
                _score.append(mcm_global_score + args.lambda_local * mcm_local_score)
            elif args.score == 'GL-MCM-L':
                mcm_global_score = -np.max(smax_global, axis=1)
                mcm_local_score = -np.max(smax_local, axis=(1, 2))
                mcm_selected_score = -np.min(np.max(smax_selected, axis=2), axis=(1))
                _score.append(mcm_global_score + args.lambda_local * mcm_local_score + args.lambda_local * mcm_selected_score)
            elif args.score == 'SA-MCM':
                pred_labels = np.argmax(smax_global, axis=1)
                mcm_global_score = -np.max(smax_global, axis=1)
                
                B, K, C = smax_selected.shape
                slot_confs = smax_selected[np.arange(B)[:, None], np.arange(K)[None, :], pred_labels[:, None]]
                min_slot_conf = np.min(slot_confs, axis=1)
                mcm_selected_score = -min_slot_conf
                
                patch_confs = smax_local[np.arange(B)[:, None], np.arange(smax_local.shape[1])[None, :], pred_labels[:, None]]
                mcm_local_score = -np.max(patch_confs, axis=1)
                
                _score.append(mcm_global_score + args.lambda_local * mcm_selected_score)
            else:
                raise NotImplementedError
    
    # Calculate ID accuracy
    all_predictions = concat(_predictions)
    all_true_labels = concat(_true_labels)
    id_accuracy = np.mean(all_predictions == all_true_labels) * 100
    
    return concat(_score)[:len(loader.dataset)].copy(), id_accuracy


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates GL-MCM Score for CLIP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dataset', default='ImageNet', type=str,
                        choices=['COCO_single', 'COCO_multi', 'VOC_single', 'ImageNet'], help='in-distribution dataset')
    parser.add_argument('--root-dir', default="./datasets", type=str,
                        help='root dir of datasets')
    parser.add_argument('--name', default="eval_ood",
                        type=str, help="unique ID for run")
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
    parser.add_argument('--score', default='MCM', type=str, 
                        choices=['MCM', 'L-MCM', 'GL-MCM', 'GL-MCM-L', 'SA-MCM', 
                                  'HYBRID', 'HYBRID-MULTI', 'HYBRID-VISUAL', 'HYBRID-MCM'], 
                        help='score options')
    parser.add_argument('--num_ood_sumple', default=-1, type=int, help="numbers of ood_sumples")
    parser.add_argument('--lambda_local', default=0.5, type=float, help='weight for local score')
    
    # Modular model parameters
    parser.add_argument('--model_path', type=str, default=None, help='path to trained modular model checkpoint')
    parser.add_argument('--selector_type', type=str, default='identity', choices=['mlp', 'slot', 'identity'], help='selector type for modular model')
    parser.add_argument('--fuser_type', type=str, default='shared_adapter', choices=['shared_adapter','mean', 'query_attn', 'self_attn'], help='fuser type for modular model')
    parser.add_argument('--num_select', type=int, default=49, help='number of features to select for modular model')
    parser.add_argument('--num_heads_selector', type=int, default=8, help='number of heads for MLP selector')
    parser.add_argument('--num_heads_fuser', type=int, default=8, help='number of heads for attention fuser')
    parser.add_argument('--templates', type=str, default="a photo of a {}", help='templates for text prompts')
    parser.add_argument('--residual_coef', type=float, default=0.2, help='residual coefficient for shared adapter (default: 0.2)')
    
    # HYBRID mode parameters
    parser.add_argument('--multimodal-weight', default=1, type=float, 
                        help='weight for multi-modal branch in HYBRID mode')
    parser.add_argument('--visual-weight', default=0.5, type=float, 
                        help='weight for visual branch in HYBRID mode')
    parser.add_argument('--shots', default=16, type=int, 
                        help='number of shots per class for computing visual prototypes (few-shot setting)')
    parser.add_argument('--text-feature-path', type=str, default=None,
                        help='path to text features file. If provided, uses external text features.')
    parser.add_argument('--FA', type=int, default=0, choices=[0, 1],
                        help='Use FA method: concatenate external prompt text features with original prompt text features (0: disabled, 1: enabled)')
    parser.add_argument('--K', type=int, default=3,
                        help='Number of times to repeat external text features in FA mode (default: 3)')
    parser.add_argument('--ood-dataset', type=str, default=None,
                        help='Test only specific OOD dataset (e.g., NINCO, Texture, iNaturalist, SUN, etc.)')
    
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
        from torchvision import datasets
        root = args.root_dir
        if args.in_dataset == "ImageNet":
            dataset = datasets.ImageFolder(os.path.join(root, 'imagenet', 'images', 'val'))
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
            'residual_coef': args.residual_coef,
        }
        
        test_loader = set_val_loader(args, preprocess)
        test_labels = get_test_labels(args)

        # Build modular model
        net = build_modular_model(cfg, test_labels, clip_model)
        
        # Move entire model to CUDA device first
        device = torch.device(f'cuda:{args.gpu}')
        net = net.to(device)
        
        # Load trained parameters if provided
        if args.model_path and os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=device)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            net.load_state_dict(state_dict, strict=False)
            log.info(f"Loaded model checkpoint from {args.model_path}")
            print(f"✓ Loaded model checkpoint from {args.model_path}")
        else:
            print(f"Warning: No model checkpoint provided or file not found. Using untrained model.")
    else:
        # Use original CLIP model
        net = clip_model
    
    net.eval()

    if args.in_dataset in ['COCO_single', 'COCO_multi']:
        out_datasets = ['iNaturalist', 'SUN', 'Texture', 'IN22k', 'ood_voc', 'NINCO']
    elif args.in_dataset in ['VOC_single']:
        out_datasets = ['iNaturalist', 'SUN', 'Texture', 'IN22k', 'ood_coco', 'NINCO']
    elif args.in_dataset in ['ImageNet']:
        out_datasets = ['iNaturalist', 'SUN', 'places365', 'Texture', 'NINCO']

    test_loader = set_val_loader(args, preprocess)
    test_labels = get_test_labels(args)
    num_classes = len(test_labels)

    # Compute visual prototypes if using HYBRID mode
    visual_prototypes = None
    if args.score.startswith('HYBRID'):
        print(f"\nComputing visual prototypes from test set ({args.shots}-shot setting)...")
        visual_prototypes = compute_visual_prototypes_from_loader(net, test_loader, num_classes, shots=args.shots)
    
    # Load external text features if provided
    external_text_features = None
    original_text_features = None
    if args.text_feature_path is not None:
        print(f"Loading external text features from: {args.text_feature_path}")
        file_ext = os.path.splitext(args.text_feature_path)[1].lower()
        
        if file_ext == '.pt':
            loaded_data = torch.load(args.text_feature_path)
            if isinstance(loaded_data, dict):
                loaded_features = loaded_data['text_features']
            else:
                loaded_features = loaded_data
        elif file_ext == '.npy':
            loaded_features = np.load(args.text_feature_path)
            loaded_features = torch.from_numpy(loaded_features).float()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Expected .pt or .npy")
        
        if loaded_features.shape[0] != num_classes:
            raise ValueError(f"Text features shape mismatch: expected {num_classes} classes, got {loaded_features.shape[0]}")
        
        external_text_features = loaded_features.cuda().type(net.dtype)
        print(f"✓ Loaded external text features with shape: {loaded_features.shape}")
        
        # FA mode: concatenate external prompt text features with original prompt text features
        if args.FA == 1:
            print(f"FA mode enabled: concatenating external prompt with original prompt (K={args.K})")
            import clip
            tokenizer = clip.tokenize
            with torch.no_grad():
                # Get original text features using standard CLIP template
                text_inputs = tokenizer([f"a photo of a {c}" for c in test_labels])
                original_text_features = clip_model.encode_text(text_inputs.cuda()).half()
                original_text_features /= original_text_features.norm(dim=-1, keepdim=True)
            
            # Repeat original text features K times (each class repeated K times) and concatenate with external
            # original_text_features shape: (num_classes,512)
            # After repeat: (K * num_classes,512) - each class repeated K times
            original_text_features_repeated = original_text_features.repeat_interleave(args.K, dim=0)
            
            # Concatenate: external features first, then repeated original features
            # Final shape: (num_classes + K * num_classes,512) = ((K+1) * num_classes,512)
            external_text_features = torch.cat([external_text_features, original_text_features_repeated], dim=0)
            print(f"✓ Concatenated text features shape: {external_text_features.shape} (K={args.K})")
    
    # Choose appropriate scoring function based on score type
    if args.score.startswith('HYBRID'):
        print(f"\nUsing HYBRID mode: {args.score}")
        print(f"  Multi-modal weight: {args.multimodal_weight}")
        print(f"  Visual weight: {args.visual_weight}")
        in_score, id_accuracy = get_ood_scores_clip_hybrid(args, net, test_loader, test_labels, visual_prototypes, external_text_features)
    else:
        print(f"\nUsing {args.score} score method")
        in_score, id_accuracy = get_ood_scores_clip_with_accuracy(args, net, test_loader, test_labels, external_text_features)
    
    # Print ID accuracy
    print(f"\n{'='*60}")
    print(f"ID Classification Accuracy: {id_accuracy:.2f}%")
    print(f"{'='*60}\n")

    auroc_list, aupr_list, fpr_list = [], [], []
    
    # Use specified OOD dataset or all datasets
    if args.ood_dataset is not None:
        test_datasets = [args.ood_dataset]
        print(f"Testing only specified OOD dataset: {args.ood_dataset}")
    else:
        test_datasets = out_datasets
        print(f"Testing all OOD datasets: {test_datasets}")
    
    for out_dataset in test_datasets:
        log.debug(f"Evaluting OOD dataset {out_dataset}")
        ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess, root=args.root_dir)
        
        if args.score.startswith('HYBRID'):
            out_score, _ = get_ood_scores_clip_hybrid(args, net, ood_loader, test_labels, visual_prototypes, external_text_features)
        else:
            out_score, _ = get_ood_scores_clip_with_accuracy(args, net, ood_loader, test_labels, external_text_features)
            
        log.debug(f"in scores: {stats.describe(in_score)}")
        log.debug(f"out scores: {stats.describe(out_score)}")
        plot_distribution(args, in_score, out_score, out_dataset)
        get_and_print_results(args, log, in_score, out_score,
                              auroc_list, aupr_list, fpr_list)
    
    if len(test_datasets) > 1:
        log.debug('\n\nMean Test Results')
        print_measures(log, np.mean(auroc_list), np.mean(aupr_list),
                       np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, test_datasets, fpr_list, auroc_list, aupr_list)


def compute_visual_prototypes_from_loader(net, loader, num_classes, shots=None):
    """
    Compute visual prototypes from data loader.
    Supports few-shot setting.
    Prototype is computed as the average of (global_features + local_features_mean) across samples.
    """
    device = next(net.parameters()).device
    prototypes = torch.zeros(num_classes, 512, device=device)
    class_counts = torch.zeros(num_classes, device=device)
    
    if shots is not None and shots > 0:
        class_samples = [[] for _ in range(num_classes)]
        
        net.eval()
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Collecting samples for few-shot"):
                images = images.to(device)
                labels = labels.to(device)
                
                res = net(images)
                global_features = res['global_features']
                local_features = res['local_features']
                
                global_features = global_features / global_features.norm(dim=-1, keepdim=True)
                # local_features = local_features / local_features.norm(dim=-1, keepdim=True)
                
                # local_features_mean = local_features.mean(dim=1)
                # combined_features = global_features + local_features_mean
                # combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
                
                for i in range(len(labels)):
                    label = labels[i].item()
                    if len(class_samples[label]) < shots:
                        class_samples[label].append(global_features[i])
        
        for cls in range(num_classes):
            if len(class_samples[cls]) > 0:
                class_samples_tensor = torch.stack(class_samples[cls])
                prototypes[cls] = class_samples_tensor.mean(dim=0)
                class_counts[cls] = len(class_samples[cls])
    else:
        net.eval()
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Computing visual prototypes"):
                images = images.to(device)
                labels = labels.to(device)
                
                res = net(images)
                global_features = res['global_features']
                local_features = res['local_features']
                
                global_features = global_features / global_features.norm(dim=-1, keepdim=True)
                # local_features = local_features / local_features.norm(dim=-1, keepdim=True)
                
                # local_features_mean = local_features.mean(dim=1)
                # combined_features = global_features + local_features_mean
                # combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
                
                for i in range(len(labels)):
                    label = labels[i].item()
                    prototypes[label] += global_features[i]
                    class_counts[label] += 1
        
        for cls in range(num_classes):
            if class_counts[cls] > 0:
                prototypes[cls] /= class_counts[cls]
    
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
    
    # Convert to model's dtype to avoid dtype mismatch
    model_dtype = next(net.parameters()).dtype
    prototypes = prototypes.to(model_dtype)
    
    mode_str = f"{shots}-shot" if shots is not None and shots > 0 else "all samples"
    print(f"✓ Visual prototypes computed for {num_classes} classes ({mode_str})")
    print(f"  Prototype shape: {prototypes.shape}")
    print(f"  Samples per class (min/mean/max): {class_counts.min().item():.0f}/{class_counts.mean().item():.0f}/{class_counts.max().item():.0f}")
    
    return prototypes


if __name__ == '__main__':
    main()
