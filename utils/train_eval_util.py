import os
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import random
from torch.utils.data.dataset import Subset
import clip


def set_model_clip(args):
    """
    Load CLIP model using the specified backbone.
    Handles both string input (from train_eval.py) and args object (from eval_ood_detection.py)
    """
    # Check if args is a string (from train_eval.py) or an args object (from eval_ood_detection.py)
    if isinstance(args, str):
        backbone = args
    elif hasattr(args, 'CLIP_ckpt'):
        backbone = args.CLIP_ckpt
    elif hasattr(args, 'backbone'):
        backbone = args.backbone
    else:
        raise ValueError("set_model_clip expects either a string or an args object with 'CLIP_ckpt' or 'backbone' attribute")
    
    model, preprocess = clip.load(backbone)
    return model, preprocess


def set_val_loader(args, preprocess=None):
    root = args.root_dir
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 8, 'pin_memory': True, 'persistent_workers': True}
    if args.in_dataset == "ImageNet":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(root, 'imagenet','images', 'val'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == 'COCO_single':
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(root, 'ID_COCO_single'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == 'COCO_multi':
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(root, 'ID_COCO_multi'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == 'VOC_single':
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(root, 'ID_VOC_single'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    return val_loader


def get_subset_with_len(dataset, length, shuffle=False):
    dataset_size = len(dataset)
    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def set_ood_loader_ImageNet(args, out_dataset, preprocess, root):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if out_dataset == 'iNaturalist':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'IN22k':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'ImageNet-22K'), transform=preprocess)
    elif out_dataset == 'ood_voc':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'OOD_VOC'), transform=preprocess)
    elif out_dataset == 'ood_coco':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'OOD_COCO'), transform=preprocess)
    elif out_dataset == 'places365':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'Places'), transform=preprocess)    
    elif out_dataset == 'Texture':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'),
                                        transform=preprocess)
    elif out_dataset == 'NINCO':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'NINCO', 'NINCO_OOD_classes'),
                                        transform=preprocess)

    if hasattr(args, 'num_ood_sumple') and args.num_ood_sumple > 0 and out_dataset != 'ood_voc':
        testsetout = get_subset_with_len(testsetout, length=args.num_ood_sumple, shuffle=True)
    
    # Use consistent kwargs with GL-MCM style
    kwargs = {'num_workers': 8, 'pin_memory': True, 'persistent_workers': True}
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 64  # Default to 64 if not provided
    
    testloaderOut = torch.utils.data.DataLoader(
        testsetout, batch_size=batch_size, shuffle=False, **kwargs
    )
    return testloaderOut


def set_train_loader(args, transform=None):
    """
    Set up training data loader using GL-MCM style.
    """
    root = args.root_dir
    if transform is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1),
                                        interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 8, 'pin_memory': True, 'persistent_workers': True}
    
    # Load full training dataset
    if args.in_dataset == "ImageNet":
        full_train_dataset = datasets.ImageFolder(os.path.join(root, 'imagenet','images', 'train'), transform=transform)
    elif args.in_dataset == 'COCO_single':
        full_train_dataset = datasets.ImageFolder(os.path.join(root, 'ID_COCO_single'), transform=transform)
    elif args.in_dataset == 'COCO_multi':
        full_train_dataset = datasets.ImageFolder(os.path.join(root, 'ID_COCO_multi'), transform=transform)
    elif args.in_dataset == 'VOC_single':
        full_train_dataset = datasets.ImageFolder(os.path.join(root, 'ID_VOC_single'), transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.in_dataset}")
    
    # Generate few-shot dataset if shots > 0
    if hasattr(args, 'shots') and args.shots > 0:
        train_dataset = generate_fewshot_dataset_by_class(full_train_dataset, args.shots, args.seed)
    else:
        train_dataset = full_train_dataset
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        prefetch_factor=2,
        **kwargs
    )
    
    return train_loader, full_train_dataset


def generate_fewshot_dataset_by_class(dataset, num_samples_per_class, random_seed=42):
    """
    Generate a few-shot dataset by selecting num_samples_per_class instances from each class.
    If a class has fewer than num_samples_per_class instances, repeat samples from that class.
    
    Args:
        dataset (torchvision.datasets.ImageFolder): Original dataset
        num_samples_per_class (int): Number of samples to select per class
        random_seed (int): Random seed for reproducibility
        
    Returns:
        torch.utils.data.Subset: Subset of the original dataset with few-shot samples
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get class indices
    class_indices = {}
    for idx, (_, label) in enumerate(dataset.imgs):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Select few-shot samples for each class
    fewshot_indices = []
    for label, indices in class_indices.items():
        class_size = len(indices)
        if class_size >= num_samples_per_class:
            # Randomly select num_samples_per_class indices from the class
            selected_indices = np.random.choice(indices, size=num_samples_per_class, replace=False)
        else:
            # If not enough samples, repeat samples to reach num_samples_per_class
            selected_indices = np.random.choice(indices, size=num_samples_per_class, replace=True)
        fewshot_indices.extend(selected_indices.tolist())
    
    # Create subset from selected indices
    fewshot_dataset = Subset(dataset, fewshot_indices)
    
    return fewshot_dataset