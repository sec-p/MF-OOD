import torch
import os
import numpy as np
import random


def read_imagenet_classes(file_path):
    class_names = []
    # 打开文件，按行读取（兼容不同编码）
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # 去除行首尾的空白字符（换行、空格、制表符）
            clean_line = line.strip()
            # 跳过空行或注释行（以#开头的行）
            if not clean_line or clean_line.startswith('#'):
                continue
            
            # 关键：分割WNID和类名（仅分割第一个空格，兼容类名含空格的情况）
            # 例如：n03777568 ford model t → 分割为 ['n03777568', 'ford model t']
            parts = clean_line.split(maxsplit=1)
            if len(parts) < 2:
                # 跳过格式错误的行（无类名），并提示
                print(f"警告：第{line_num}行格式错误，跳过 → 内容：{clean_line}")
                continue
            
            # 提取类名（第二个部分）并加入列表
            class_name = parts[1]
            class_names.append(class_name)
    
    return class_names




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_test_labels(args):
    if args.in_dataset == 'ImageNet':
        test_labels = obtain_ImageNet_classes()
    elif args.in_dataset == 'COCO_single':
        test_labels = obtain_COCO_single_classes()
    elif args.in_dataset == 'COCO_multi':
        test_labels = obtain_COCO_multi_classes()
    elif args.in_dataset == 'VOC_single':
        test_labels = obtain_VOC_single_classes()
    return test_labels


def obtain_ImageNet_classes():
    # Try to load from the standard location first
    loc = "/amax/yeliu/data/imagenet"
    # class_file = os.path.join(loc, 'imagenet_class_clean.npy')
    class_file = os.path.join(loc, 'classnames.txt')

    
    if os.path.exists(class_file):
        with open(class_file, 'rb') as f:
            # imagenet_cls = np.load(f)
            imagenet_cls = read_imagenet_classes(class_file)
        return imagenet_cls
    else:
        # Fallback: Return a placeholder list for ImageNet classes
        # This ensures the code can run even without the class file
        print(f"Warning: Could not find {class_file}")
        print("Using placeholder ImageNet class names (1000 classes)")
        return [f"class_{i}" for i in range(1000)]


def obtain_COCO_single_classes():
    class_name = [
       'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    return class_name


def obtain_COCO_multi_classes():
    class_name = [
       'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'book', 'clock', 'vase', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    return class_name


def obtain_VOC_single_classes():
    class_name = [
         'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    return class_name


def accuracy(output, target, topk=(1,)):
    '''Computes the precision@k for the specified values of k'''
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def read_file(file_path, root='corpus'):
    corpus = []
    with open(os.path.join(root, file_path)) as f:
        for line in f:
            corpus.append(line[:-1])
    return corpus


def calculate_cosine_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    return similarity


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count