import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sklearn.metrics as sk
import clip
from torch.cuda.amp import GradScaler, autocast

def print_measures(log, auroc, aupr, fpr, method_name='Ours', recall_level=0.95):
    if log == None: 
        print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
        print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
        print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))
    else:
        log.debug('\t\t\t' + method_name)
        log.debug('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
        log.debug('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def get_ood_scores_clip(args, net, loader, test_labels):

    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    
    # Check if model is HybridCLIP
    is_hybrid = hasattr(net, 'ood_score_combined')
    
    # Use cached text features from model (already computed during initialization)
    # No need to recompute - this avoids redundant computation
    if hasattr(net, 'text_features') and net.text_features is not None:
        text_features = net.text_features
    else:
        # Fallback: compute text features if not cached
        tokenizer = clip.tokenize
        with torch.no_grad():
            text_inputs = tokenizer([f"a photo of a {c}" for c in test_labels])
            text_features = net.encode_text(text_inputs.cuda()).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
    
    tqdm_object = tqdm(loader, total=len(loader))
    with torch.no_grad():
        with autocast():
            for batch_idx, (images, labels, *id_flag) in enumerate(tqdm_object):
                bz = images.size(0)
                labels = labels.long().cuda()
                images = images.cuda()
                
                # For HybridCLIP, compute OOD scores using MCM/GL-MCM methods
                if is_hybrid:
                    res = net(images)
                    
                    # Get logits from both branches
                    logits_multimodal = res['logits_multimodal']  # (B, C)
                    logits_visual = res['logits_visual']  # (B, C)
                    
                    # Get features for local scores
                    local_features = res['local_features']  # (B, N, 768)
                    local_features = local_features / local_features.norm(dim=-1, keepdim=True)
                    
                    # Compute softmax for both branches
                    smax_multimodal = to_np(F.softmax(logits_multimodal / args.T, dim=1))  # (B, C)
                    smax_visual = to_np(F.softmax(logits_visual / args.T, dim=1))  # (B, C)
                    
                    # Compute local scores using text features (512D)
                    output_local = local_features @ text_features.T  # (B, N, C)
                    smax_local = to_np(F.softmax(output_local / args.T, dim=-1))  # (B, N, C)
                    
                    # Compute OOD scores based on score type
                    if args.score == 'HYBRID':
                        # Use GL-MCM for both branches and combine
                        mcm_global_multi = -np.max(smax_multimodal, axis=1)
                        mcm_global_visual = -np.max(smax_visual, axis=1)
                        mcm_local_multi = -np.max(smax_local, axis=(1, 2))
                        mcm_local_visual = -np.max(smax_local, axis=(1, 2))
                        
                        # Combine scores
                        ood_scores_multi = mcm_global_multi + args.lambda_local * mcm_local_multi
                        ood_scores_visual = mcm_global_visual + args.lambda_local * mcm_local_visual
                        
                        # Weighted combination
                        ood_scores = net.multimodal_weight * ood_scores_multi + net.visual_weight * ood_scores_visual
                        
                    elif args.score == 'HYBRID-MULTI':
                        # Use GL-MCM for multi-modal branch only
                        mcm_global = -np.max(smax_multimodal, axis=1)
                        mcm_local = -np.max(smax_local, axis=(1, 2))
                        ood_scores = mcm_global + args.lambda_local * mcm_local
                        
                    elif args.score == 'HYBRID-VISUAL':
                        # Use GL-MCM for visual branch only
                        mcm_global = -np.max(smax_visual, axis=1)
                        mcm_local = -np.max(smax_local, axis=(1, 2))
                        ood_scores = mcm_global + args.lambda_local * mcm_local
                        
                    elif args.score == 'HYBRID-MCM':
                        # Use MCM for both branches and combine
                        mcm_multi = -np.max(smax_multimodal, axis=1)
                        mcm_visual = -np.max(smax_visual, axis=1)
                        ood_scores = net.multimodal_weight * mcm_multi + net.visual_weight * mcm_visual
                        
                    elif args.score == 'HYBRID-SA-MCM':
                        # Use SA-MCM for multi-modal branch
                        pred_labels = np.argmax(smax_multimodal, axis=1)  # (B,)
                        mcm_global = -np.max(smax_multimodal, axis=1)
                        
                        # For SA-MCM, we need selected features
                        selected_feats = res['selected_feats']  # (B, N, 768)
                        selected_feats = selected_feats / selected_feats.norm(dim=-1, keepdim=True)
                        output_selected = selected_feats @ text_features.T  # (B, N, C)
                        smax_selected = to_np(F.softmax(output_selected / args.T, dim=-1))  # (B, N, C)
                        
                        # Extract slot confidences for predicted class
                        B, N, C = smax_selected.shape
                        slot_confs = smax_selected[np.arange(B)[:, None], np.arange(N)[None, :], pred_labels[:, None]]
                        min_slot_conf = np.min(slot_confs, axis=1)
                        mcm_selected = -min_slot_conf
                        
                        ood_scores = mcm_global + args.lambda_local * mcm_selected
                        
                    else:
                        # Default to GL-MCM combined
                        mcm_global_multi = -np.max(smax_multimodal, axis=1)
                        mcm_global_visual = -np.max(smax_visual, axis=1)
                        mcm_local = -np.max(smax_local, axis=(1, 2))
                        
                        ood_scores_multi = mcm_global_multi + args.lambda_local * mcm_local
                        ood_scores_visual = mcm_global_visual + args.lambda_local * mcm_local
                        ood_scores = net.multimodal_weight * ood_scores_multi + net.visual_weight * ood_scores_visual
                    
                    _score.append(ood_scores)
                    continue
                
                # Original logic for non-hybrid models
                # global_features, local_features = net.encode_image(images)  # .float()
                res=net(images)
                global_features = res['global_features']  # .float()
                local_features = res['local_features']  # .float()
                # selected_feats = res['selected_feats']

                # Remove unnecessary FP32 conversion - let autocast handle precision
                # global_features = global_features.float()
                # local_features = local_features.float()
                # selected_feats = selected_feats.float()

                global_features /= global_features.norm(dim=-1, keepdim=True)
                local_features /= local_features.norm(dim=-1, keepdim=True)
                # selected_feats /= selected_feats.norm(dim=-1, keepdim=True)+1e-8

                # Use cached text features from model (no recomputation needed)
                output_global = global_features @ text_features.T
                output_local = local_features @ text_features.T
                # output_selected = selected_feats @ text_features.T
                # import pdb
                # pdb.set_trace()

                smax_global = to_np(F.softmax(output_global/ args.T, dim=1))
                smax_local = to_np(F.softmax(output_local/ args.T, dim=-1))  # batch, grid, grid, class
                # smax_selected = to_np(F.softmax(output_selected/ args.T, dim=-1))

                if args.score == 'MCM':
                    _score.append(-np.max(smax_global, axis=1)) 
                elif args.score == 'L-MCM':
                    mcm_local_score = -np.max(smax_local, axis=(1, 2))
                    _score.append(mcm_local_score) 
                elif args.score == 'GL-MCM':
                    mcm_global_score = -np.max(smax_global, axis=1)
                    # import pdb
                    # pdb.set_trace()
                    mcm_local_score = -np.max(smax_local, axis=(1, 2))
                    _score.append(mcm_global_score+args.lambda_local*mcm_local_score)
                elif args.score == 'GL-MCM-L':
                    # import pdb
                    # pdb.set_trace()
                    mcm_global_score = -np.max(smax_global, axis=1)

                    #解法一：先在 Patch 维度取 Max，再在 Batch 维度取 Mean
                    # patch_confidences = np.max(smax_local, axis=2) 
                    # mcm_local_score = -np.mean(patch_confidences, axis=1)

                    mcm_local_score = -np.max(smax_local, axis=(1, 2))

                    # mcm_selected_score= -np.min(np.max(smax_selected,axis=2), axis=(1))
                    


                    _score.append(mcm_global_score+args.lambda_local*mcm_local_score+args.lambda_local*mcm_selected_score)

                elif args.score == 'SA-MCM': # 推荐使用这个新名字：Structure-Aware MCM
                    # 1. 确定目标类别 (基于最准的 Global 分数)
                    # smax_global: (B, C)
                    pred_labels = np.argmax(smax_global, axis=1) # (B,)
                    
                    # 2. 计算 Global Score
                    mcm_global_score = -np.max(smax_global, axis=1) # 负号表示分数越高越ID(或越OOD，看你定义)
                    
                    # 3. 计算 Local Score (Selected Slots) - 修正部分
                    # smax_selected: (B, K, C) -> 我们只关心 pred_labels 那一列
                    B, K, C = smax_selected.shape
                    
                    # 使用花式索引提取：每个样本、所有Slot、对应的预测类
                    # 结果形状: (B, K) -> 每个 Slot 对"预测类"的信心
                    slot_confs = smax_selected[np.arange(B)[:, None], np.arange(K)[None, :], pred_labels[:, None]]
                    
                    # 4. 体现支撑集思想
                    # 思想：木桶效应。如果是一个完整的 ID 物体，它的所有 Slot (头、腿、身) 都应该支持这个类。
                    # 如果有一个 Slot 支持度极低 (比如腿不对)，说明结构不完整 -> OOD。
                    min_slot_conf = np.min(slot_confs, axis=1) # (B,)
                    
                    mcm_selected_score = -min_slot_conf
                    
                    # 5. 计算原始 Local (Patch) Score - 修正部分
                    # smax_local: (B, N, C)
                    # 同样，我们只看预测类别的信心，而不是全局最大
                    patch_confs = smax_local[np.arange(B)[:, None], np.arange(smax_local.shape[1])[None, :], pred_labels[:, None]]
                    # 选最强的 Top-K Patch 平均 (排除背景)
                    # topk_patch_conf = np.sort(patch_confs, axis=1)[:, -16:].mean(axis=1)
                    # 或者简单点，用 Max
                    mcm_local_score = -np.max(patch_confs, axis=1)

                    # 6. 融合
                    # 这里你可以调权重，min_slot_score 现在非常有意义了
                    _score.append(mcm_global_score + args.lambda_local * mcm_selected_score)

                else:
                    raise NotImplementedError
    return concat(_score)[:len(loader.dataset)].copy()   


def get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list):
    '''
    1) evaluate detection performance for a given OOD test set (loader)
    2) print results (FPR95, AUROC, AUPR)
    '''
    aurocs, auprs, fprs = [], [], []
    measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(f'in score samples (random sampled): {in_score[:3]}, out score samples: {out_score[:3]}')

    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)  # used to calculate the avg over multiple OOD test sets
    print_measures(log, auroc, aupr, fpr, args.score)