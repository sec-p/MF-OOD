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
    
    # Use cached text features from model (already computed during initialization)
    # This ensures consistency with training: both use the same templates
    if hasattr(net, 'text_features') and net.text_features is not None:
        text_features = net.text_features
    else:
        # Fallback: compute text features if not cached
        # Use the same template as training: 'a photo of a {}'
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
                # global_features, local_features = net.encode_image(images)  # .float()
                res=net(images)
                global_features = res['global_features']  # .float()
                local_features = res['local_features']  # .float()
                selected_feats = res['selected_feats']

                # Remove unnecessary FP32 conversion - let autocast handle precision
                # global_features = global_features.float()
                # local_features = local_features.float()
                # selected_feats = selected_feats.float()

                global_features /= (global_features.norm(dim=-1, keepdim=True) + 1e-8)
                local_features /= (local_features.norm(dim=-1, keepdim=True) + 1e-8)
                selected_feats /= (selected_feats.norm(dim=-1, keepdim=True) + 1e-8)

                # Use cached text features from model (no recomputation needed)
                output_global = global_features @ text_features.T
                output_local = local_features @ text_features.T
                output_selected = selected_feats @ text_features.T
                # import pdb
                # pdb.set_trace()

                smax_global = to_np(F.softmax(output_global/ args.T, dim=1))
                smax_local = to_np(F.softmax(output_local/ args.T, dim=-1))  # batch, grid, grid, class
                smax_selected = to_np(F.softmax(output_selected/ args.T, dim=-1))

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

                    mcm_selected_score= -np.min(np.max(smax_selected,axis=2), axis=(1))
                    


                    _score.append(mcm_global_score+args.lambda_local*mcm_local_score+args.lambda_local*mcm_selected_score)

                elif args.score == 'SA-MCM': # Structure-Aware MCM with Local Classifier
                    # 1. Determine target class (based on most accurate Global score)
                    # smax_global: (B, C)
                    pred_labels = np.argmax(smax_global, axis=1) # (B,)
                    
                    # 2. Compute Global Score
                    mcm_global_score = -np.max(smax_global, axis=1) # Negative: higher score = more ID-like
                    
                    # 3. Compute Local Score using local_classifier
                    # Get patch logits from local_classifier
                    patch_logits = net.local_classifier(selected_feats)  # (B, N, C)
                    patch_probs = to_np(F.softmax(patch_logits / args.T, dim=-1))  # (B, N, C)
                    
                    # Extract confidence for predicted class
                    B, N, C = patch_probs.shape
                    patch_confs = patch_probs[np.arange(B)[:, None], np.arange(N)[None, :], pred_labels[:, None]]  # (B, N)
                    
                    # 4. Top-K Filter (ignore background tokens)
                    # Get num_select from config (default to 49 if not available)
                    num_select = getattr(net, 'num_select', 49)
                    if hasattr(net, 'cfg') and net.cfg is not None:
                        num_select = net.cfg.get('num_select', 49)
                    
                    # For each sample, sort patch confidences and take Top-K
                    # Top-K represents the most confident foreground patches
                    topk_confs = np.sort(patch_confs, axis=1)[:, -num_select:]  # (B, num_select)
                    
                    # 5. Compute local score using Min (bucket effect)
                    # If all top-K patches support the predicted class -> ID
                    # If any patch has low support -> OOD (incomplete structure)
                    mcm_local_score = -np.min(topk_confs, axis=1)  # (B,)
                    
                    # 6. Final Score: Global + lambda * Local
                    _score.append(mcm_global_score + args.lambda_local * mcm_local_score)

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