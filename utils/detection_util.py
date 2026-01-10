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
    # No need to recompute - this avoids redundant computation
    # if hasattr(net, 'text_features') and net.text_features is not None:
    #     text_features = net.text_features
    # else:
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
                # global_features, local_features = net.encode_image(images)  # .float()
                res=net(images)

                final_feats = res['final_feats']
                global_features = res['global_features']  # .float()
                local_features = res['local_features']  # .float()
                selected_feats = res['selected_feats']

                # Remove unnecessary FP32 conversion - let autocast handle precision
                # global_features = global_features.float()
                # local_features = local_features.float()
                # selected_feats = selected_feats.float()

                global_features /= global_features.norm(dim=-1, keepdim=True)
                final_feats /= final_feats.norm(dim=-1, keepdim=True)
                local_features /= local_features.norm(dim=-1, keepdim=True)
                selected_feats /= selected_feats.norm(dim=-1, keepdim=True)+1e-8

                # Use cached text features from model (no recomputation needed)
                output_global = global_features @ text_features.T
                output_local = local_features @ text_features.T
                output_selected = selected_feats @ text_features.T
                output_final = final_feats @ text_features.T

                # output_global = net.logit_scale * global_features @ text_features.T
                # output_local = net.logit_scale * local_features @ text_features.T
                # output_selected = net.logit_scale * selected_feats @ text_features.T
                # output_final = net.logit_scale * final_feats @ text_features.T

                # import pdb
                # pdb.set_trace()

                smax_global = to_np(F.softmax(output_global/ args.T, dim=1))
                smax_local = to_np(F.softmax(output_local/ args.T, dim=-1))  # batch, grid, grid, class
                smax_selected = to_np(F.softmax(output_selected/ args.T, dim=-1))
                smax_final = to_np(F.softmax(output_final/ args.T, dim=1))

                bg_mask = to_np(res['bg_mask'])  # (B, N, 1)
                bg_mask_flat = bg_mask.squeeze()  # (B, N)

                selected_scores = res['selected_scores']

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
                    
                    mcm_global_score = -np.max(smax_global, axis=1)

                    #解法一：先在 Patch 维度取 Max，再在 Batch 维度取 Mean
                    # patch_confidences = np.max(smax_local, axis=2) 
                    # mcm_local_score = -np.mean(patch_confidences, axis=1)

                    mcm_local_score = -np.max(smax_local, axis=(1, 2))

                    mcm_selected_score= -np.min(np.max(smax_selected,axis=2), axis=(1))
                    
                    avg_score = selected_scores.mean(dim=-1) # (B, N)
                    # 计算熵
                    probs = F.softmax(avg_score / args.T, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1) # (B,)

                    _score.append(mcm_global_score+args.lambda_local*mcm_local_score+args.lambda_local*mcm_selected_score- 0.1 * entropy.cpu().numpy())

                elif args.score == 'SA-MCM': # Structure-Aware MCM with Local Classifier
                    # 1. Determine target class (based on most accurate Global score)
                    # smax_global: (B, C)
                    

                    pred_labels = np.argmax(smax_global, axis=1) # (B,)
                    
                    # 2. Compute Global Score
                    mcm_global_score = -np.max(smax_global, axis=1) # Negative: higher score = more ID-like

                    mcm_local_score = -np.max(smax_local, axis=(1, 2))

                    mcm_selected_score= -np.min(np.max(smax_selected,axis=2), axis=(1))
                    
                    # 3. Compute Local Score for each sample individually
                    # Process each sample separately to avoid masked feature noise
                    B = selected_feats.shape[0]
                    mcm_local_score_list = []
                    
                    for i in range(B):
                        # a. Get mask for this sample
                        valid_mask_i = bg_mask_flat[i] > 0.5  # (N,)
                        num_valid = valid_mask_i.sum()
                        
                        if num_valid > 0:
                            # b. Get valid features for this sample
                            valid_feats_i = selected_feats[i][valid_mask_i]  # (num_valid, D)
                            
                            # c. Normalize valid features
                            valid_feats_i = valid_feats_i / valid_feats_i.norm(dim=-1, keepdim=True)
                            
                            # d. Compute scores for valid features
                            output_selected_i = valid_feats_i @ text_features.T  # (num_valid, C)
                            
                            # e. Apply temperature and softmax
                            smax_selected_i = to_np(F.softmax(output_selected_i / args.T, dim=-1))  # (num_valid, C)
                            
                            # f. Extract confidence for predicted class
                            patch_confs_i = smax_selected_i[:, pred_labels[i]]  # (num_valid,)
                            
                            # g. Compute local score (using Max for now, can be adjusted to min/mean)
                            mcm_local_score_i = -np.mean(patch_confs_i)
                        else:
                            # If no valid features, use a default score
                            mcm_local_score_i = 0.0
                        
                        mcm_local_score_list.append(mcm_local_score_i)
                    
                    mcm_selected_score = np.array(mcm_local_score_list)  # (B,)
                    import pdb
                    pdb.set_trace()
                    # 4. Final Score: Global + lambda * Local
                    _score.append(mcm_global_score + args.lambda_local*mcm_local_score + args.lambda_local * mcm_selected_score)

                elif args.score == 'AL-MCM': # Attribute-Local MCM
                    # 1. Compute Global Score (standard MCM)
                    mcm_global_score = -np.max(smax_global, axis=1)
                    
                    # 2. Compute Local Attribute Score
                    # Retrieve cached attribute features from model
                    if hasattr(net, 'attribute_features'):
                        attribute_features = net.attribute_features  # [1000, D]
                    else:
                        # Fallback to text features if attribute features not available
                        attribute_features = text_features
                    
                    # Compute Similarity: selected_feats [B, N, D] @ attribute_features.T [D, 1000] -> [B, N, 1000]
                    sim = selected_feats @ attribute_features.T
                    
                    # Max-Pooling: Find best matching patch for each class attribute
                    val, _ = sim.max(dim=1)  # [B, 1000]
                    
                    # Apply temperature and softmax
                    s_attr = to_np(F.softmax(val / args.T, dim=1))
                    
                    # MCM score for attributes
                    mcm_attr_score = -np.max(s_attr, axis=1)
                    
                    # 3. Fusion: Global + lambda * Local Attribute
                    _score.append(mcm_global_score + args.lambda_local * mcm_attr_score)

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