# Experiment Plan for ViVer / Modular CLIP Few-shot OOD

## Goal
Improve the current method along two axes:

1. **Training-time robustness**
   - reduce texture bias
   - improve patch-level evidence reliability
   - improve hard-negative semantic exclusion

2. **Inference-time visual verification**
   - replace simple prototype mean suppression with stronger frozen-visual-space support verification
   - keep visual verification in the **original frozen visual feature space**, not in the adapted multimodal-aligned space

---

## Global Notes
- Keep the current design principle:
  - **adapted features** are used for semantic classification / OOD-aware training losses
  - **original frozen visual features** are used for visual verification at inference
- All new experiments should be run under:
  - 1-shot and 16-shot
  - same backbone as current main setting
  - same OOD datasets: iNaturalist / SUN / Places / Texture
- Report:
  - ID accuracy
  - AUROC / FPR95 for each OOD dataset
  - average AUROC / average FPR95
- For each ablation, run at least **3 seeds**
- Save full config and best checkpoint

---

# Group A. Training Loss Improvements

## A1. Style / Texture Consistency Loss
### Motivation
Current training does not explicitly address texture bias. Add style-perturbed view consistency.

### Implementation
For each training image, generate an extra augmented version `x_style`.
Use stronger style-changing augmentation than the default train transform.

Recommended augmentation candidates:
- ColorJitter (stronger than current)
- RandomGrayscale
- GaussianBlur
- Solarization
- RandomAutocontrast
- Optional advanced version: Fourier amplitude perturbation

### Loss options
#### A1.1 Logit consistency
`L_style_kl = KL(p(x) || p(x_style)) + KL(p(x_style) || p(x))`

#### A1.2 Feature consistency
`L_style_feat = 1 - cosine(final_feat(x), final_feat(x_style))`

### Hyperparameters to search
- `lambda_style`: [0.01, 0.05, 0.1, 0.2]
- consistency type: [`kl`, `feat`]

### Priority
High

---

## A2. Dual Patch Reliability Loss
### Motivation
Current LoCoOp-style loss only enforces high entropy on pseudo-OOD patches.
Need a positive term for pseudo-ID patches.

### Implementation
Use adapted patch features for patch loss.

Define:
- pseudo-OOD patches: GT label not in top-k predictions
- pseudo-ID patches: GT label in top-k predictions

### Loss options
#### A2.1 ID patch cross-entropy + OOD patch entropy
`L_patch = lambda_patch_ood * L_patch_ood + lambda_patch_id * L_patch_id_ce`

Where:
- `L_patch_ood = - mean entropy over pseudo-OOD patches`
- `L_patch_id_ce = mean CE over pseudo-ID patches`

#### A2.2 ID patch / global consistency + OOD patch entropy
`L_patch = lambda_patch_ood * L_patch_ood + lambda_patch_id * L_patch_id_kl`

Where:
- `L_patch_id_kl = mean KL(q_patch || p_global)` over pseudo-ID patches

### Hyperparameters to search
- `lambda_patch_id`: [0.01, 0.05, 0.1, 0.2]
- `lambda_patch_ood`: [0.05, 0.1, 0.2]
- `top_k`: [50, 100, 200]
- ID patch mode: [`ce`, `kl`]

### Priority
Very High

---

## A3. Hard Negative Semantic Exclusion
### Motivation
Current semantic exclusion uses all negatives. Try harder and more focused negatives.

### Implementation
Instead of using all negative classes in logsumexp, only use top-M hardest negatives based on current semantic logits.

### Variants
#### A3.1 Top-M hard negatives
Use top-M semantic negatives for each sample.

#### A3.2 Class-similarity-aware negatives
Use text-anchor similarity to pre-select semantically adjacent negative classes, then apply SE.

### Hyperparameters to search
- `hard_negative_M`: [5, 10, 20, all]
- `margin`: [0.05, 0.1, 0.2, 0.3]

### Priority
Medium-High

---

## A4. Fix and Re-evaluate Intra-class Consistency
### Motivation
Check the current bug in `compute_intra_class_consistency_loss`.
The condition `if num_pairs > 0: return 0` appears incorrect.

### Implementation
Fix:
- if `num_pairs == 0`, return zero
- otherwise compute the loss normally

### Hyperparameters to search
- `lambda_intra_class`: [0.0, 0.01, 0.05, 0.1]
- `intra_class_temp`: [0.05, 0.1, 0.2]

### Priority
Medium

---

# Group B. Adapter Architecture Ablations

## B1. Residual Adapter Form
### Motivation
Check whether the current adapter formulation is optimal.

### Variants
#### B1.1 Current implementation
`output = LN(adapter(x))`, residual added outside

#### B1.2 Standard residual adapter
`output = x + alpha * adapter(LN(x))`

#### B1.3 Post-norm residual adapter
`output = LN(x + alpha * adapter(x))`

### Hyperparameters to search
- `residual_coef`: [0.05, 0.1, 0.2, 0.5]
- `adapter_ratio`: [0.25, 0.5, 1.0]

### Priority
Medium

---

# Group C. Visual Verification Improvements
Important: all visual verification in this group must use **original frozen visual features**, not adapted features.

## C1. Replace Mean Prototype with Support-Set Matching
### Motivation
Mean prototype is too simple and can be noisy in few-shot.
Use support-set verification directly in frozen visual feature space.

### Support feature space
Use:
- original `image_features` from frozen visual encoder
- no adapter applied

### Variants
#### C1.1 Top-1 support matching
For each class:
`score_c = max cosine(query_feat, support_feats_c)`

#### C1.2 Top-r support matching
For each class:
`score_c = mean of top-r cosine similarities`

Recommended `r <= shots`

#### C1.3 LogSumExp support matching
For each class:
`score_c = tau_v * logmeanexp(sim / tau_v)`

### Hyperparameters to search
- matching type: [`top1`, `topr`, `lse`]
- `r`: [1, 2, 4] (only valid when shots >= r)
- `tau_v`: [0.01, 0.05, 0.1, 0.2]

### Priority
Very High

---

## C2. Replace Mean Suppression with Veto Fusion
### Motivation
Current mean suppression is heuristic.
Use visual branch as a conservative veto / penalty signal.

### Inputs
- semantic logits from adapted branch: `z_sem`
- visual verification scores from frozen visual branch: `s_vis`

### Variants
#### C2.1 Penalty fusion
`z_fused[c] = z_sem[c] - lambda_veto * relu(delta - s_vis[c])`

#### C2.2 Multiplicative discount
Convert `s_vis[c]` to reliability `r[c] = sigmoid(a * s_vis[c] + b)`
Then:
`p_fused[c] proportional to p_sem[c] * r[c]^beta`

#### C2.3 Linear fusion baseline
Keep current linear fusion as reference baseline

### Hyperparameters to search
For penalty fusion:
- `lambda_veto`: [0.1, 0.3, 0.5, 1.0]
- `delta`: [0.0, 0.1, 0.2, 0.3]

For multiplicative discount:
- `beta`: [0.5, 1.0, 2.0]
- sigmoid temperature / scale: [5, 10, 20]

### Priority
Very High

---

## C3. Robust Prototype Variants
### Motivation
If prototype-style verification is retained, make prototypes more robust.

### Variants
#### C3.1 Mean prototype
Current baseline

#### C3.2 Medoid prototype
Choose the support sample with the highest average similarity to other supports

#### C3.3 Trimmed mean prototype
Remove the farthest support sample(s) from the class center before averaging

#### C3.4 Weighted mean prototype
Weight support samples by support-support consistency

### Hyperparameters to search
- prototype type: [`mean`, `medoid`, `trimmed_mean`, `weighted_mean`]
- for trimmed mean: trim count [1] or [1, 2] if shots allow

### Priority
Medium-High

---

## C4. Dynamic Semantic-Visual Fusion Weight
### Motivation
Fixed fusion weights may be suboptimal.

### Variants
#### C4.1 Margin-aware fusion
Use semantic margin:
`margin_sem = top1_logit - top2_logit`
Set visual fusion strength larger when margin is small

Example:
`lambda_vis(x) = lambda0 * exp(-gamma * margin_sem)`

#### C4.2 Entropy-aware fusion
Use semantic entropy instead of margin

### Hyperparameters to search
- `lambda0`: [0.1, 0.3, 0.5]
- `gamma`: [1, 5, 10]

### Priority
Medium

---

# Group D. Combined Best Config Search

## D1. Best Training + Best Verification Combination
### Goal
After finishing Groups A and C, combine the best training-side and inference-side improvements.

### Suggested combinations
Run these combinations explicitly:

1. baseline current best
2. + style consistency
3. + dual patch reliability
4. + style consistency + dual patch reliability
5. + support-set matching
6. + support-set matching + veto fusion
7. + style consistency + dual patch reliability + support-set matching + veto fusion

### Priority
Very High

---

# Group E. Diagnostics / Additional Analysis

## E1. Per-dataset effect analysis
### Goal
Check which components help:
- Texture OOD
- Scene OOD (SUN / Places)
- Fine-grained / iNaturalist

### Required output
For each method:
- delta AUROC / FPR95 relative to baseline on each OOD dataset
- comment whether the gain is mainly from texture robustness, scene robustness, or both

### Priority
High

---

## E2. Confidence / calibration analysis
### Goal
Check if new methods reduce overconfidence.

### Metrics
- max softmax score histograms for ID vs OOD
- ECE if easy to compute
- semantic margin distributions for ID vs OOD

### Priority
Medium

---

## E3. Support-shot sensitivity
### Goal
Check if visual verification improvements are more useful in lower-shot settings.

### Settings
- 1-shot
- 2-shot
- 4-shot
- 8-shot
- 16-shot

### Priority
Medium-High

---

# Minimal Recommended Execution Order
If compute budget is limited, run in this order:

1. A2 Dual Patch Reliability
2. A1 Style Consistency
3. C1 Support-Set Matching
4. C2 Veto Fusion
5. D1 Combined Best Config Search
6. E1 Per-dataset analysis

---

# Concrete Code Change Suggestions

## For A1
Need to add:
- second augmented image view in training loop
- second forward pass
- consistency loss function
- config flags:
  - `use_style_consistency`
  - `style_consistency_type`
  - `lambda_style`

## For A2
Need to add:
- function to split pseudo-ID and pseudo-OOD patches
- new patch ID loss
- config flags:
  - `use_dual_patch_loss`
  - `patch_id_mode`
  - `lambda_patch_id`
  - `lambda_patch_ood`

## For C1
Need to add:
- support feature cache using original frozen `image_features`
- class-wise support feature bank
- matching backend: top1 / topr / lse
- config flags:
  - `vis_verify_type`
  - `vis_topr`
  - `vis_tau`

## For C2
Need to add:
- fusion function for semantic logits and visual verification scores
- support penalty fusion and multiplicative discount
- config flags:
  - `fusion_type`
  - `lambda_veto`
  - `delta_veto`
  - `beta_discount`

---

# Important Bug Fixes to Do First
1. Fix `compute_intra_class_consistency_loss`:
   - `if num_pairs == 0: return 0`
2. Check `SelfAttentionFuser.forward()` return value:
   - currently returns `x`, likely should return `x_out`
3. Verify whether current visual verification path truly uses original frozen visual features during evaluation; if not, add an explicit flag and separate branch

---

# Recommended Final Objective
Target a stronger version of the method with this structure:

- training branch:
  - CE
  - semantic exclusion
  - dual patch reliability
  - style consistency

- inference branch:
  - semantic logits from adapted branch
  - visual verification from frozen visual support-set matching
  - veto-style fusion

This should match the intended “semantic adaptation + frozen visual verification” design most cleanly.

---