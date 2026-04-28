```md
# Project Brief: Dual-Space Few-Shot OOD Detection with CLIP

## 1. Goal

Create a **new project folder** for extending an existing CLIP-based few-shot OOD detection system.

Important requirement:
- **Do not rewrite the whole old codebase.**
- Only implement the **necessary new code** and keep the project lightweight and modular.
- Reuse existing checkpoints, text features, and evaluation logic where helpful.
- Before running experiments, check available GPUs with `nvidia-smi` and try to utilize **all 8 GPUs** efficiently.

The main objective is to improve:
1. texture robustness
2. scene/context robustness
3. visual verification quality

under a **dual-space design**:
- **adapted features**: used for training-time semantic classification and regularization
- **original frozen visual features**: used for inference-time visual verification

---

## 2. Existing Assets and Environment

### 2.1 Existing prompt-learning text features

Prompt-learning methods (FA / LoCoOp / SCT / etc.) have already been trained separately.
Their learned text features are stored under:

```bash
/amax/yeliu/checkpoints/
```

Typical structure:

```bash
/amax/yeliu/checkpoints/fa/ViT-B-16/16shots/seed1/
/amax/yeliu/checkpoints/locoop/ViT-B/16/1shots/seed1/
/amax/yeliu/checkpoints/locoop/ViT-B/16/16shots/seed1/
/amax/yeliu/checkpoints/sct/ViT-B/16/1shots/seed1/
/amax/yeliu/checkpoints/sct/ViT-B/16/16shots/seed1/
```

Each directory typically contains:
- `class_names_xxx.json`
- `feature_info_xxx.json`
- `text_features_xxx.npy`
- `text_features_xxx.pt`

These should be loaded through `--text_features_path`.

---

### 2.2 Existing visual-side training outputs

Current visual-side experiment logs/checkpoints are stored under:

```bash
/home/yeliu/huhuhu/data/ICML2026/GL_MCM_FA/logs/
```

Typical structure:

```bash
/home/yeliu/huhuhu/data/ICML2026/GL_MCM_FA/logs/coop-1_identity_1_20260129_125354/
```

Best checkpoint is usually:

```bash
checkpoints/epoch_999.pt
```

---

### 2.3 Existing codebase references

Existing codebase path:

```bash
/home/yeliu/huhuhu3/MF-OOD/
```

Key files there:
- training model:
  - `src/model_modular.py`
- training entry:
  - `src/train_eval.py`
- evaluation entry:
  - `src/eval_ood_detection.py`
- current search scripts:
  - `scripts/common_params_locoop.sh`
  - `scripts/search_shared_adapter_locoop_v1.sh`

The new project should **not modify the old project aggressively**.
Instead, create a **new project** and only copy/adapt the necessary parts.

---

## 3. New Project Requirements

Create a **new standalone project** with a clean structure, for example:

```bash
dual_space_viver/
  README.md
  src/
    model.py
    losses.py
    train.py
    eval.py
    support_verifier.py
    utils.py
  scripts/
    common_params.sh
    search_*.sh
  configs/
  logs/
```

You may also organize it differently, but it should be:
- clean
- minimal
- modular
- only include what is needed

---

## 4. Core Method Design

### 4.1 Dual-space principle

This is the most important design rule.

#### Adapted semantic branch
Use adapted features for:
- classification
- semantic exclusion
- patch-level OOD / patch reliability losses
- style consistency loss

#### Frozen visual verification branch
Use original frozen visual features for:
- support-set verification
- prototype or support matching
- visual veto / discount fusion at inference

Do **not** default to using adapted features for visual verification.

---

### 4.2 Current feature semantics from the old code

In the old model forward output:
- `global_feature`: original frozen visual global feature
- `local_features`: original frozen patch features
- `global_features`: adapted global feature after residual adapter
- `final_feats`: normalized final classification feature
- `selected_feats`: patch features used for loss

In the new project, keep this semantics clear.

---

## 5. Required Improvements

Implement the following components.

### 5.1 Style consistency loss
Purpose:
- reduce texture bias

Implementation:
- create a style-perturbed image view during training
- add consistency loss between original and style-perturbed predictions/features

Support:
- `KL` consistency
- `feature cosine` consistency

Recommended CLI/config args:
- `use_style_consistency`
- `style_consistency_type`
- `lambda_style`

---

### 5.2 Dual patch reliability loss
Purpose:
- not only suppress pseudo-OOD/background patches
- also promote pseudo-ID / reliable patches

Implementation:
- identify pseudo-OOD patches:
  GT label not in patch top-k predictions
- identify pseudo-ID patches:
  GT label is in patch top-k predictions

Support:
- pseudo-OOD patch high-entropy loss
- pseudo-ID patch CE or KL-to-global loss

Recommended CLI/config args:
- `use_dual_patch_loss`
- `patch_id_mode`
- `lambda_patch_id`
- `lambda_patch_ood`
- `patch_topk`

---

### 5.3 Hard-negative semantic exclusion
Purpose:
- focus semantic exclusion on hard negatives instead of all negatives

Recommended CLI/config args:
- `use_hard_negative_se`
- `hard_negative_M`
- `margin`

---

### 5.4 Frozen-visual support-set verification
Purpose:
- replace simple mean prototype verification
- perform stronger support-based visual verification

Must support:
- `prototype_mean`
- `prototype_medoid`
- `support_top1`
- `support_topr`
- `support_lse`

Important:
- default feature source for this module should be:
  - `frozen`

Recommended CLI/config args:
- `visual_feature_source`
- `vis_verify_type`
- `vis_topr`
- `vis_tau`

---

### 5.5 Veto-style semantic-visual fusion
Purpose:
- replace heuristic mean suppression with more interpretable fusion

Support:
- `linear`
- `mean_suppress`
- `veto_penalty`
- `multiplicative_discount`

Recommended CLI/config args:
- `fusion_type`
- `lambda_veto`
- `delta_veto`
- `beta_discount`

---

## 6. Bugs / Fixes to Handle

### 6.1 Intra-class consistency bug
In the old code, `compute_intra_class_consistency_loss()` likely has a logic bug:
- it appears to return zero when valid same-class pairs exist

Fix:
- only return zero when the number of same-class pairs is zero

---

### 6.2 SelfAttentionFuser return bug
Check whether the old `SelfAttentionFuser.forward()` returns the wrong variable.

---

### 6.3 Explicit visual feature source control
Evaluation must support explicit selection between:
- `frozen`
- `adapted`

for visual verification.

---

## 7. Data / Evaluation Setup

### ID dataset
- `ImageNet`

### OOD datasets
- `iNaturalist`
- `SUN`
- `places365`
- `Texture`

### Shots
- `1-shot`
- `16-shot`

### Seeds
- at least `1, 2, 3` for important experiments

### Metrics
- ID Accuracy
- AUROC
- FPR95
- average AUROC
- average FPR95

---

## 8. GPU Usage Requirement

There are **8 GPUs** available.

Before running experiments:
1. run `nvidia-smi`
2. inspect free GPUs / memory
3. schedule jobs to maximize usage of all 8 GPUs if possible

Requirements:
- support specifying GPU ids
- preferably support launching multiple experiments in parallel across GPUs
- log which GPU is used by each run

Optional:
- implement a lightweight launcher that distributes experiments to available GPUs

---

## 9. Coding Requirements

### General
- write a new project
- only include necessary code
- keep code readable and modular
- minimize dependency on old project internals
- but feel free to borrow useful code patterns

### Output
Please produce:
1. project structure
2. core code files
3. training script(s)
4. evaluation script(s)
5. bash search scripts
6. logging and checkpoint saving
7. concise README

---

## 10. Initial Baseline Behavior to Preserve

The new project should be able to reproduce the current style of workflow:

1. load external text features from prompt-learning methods
2. freeze CLIP encoders
3. train visual-side adaptation module
4. save best checkpoint
5. evaluate using:
   - saved model checkpoint
   - external text features
   - chosen OOD score / fusion strategy

---

## 11. Suggested Minimal New Project Files

Suggested implementation files:

```bash
src/
  model.py                 # CLIP wrapper + adapter branch
  losses.py                # CE / style / patch / semantic exclusion
  verifier.py              # support-set verification in frozen visual space
  fusion.py                # linear / mean_suppress / veto / discount fusion
  train.py                 # training entry
  eval.py                  # evaluation entry
  data.py                  # dataloader helpers
  utils.py                 # misc helpers
scripts/
  common_params.sh
  search_style.sh
  search_patch.sh
  search_verifier.sh
  search_fusion.sh
  search_combined.sh
```

---

## 12. Important Final Reminder

Do not over-engineer.
The goal is:
- a **clean new project**
- implementing only the **necessary extensions**
- preserving the core dual-space idea:
  - adapted semantic training
  - frozen visual verification
```

---
