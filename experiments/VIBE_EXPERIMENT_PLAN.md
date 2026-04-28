```md
# Experiment Plan: Dual-Space CLIP Few-Shot OOD

## 1. Global Experimental Rules

### Dataset / Backbone
- ID dataset: `ImageNet`
- OOD datasets:
  - `iNaturalist`
  - `SUN`
  - `places365`
  - `Texture`
- backbone: `ViT-B/16`

### Few-shot settings
- `1-shot`
- `16-shot`

### Seeds
- important experiments must run with:
  - `seed1`
  - `seed2`
  - `seed3`

### Metrics
For every experiment report:
- ID Accuracy
- AUROC on each OOD dataset
- FPR95 on each OOD dataset
- average AUROC
- average FPR95

### General principle
- adapted features are used for training losses and semantic scoring
- frozen visual features are used for visual verification by default

### GPU requirement
Before launching experiments:
1. run `nvidia-smi`
2. detect available GPUs
3. try to use all 8 GPUs by running multiple jobs in parallel

---

## 2. Existing External Text Features

Use external text features from:
- `/amax/yeliu/checkpoints/fa/...`
- `/amax/yeliu/checkpoints/locoop/...`
- `/amax/yeliu/checkpoints/sct/...`

Each experiment should support specifying:
- prompt-learning method
- shot
- seed
- `text_features_path`

Main prompt-learning baselines to test:
- `FA`
- `LoCoOp`
- `SCT`

Optional:
- `CoOp` if available in matching format

---

## 3. Experiment Groups Overview

We want a relatively complete experimental coverage:

1. sanity checks and bug fixes
2. baseline reproduction
3. training-side improvements
4. visual verification improvements
5. fusion improvements
6. combined best models
7. diagnostics and analysis
8. shot sensitivity / seed stability

---

## 4. Group G0: Sanity Checks and Fixes

### G0.1 Intra-class consistency bug fix
- fix the logic bug in intra-class consistency loss
- verify training still runs correctly
- compare:
  - before bug fix
  - after bug fix

Report:
- ID Acc
- avg AUROC
- avg FPR95

Priority: medium

---

### G0.2 SelfAttentionFuser return-value check
- verify whether the old self-attention fuser returns the wrong tensor
- only needed if self-attention fuser is used in the new project

Priority: low-to-medium

---

### G0.3 Visual feature source sanity
Compare visual verification using:
- `frozen` visual feature
- `adapted` visual feature

This is a very important ablation.

Report:
- per-dataset AUROC/FPR95
- average AUROC/FPR95

Priority: very high

---

## 5. Group G1: Baseline Reproduction

### G1.1 Current baseline reproduction
Reproduce current visual-side shared adapter baseline using:
- external text features
- semantic exclusion
- LoCoOp-style patch OOD loss
- current GL-MCM style evaluation

Run for:
- prompt methods: `FA`, `LoCoOp`, `SCT`
- shots: `1`, `16`
- seeds: at least `1`

Priority: very high

---

### G1.2 Current hybrid/prototype verification reproduction
Reproduce the current prototype-style or hybrid evaluation behavior.

Run for:
- best baseline setting from G1.1
- frozen vs adapted visual feature source

Priority: high

---

## 6. Group G2: Training-Side Improvements

### G2.1 Style consistency
#### Goal
Improve texture robustness.

#### Variants
- `style_consistency_type = kl`
- `style_consistency_type = feat`

#### Hyperparameters
- `lambda_style in {0.01, 0.05, 0.1, 0.2}`

#### Recommended augmentation strength variants
- mild style augmentation
- strong style augmentation

Suggested ops:
- ColorJitter
- RandomGrayscale
- GaussianBlur
- Solarization
- RandomAutocontrast

#### Run settings
- first run on `LoCoOp text features`
- then test on `FA` and `SCT`

Priority: very high

---

### G2.2 Dual patch reliability
#### Goal
Improve scene/context robustness by:
- suppressing pseudo-OOD patches
- encouraging pseudo-ID patches

#### Variants
- `patch_id_mode = ce`
- `patch_id_mode = kl`

#### Hyperparameters
- `lambda_patch_id in {0.01, 0.05, 0.1, 0.2}`
- `lambda_patch_ood in {0.05, 0.1, 0.2}`
- `patch_topk in {50, 100, 200}`

#### Expected focus
- SUN
- places365

Priority: very high

---

### G2.3 Hard-negative semantic exclusion
#### Goal
Reduce over-permissive semantic matching.

#### Variants
- all negatives
- hard negatives only

#### Hyperparameters
- `hard_negative_M in {5, 10, 20}`
- `margin in {0.05, 0.1, 0.2, 0.3}`

#### Expected focus
- iNaturalist
- Texture
- general calibration

Priority: high

---

### G2.4 Training-side module combinations
Evaluate combinations:
1. baseline
2. baseline + style consistency
3. baseline + dual patch reliability
4. baseline + hard-negative SE
5. baseline + style + patch
6. baseline + style + hard-negative SE
7. baseline + patch + hard-negative SE
8. baseline + style + patch + hard-negative SE

Priority: very high

---

## 7. Group G3: Visual Verification Improvements

Important:
All visual verification experiments here should use:
- **frozen visual features by default**

### G3.1 Verification type ablation
Compare:
- `prototype_mean`
- `prototype_medoid`
- `support_top1`
- `support_topr`
- `support_lse`

#### Hyperparameters
- `vis_topr in {1, 2, 4}`
- `vis_tau in {0.01, 0.05, 0.1, 0.2}`

Notes:
- for `1-shot`, `topr=1` only
- for `16-shot`, test `topr=1,2,4`

Priority: very high

---

### G3.2 Prototype robustness ablation
If prototype methods are retained, compare:
- mean prototype
- medoid prototype
- trimmed mean prototype
- weighted mean prototype

#### Hyperparameters
- trim count:
  - `1` for 16-shot
- weight mode:
  - similarity-based
  - uniform baseline

Priority: medium-high

---

### G3.3 Visual feature source ablation inside verifier
For the best verification methods, compare:
- frozen feature source
- adapted feature source

Priority: high

---

## 8. Group G4: Fusion Improvements

### G4.1 Fusion type comparison
Compare:
- `linear`
- `mean_suppress`
- `veto_penalty`
- `multiplicative_discount`

Priority: very high

---

### G4.2 Veto penalty hyperparameter search
#### Formula idea
`z_fused[c] = z_sem[c] - lambda_veto * relu(delta_veto - s_vis[c])`

#### Hyperparameters
- `lambda_veto in {0.1, 0.3, 0.5, 1.0}`
- `delta_veto in {0.0, 0.1, 0.2, 0.3}`

Priority: very high

---

### G4.3 Multiplicative discount hyperparameter search
#### Hyperparameters
- `beta_discount in {0.5, 1.0, 2.0}`
- optional sigmoid scale:
  - `{5, 10, 20}`

Priority: medium-high

---

### G4.4 Dynamic fusion strength (optional)
Use semantic uncertainty to modulate visual fusion strength.

Possible signals:
- semantic margin
- semantic entropy

Priority: optional / exploratory

---

## 9. Group G5: Full Combined Models

### G5.1 Best training-only model
Combine the best settings from:
- style consistency
- dual patch reliability
- hard-negative SE

Run:
- prompt methods: FA / LoCoOp / SCT
- shots: 1 and 16
- seeds: 1,2,3

Priority: very high

---

### G5.2 Best verification-only model
Combine the best settings from:
- support-set verifier
- frozen visual feature source
- best fusion strategy

Run:
- best training baseline
- shots: 1 and 16
- seeds: 1,2,3

Priority: very high

---

### G5.3 Final full best model
Combine:
- best training-side improvements
- best verification-side improvements

Compare against:
1. current baseline
2. current best shared-adapter setting
3. final best model

Run:
- prompt methods: FA / LoCoOp / SCT
- shots: 1 and 16
- seeds: 1,2,3

Priority: highest

---

## 10. Group G6: Prompt-Method Generalization

### G6.1 Cross prompt-feature source comparison
Use the same visual-side method with different external text features:
- FA text features
- LoCoOp text features
- SCT text features

Check:
- which prompt-learning source works best with the new dual-space method
- whether gains are consistent across prompt sources

Priority: high

---

## 11. Group G7: Diagnostics and Analysis

### G7.1 Per-dataset gain analysis
For all strong variants, compute performance change relative to baseline on:
- iNaturalist
- SUN
- places365
- Texture

Goal:
- identify which modules help texture robustness
- identify which modules help scene/context robustness

Priority: very high

---

### G7.2 Score distribution plots
For baseline vs best model:
- plot ID score distribution
- plot OOD score distribution
- do this for each OOD dataset if possible

Priority: medium-high

---

### G7.3 ID/OOD confidence analysis
Compare:
- max softmax on ID vs OOD
- semantic margin on ID vs OOD
- if easy: ECE

Priority: medium

---

### G7.4 Support verification diagnostics
For best verifier:
- inspect class-wise verification scores
- compare:
  - ID queries
  - OOD queries
- check whether visual verifier acts as a true veto

Priority: medium

---

## 12. Group G8: Shot Sensitivity and Stability

### G8.1 Shot sensitivity
Run:
- 1-shot
- 2-shot
- 4-shot
- 8-shot
- 16-shot

At least for:
- current baseline
- final best model

Priority: high

---

### G8.2 Seed stability
For final best model:
- seed1
- seed2
- seed3

Report:
- mean and std for ID Acc / AUROC / FPR95

Priority: very high

---

## 13. Recommended Execution Order

If compute budget is limited, run in this order:

1. G0.3 visual feature source sanity
2. G1.1 baseline reproduction
3. G2.1 style consistency
4. G2.2 dual patch reliability
5. G3.1 verification type ablation
6. G4.1 fusion type comparison
7. G5.3 final full best model
8. G7.1 per-dataset gain analysis
9. G8.2 seed stability

---

## 14. Suggested Hyperparameter Search Strategy

### Stage 1: one-factor-at-a-time
Fix a strong baseline and vary one new module at a time.

### Stage 2: local combination search
Take top-performing settings from Stage 1 and combine them.

### Stage 3: multi-seed confirmation
For promising configs, run:
- seed1
- seed2
- seed3

### Stage 4: final report table generation
Aggregate:
- per-dataset AUROC/FPR95
- average AUROC/FPR95
- ID Acc
- mean/std over seeds

---

## 15. Strong Default Starting Point

Use this as a starting reference:

- backbone: `ViT-B/16`
- shots: `1` or `16`
- seeds: `1`
- text features: LoCoOp or SCT external text features
- adapter ratio: `0.5`
- residual coef: `0.2`
- patch topk: `200`
- initial style lambda: `0.05`
- initial patch id lambda: `0.05`
- initial patch ood lambda: `0.1`
- hard negative M: `10`
- margin: `0.1`
- verifier: `support_lse`
- visual feature source: `frozen`
- fusion: `veto_penalty`
- lambda_veto: `0.3`
- delta_veto: `0.1`

---

## 16. Output Expectations

For every completed experiment group, save:
1. config
2. log
3. best checkpoint
4. evaluation summary
5. final csv/json table

For the final report, generate:
- main result table
- ablation table
- prompt-source comparison table
- per-dataset gain table
- shot sensitivity table
- seed mean/std table
```

---