```md
# Experiment Parameters and Results

## 1. Common Parameters
The following parameters are set for all hyperparameter search scripts:

```bash
# Project root (relative to scripts directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default training parameters
DEFAULT_EPOCHS=30
DEFAULT_BATCH_SIZE=1024
DEFAULT_LR=0.001
DEFAULT_SEED=1
DEFAULT_BACKBONE="ViT-B/16"
DEFAULT_ROOT_PATH="/amax/yeliu/data"
DEFAULT_SHOTS=1
DEFAULT_CLASS_NEGATIVES_PATH="/root/huhuhu2/MF-OOD/negatives_non_photographic.json"

# Hyperparameter search grids
LEARNING_RATES=(0.00005)
BATCH_SIZES=(32)
SEEDS=(1)

# Loss function coefficients grids
LAMBDA_LLM_NEGATIVES=(5)
LAMBDA_MIXUP=(0)
MARGIN_VALUES=(0.025)
LAMBDA_INTRA_CLASS=(0)
INTRA_CLASS_TEMP=(0)

# LoCoOp OOD regularization parameters
LAMBDA_LOCOOP_OOD=(2)
LOCOOP_TOPK=(200)
LAMBDA_LOCOOP_CLS=(0.5)
LAMBDA_LOCOOP_PATCH=(0.5)

# Selector parameters grid
NUM_SELECT_VALUES=(64)

# Advanced parameters grid
SELECTOR_TEMPERATURE=(1.0)
PATCHES_PER_SLOT_ATTN=(4)

# Dimension parameters
MLP_HIDDEN_RATIO=(1)
SLOT_FFN_RATIO=(4.0)
FUSER_FFN_RATIO=(4)
ADAPTER_RATIO=(0.5)
RESIDUAL_COEF=(0.2)

# OOD score parameters
SCORE_TYPE="GL-MCM"
TEMPERATURE=1.0
LAMBDA_LOCAL=0.5

# Warmup parameters
WARMUP_EPOCHS=1
WARMUP_TYPE="constant"
WARMUP_CONS_LR=1e-5
```

---

## 2. Main Results on ImageNet-1K OOD Benchmarks
The following table summarizes results for various methods using the given parameters.

\begin{table*}[t]
\centering
\caption{\textbf{Main Results on ImageNet-1K OOD Benchmarks.} We report FPR95 ($\downarrow$), AUROC ($\uparrow$), and ID Accuracy ($\uparrow$). The results are organized by the number of support shots (1-shot vs. 16-shot). ``+ ViVer'' denotes our full visual-side training method applied to the baseline. ``+ PV'' denotes our training-free Prototype Verification inference strategy. All results are averaged over three random seeds.}
\label{tab:main_results}
\scriptsize
\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{l|cc|cc|cc|cc|cc|c}
\toprule
\multirow{2}{*}{\textbf{Method}} &
\multicolumn{2}{c|}{\textbf{iNaturalist}} &
\multicolumn{2}{c|}{\textbf{SUN}} &
\multicolumn{2}{c|}{\textbf{Places}} &
\multicolumn{2}{c|}{\textbf{Textures}} &
\multicolumn{2}{c|}{\textbf{Average}} &
\multirow{2}{*}{\textbf{ID Acc (\%)}} \\
& FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC &  \\
\midrule

\multicolumn{12}{l}{\textit{\textbf{Reference: Zero-shot Methods}}} \\
MCM~\cite{mcm}      & 31.95 & 94.16 & 37.22 & 92.55 & 42.98 & 90.10 & 58.35 & 85.83 & 42.63 & 90.66 & 67.03 \\
GL-MCM~\cite{glmcm} & \textbf{15.09} & \textbf{96.72} & 29.08 & 93.41 & 37.07 & 90.37 & 58.94 & 83.11 & 35.04 & 90.90 & 67.03 \\
\midrule
\midrule

\multicolumn{12}{c}{\textbf{1-Shot Evaluation} (Extreme Low-Data Regime)} \\
\midrule
\multicolumn{12}{l}{\textit{Training-free Prototype Verification}} \\
GL-MCM \textbf{+ PV (Ours)} & 17.25 & 96.46 & 30.90 & 93.44 & 37.76 & 90.63 & 55.43 & 85.55 & 35.33 & 91.52 & 67.04 \\
\cmidrule(lr){1-12}

\multicolumn{12}{l}{\textit{State-of-the-Art Prompt Learning Methods}} \\
IDLike~\cite{idlike}       & 17.73 & \textbf{96.68} & 48.17 & 89.53 & 50.43 & 88.27 & 29.12 & 93.25 & 36.36 & 91.93 & 68.17 \\
NegPrompt~\cite{negprompt} & 65.03 & 84.56 & 44.39 & 89.63 & 51.31 & 86.55 & 87.60 & 63.76 & 62.08 & 81.13 & 60.14 \\
LSN~\cite{lsn}             & 59.28 & 87.20 & 40.15 & 91.47 & 46.11 & 88.74 & 60.34 & 83.92 & 51.47 & 87.84 & 64.79 \\
\cmidrule(lr){1-12}
CoOp~\cite{coop}           & 23.40 & 95.06 & \textbf{25.84} & 94.10 & 34.42 & 91.50 & 50.66 & 86.15 & 33.58 & 91.70 & 68.98 \\
\hspace{2mm}\textbf{+ ViVer (Ours)} & \textbf{23.22} & 95.07 & \textbf{24.12} & \textbf{94.53} & \textbf{30.48} & \textbf{92.19} & \textbf{46.65} & \textbf{87.60} & \textbf{31.12} & \textbf{92.34} & \textbf{69.58} \\
\cmidrule(lr){1-12}
LoCoOp~\cite{miyai2023locoop} & 30.55 & 93.94 & 31.08 & 93.93 & 39.05 & 90.76 & 46.51 & 89.17 & 36.80 & 91.95 & 67.30 \\
\hspace{2mm}\textbf{+ ViVer (Ours)} & \textbf{27.54} & \textbf{94.38} & \textbf{25.53} & \textbf{94.36} & \textbf{32.10} & \textbf{91.75} & \textbf{46.22} & 88.26 & \textbf{32.85} & \textbf{92.19} & \textbf{69.29} \\
\cmidrule(lr){1-12}
SCT~\cite{sct}             & \textbf{27.43} & \textbf{94.44} & 28.39 & 93.91 & 36.39 & 90.84 & 47.25 & 87.95 & 34.87 & 91.79 & 68.18 \\
\hspace{2mm}\textbf{+ ViVer (Ours)} & 31.04 & 93.87 & \textbf{26.48} & \textbf{94.31} & \textbf{33.53} & \textbf{91.46} & 47.45 & \textbf{88.30} & \textbf{34.62} & \textbf{91.99} & \textbf{69.17} \\
\cmidrule(lr){1-12}
FA~\cite{fa}               & \textbf{19.66} & \textbf{95.44} & 31.07 & 92.58 & 34.53 & 91.23 & 33.03 & 91.75 & 38.35 & 92.75 & 69.10 \\
\hspace{2mm}\textbf{+ ViVer (Ours)} & 24.41 & 94.93 & \textbf{30.61} & \textbf{93.57} & \textbf{32.55} & \textbf{92.40} & \textbf{28.94} & \textbf{93.63} & \textbf{29.13} & \textbf{93.63} & \textbf{69.31} \\
\midrule
\midrule

\multicolumn{12}{c}{\textbf{16-Shot Evaluation} (Standard Few-Shot Regime)} \\
\midrule
\multicolumn{12}{l}{\textit{Training-free Prototype Verification}} \\
GL-MCM \textbf{+ PV (Ours)} & 16.72 & 96.65 & 27.64 & 94.21 & 35.39 & 91.40 & 56.10 & 85.60 & 33.96 & 91.96 & 67.06 \\
\cmidrule(lr){1-12}

\multicolumn{12}{l}{\textit{State-of-the-Art Prompt Learning Methods}} \\
IDLike~\cite{idlike}       & 19.23 & \textbf{96.70} & 54.15 & 87.64 & 56.63 & 85.86 & 34.69 & 91.90 & 41.18 & 90.53 & 69.46 \\
NegPrompt~\cite{negprompt} & 37.79 & 90.49 & 32.11 & 92.25 & 35.52 & 91.16 & 43.93 & 88.38 & 37.34 & 90.57 & 67.88 \\
LSN~\cite{lsn}             & 36.17 & 92.66 & 34.27 & 93.53 & 41.47 & 90.52 & 46.43 & 89.38 & 39.58 & 91.53 & 68.55 \\
\cmidrule(lr){1-12}
CoOp~\cite{coop}           & 17.99 & 95.72 & 30.72 & 92.43 & 37.87 & 90.32 & 45.09 & 88.10 & 32.92 & 91.64 & 70.95 \\
\hspace{2mm}\textbf{+ ViVer (Ours)} & \textbf{15.96} & \textbf{96.74} & \textbf{26.82} & \textbf{94.19} & \textbf{33.01} & \textbf{92.14} & \textbf{34.86} & \textbf{93.12} & \textbf{27.66} & \textbf{94.05} & \textbf{75.05} \\
\cmidrule(lr){1-12}
LoCoOp~\cite{miyai2023locoop} & 18.22 & 96.13 & 24.47 & 94.93 & 32.61 & 91.95 & 38.91 & 91.46 & 28.55 & 93.62 & 71.70 \\
\hspace{2mm}\textbf{+ ViVer (Ours)} & 21.49 & 95.28 & \textbf{19.87} & \textbf{96.15} & \textbf{25.51} & \textbf{93.89} & \textbf{37.78} & \textbf{92.47} & \textbf{26.16} & \textbf{94.45} & \textbf{72.43} \\
\cmidrule(lr){1-12}
SCT~\cite{sct}             & 16.89 & 96.50 & 20.08 & 95.73 & 28.87 & 92.63 & 39.38 & 90.90 & 26.30 & 93.94 & 71.59 \\
\hspace{2mm}\textbf{+ ViVer (Ours)} & 17.63 & 95.96 & \textbf{19.94} & \textbf{96.11} & \textbf{27.18} & \textbf{93.65} & \textbf{32.66} & \textbf{92.99} & \textbf{24.36} & \textbf{94.67} & \textbf{74.91} \\
\cmidrule(lr){1-12}
FA~\cite{fa}               & \textbf{12.84} & \textbf{96.85} & 28.67 & 93.24 & 29.55 & 92.89 & 30.18 & 92.92 & 25.31 & 93.97 & 70.96 \\
\hspace{2mm}\textbf{+ ViVer (Ours)} & 15.17 & 96.46 & 31.16 & \textbf{93.48} & 29.94 & \textbf{93.33} & \textbf{23.74} & \textbf{95.06} & \textbf{25.00} & \textbf{94.58} & \textbf{71.84} \\
\bottomrule
\end{tabular}
\end{table*}
```