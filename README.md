# DysarthriaNSR — Neuro-Symbolic ASR for Dysarthric Speech

> **Target venue:** SPCOM 2026  
> **Current status:** Implementation complete; LOSO-CV pending for publication-valid results

---

## Project Description

DysarthriaNSR is a **neuro-symbolic automatic speech recognition** system designed for people with **dysarthria** — a motor speech disorder causing reduced intelligibility. The system combines a pretrained HuBERT speech encoder with a symbolic phoneme-constraint layer grounded in articulatory phonetics, enabling clinically interpretable recognition that adapts to individual speaker severity.

The key technical contribution is the **SymbolicConstraintLayer**: a differentiable, learnable confusion-probability matrix initialized from articulatory feature distances, which blends neural phoneme posteriors with phonologically-motivated priors in a severity-adaptive manner. This bridges data-driven deep learning with explicit linguistic knowledge — a core requirement for clinical deployment.

---

## Research Motivation & Key Contributions

**Why dysarthria?** Dysarthric speakers are dramatically underserved by commercial ASR systems. TORGO corpus speakers show 2–98% intelligibility, with severe speakers effectively incomprehensible to standard ASR. Clinical speech-language pathology requires not only accurate recognition but *explainable* error patterns.

**Key contributions:**
1. **Learnable Constraint Matrix (P2):** End-to-end trainable phoneme confusion matrix `C`, initialized from articulatory priors and anchored by a KL-divergence loss during training.
2. **Cross-Attention Severity Adapter (P3):** Projects continuous speaker severity `[0, 5]` to a `[B, 1, 768]` context vector; cross-attends with HuBERT hidden states for spatially-aware severity conditioning.
3. **Ordinal Contrastive Loss (P1):** Pairwise contrastive loss with margins proportional to severity distance, learning graded dysarthria representations.
4. **Articulatory auxiliary heads:** Manner, place, and voice classification heads supervised at the utterance level via global average pooling — providing articulatory interpretability with correct CTC semantics.
5. **Blank-Prior KL regularisation:** Controls CTC insertion bias by regularising the mean blank emission rate toward an empirically calibrated target (0.75).

---

## System Overview

```
Raw Audio [B, T_audio]
    ↓
HuBERT Encoder (facebook/hubert-base-ls960, frozen CNN + progressive transformer unfreezing)
    → [B, T', 768]  where T' ≈ T_audio / 320
    ↓
SpecAugmentLayer (time/freq masking on hidden states, training only)
    ↓
SeverityAdapter (cross-attention conditioning on continuous severity [0, 5])
    → [B, T', 768]
    ↓
TemporalDownsampler (stride-2 Conv1d, ~50 Hz → ~25 Hz)
    → [B, T'//2, 768]
    ↓
PhonemeClassifier (768 → LayerNorm → GELU → Dropout → 512 → |V|)
    → logits_neural [B, T, |V|]
    ↓
SymbolicConstraintLayer
    → P_final = β·(P_neural @ C) + (1-β)·P_neural
    → log_probs_constrained [B, T, |V|]
    ↓
CTC Decoding → Phoneme sequence → PER
```

The vocabulary `|V|` is 47 tokens: 44 stress-agnostic ARPABET phonemes + `<BLANK>`, `<PAD>`, `<UNK>`.

---

## Quick Start

### Installation

```bash
git clone <repo-url>
cd DysarthriaNSR
pip install -r requirements.txt
```

**Core dependencies (pinned):**
```
torch==2.9.0 / torchaudio==2.9.0
transformers==4.57.1
pytorch-lightning==2.6.0
mlflow==3.6.0
g2p-en==2.1.0
```

### Dataset Setup

Download the TORGO corpus and generate the manifest:

```bash
# 1. Download audio files (see docs/dataset.md for access instructions)
python src/data/download.py

# 2. Generate manifest (phoneme annotations, speaker metadata)
python src/data/manifest.py
# Output: data/processed/torgo_neuro_symbolic_manifest.csv
```

### Training

```bash
# Full training + evaluation
python run_pipeline.py --run-name experiment_v1

# Train only (skip evaluation)
python run_pipeline.py --run-name experiment_v1 --skip-eval

# Smoke test (1 epoch, 5 batches — CI/sanity check)
python run_pipeline.py --run-name smoke --smoke
```

### Evaluation

```bash
# Evaluate a saved checkpoint
python run_pipeline.py --run-name experiment_v1 --skip-train

# Beam search decoding + explainability
python run_pipeline.py --run-name experiment_v1 --skip-train \
    --beam-search --beam-width 10 --explain

# Ablation: neural-only baseline
python run_pipeline.py --run-name neural_only_v1 --ablation neural_only
```

### Reproduce Main Results

```bash
# Standard single-split training
python run_pipeline.py --run-name baseline_v6

# Full LOSO cross-validation (publication-required; ~32h on RTX 4060)
python run_pipeline.py --run-name loso_v1 --loso

# Resume LOSO from last completed fold
python run_pipeline.py --run-name loso_v1 --loso --resume-loso
```

---

## Repository Structure

```
DysarthriaNSR/
├── run_pipeline.py          # Entry point: train + eval orchestrator
├── train.py                 # DysarthriaASRLightning, run_loso()
├── evaluate.py              # evaluate_model(), BeamSearchDecoder, PER
├── requirements.txt
├── README.md
├── ROADMAP.md
│
├── src/
│   ├── models/
│   │   ├── model.py         # NeuroSymbolicASR, all architectural components
│   │   ├── losses.py        # OrdinalContrastiveLoss, BlankPriorKLLoss, SymbolicKLLoss
│   │   └── uncertainty.py   # UncertaintyAwareDecoder (MC-Dropout)
│   ├── data/
│   │   ├── dataloader.py    # TorgoNeuroSymbolicDataset, NeuroSymbolicCollator
│   │   ├── manifest.py      # TORGO manifest generation (G2P → ARPABET)
│   │   ├── download.py      # Audio download script
│   │   └── warm_feature_cache.py
│   ├── utils/
│   │   ├── config.py        # All hyperparameters (single source of truth)
│   │   └── sequence_utils.py
│   ├── explainability/
│   │   ├── attribution.py   # PhonemeAttributor
│   │   ├── rule_tracker.py  # SymbolicRuleTracker
│   │   └── output_format.py # ExplainableOutputFormatter → explanations.json
│   └── visualization/
│       └── experiment_plots.py
│
├── scripts/
│   ├── smoke_test.py        # 7 automated sanity tests
│   ├── generate_figures.py  # Publication-quality figure CLI
│   └── tree.md, tree.sh, organize.py, cleanup.py
│
├── docs/
│   ├── architecture.md      # Model architecture (this suite)
│   ├── pipeline.md          # End-to-end pipeline
│   ├── experiments.md       # Results and ablations
│   ├── reproducibility.md   # Step-by-step reproduction guide
│   └── dataset.md           # TORGO corpus guide
│
├── data/
│   ├── raw/audio/           # Downloaded TORGO .wav files
│   └── processed/           # torgo_neuro_symbolic_manifest.csv, feature_cache/
│
├── checkpoints/             # Saved model checkpoints per run_name
├── results/                 # Evaluation artifacts per run_name
└── mlruns/                  # MLflow experiment tracking
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{dysarthriaNSR2026,
  title     = {Neuro-Symbolic Phoneme Recognition for Dysarthric Speech with
               Articulatory Constraints and Severity-Adaptive Fusion},
  author    = {},
  booktitle = {Proceedings of SPCOM 2026},
  year      = {2026},
}
```

*(Full citation to be updated upon acceptance.)*

---

## License

[MIT License](LICENSE)