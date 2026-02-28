# DysarthriaNSR
Neuro-symbolic self-supervised speech recognition for dysarthric speech with phoneme-level explainability.

## Overview

DysarthriaNSR is an end-to-end ASR system designed specifically for dysarthric (speech-impaired) speech recognition. The system combines:

- **Self-supervised learning (HuBERT)**: Pre-trained acoustic-phonetic representations from 960 hours of unlabeled speech
- **Symbolic constraints**: Articulatory feature-based phoneme similarity matrix for linguistically principled reasoning
- **Multi-task learning**: CTC phoneme alignment + frame-level CE + articulatory auxiliary heads (manner/place/voicing)
- **Clinical focus**: TORGO dataset with dysarthric and control (normal) speech; speaker-independent evaluation
- **Hardware-optimized**: Tuned for RTX 4060 (8GB VRAM) with mixed precision training

For detailed architecture and research context, see [ROADMAP.md](ROADMAP.md).

## Recent Updates (February 2026)

- **Configuration overhaul**: All hyperparameters moved to dataclasses in [src/utils/config.py](src/utils/config.py) with explicit VRAM-aware defaults (RTX 4060 target)
  - Mixed precision: `bf16-mixed` (for Ada architecture) or `16-mixed` 
  - Batch size: 4 with gradient accumulation 8 → effective batch 32
  - Encoder freezing: First 10/12 layers frozen (unfrozen after 3 warmup epochs)
  - Max audio: 6.0s covers ~99% of TORGO samples
- **Dataloader enhancements**: 
  - Inverse-frequency phoneme weights for class balancing
  - Articulatory class labels (manner/place/voicing) for auxiliary heads
  - Weighted random sampling for dysarthric/control balance
- **Training pipeline**: PyTorch Lightning with MLflow logging; evaluation runs automatically at end of training  
- **Evaluation**: Full PER/WER with confusion matrices, bootstrap confidence intervals, beam search decoding

## Quick Start

### Prerequisites
- CUDA 11.8+ (for GPU training)
- Python 3.9+
- ~40GB disk space for TORGO dataset download

### 1) Install Dependencies
```bash
pip install -r requirements.txt
```

### 2) Download Dataset
```bash
python src/data/download.py
# Downloads ~26GB TORGO audio, extracts to data/raw/audio/
# Creates raw_extraction_map.csv for debugging
```

### 3) Generate Neuro-Symbolic Manifest
```bash
python src/data/manifest.py
# Runs g2p_en for phoneme extraction
# Computes articulatory features (manner/place/voicing)
# Outputs: data/processed/torgo_neuro_symbolic_manifest.csv (16.5k samples)
```

### 4) Train Model
```bash
# Single run
python train.py

# With custom run name (saved to checkpoints/{run_name})
python train.py --run-name my_experiment_v1
```

Training outputs are saved to:
- **Checkpoints**: `checkpoints/{run_name}/last.ckpt`
- **Evaluation results**: `results/{run_name}/evaluation_results.json`
- **MLflow logs**: `mlruns/{exp_name}/` (metrics, hyperparams, artifacts)

### 5) Evaluate Model
Evaluation runs automatically at the end of `train.py`, but you can also:
```bash
# Evaluate a checkpoint explicitly (optional)
python evaluate.py  # Self-test only; for custom evaluation, use train.py output
```

## Key Features

### Neural Architecture
- **HuBERT encoder** (758M base): 12-layer transformer with gradient checkpointing
- **Selective layer freezing**: First 10 layers frozen (unfrozen after warmup) for VRAM efficiency
- **Phoneme classifier**: 768→512→vocab projection with dropout regularization
- **Symbolic constraint layer**: Learnable neural-symbolic fusion with severity-adaptive weighting

### Learning Objectives
- **CTC loss** (λ=0.8): Handles variable-length phoneme alignment without forced alignment
- **Frame-level CE** (λ=0.2): Auxiliary phoneme classification for better gradient signal
- **Articulatory CE** (λ=0.1): Manner/place/voicing auxiliary heads for implicit regularization
- **Class weighting**: Inverse-frequency for phoneme imbalance + WeightedRandomSampler for speaker balance

### Symbolic Constraints
- **Articulatory feature matrix**: 44×44 similarity matrix based on manner/place/voicing
- **Dysarthric substitution rules**: Evidence-based confusion probabilities (e.g., devoicing, fronting, gliding)
- **Adaptive beta**: Learnable blending weight (initialized 0.05) with severity-dependent scaling

### Hardware Optimization
- **RTX 4060 target** (8GB VRAM, ~50W TDP)
- **Mixed precision**: BF16 for Ada cards, FP16 for older
- **Batch size 4 + gradient accumulation 8** = effective batch 32
- **Max audio 6.0s**: Covers 99% of TORGO; typical sample ~2–3s

### Evaluation Metrics
- **PER (Phoneme Error Rate)**: Substitutions + deletions + insertions / total phonemes
- **WER (Word Error Rate)**: For fairness comparison (phoneme→word lexicon mapping)
- **Confusion matrices**: Per-speaker, per-phoneme, stratified by dysarthria severity
- **Bootstrap confidence intervals**: 1000 resamples for statistical rigor
- **Beam search decoding**: Width=10, phoneme prefix pruning (vs greedy baseline)

## Project Structure

```
DysarthriaNSR/
├── train.py
├── evaluate.py
├── data/
├── checkpoints/
├── results/
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
│   └── visualization/
└── ROADMAP.md
```

## Architecture (High-Level)

```
Audio Waveform (16 kHz)
  ↓ [Peak normalize + truncate to 6s]
HuBERT Feature Extractor (frozen)
  ↓ [1024-D → 50 Hz frame rate]
HuBERT Encoder (12 layers, first 10 frozen)
  ↓ [768-D contextualized representations]
Phoneme Classifier (768 → 512 → 44)
  ↓ [Logits for 44 ARPABET phonemes]
  ├─ CTC Alignment Head (logits → CTC loss)
  ├─ Frame-level CE (logits → CE loss)
  └─ Articulatory Heads
      ├─ Manner classification (logits → manner CE)
      ├─ Place classification (logits → place CE)
      └─ Voicing classification (logits → voicing CE)
Symbolic Constraint Layer
  ├─ Build 44×44 constraint matrix from:
  │  ├─ Articulatory similarity (manner/place/voicing distance)
  │  └─ Dysarthric substitution rules (hard-coded probabilities)
  └─ Learnable blending: β·P_neural + (1-β)·(C @ P_neural)
      (β: initialized 0.05, clamped <0.8, severity-adaptive)
  ↓ [Final phoneme probabilities]
CTC Beam Search Decoder (width=10)
  ↓ [Phoneme sequence + confidence scores]
Word Lexicon Lookup (CMU + dysarthric variants)
  ↓ [Word sequence (optional, for WER evaluation)]
```

## Data Pipeline

1. **Download** (`src/data/download.py`):
   - Pulls HF dataset `abnerh/TORGO-database` (26GB)
   - Extracts audio to `data/raw/audio/{speaker}_{hash}_{filename}.wav`
   - Outputs `data/processed/raw_extraction_map.csv` for debugging

2. **Manifest** (`src/data/manifest.py`):
   - Matches HF metadata with local audio files
   - Runs g2p_en for phoneme extraction → ARPABET (stress-normalized)
   - Computes articulatory classes (manner/place/voicing) for each phoneme
   - Calculates audio metrics (RMS energy, duration, peak amplitude)
   - Outputs `data/processed/torgo_neuro_symbolic_manifest.csv`

3. **Dataloader** (`src/data/dataloader.py`):
   - Loads audio, peak-normalizes, truncates to max_audio_length
   - Computes HuBERT features at 50 Hz frame rate
   - Builds vocabularies: `<BLANK>`=0, `<PAD>`=1, `<UNK>`=2, phonemes=3+
   - Pads labels with -100 (ignored by CTC/CE loss)
   - Inverse-frequency weights for class balancing
   - WeightedRandomSampler for speaker balance

## Evaluation

After each training run, `train.py` automatically executes `evaluate_model()` which generates:
- **Evaluation results**: `results/{run_name}/evaluation_results.json`
  - PER, WER with bootstrap confidence intervals
  - Per-speaker, per-phoneme breakdowns
  - Confusion matrices (text + visualization)
  - Error distribution (substitutions/deletions/insertions)
- **Confusion matrix PNG**: `results/{run_name}/confusion_matrix.png`
- **Metrics summary**: Printed to stdout

## Configuration and Training

**Single source of truth**: [src/utils/config.py](src/utils/config.py) contains all hyperparameters as dataclasses:

- **ModelConfig**: Architecture (freeze layers, hidden dim, constraint init)
- **TrainingConfig**: Optimization (LR, batch size, precision, loss weights, early stopping)
- **DataConfig**: Dataset limits (max audio length, split ratios, sampling rate)
- **ExperimentConfig**: MLflow tracking and logging behavior
- **SymbolicConfig**: Dysarthric substitution rules and articulatory weights

### Key Defaults (RTX 4060 optimized)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 3e-5 | OneCycleLR with 5% warmup (250 steps) |
| Batch size | 4 | Effective 32 with gradient accumulation |
| Gradient accumulation | 8 | 8GB VRAM constraint |
| Max audio length | 6.0s | Covers ~99% of TORGO utterances |
| Precision | bf16-mixed (Ada) or 16-mixed | Mixed precision for speed & stability |
| Frozen layers | 0–9 (of 12) | Unfrozen after 3 warmup epochs |
| Max epochs | 30 | Early stopping on val/per (patience=8) |
| Loss weights | CTC: 0.8, CE: 0.2, Art: 0.1 | Multi-task balancing |

### Baseline Results (`baseline_v1`)
- **Dataset**: 16,552 samples (13.68 hours) across 15 TORGO speakers
- **Train/Val/Test split**: 10 / 2 / 3 speakers (speaker-stratified, no speaker overlap)
- **Model**: 94.8M params (416K trainable with encoder frozen)
- **Best epoch**: 17/50 (val/per = 0.503)

| Metric | Value | Notes |
|--------|-------|-------|
| **Test PER** | 0.567 ± 0.365 | 3,553 test samples (3 speakers) |
| Dysarthric PER | 0.541 | ~3% better than control |
| Control PER | 0.575 | Expected baseline |
| **Correct** | 8,853 (72.1%) | |
| **Substitutions** | 2,833 (23.1%) | Aligned with dysarthric phonology |
| **Deletions** | 376 (3.1%) | Low deletion rate |
| **Insertions** | 21,290 (high) | **⚠️ Known issue: CTC blank suppression** |
| **Constraint weight (β)** | ~0.50 | Learned balance between neural & symbolic |

### CTC Insertion Problem & Mitigation
The high insertion rate (21,290 insertions) indicates the model emits non-blank frames more than CTC expects. Strategies:
1. **Blank prior regularization**: Increase blank class weight in frame-level CE
2. **Blank posterior targeting**: Add KL penalty to force mean blank probability ~0.3
3. **Length-penalized decoding**: Apply insertion cost during inference
4. **Symbolic constraint boosting**: Increase β on silence frames

## Diagnostics & Next Steps

### Identified Issues

#### 1. CTC Insertion Bias (High Priority)
The baseline exhibits **21,290 insertions** vs. 376 deletions, suggesting blank tokens are under-represented.

**Investigation plan**:
- Analyze blank posterior statistics (histogram, mean per speaker)
- Compare blank probability distributions: dysarthric vs. control
- Check per-frame entropy to detect overconfident non-blank emissions
- Verify `-100` label padding is working correctly

**Mitigation strategies**:
- Increase `blank_priority_weight` in TrainingConfig (currently 1.5)
- Add KL regularizer: force mean(P_blank) → 0.3 target
- Apply length penalties during CTC beam search
- Reduce CE loss weight if it suppresses blank gradients

#### 2. Dysarthric vs. Control Performance Gap
Dysarthric PER (0.541) is slightly better than control (0.575), which is surprising.

**Investigation plan**:
- Stratify by phoneme count (0–5, 6–10, 11–20, 21+ phonemes)
- Check speaker-level variance; may be speaker effects, not dysarthria
- Analyze speaker-specific error rates with paired statistical tests

#### 3. Symbolic Constraint Weight Evolution
β converges ~0.5, suggesting balanced use of constraints.

**Analysis**:
- Monitor β per speaker: does it adapt to severity?
- Compare β=0.3 (neural-heavy) vs. β=0.7 (symbolic-heavy) ablations
- Validate substitution rules match ground-truth confusion frequencies

### Recommended Experiments

1. **Ablation studies**:
   - Neural-only (freeze symbolic constraints, β=1.0)
   - Symbolic-only (β=0.0, rely on constraint matrix)
   - No auxiliary heads (articulatory CE removed)
   - Varying β sweeps (0.3, 0.5, 0.7, 0.9)

2. **Data stratification**:
   - Length-stratified PER analysis
   - Speaker-level confidence intervals
   - Per-phoneme error rates

3. **Insertion mitigation**:
   - Test blank weight scaling (1.5 → 2.0 → 3.0)
   - Blank prior KL regularization
   - Beam search with insertion penalty (Kaldi-style)

4. **Symbolic rule discovery**:
   - Extract top-K confusion pairs from baseline confusion matrix
   - Compare to PHONEME_DETAILS in manifest.py
   - Validate clinical phonology literature alignment

- Export HuBERT + classifier + symbolic layer to ONNX with dynamic axes.
- Validate parity on a small batch, then run ONNX Runtime CPU inference.

## Dataset: TORGO

Speakers:

- Dysarthric: F01, F03, F04, M01-M05
- Control: FC01-03, MC01-04 (prefixed with C)

Features:

- Multi-session recordings with array and head microphones.
- Phoneme-level alignments in ARPABET.
- Articulatory metadata (manner, place, voicing).
- RMS energy for signal quality assessment.

## Research Context

- Explainability: phoneme-level outputs and rule tracking for clinical analysis.
- Clinical relevance: articulatory features tied to dysarthric error patterns.
- Small-dataset awareness: speaker-independent splits, transfer learning.
- Modularity: neural and symbolic components can be ablated independently.

## What Is Implemented

- TORGO download and neuro-symbolic manifest generation with articulatory metadata.
- HuBERT-based dataset and collator for CTC-ready batching.
- NeuroSymbolicASR model with adaptive symbolic constraints.
- PyTorch Lightning training pipeline with MLflow tracking.
- Evaluation with PER, WER, confusion matrices, and per-speaker analysis.

## Known Issues and Next Steps

- High insertion rate (21,290 insertions vs. 376 deletions) suggests blank over-suppression.
- Dysarthric PER lower than control PER suggests length or speaker effects; length-stratified analysis is recommended.

Planned work:

- Analyze blank probabilities and add insertion regularization.
- Add ablation scripts (neural-only vs. symbolic-only).
- Expand speaker-level stratified metrics and statistical testing.
- Prepare ONNX export for deployment.

## References

- Cui et al. (2019) "Class-Balanced Loss Based on Effective Number of Samples"