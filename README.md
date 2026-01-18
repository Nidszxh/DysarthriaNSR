# DysarthriaNSR
**Neuro-symbolic self-supervised speech recognition for dysarthric speech with phoneme-level explainability.**

## Overview

DysarthriaNSR combines **neural** and **symbolic** approaches to build an explainable automatic speech recognition (ASR) system for dysarthric (impaired) speech. The system leverages:

- **Self-supervised learning** (HuBERT) for robust acoustic-phonetic representations
- **Symbolic constraints** (articulatory features) for phoneme-level reasoning and clinical interpretability
- **TORGO dataset** with dysarthric and control (normal) speech samples
- **Multi-task learning** for joint phoneme recognition and dysarthria classification

## Quick Start

### 1. Download Dataset
```bash
python download.py  # Downloads TORGO to ./data/
```

### 2. Generate Manifest
```bash
python src/data/manifest.py --data-dir ./data --out ./data/processed/torgo_neuro_symbolic_manifest.csv
```

### 3. Train Model
```bash
# Run with default config
python train.py

# Or with custom run name
python train.py --run-name experiment_v1
```

### 4. Results
After training completes, results are automatically saved to:
- **Checkpoints**: `./checkpoints/{run_name}/` — best 3 models + last checkpoint
- **Evaluation**: `./results/{run_name}/` — visualizations, metrics, confusion matrices
- **MLflow logs**: `./mlruns/` — hyperparameters and training curves

## Key Features

- **HuBERT Encoder**: 768-dim self-supervised representations with selective layer freezing + gradient checkpointing
- **Symbolic Constraint Layer**: Articulatory feature-based phoneme similarity matrix (initial β=0.5) with adaptive dysarthria weighting
- **Multi-Task Learning**: CTC loss (phoneme alignment) + CE loss (frame-level auxiliary) with inverse-frequency class weights
- **Class Balancing**: Inverse-frequency CE weights + weighted sampler for dysarthric/control balance
- **VRAM Optimized**: Audio truncation (8s max), batch_size=2, gradient_accumulation=8 (effective=16), fp16 mixed precision
- **MLflow Integration**: Experiment tracking with safe parameter flattening
- **Evaluation Metrics**: PER, WER, phoneme confusion matrices, per-speaker analysis, rule activation counts

## Current Performance (baseline_v1)

**Dataset**: 16,552 samples (13.68 hours) across 15 TORGO speakers  
**Model**: 94.8M params (416K trainable, encoder frozen initially)  
**Training**: 10 speakers train, 2 val, 3 test | Best epoch: 17/50

**Test Set Results**:
- **Overall PER**: 0.567 ± 0.365
- **Dysarthric PER**: 0.541 (n=810 samples)
- **Control PER**: 0.575 (n=2,743 samples)

**Error Breakdown**:
- Correct: 8,853
- Substitutions: 2,833
- Deletions: 376
- Insertions: 21,290 ⚠️ (high - see notes below)

**Key Observations**:
- High insertion rate indicates model over-predicting phonemes vs. CTC blanks
- Dysarthric speakers show slightly better PER than controls (counter-intuitive; may indicate dataset imbalance)
- Symbolic constraint weight (β) converged to 0.5, suggesting balanced neural-symbolic fusion

## Project Structure

```
DysarthriaNSR/
├── download.py              # TORGO dataset download
├── manifest.py              # Neuro-symbolic manifest generation
├── dataloader.py            # PyTorch dataset & collator (CTC-ready)
├── model.py                 # NeuroSymbolicASR architecture
├── config.py                # Configuration (ModelConfig, TrainingConfig, SymbolicConfig)
├── train.py                 # PyTorch Lightning training pipeline
├── evaluate.py              # Evaluation metrics & visualization
├── data/
│   ├── torgo_neuro_symbolic_manifest.csv  # Generated manifest
│   └── abnerh___torgo-database/           # HuggingFace cache
├── checkpoints/             # Model checkpoints (best model from early stopping)
├── results/                 # Evaluation outputs & visualizations
└── ROADMAP.md               # Detailed low-level architecture
```

## Architecture

```
Audio Input (Dysarthric/Control)
    ↓
HuBERT Encoder (768-dim SSL representations)
    ↓
Projection Layer (768 → 512-dim)
    ↓
[Phoneme Classifier] ──→ Logits (num_phonemes)
    ↓
Symbolic Constraint Aggregation (articulatory similarity)
    ↓
Constrained Logits (dysarthria-aware weighting)
    ↓
[CTC Decoder] ──→ Phoneme Sequence
    ↓
[Dysarthria Classifier] ──→ Binary Label
```

## Configuration & Training

**Key Hyperparameters** (see [src/utils/config.py](src/utils/config.py)):
- Learning rate: 5e-5 (OneCycleLR with 10% warmup)
- Batch size: 2 (effective: 16 with gradient accumulation=8)
- Max epochs: 50 (early stopping patience=10)
- Max audio length: 8 seconds (truncated for memory)
- Dropout: 0.1 (classifier), 0.05 (layer)
- Label smoothing: 0.1
- Precision: fp16-mixed for GPU memory efficiency
- Symbolic constraint init: β=0.5 (learnable, adaptive by speaker severity)

**Memory Optimizations**:
- **Gradient Checkpointing**: Enabled on HuBERT encoder (trades compute for memory)
- **Audio Truncation**: Long utterances capped at 8 seconds
- **Frozen Encoder Layers**: First 8 of 12 HuBERT layers frozen (reduces parameters)
- **Fragmentation Mitigation**: `PYTORCH_ALLOC_CONF=expandable_segments:True`

**Loss Weighting**:
- CTC loss (phoneme alignment): 0.7
- CE loss (frame-level auxiliary): 0.3

**Callbacks**:
- EarlyStopping (patience=10, metric=val/per)
- ModelCheckpoint (save top 3 models)
- LearningRateMonitor

## Dataset: TORGO

**Speakers**:
- Dysarthric: F01, F03, F04, M01-M05 (8 total)
- Control: FC01-03, MC01-04 (7 total, prefixed with 'C')

**Features**:
- ~16K+ samples across multiple recording sessions
- Multiple microphone types (arrayMic, headMic)
- Phoneme-level alignments (ARPAbet)
- Articulatory metadata (manner, place, voicing)
- RMS energy for signal quality assessment

## Research Context

This project prioritizes:
- **Explainability**: Phoneme-level outputs over word-level black boxes
- **Clinical relevance**: Features tied to dysarthria characteristics (articulation, timing)
- **Small-dataset awareness**: Speaker-independent splits, transfer learning
- **Modularity**: Neural and symbolic components separately ablatable

## What's Implemented

- Data pipeline (TORGO download + neuro-symbolic manifest with articulatory features)
- Neural dataset & dataloader (HuBERT feature extraction, CTC-compatible batching, audio truncation, inverse-frequency weights)
- NeuroSymbolicASR model (HuBERT encoder + phoneme classifier + symbolic constraint layer with adaptive β)
- Training infrastructure (PyTorch Lightning, multi-task CTC+CE loss, weighted sampler, MLflow logging)
- Evaluation pipeline (PER, WER, confusion matrices, per-speaker analysis, rule hit-rates)
- Result persistence (checkpoints to disk, evaluation artifacts with visualizations)
- **Baseline model**: PER 0.567 on TORGO test set (3 speakers, 3,553 samples)

## Known Issues & Future Work

**Current Limitations**:
- High insertion rate (21,290 insertions vs. 376 deletions) suggests model is over-generating phonemes
  - **Potential fixes**: Increase CTC blank penalty, adjust CE loss `<BLANK>` weight further, or add insertion-specific regularization
- Counter-intuitive dysarthric vs. control PER (0.541 vs. 0.575) may indicate:
  - Dysarthric samples are shorter/simpler utterances on average
  - Dataset imbalance or speaker-specific effects
  - Need phoneme-length stratified analysis

**Roadmap**:
- [ ] Investigate insertion bias: analyze CTC blank probabilities, add explicit blank encouragement
- [ ] Phoneme-length distribution logging and stratified PER analysis
- [ ] Ablation studies: neural-only vs. symbolic-only vs. neuro-symbolic fusion
- [ ] Extend to word-level decoding with language model integration
- [ ] ONNX export for clinical deployment