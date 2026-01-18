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
python manifest.py --data-dir ./data --out ./data/torgo_neuro_symbolic_manifest.csv
```

### 3. Train Model
```bash
python train.py --config config.yaml
```

### 4. Evaluate
```python
from evaluate import compute_per, compute_wer
# Load checkpoints/best_model.ckpt and evaluate on test set
```

## Key Features

- ✅ **HuBERT Encoder**: 768-dim self-supervised representations with selective layer freezing
- ✅ **Symbolic Constraint Layer**: Articulatory feature-based phoneme similarity matrix
- ✅ **Multi-Task Learning**: CTC loss (phoneme) + Focal loss (dysarthria classification) + KL constraint loss
- ✅ **VRAM Optimized**: Gradient accumulation (batch_size=2, accumulation_steps=12) for RTX 4060 8GB
- ✅ **MLflow Integration**: Experiment tracking with safe parameter flattening
- ✅ **Evaluation Metrics**: PER, WER, phoneme confusion matrices, per-speaker error analysis

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
├── .github/
│   └── copilot-instructions.md  # AI agent guide
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

**Key Hyperparameters** (see [config.py](config.py)):
- Learning rate: 5e-5 (cosine annealing + 500-step warmup)
- Batch size: 2 (effective: 24 with gradient accumulation)
- Max epochs: 2 (experimentally configured)
- Dropout: 0.1 (classifier), 0.05 (layer)
- Label smoothing: 0.1

**Loss Weighting**:
- CTC loss (phoneme): 1.0
- Focal loss (dysarthria): 0.1
- KL constraint loss: 0.05

**Callbacks**:
- EarlyStopping (patience=5, metric=val_per)
- ModelCheckpoint (save best)
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

## Development

For detailed implementation patterns, cross-component communication, and troubleshooting, see [`.github/copilot-instructions.md`](.github/copilot-instructions.md).

## Research Context

This project prioritizes:
- **Explainability**: Phoneme-level outputs over word-level black boxes
- **Clinical relevance**: Features tied to dysarthria characteristics (articulation, timing)
- **Small-dataset awareness**: Speaker-independent splits, transfer learning
- **Modularity**: Neural and symbolic components separately ablatable

## What's Implemented

- ✅ Data pipeline (TORGO download + neuro-symbolic manifest)
- ✅ Neural dataset & dataloader (HuBERT feature extraction, CTC batching)
- ✅ NeuroSymbolicASR model (HuBERT + phoneme classifier + symbolic constraints)
- ✅ Training infrastructure (PyTorch Lightning, multi-task learning, MLflow)
- ✅ Evaluation (PER, WER, confusion matrices, per-speaker analysis)

## Future Work

- Explainability dashboard (phoneme confusion heatmaps, rule activations)
- Pretrained model checkpoints (dysarthric/control baselines)
- Symbolic rule discovery (auto-extract substitution patterns)
- Clinical interface (ONNX export, streaming inference)
- Ablation studies (neural vs. symbolic component contribution)

## License

See LICENSE file.
