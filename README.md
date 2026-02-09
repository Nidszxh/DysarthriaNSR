# DysarthriaNSR
Neuro-symbolic self-supervised speech recognition for dysarthric speech with phoneme-level explainability.

## Overview

DysarthriaNSR combines neural and symbolic approaches to build an explainable ASR system for dysarthric (impaired) speech. The system leverages:

- Self-supervised learning (HuBERT) for robust acoustic-phonetic representations.
- Symbolic constraints (articulatory features) for phoneme-level reasoning and clinical interpretability.
- TORGO dataset with dysarthric and control (normal) speech samples.
- Multi-task learning for phoneme alignment plus articulatory auxiliary heads.

For a detailed architecture walkthrough, see [ROADMAP.md](ROADMAP.md).

## Recent Updates (Feb 2026)

- Training defaults moved into dataclasses in [src/utils/config.py](src/utils/config.py) (VRAM-aware batch/precision settings, warmup, and loss weights).
- Dataloader now computes inverse-frequency phoneme weights and adds articulatory class labels (manner/place/voice).
- Training pipeline logs MLflow metrics and runs full evaluation at the end of training.

## Quick Start

### 1) Download Dataset
```bash
python download.py
```

### 2) Generate Manifest
```bash
python src/data/manifest.py
```

### 3) Train Model
```bash
python train.py
python train.py --run-name experiment_v1
```

### 4) Evaluate
```bash
python evaluate.py
```

Results are saved to:

- Checkpoints: [checkpoints](checkpoints)
- Evaluation outputs: [results](results)
- MLflow logs: [mlruns](mlruns)

## Key Features

- HuBERT encoder with selective layer freezing and gradient checkpointing.
- Symbolic constraint layer using articulatory similarity with adaptive beta weighting.
- Multi-task learning (CTC phoneme alignment + frame-level CE + articulatory CE heads).
- Class balancing via inverse-frequency weights and weighted sampling.
- VRAM-aware defaults: 8s truncation, batch size 4, gradient accumulation 16, mixed precision.
- Evaluation metrics: PER, WER, confusion matrices, per-speaker analysis, and rule activation counts.

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
Audio Input
  -> HuBERT Encoder (SSL representations)
  -> Projection Layer
  -> Phoneme Classifier
  -> Symbolic Constraint Aggregation
  -> CTC Decoder -> Phoneme Sequence
```

## Configuration and Training

Key hyperparameters live in [src/utils/config.py](src/utils/config.py). Defaults are chosen for 8GB VRAM with mixed precision:

- Learning rate: 3e-5 (OneCycleLR with warmup ratio 0.05)
- Batch size: 4 (effective 64 with gradient accumulation 16)
- Max epochs: 50 (early stopping on val/per)
- Max audio length: 8 seconds
- Loss weights: CTC 0.8, CE 0.2, articulatory 0.1
- Symbolic constraint init: beta=0.05 (learnable, severity-adaptive)
- Frozen layers: feature extractor + encoder layers 0-5; unfreeze after 5 warmup epochs

## Evaluation

Evaluation provides PER/WER, confusion matrices, and stratified analysis. The full evaluation run happens at the end of `train.py` via `evaluate_model()`; [evaluate.py](evaluate.py) is a utility module (its `main()` is a small self-test).

## Baseline Performance (baseline_v1)
- Dataset: 16,552 samples (13.68 hours) across 15 TORGO speakers.
- Training split: 10 speakers train, 2 val, 3 test. Best epoch: 17/50.
- Test PER: 0.567 ± 0.365 on 3,553 samples (3 TORGO speakers)
- Dysarthric PER: 0.541
- Control PER: 0.575
- Model: 94.8M params (416K trainable with frozen encoder layers)

Error Breakdown:

- Correct: 8,853
- Substitutions: 2,833
- Deletions: 376
- Insertions: 21,290

Key Observations:

- High insertion rate indicates over-prediction vs. CTC blanks.
- Dysarthric PER slightly lower than control PER suggests length or speaker effects.
- Constraint weight converges near 0.5, suggesting balanced neural-symbolic fusion.

## Diagnostics and Next-Step Plan

Insertion diagnosis:

- Inspect blank posterior statistics (mean, histogram, blank vs non-blank ratio).
- Compare blank probabilities between dysarthric vs control samples.
- Check per-frame entropy to detect overconfident non-blank emissions.

Insertion reduction strategies (minimal params):

- Increase blank weight in frame-level CE or reduce non-blank weights.
- Add a blank prior regularizer with a target blank rate.
- Apply length-penalized decoding or insertion penalties during inference.
- Lower CE weight if it suppresses CTC blanks.

Ablations:

- Neural-only (beta = 1.0, symbolic disabled).
- Symbolic-only (beta = 0.0, neural logits frozen).
- Fixed beta sweeps (0.3, 0.5, 0.7).

Stratified evaluation:

- Length buckets: 0-5, 6-10, 11-20, 21+ phonemes.
- Speaker-level PER with statistical testing (paired tests + correction).

Deployment outline:

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