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

## Recent Updates (March 2026)

- **`baseline_v5` trained and evaluated** (March 6, 2026): 30 epochs, 66.4M trainable params. Best val/per = **0.505** (epoch 28). Beam-search test PER = **0.4750**, WER = 0.665. I/D = 0.9×. Identical to v4 — confirms symbolic constraint layer is the bottleneck (`per_neural=0.305` vs `per_constrained=0.475`, Δ=−0.170). LOSO-CV not yet run.
- **HuBERT model revision pinned** (R-01, March 2026): `facebook/hubert-base-ls960` is now pinned to commit `dba3bb02fda4248b6e082697eee756de8fe8aa8a` via `ModelConfig.hubert_model_revision`. Set this field to `None` to always pull the latest Hub weights. Prevents silent result drift between runs if the Hub model card is updated.
- **B13–B23 fixed** (March 6, 2026 — current session): see table below. Key fixes:
  - **B13** `SpecAugmentLayer`: was applying the same mask to every sample in the batch; now applies independent per-sample masks.
  - **B14** Stage-2 unfreezing: was calling `unfreeze_after_warmup()` (all layers 4–11); now correctly calls `unfreeze_encoder(layers=[6,7,8,9,10,11])`.
  - **B15** `val/per` metric: was a batch-mean PER (over-weighted high-utterance speakers); now computed as proper macro-speaker PER.
  - **B16** CTC attention-mask stride: was hardcoded to 320 even when `TemporalDownsampler` is active (effective stride = 640); fixed.
  - **B17** `validation_step` / `test_step`: were passing `attention_mask=None` to `compute_loss`; new `_downsample_attn_mask()` helper now provides the correct downsampled mask.
  - **B18** Vocabulary not persisted in checkpoints: `on_save_checkpoint` / `on_load_checkpoint` now save vocab and warn on mismatch.
  - **B19** Beam search length-norm bug: divisor was applied to LM bonus too; fixed to normalise acoustic score only.
  - **B20** `decode_predictions`: `beam_length_norm_alpha` lookup raised `AttributeError` (config not in scope); replaced with hardcoded 0.6.
  - **B21** `LearnableConstraintMatrix` init: `log(C_static)` through softmax produced a flat distribution; fixed with temperature-sharpened init (`log_init = log(C) / 0.5`).
  - **B22** `lambda_symbolic_kl` raised from 0.05 → **0.5** (was too weak; effective per-row weight ≈ 0.001).
  - **B23** Articulatory place labels corrected: SH/ZH/CH/JH `palatal` → `postalveolar`; R `palatal` → `alveolar`; W `bilabial` → `labio-velar`.
- **New evaluation metrics** (March 6, 2026):
  - `constraint_precision`: per-utterance helpful/neutral/harmful rate for the symbolic layer.
  - `by_severity`: PER bucketed by clinical severity (mild / moderate / severe).
  - MLflow now logs `val/constraint_row_entropy` and `val/constraint_kl_from_prior` per epoch.
- **New ablation modes**: `--ablation no_constraint_matrix`, `no_spec_augment`, `no_temporal_ds` added to `run_pipeline.py` and `train.py`.
- **Speaker-balanced sampler**: `WeightedRandomSampler` now weights by *speaker* frequency (not dysarthric/control class), so high-utterance speakers no longer dominate gradient signal.
- **`baseline_v4` trained and evaluated** (March 5, 2026): batch=8, 40 epochs, staged progressive unfreezing, 66.4M trainable params. Best val/per = **0.504** (epoch 28). Beam-search test PER = **0.4748**, WER = 0.664. First full end-to-end valid evaluation.

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

### 4) Train and Evaluate
```bash
# Full pipeline: train 30 epochs + greedy evaluation
python run_pipeline.py --run-name my_experiment_v1

# Skip training; run beam-search evaluation with explainability + uncertainty
python run_pipeline.py --run-name my_experiment_v1 --skip-train \
    --explain --uncertainty --beam-search --beam-width 25

# Smoke test (5 batches, confirms pipeline works)
python run_pipeline.py --run-name smoke_check --smoke-test
```

Training outputs are saved to:
- **Checkpoints**: `checkpoints/{run_name}/epoch=*-val_per=*.ckpt` + `last.ckpt`
- **Evaluation results**: `results/{run_name}/evaluation_results.json`
- **Explanations**: `results/{run_name}/explanations.json` (with `--explain`)
- **MLflow logs**: `mlruns/` (metrics, hyperparams, artifacts)

### 5) Generate Publication Figures
```bash
python scripts/generate_figures.py --run-name my_experiment_v1
# → results/my_experiment_v1/figures/ (6 diagnostic plots)

# Compare multiple ablation runs
python scripts/generate_figures.py --run-name baseline_v2 \
    --compare neural_only no_art_heads no_constraint_matrix
```

### 6) Run Unit Smoke Tests
```bash
python scripts/smoke_test.py  # All 7 tests should pass
```

## Key Features

### Neural Architecture
- **HuBERT encoder** (base): 12-layer transformer with gradient checkpointing
- **Staged progressive unfreezing**: entire encoder frozen at epoch 0; layers 8-11 unfrozen at epoch 1; layers 6-11 at epoch 6 (layers 0-3 remain frozen) for VRAM efficiency
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
├── run_pipeline.py          ← end-to-end orchestrator (entry point)
├── train.py                 ← training-only; called by run_pipeline
├── evaluate.py              ← evaluation-only; called by run_pipeline
├── data/
├── checkpoints/
├── results/
│   └── {run_name}/
│       ├── evaluation_results.json
│       ├── explanations.json
│       ├── confusion_matrix.png
│       └── figures/         ← from generate_figures.py
├── scripts/
│   ├── generate_figures.py  ← publication-quality diagnostic plots
│   └── smoke_test.py        ← 7 automated end-to-end tests
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
│   ├── explainability/
│   └── visualization/
├── RESEARCH_BRIEF.md        ← compressed project context (current state)
└── ROADMAP.md
```

> **Orchestrator**: `run_pipeline.py` is the recommended entry point. `train.py` and `evaluate.py` remain callable directly but evaluation is no longer invoked inside `train()`.

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

After each `run_pipeline.py` run, `evaluate_model()` generates:
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

### Key Defaults (RTX 4060 optimized — baseline_v4 configuration)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 3e-5 | OneCycleLR with 5% warmup |
| Batch size | 8 | Effective 64 with gradient accumulation |
| Max audio length | 6.0s | Covers ~99% of TORGO utterances |
| Precision | bf16-mixed | Mixed precision for speed & stability |
| Frozen layers | 0–11 → 4–11 (ep1) → 0–5 (ep6) | Staged progressive unfreezing |
| Max epochs | 40 | Early stopping on val/per (patience=8) |
| Loss weights | CTC: 0.8, CE: 0.2, Art: 0.1 | Multi-task balancing |

### Baseline Results

#### `baseline_v5` (March 6, 2026) — **Current Reference**
- **Dataset**: 16,531 samples, 15 TORGO speakers (manifest with correct speaker IDs)
- **Train/Val/Test split**: speaker-stratified; Train 11,654 / Val 1,329 / Test 3,548 (10/2/3 speakers)
- **Model**: 98.9M params total / 66.4M trainable — 30 epochs, same staged unfreezing as v4
- **Best checkpoint**: epoch 28 (val/per = 0.505)
- **Evaluation**: beam-search (width=25) + `--explain` + `--uncertainty` (20 MC samples)

| Metric | Value | Notes |
|--------|-------|-------|
| **Beam-search test PER** (macro-speaker) | 0.4750 | 3,548 test samples, 3 speakers |
| **WER** | 0.6646 | |
| 95% CI | [0.448, 0.503] | Bootstrap |
| Dysarthric PER | ~0.45 | M03 |
| Control PER (avg) | ~0.49 | MC02, MC04 |
| Substitutions / Deletions / Insertions | 13,821 / 4,338 / 3,752 | **I/D = 0.9×** |
| Articulatory accuracy | manner 78.3%, place 79.3%, voice 92.3% | |
| per_neural / per_constrained | 0.305 / 0.475 | Δ=−0.170 — constraint layer hurts |
| Wilcoxon p (Holm) | 0.0027 (0.0050) | Significant; n=3 speakers only |

> ⚠️ **Critical finding — identical to v4**: `per_neural = 0.305` vs `per_constrained = 0.475` (Δ = −0.170). The symbolic constraint layer consistently degrades PER by ~57% relative across both v4 and v5. Ablation and LOSO-CV are the immediate next steps.

#### `baseline_v4` (March 5, 2026) — Previous Reference
- **Model**: 98.9M params / 66.4M trainable (67.1%), 40 epochs, staged progressive unfreezing
- **Best val/per**: 0.504 (epoch 28); beam PER 0.4748, WER 0.664

#### `baseline_v3` (March 4, 2026) — Previous Valid Run
- **Model**: 98.9M params / 23.9M trainable (24.2%), 30 epochs
- **Best val/per**: 0.574 (epoch 26); no separate test evaluation run

#### `baseline_v2` (March 2026) — Historical Reference (⚠️ Invalid Speaker Split)
- **⚠️ Data leakage**: manifest had `speaker='unknown'` for all samples (B12 unresolved at train time); speaker-stratified split was ineffective

| Metric | Value | Notes |
|--------|-------|-------|
| Greedy test PER | 0.215 | Inflated; split was not speaker-independent |
| Beam-search PER (width=25) | 0.243 | |

#### `baseline_v1` (January 2026) — Superseded
| Metric | Value | Notes |
|--------|-------|-------|
| Test PER | 0.567 ± 0.365 | B3 attention mask stride bug active |
| Insertions | 21,290 | BlankPriorKL receiving incorrect mask |

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

**Active issues (March 2026)**:
- **Symbolic constraint layer degrades performance** (CRITICAL): neural-only PER = 0.305 vs constrained PER = 0.475 (Δ = −0.170) — consistent across v4 and v5. Root-cause fixes (B21 temperature init, B22 `lambda_symbolic_kl`→0.5) have been applied but need a re-run to confirm.
- **LOSO-CV not yet run**: test split = 3 speakers only → severity correlation p=0.347 (n.s.); statistically valid macro-PER requires full 15-fold LOSO (~15–22h)
- `SymbolicRuleTracker` still reports low confidence (avg 0.131); mostly X→`<BLANK>` activations — should improve after B21/B22 re-run

**Immediate next steps**:
1. Train **baseline_v6** with all B13–B23 fixes applied: `python run_pipeline.py --run-name baseline_v6`
2. Verify `constraint_precision.helpful_rate > harmful_rate` in `evaluation_results.json`
3. Run neural-only ablation: `python run_pipeline.py --run-name ablation_neural_only --ablation neural_only`
4. Run LOSO-CV after constraint fix confirmed: `python run_pipeline.py --run-name loso_v1 --loso`
5. Regenerate figures: `python scripts/generate_figures.py --run-name baseline_v6`

## Reproducibility

All experiments use a fixed global seed and deterministic settings to minimise run-to-run variance.

| Factor | Value |
|--------|-------|
| Random seed | `42` (`pl.seed_everything(42, workers=True)`) |
| Python | 3.12 |
| PyTorch | 2.9.0 |
| CUDA | 12.x |
| HuBERT checkpoint | `facebook/hubert-base-ls960` |
| Deterministic ops | `torch.use_deterministic_algorithms(True, warn_only=True)` |
| Precision | `bf16-mixed` (ADA / Ampere) |

**Config snapshot**: every run writes `config.yaml` into both `checkpoints/<run_name>/` and
`results/<run_name>/` so the exact hyperparameters used for any checkpoint are always recoverable.

**Dependency pinning**: see `requirements.txt` for exact package versions.

**Known non-determinism**: HuBERT uses `attn_implementation="eager"` which is not fully
deterministic across all CUDA drivers. `warn_only=True` allows these operations to
proceed silently. Expect ±0.005 PER run-to-run variance on identical hardware.

To reproduce `baseline_v5`:
```bash
python run_pipeline.py --run-name baseline_v5_repro \
    --beam-search --beam-width 25 --explain --uncertainty
```

## References

- Cui et al. (2019) "Class-Balanced Loss Based on Effective Number of Samples"