# AGENTS.md — DysarthriaNSR

## Environment

```bash
source ~/.ichigo/bin/activate
```

## Entry Point

```bash
python run_pipeline.py --run-name <experiment_name>
```

**Never call `train.py` or `evaluate.py` directly** — they don't run train/eval when invoked directly. Use `run_pipeline.py` for end-to-end runs.

## Essential Commands

```bash
# Full train + eval (single-split)
python run_pipeline.py --run-name v4_final

# LOSO cross-validation (15 folds, ~32h)
python run_pipeline.py --run-name loso_v1 --loso

# Resume (normal / LOSO)
python run_pipeline.py --run-name v4_final --resume
python run_pipeline.py --run-name v4_final --resume epoch=37-val_per=0.508.ckpt
python run_pipeline.py --run-name loso_v1 --loso --resume-loso

# Force re-run specific LOSO folds
python run_pipeline.py --run-name loso_v1 --loso --loso-force-speakers M01,MC02

# Eval-only with checkpoint
python run_pipeline.py --run-name v4_final --skip-train --beam-search --beam-width 25

# Eval with uncertainty / calibration / LM fusion
python run_pipeline.py --run-name v4_final --skip-train --beam-search --beam-width 25 --uncertainty --uncertainty-samples 20 --calibrate-temperature
python run_pipeline.py --run-name v4_final --skip-train --beam-search --beam-width 25 --lm-weight 0.3

# LOSO multi-mode eval
python scripts/loso_eval_modes.py --run-name loso_v1 --modes symbolic,neural,neural_beam

# Ablation modes: neural_only, no_constraint_matrix, no_severity_adapter, no_spec_augment, no_temporal_ds
python run_pipeline.py --run-name <name> --ablation neural_only

# Smoke test (1 epoch, 5 batches)
python run_pipeline.py --run-name smoke --smoke

# Unit tests
python scripts/smoke_test.py --profile unit      # 8 fast checks
python scripts/smoke_test.py --profile pipeline   # 1 full CLI smoke
python scripts/smoke_test.py --profile all        # both

# Data setup (order matters — download first)
python src/data/download.py && python src/data/manifest.py
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"

# Feature cache (safe speed optimization)
python run_pipeline.py --run-name cache_warmup --warm-cache
python run_pipeline.py --run-name cache_warmup --warm-cache --warm-cache-only
```

## Critical Conventions

### Phoneme Normalization
**Always** normalize phonemes before comparison or vocab building:
```python
from src.utils.config import normalize_phoneme
normalize_phoneme("AH0")  # → "AH" (strips stress markers 0/1/2)
```
Manifest uses ARPABET with stress; model vocab is stress-agnostic.

### Vocabulary IDs (Fixed — never change)
```
<BLANK> = 0  # CTC blank (never a true label)
<PAD>   = 1  # Padding sentinel (valid token)
<UNK>   = 2  # Unknown fallback
IY, AE… = 3-46  # ARPABET phonemes (47 total)
```
Built in `TorgoNeuroSymbolicDataset._build_vocabularies()`.

### Label Padding
Labels use **-100** sentinel (never 0 or 1 — those are valid token IDs):
```python
labels = torch.nn.functional.pad(labels, (0, max_len - labels.size(1)), value=-100)
```

### BF16 Epsilon Safety
All numerical epsilons use **1e-6** (not 1e-8) for BF16 numerical safety.

### Key Config Defaults (TrainingConfig / SymbolicConfig)

| Parameter | Default | Note |
|---|---|---|
| `lambda_blank_kl` | 0.20 | Staged: 0.10→epoch10, 0.15→epoch20, 0.20 |
| `lambda_ce` | 0.15 | Frame-CE weight |
| `lambda_symbolic_kl` | 0.5 | Staged: 0.1→epoch5, 0.3→epoch15, 0.5 |
| `lambda_ordinal` | 0.05 | Staged: 0.01→epoch10, 0.03→epoch20 |
| `use_forced_alignment` | True | torchaudio forced_align for frame-CE |
| `frame_ce_start_epoch` | 0 | Active from epoch 0 (forced alignment removes gate) |
| `blank_constraint_threshold` | 0.25 | Constraint bypassed on blank-dominant frames |
| `max_beta` | 0.8 | Upper bound for adaptive β clamping |
| `encoder_lr_multiplier` | 0.1 | HuBERT encoder LR = lr × multiplier |
| `symbolic_lr_multiplier` | 0.5 | Symbolic layer LR = lr × multiplier |
| `batch_size` / `grad_accum` | 12 / 3 | Effective batch = 36 |
| `loso_early_stopping_patience` | 22 | Higher patience for LOSO fold variance |

## Important Bug History

| Bug | Fix |
|---|---|
| B12 | Speaker ID — use `path.name.split('_')[2]` not `[0]` |
| B13 | SpecAugment — apply per-sample, not batch-wide |
| B14 | Stage 2 unfreeze — explicit `unfreeze_encoder(layers=[6,7,8,9,10,11])` |
| B15 | Macro-speaker PER for val/per, test/per; dys/control stratified PER uses per-utterance mean |
| B16 | CTC stride — `320 * (2 if temporal_downsampler else 1)` |
| B17 | Attention mask — pass downsampled mask to `BlankPriorKLLoss` in val/test |
| B21 | Constraint init — temperature-sharpened: `log(C) / 0.5` |
| B22 | KL weight — λ=0.5 with sum/V normalization (not batchmean) |

## Hardware & Data

- RTX 4060 8GB, CUDA 12.x, VRAM ~6.5-7.2 GB peak (batch=12, bf16-mixed).
- Data pipeline: `download.py → manifest.py → dataloader.py` (16,531 manifest rows).
- Splits: speaker-stratified (no speaker overlap between train/val/test).

## Known Open Issues

- Symbolic constraint Δ negligible with fair decoder comparison (v4_final: per_neural=0.131 vs per_constrained=0.133)
- High LOSO variance across dysarthric folds (M01/M02/M04/M05/F01)
- Substitution/deletion dominant error pattern
- `SymbolicRuleTracker` logs 0 activations (logging issue, model works)

## Docs

| File | Content |
|---|---|
| `docs/architecture.md` | Model architecture |
| `docs/training.md` | Training loop details |
| `docs/evaluation.md` | Evaluation methodology |
| `docs/experiments.md` | Experiment results & analysis |
| `docs/data.md` | Data pipeline details |
| `CHANGELOG.md` | Release history |
