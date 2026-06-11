# AGENTS.md — DysarthriaNSR

## Environment
```bash
source ~/.ichigo/bin/activate
```

## Canonical Entry Point
```bash
python run_pipeline.py --run-name <experiment_name>
```
**Never call `train.py` or `evaluate.py` directly** — they don't run evaluation/train automatically. Use `run_pipeline.py` for end-to-end runs.

## Essential Commands
```bash
# Full train + eval (single-split — best result)
python run_pipeline.py --run-name v4_final

# LOSO cross-validation (15 folds, ~32h)
python run_pipeline.py --run-name loso_v1 --loso

# Resume normal training from last.ckpt (or specify a filename)
python run_pipeline.py --run-name v4_final --resume
python run_pipeline.py --run-name v4_final --resume epoch=37-val_per=0.508.ckpt

# Resume or force re-run specific LOSO folds
python run_pipeline.py --run-name loso_v1 --loso --resume-loso
python run_pipeline.py --run-name loso_v1 --loso --loso-force-speakers M01,MC02

# Eval-only with checkpoint
python run_pipeline.py --run-name v4_final --skip-train
python run_pipeline.py --run-name v4_final --skip-train --beam-search --beam-width 25 --explain --uncertainty --uncertainty-samples 20 --calibrate-temperature

# Eval with per-speaker temperature calibration (no uncertainty/explain)
python run_pipeline.py --run-name v4_final --skip-train --beam-search --beam-width 25 --calibrate-temperature

# Earlier reference
python run_pipeline.py --run-name baseline_v6

# Ablation modes: neural_only, no_constraint_matrix, no_severity_adapter, no_spec_augment, no_temporal_ds
python run_pipeline.py --run-name <name> --ablation neural_only

# Smoke test (1 epoch, 5 batches)
python run_pipeline.py --run-name smoke --smoke

# Tests
python scripts/smoke_test.py --profile unit      # 8 fast checks
python scripts/smoke_test.py --profile pipeline   # 1 full CLI smoke
python scripts/smoke_test.py --profile all        # both
python -m pytest tests/ -v                       # all pytests
python -m pytest tests/test_training_step.py -v  # training step + e2e data flow

# Figures
python scripts/generate_figures.py --run-name v4_final
python scripts/generate_figures.py --run-name loso_v1
python scripts/generate_figures.py --run-name v4_final --compare ablation_neural_only_v7

# Data setup (order matters):
python src/data/download.py && python src/data/manifest.py

# Post-install requirement:
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
```

## Critical Conventions

### Phoneme Normalization
**ALWAYS** normalize phonemes before comparison or vocab building:
```python
from src.utils.config import normalize_phoneme
normalize_phoneme("AH0")  # → "AH" (strips stress)
```
Manifest uses ARPABET with stress, but model vocab is stress-agnostic.

### Vocabulary IDs (Fixed)
```
<BLANK> = 0  # CTC blank (never a true label)
<PAD>   = 1  # Padding sentinel (valid token)
<UNK>   = 2  # Unknown fallback
IY, AE… = 3-46  # ARPABET phonemes (47 total)
```
Built in `TorgoNeuroSymbolicDataset._build_vocabularies()` — never change.

### Label Padding
Labels use **-100** sentinel, not 0 or 1:
```python
labels = torch.nn.functional.pad(labels, (0, max_len - labels.size(1)), value=-100)
```
CTC/CE losses automatically ignore -100. Never use 0 or 1 for padding — those are valid token IDs.

### Speaker Severity
`batch['status']`: 0 (control) → severity 0.0, 1 (dysarthric) → severity 5.0
```python
severity = batch['status'].float() * 5.0  # fallback; prefers get_speaker_severity()
```

### Config YAML Save/Load
Config auto-saved to `results/{run_name}/config.yaml` each run. Reload:
```bash
python run_pipeline.py --run-name rerun --config results/previous_run/config.yaml
```
Only keys present in YAML override defaults — forward-compatible.

## Key Architecture Notes

- **HuBERT encoder** (`facebook/hubert-base-ls960`, pinned commit `dba3bb02f…`):
  CNN frozen forever. Layers 0-3 permanently frozen. Progressive unfreeze:
  - Epoch ≥ 1: unfreeze [8,9,10,11]
  - Epoch ≥ 6: unfreeze [6,7,8,9,10,11]
  - Epoch ≥ 12: unfreeze [4,5,6,7,8,9,10,11]
- **TemporalDownsampler**: stride-2 Conv1d halves frame rate (~50Hz → ~25Hz), effective CTC stride = 640
- **SymbolicConstraintLayer**: severity-adaptive β = clamp(0.05 + 0.2·s/norm, 0.0, 0.8); norm from `severity_normalization_constant` (default 5.0)
- **Constraint matrix**: 47×47 learnable, initialized from articulatory priors, KL-anchored with λ=0.5; row-entropy penalty (λ=0.05) prevents degenerate rows
- **LR scheduler**: CosineAnnealingWarmRestarts (T_0=1, T_mult=2, interval='epoch')
- **StratifiedMicroBatchSampler**: tiles minority class to match majority; ensures each micro-batch has both dysarthric and control samples at 3:1 ratio
- **TF32 enabled** via `torch.set_float32_matmul_precision('high')` on Ampere+

## Critical Config Parameters (`src/utils/config.py`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `lambda_symbolic_kl` | 0.5 | KL anchor for constraint matrix |
| `blank_target_prob` | 0.75 | Blank-prior KL target |
| `lambda_ctc` / `lambda_ce` | 0.8 / 0.15 | Primary losses (CE was 0.1 in earlier versions) |
| `lambda_articulatory` | 0.08 | Reduced from 0.15 (accuracy already 78-92%) |
| `lambda_blank_kl` | 0.20 | Full-stage blank KL target (staged from 0.10) |
| `blank_constraint_threshold` | 0.25 | Was 0.5 — lowered so constraint acts on more frames |
| `batch_size` / `grad_accum` | 12 / 3 | Effective batch = 36 |
| `precision` | bf16-mixed | Auto-detected; use `16-mixed` for pre-Ampere |
| `use_gradient_checkpointing` | True | Disable with `--no-gradient-checkpointing` for speed |
| `hubert_model_revision` | dba3bb02f… | Pinned for reproducibility |
| `severity_normalization_constant` | 5.0 | Normalization for adaptive β computation |
| `constraint_entropy_penalty_weight` | 0.05 | Row-entropy regularization for constraint matrix |
| `use_stratified_micro_batch` | True | Per-batch dysarthric/control balancing |
| `stratified_dysarthric_ratio` | 0.75 | Target dysarthric fraction in each micro-batch |

## Staged Loss Warmup (config: TrainingConfig)

| Loss | Stage 1 | Stage 2 | Stage 3 |
|------|---------|---------|---------|
| blank_kl | 0.10 → epoch 10 | 0.15 → epoch 20 | 0.20 |
| symbolic_kl | 0.1 → epoch 5 | 0.3 → epoch 15 | 0.5 |
| ordinal | 0.01 → epoch 10 | 0.03 → epoch 20 | 0.05 |

## Important Bug History (Still Relevant)

| Bug | Fix |
|-----|-----|
| B12 | Speaker ID extraction — use `path.name.split('_')[2]` not `[0]` |
| B13 | SpecAugment — apply per-sample, not batch-wide |
| B14 | Stage 2 unfreeze — explicit `unfreeze_encoder(layers=[6,7,8,9,10,11])`, not `unfreeze_after_warmup()` |
| B15 | Macro-speaker PER — average per-speaker first, not batch-mean |
| B16 | CTC stride — `320 * (2 if temporal_downsampler else 1)` |
| B17 | Attention mask — pass downsampled mask to `BlankPriorKLLoss` in val/test |
| B21 | Constraint init — temperature-sharpened: `log(C) / 0.5` |
| B22 | KL weight — λ=0.5 with sum/V normalization (not batchmean) |

## Hardware & Data

- RTX 4060 8GB, CUDA 12.x. VRAM ~6.5-7.2 GB at peak (batch=12, bf16-mixed).
- Data pipeline: `download.py → manifest.py → dataloader.py` (16,531 rows in manifest).
- Data splits: speaker-stratified (no speaker overlap between train/val/test).

## Known Open Issues (Still Relevant)

- Symbolic constraint Δ not statistically significant (v4_final: per_neural=0.134 vs per_constrained=0.137, p=0.1114)
- High variance across dysarthric LOSO folds (M01/M02/M04/M05/F01)
- Substitution/deletion dominant error pattern
- SymbolicRuleTracker logs 0 activations (logging issue, model works)

## References

- Full bug history & architecture details: `.github/copilot-instructions.md`
- Architecture: `docs/architecture.md` | Training: `docs/training.md` | Evaluation: `docs/evaluation.md`
