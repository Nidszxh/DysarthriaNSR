# docs/training.md — Training Guide

> Cross-references: [architecture.md](architecture.md) for model internals, [experiments.md](experiments.md) for result interpretation.

---

## Hardware Requirements

| Requirement | Minimum | Tested |
|---|---|---|
| GPU | NVIDIA GPU (8+ GB VRAM) | RTX 4060 8GB |
| CUDA | 11.8+ | CUDA 12.x |
| CPU RAM | 16 GB | 32 GB recommended |
| Storage | 50 GB free | NVMe SSD recommended |

**VRAM budget:** `Config._print_vram_status()` (called automatically by `run_pipeline.py` after config overrides) prints a rough VRAM safety estimate. The formula sums: `param_memory + gradient_memory + optimizer_states + activation_memory + 1.2 GB runtime reserve`. With `batch_size=12`, `bf16-mixed`, and `use_temporal_downsample=True`, estimated peak VRAM is approximately 6.5–7.2 GB during Stage 3 (layers 4–11 unfrozen). The call to `print_vram_status()` must come after all config overrides are applied so ablation mode is correctly reflected.

**BF16 check:**
```python
import torch
print(torch.cuda.is_bf16_supported())  # True for RTX 30xx+ (Ampere and later)
```
If BF16 is unavailable, set `config.training.precision = "16-mixed"`.

---

## Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd DysarthriaNSR

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# 3. Install all pinned dependencies
pip install -r requirements.txt

# 4. Download NLTK data required by g2p_en
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"

# 5. (Optional) Pre-download HuBERT model (~360 MB, avoids first-run network latency)
python -c "
from transformers import HubertModel
HubertModel.from_pretrained(
    'facebook/hubert-base-ls960',
    revision='dba3bb02fda4248b6e082697eee756de8fe8aa8a'
)
print('HuBERT cached successfully')
"
```

**Why the revision is pinned:** `dba3bb02fda4248b6e082697eee756de8fe8aa8a` was verified from the local cache on 2026-03-06 and ensures exact weight reproducibility. Setting `hubert_model_revision=None` disables pinning (development/CI use only).

**Key pinned dependencies:**
`torch==2.9.0`, `torchaudio==2.9.0`, `transformers==4.57.1`, `pytorch-lightning==2.6.0`, `mlflow==3.6.0`, `statsmodels==0.14.6` (hard dependency — required for Holm-Bonferroni correction; no silent fallback), `rapidfuzz==3.9.7` (C-extension fast path for phoneme alignment, falls back to pure Python if unavailable).

Full pinned stack in `requirements.txt`.

---

## Configuration System

All hyperparameters are defined in `src/utils/config.py`. The `Config` class aggregates six dataclasses: `ModelConfig`, `TrainingConfig`, `DataConfig`, `ExperimentConfig`, `SymbolicConfig`, and `ProjectPaths`.

**Golden rules:**
- Never hardcode hyperparameters outside `config.py` — all parameters must come from config and are logged to MLflow automatically via `flatten_config_for_mlflow()`.
- Never hardcode file paths — use `ProjectPaths` from `config.py` (`get_project_root() / "results" / config.experiment.run_name`).
- The deferred `config.print_vram_status()` call in `run_pipeline.py` must come after all overrides are applied.

**Config save/load round-trip:**
```python
from src.utils.config import Config
from pathlib import Path

config = Config()
config.save(Path("results/my_run/config.yaml"))  # Saves to YAML

config_loaded = Config.load(Path("results/my_run/config.yaml"))  # Restores from YAML
```

Config is saved to both `checkpoints/{run_name}/config.yaml` (during training) and `results/{run_name}/config.yaml` (during evaluation). Note that `substitution_rules` tuple keys are serialized as `"B_P"` strings in YAML and reconstructed as `('B', 'P')` tuples on load.

### 15 Most Commonly Overridden Parameters

| Config key | Default | CLI flag | Notes |
|---|---|---|---|
| `experiment.run_name` | `rtx4060_optimized_v1` | `--run-name STR` | Single source of truth for all output paths |
| `training.ablation_mode` | `full` | `--ablation MODE` | One of: full, neural_only, no_constraint_matrix, no_art_heads, no_spec_augment, no_temporal_ds, symbolic_only |
| `training.max_epochs` | `40` | `--max-epochs INT` | Maximum training epochs |
| `training.batch_size` | `12` | `--batch-size INT` | Per-GPU batch size |
| `training.gradient_accumulation_steps` | `3` | `--grad-accum INT` | Effective batch = batch_size × grad_accum |
| `training.early_stopping_patience` | `8` | `--early-stopping-patience INT` | Epochs without improvement (paper full-system runs use 6; ablations use 8) |
| `training.check_val_every_n_epoch` | `1` | `--check-val-every-n-epoch INT` | Set to 2 to halve validation overhead for LOSO |
| `training.encoder_warmup_epochs` | `1` | (modify config.py) | Stage 1 unfreeze trigger |
| `training.lambda_symbolic_kl` | `0.5` | (modify config.py) | KL anchor strength |
| `training.blank_target_prob` | `0.75` | (modify config.py) | CTC blank target probability |
| `model.use_learnable_constraint` | `True` | (modify config.py) | Disable for static constraint matrix |
| `model.use_severity_adapter` | `True` | (modify config.py) | Disable for no severity conditioning |
| `model.freeze_encoder_layers` | `[0,1,2,3]` | (modify config.py) | Permanently frozen transformer layers |
| `model.use_gradient_checkpointing` | `True` | `--no-gradient-checkpointing` | Disable for faster training with more VRAM |
| `experiment.seed` | `42` | (modify config.py) | Random seed for all splits and augmentation |

---

## Run Commands Reference

### Full Pipeline

```bash
# Full train + evaluation (canonical single-split run)
python run_pipeline.py --run-name experiment_v1

# Train only (skip evaluation)
python run_pipeline.py --run-name experiment_v1 --skip-eval

# Eval only (checkpoint must already exist)
python run_pipeline.py --run-name experiment_v1 --skip-train

# Smoke test: max_epochs=1, limit_train_batches=5 (fast CI check)
python run_pipeline.py --run-name smoke --smoke

# Smoke test: unit profile (7 automated checks)
python scripts/smoke_test.py --profile unit

# Smoke test: pipeline CLI (trains 1 epoch, verifies end-to-end)
python scripts/smoke_test.py --profile pipeline
```

### Ablation Modes

```bash
# Neural-only baseline (bypasses SeverityAdapter + SymbolicConstraintLayer entirely)
python run_pipeline.py --run-name ablation_neural_only --ablation neural_only

# No learnable constraint matrix (keeps SeverityAdapter, removes constraint blending)
python run_pipeline.py --run-name ablation_no_cm --ablation no_constraint_matrix

# No articulatory auxiliary heads
python run_pipeline.py --run-name ablation_no_art --ablation no_art_heads

# No SpecAugment
python run_pipeline.py --run-name ablation_no_aug --ablation no_spec_augment

# No TemporalDownsampler
python run_pipeline.py --run-name ablation_no_ds --ablation no_temporal_ds

# Symbolic-only (CTC/CE disabled, tests pure symbolic signal)
python run_pipeline.py --run-name ablation_symbolic_only --ablation symbolic_only
```

### LOSO Cross-Validation

```bash
# Full 15-fold sweep (publication result, ~32h on RTX 4060)
python run_pipeline.py --run-name loso_v1 --loso

# Resume from last completed fold (reads loso_progress.json)
python run_pipeline.py --run-name loso_v1 --loso --resume-loso

# Force re-run specific folds (clears old checkpoint + results dirs)
python run_pipeline.py --run-name loso_v1 --loso --resume-loso \
    --loso-force-speakers M01,F01

# LOSO with faster validation (validate every 2 epochs)
python run_pipeline.py --run-name loso_v1 --loso --check-val-every-n-epoch 2
```

### Evaluation Options

```bash
# Beam search decoding (width=25, more accurate than greedy)
python run_pipeline.py --run-name baseline_v6 --skip-train \
    --beam-search --beam-width 25

# Beam search + bigram LM shallow fusion (λ=0.3)
python run_pipeline.py --run-name baseline_v6 --skip-train \
    --beam-search --beam-width 25 --lm-weight 0.3

# Per-utterance explainability output (generates explanations.json)
python run_pipeline.py --run-name baseline_v6 --skip-train --explain

# MC-Dropout uncertainty estimation (20 forward passes)
python run_pipeline.py --run-name baseline_v6 --skip-train --uncertainty \
    --uncertainty-samples 20

# Full evaluation suite
python run_pipeline.py --run-name baseline_v6 --skip-train \
    --beam-search --beam-width 25 --explain --uncertainty
```

### Figures and Cache

```bash
# Generate publication figures for a run
python scripts/generate_figures.py --run-name loso_v1

# Compare multiple runs in a single figure suite
python scripts/generate_figures.py --run-name baseline_v6 \
    --compare ablation_neural_only_v7 ablation_no_constraint_matrix_v6

# Pre-warm HuBERT feature cache (speeds up first epoch)
python run_pipeline.py --run-name cache_warmup --warm-cache --warm-cache-only
```

---

## Training Dynamics Deep Dive

### Differential Learning-Rate Groups (paper configuration)

Paper runs use AdamW with three parameter groups under OneCycleLR (5% warmup, cosine annealing):

| Parameter group | LR multiplier | Base LR | Effective peak LR |
|---|---:|---:|---:|
| HuBERT encoder | 0.1× | 3e-5 | 3e-6 |
| PhonemeClassifier + SeverityAdapter | 1.0× | 3e-5 | 3e-5 |
| Constraint layer (`LearnableConstraintMatrix`) | 0.5× | 3e-5 | 1.5e-5 |

This split keeps pretrained HuBERT updates conservative while allowing classifier/adapter adaptation and moderate constraint-matrix plasticity.

### Three-Stage Freeze Schedule

```
Epoch 0      ──────────────────────────────── Warmup
  Frozen: layers 0-11 + CNN
  Trainable: PhonemeClassifier, SeverityAdapter, SymbolicLayer, art heads
  VRAM: ~3.5 GB

Epoch 1      ──────────────────────────────── Stage 1 Unfreeze
  Newly trainable: layers 8, 9, 10, 11
  Adam state cleared for these params (_reset_hubert_lr_warmup)
  VRAM: ~4.2 GB

Epoch 6      ──────────────────────────────── Stage 2 Unfreeze
  Newly trainable: layers 6, 7 (layers 8-11 already trainable)
  Adam state cleared again
  VRAM: ~5.0 GB

Epoch 12     ──────────────────────────────── Stage 3 Unfreeze
  Newly trainable: layers 4, 5 (layers 6-11 already trainable)
  Adam state cleared again
  VRAM: ~6.5 GB

Epoch 40     ──────────────────────────────── Max epochs
  Layers 0-3 remain frozen throughout
```

### `_reset_hubert_lr_warmup()` — Adam State Reset

When encoder layers are unfrozen, the `hubert_encoder` parameter group in AdamW has its per-parameter state cleared: `optimizer.state[p].clear()` for each parameter `p` in the group. This clears the **entire** state dict entry (exp_avg + exp_avg_sq + step), not individual keys. Clearing only `exp_avg` leaves a partially-initialized state that causes `KeyError('exp_avg')` on the next `optimizer.step()`. The full `.clear()` ensures Adam reinitializes cleanly on first use after unfreeze.

**Known limitation (T-04/O-2):** OneCycleLR's step counter is not reset. Newly unfrozen parameters enter at the current (potentially decayed) LR position rather than at the original peak. Full fix requires per-group `CosineAnnealingWarmRestarts` schedulers (planned post-submission).

### Staged `lambda_blank_kl` Ramp (I2)

Without the ramp, aggressively suppressing blank probability from epoch 0 forces the model to emit phonemes before it has learned basic acoustic representations, causing CTC to collapse to repeated phoneme insertions.

| Epoch range | λ_blank_kl |
|---|---|
| 0–9 | 0.10 |
| 10–19 | 0.15 |
| ≥ 20 | 0.20 |

The current epoch's value is stored as `self._current_lambda_blank_kl` on the Lightning module in `on_train_epoch_start()` and read by `compute_loss()`. This avoids mutating the config object, which would break per-fold isolation in LOSO.

### `valid_mask` Filtering in `training_step()`

Samples where `label_lengths > input_lengths` are dropped before loss computation. CTC loss would return `inf` for these samples, producing NaN gradients that crash training. The fraction of dropped samples is logged as `train/ctc_invalid_frac`. `input_lengths` used here are the exact `outputs['output_lengths']` from the model's live forward pass — not the approximate `batch['input_lengths']` from the collator (B4 fix).

---

## LOSO Operational Guide

### Deterministic Fold Ordering

`run_loso()` sorts speakers via `_split_speaker_id()`:
```python
def _split_speaker_id(spk: str) -> Tuple[str, int]:
    m = re.match(r"([A-Za-z]+)(\d+)$", str(spk))
    return (m.group(1), int(m.group(2)))  # e.g., FC01 → ("FC", 1)
```

This produces a deterministic order independent of manifest regeneration: FC01, FC02, FC03, M01, M02, M03, M04, M05, MC01, MC02, MC03, MC04, F01, F03, F04.

### Crash-Safe Progress JSON

The progress file `results/{run_name}_loso_progress.json` records completed fold metadata atomically (via `.tmp` rename) after each fold completes. On resume, completed folds are read from this file and skipped — fold PER, WER, and n_samples are used directly from the progress file without re-evaluating.

### Resume Paths

`run_loso()` handles three resume scenarios:

1. **Completed fold skip:** If the speaker appears in `completed_folds` (from progress JSON), skip entirely and use cached metrics.

2. **Normal checkpoint resume:** If `checkpoints/{fold_run_name}/last.ckpt` exists and the scheduler is not exhausted, continue training from the checkpoint.

3. **Weights-only resume (scheduler exhausted):** If `last.ckpt` epoch ≥ max_epochs-1, or if the OneCycleLR scheduler's `last_epoch ≥ total_steps`, the model weights are loaded but the optimizer/scheduler are fresh. `resume_epoch_offset` is set to preserve staged unfreezing behavior. A `weights_only_resume.pt` marker file is written to the fold checkpoint directory.

**`--loso-force-speakers`:** Clears `checkpoints/{fold_run_name}/` and `results/{fold_run_name}/` before rerunning the specified folds. Use to recover from corrupted checkpoints or to rerun with code fixes.

### H-1 VRAM Cleanup Between Folds

```python
del trainer, lm, model
torch.cuda.empty_cache()
```

This explicit cleanup after each fold prevents VRAM fragmentation accumulation on the 8 GB RTX 4060.

### Per-Fold Timing on RTX 4060

Approximately 32 total hours for 15 folds at 40 epochs each (estimated ~2h/fold). The `_CompactFoldProgressCallback` prints ETA after each completed fold based on average elapsed time per fold.

---

## Monitoring

### MLflow

```bash
mlflow ui --backend-store-uri file://$(pwd)/mlruns
# Open http://127.0.0.1:5000
```

All experiments log to the `DysarthriaNSR` experiment. Filter by `run_name` tag to find fold runs.

### 8 Key Metrics

| MLflow key | Healthy range | Interpretation |
|---|---|---|
| `val/per` | Decreasing over epochs | Primary quality metric; triggers checkpointing |
| `train/blank_prob_mean` | Converging toward 0.75 | CTC insertion bias diagnostic; too high → deletions dominate; too low → insertions dominate |
| `val/constraint_row_entropy` | Moderate (not near 0 or log(47)=3.85) | Near 0 → degenerate peaked constraint rows; near log(V) → uniform/useless constraint |
| `val/constraint_kl_from_prior` | < 1.0 | Drift from symbolic prior; >1.0 → increase `lambda_symbolic_kl` |
| `train/avg_beta` | ~0.05–0.25 | Adaptive β; should be higher for dysarthric batches |
| `train/loss_ctc` | Decreasing | Primary sequence alignment loss |
| `train/loss_blank_kl` | Decreasing or stable | Should respond to staged warmup ramp |
| `train/grad_norm` | < 2.0 (logged every 50 steps) | Spikes after each unfreeze event are normal; sustained spikes indicate LR too high |

---

## Checkpoints and Results

### Directory Layout

```
checkpoints/{run_name}/
├── epoch=07-val_per=0.721.ckpt    # Scored checkpoint (lower val_per wins)
├── epoch=28-val_per=0.505.ckpt    # Best checkpoint
├── last.ckpt                       # Most recent epoch
└── config.yaml                    # Exact config for reproducibility

results/{run_name}/
├── evaluation_results.json        # Primary results artifact
├── config.yaml                    # Config snapshot
├── confusion_matrix.png           # Top-30 phoneme confusion heatmap
├── per_by_length.png              # PER by utterance length × dysarthric/control
├── clinical_gap.png               # Dysarthric vs. control PER bar chart
├── rule_impact.png                # Symbolic activation analysis / C matrix heatmap (E6 fallback)
├── blank_probability_histogram.png # Blank prob distribution (target line at 0.75)
├── per_phoneme_per.png            # Per-phoneme PER breakdown (top-30 hardest)
├── articulatory_confusion.png     # Articulatory feature confusion heatmaps (3-panel)
├── severity_vs_per.png            # Scatter: severity vs. per-speaker PER + OLS
├── per_by_speaker.png             # Per-speaker PER bar chart (severity-sorted, M-6 fix)
├── rule_pair_confusion.png        # Neural vs. constrained substitution counts per rule
├── per_phoneme_per.json           # Raw per-phoneme breakdown data
├── rule_pair_confusion.json       # Raw rule-pair confusion data
├── per_by_manner.json             # Articulatory-stratified PER by manner class
├── learning_curve.png             # Train loss + val PER over epochs
└── explanations.json              # Per-utterance explainability (--explain only)
```

### `evaluation_results.json` Schema (Key Fields)

```json
{
  "avg_per": 0.137,
  "wer": 0.141,
  "overall": {
    "per_macro_speaker": 0.137,
    "per_sample_mean": 0.141,
    "wer": 0.141,
    "ci": [0.115, 0.162],
    "std": 0.089,
    "n_samples": 1200,
    "n_speakers": 3
  },
  "symbolic_impact": {
    "per_neural": 0.1451,
    "per_constrained": 0.1372,
    "delta_per": 0.0079,
    "paired_delta_constrained_minus_neural": {
      "delta_mean": -0.0079,
      "ci_95_low": -0.015,
      "ci_95_high": -0.001,
      "p_value_two_sided": 0.0
    },
    "p_value_neural_vs_constrained": 0.0
  },
  "articulatory_accuracy": { "manner": 0.786, "place": 0.791, "voice": 0.924 },
  "stratified": {
    "dysarthric": { "per_sample": 0.189, "per_speaker": 0.210, "ci": [...], "n": 600 },
    "control":    { "per_sample": 0.085, "per_speaker": 0.091, "ci": [...], "n": 600 }
  },
  "per_speaker": {
    "M01": { "per": 0.248, "ci": [0.21, 0.29], "std": 0.12, "n_samples": 312, "status": 1 }
  },
  "constraint_precision": {
    "helpful_rate": 0.0916,
    "neutral_rate": 0.8706,
    "harmful_rate": 0.0378,
    "rule_precision": 0.0916,
    "n_utterances": 1200
  },
  "error_analysis": {
    "error_counts": { "substitutions": 13821, "deletions": 4338, "insertions": 3752, "correct": 42000 }
  },
  "uncertainty": { "computed": false, "n_samples": null, "entropy_mean": null, "per_utterance": [] }
}
```

---

## Troubleshooting

| Symptom | Root Cause | Fix |
|---|---|---|
| CUDA OOM during training | Batch too large for VRAM after Stage 3 unfreeze | Reduce `batch_size` to 4–8 and increase `gradient_accumulation_steps` proportionally; or set `max_audio_length=4.0` |
| NaN CTC loss | Gradient explosion after unfreeze | Enable `--detect-anomaly`; verify `gradient_clip_val=1.0`; check `valid_mask` is filtering short sequences |
| `val/per` not improving after epoch 12 | LR too low post-Stage-3 unfreeze (OneCycleLR not reset) | Reduce `encoder_third_unfreeze_epoch`; or lower `lambda_ce` to 0.05; known T-04 limitation |
| `val/constraint_kl_from_prior` > 1.0 (diverging) | `lambda_symbolic_kl` too weak | Increase `lambda_symbolic_kl` from 0.5 to 1.0 |
| `per_constrained > per_neural` | Frame-CE alignment noise polluting constraint matrix training | Reduce `lambda_ce` further; confirm `lambda_symbolic_kl=0.5` is set |
| LOSO fold crashes with `Tried to step X times` | OneCycleLR scheduler state exhausted in checkpoint | Re-run with `--resume-loso`; pipeline auto-detects exhaustion and does weights-only resume |
| Checkpoint vocab mismatch warning at load | Manifest regenerated between train and eval with different phoneme set | Verify manifest uses same TORGO data; check `evaluation_results.json` for vocab size |
| `FileNotFoundError: Manifest not found` | Manifest not generated | Run `python src/data/manifest.py` |
| HuBERT revision not found | Offline environment or pinned hash unavailable | Set `hubert_model_revision=None` and `HF_HUB_OFFLINE=1` to use cached model |
| Zero matched samples in manifest generation | Hash mismatch between download and manifest | Re-run `python src/data/download.py` then `python src/data/manifest.py` |
| Articulatory accuracy stuck at chance | Articulatory head receiving wrong label alignment | Confirm `logits_manner.dim() == 2` (utterance-level path I5); check manner_to_id vocabulary build |
| `BlankPriorKLLoss` not decreasing | Target probability too high or warmup stage too long | Verify `blank_target_prob=0.75`; verify staged ramp is active (`_current_lambda_blank_kl` attribute) |
| `SymbolicRuleTracker` returns 0 activations | `rule_activations` list not wired to tracker | Known logging issue, not a model issue (B9 fix was applied); check `SymbolicConstraintLayer._track_activations()` |
