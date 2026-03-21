# docs/training.md — Training Guide

> Cross-references: [docs/architecture.md](architecture.md) for model internals, [docs/experiments.md](experiments.md) for result interpretation.

---

## Environment Requirements

### Hardware

| Requirement | Minimum | Tested |
|---|---|---|
| GPU | NVIDIA GPU (8+ GB VRAM) | RTX 4060 8GB |
| CUDA | 11.8+ | CUDA 12.x |
| CPU RAM | 16 GB | 32 GB recommended |
| Storage | 50 GB free | NVMe SSD recommended |

**VRAM budget:** The `Config._print_vram_status()` method (called automatically by `run_pipeline.py` after config overrides are applied) prints a rough VRAM safety estimate. The formula is: `param_memory + gradient_memory + optimizer_states + activation_memory + 1.2 GB runtime reserve`. With `batch_size=12`, `bf16-mixed`, and `use_temporal_downsample=True`, estimated peak VRAM is approximately 6.5–7.2 GB during Stage 3 (layers 4–11 unfrozen), leaving a ~0.8–1.5 GB margin on 8 GB hardware.

**BF16 support check:**
```python
import torch
print(torch.cuda.is_bf16_supported())  # True for RTX 30xx+ (Ampere and later)
```

If BF16 is unavailable, set `config.training.precision = "16-mixed"` for FP16 mixed precision.

### Full Pinned Dependency Stack
```
torch==2.9.0
torchaudio==2.9.0
transformers==4.57.1
datasets==4.4.1
pytorch-lightning==2.6.0
mlflow==3.6.0
librosa==0.11.0
soundfile==0.13.1
g2p-en==2.1.0
nltk==3.9.2
jiwer==4.0.0
editdistance==0.8.1
statsmodels==0.14.6
pandas==2.3.3
numpy==1.26.4
tqdm==4.67.1
matplotlib==3.10.8
seaborn==0.13.2
pyyaml==6.0.3
scikit-learn==1.8.0
scipy==1.15.3
rapidfuzz==3.9.7
```

`statsmodels` is a **hard dependency** (required for Holm-Bonferroni correction in `evaluate.py`). `rapidfuzz` provides a fast C-extension path for phoneme alignment (falls back to pure Python if unavailable).

---

## Installation
```bash
# 1. Clone repository
git clone <repo-url>
cd DysarthriaNSR

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Install all pinned dependencies
pip install -r requirements.txt

# 4. Download NLTK data required by g2p_en
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"

# 5. (Optional) Pre-download HuBERT model (~360 MB)
python -c "
from transformers import HubertModel
HubertModel.from_pretrained(
    'facebook/hubert-base-ls960',
    revision='dba3bb02fda4248b6e082697eee756de8fe8aa8a'
)
print('HuBERT downloaded successfully')
"
```

---

## Configuration System

All hyperparameters are defined in `src/utils/config.py`. The `Config` class aggregates six dataclasses: `ModelConfig`, `TrainingConfig`, `DataConfig`, `ExperimentConfig`, `SymbolicConfig`, and `ProjectPaths`.

**The golden rule:** Never hardcode hyperparameters outside `config.py`. All parameters must be read from the config and logged to MLflow automatically via `flatten_config_for_mlflow()`.

### Config Save/Load Round-Trip
```python
from src.utils.config import Config
from pathlib import Path

# Save
config = Config()
config.save(Path("results/my_run/config.yaml"))

# Load for reproduction
config = Config.load(Path("results/my_run/config.yaml"))
```

Config is automatically saved to both `checkpoints/{run_name}/config.yaml` (during training) and `results/{run_name}/config.yaml` (during evaluation) by `run_pipeline.py`.

### Most Commonly Overridden Parameters

| Config key | Default | CLI flag | Description |
|---|---|---|---|
| `experiment.run_name` | `rtx4060_optimized_v1` | `--run-name STR` | Run identifier for all output paths |
| `training.ablation_mode` | `full` | `--ablation MODE` | Ablation mode |
| `training.max_epochs` | `40` | `--max-epochs INT` | Maximum training epochs |
| `training.batch_size` | `12` | `--batch-size INT` | Per-GPU batch size |
| `training.gradient_accumulation_steps` | `3` | `--grad-accum INT` | Gradient accumulation steps |
| `training.early_stopping_patience` | `8` | `--early-stopping-patience INT` | ES patience |
| `training.check_val_every_n_epoch` | `1` | `--check-val-every-n-epoch INT` | Validation frequency |
| `model.use_gradient_checkpointing` | `True` | `--no-gradient-checkpointing` | Disable gradient checkpointing |

---

## Running Experiments

### Full Pipeline
```bash
# Full train + evaluation
python run_pipeline.py --run-name experiment_v1

# Train only (skip evaluation)
python run_pipeline.py --run-name experiment_v1 --skip-eval

# Eval only (checkpoint must already exist)
python run_pipeline.py --run-name experiment_v1 --skip-train

# Smoke test (max_epochs=1, limit_train_batches=5)
python run_pipeline.py --run-name smoke --smoke
```

### Ablation Modes
```bash
# Neural-only (bypasses SeverityAdapter + SymbolicConstraintLayer)
python run_pipeline.py --run-name ablation_neural_v1 --ablation neural_only

# No learnable constraint matrix (keeps SeverityAdapter)
python run_pipeline.py --run-name ablation_no_cm_v1 --ablation no_constraint_matrix

# No articulatory auxiliary heads
python run_pipeline.py --run-name ablation_no_art_v1 --ablation no_art_heads

# No SpecAugment
python run_pipeline.py --run-name ablation_no_aug_v1 --ablation no_spec_augment

# No TemporalDownsampler
python run_pipeline.py --run-name ablation_no_ds_v1 --ablation no_temporal_ds
```

### LOSO Cross-Validation
```bash
# Full 15-fold sweep (required for publication; ~32h on RTX 4060)
python run_pipeline.py --run-name loso_v1 --loso

# Resume from last completed fold
python run_pipeline.py --run-name loso_v1 --loso --resume-loso

# Force re-run specific folds (clears old checkpoint + results first)
python run_pipeline.py --run-name loso_v1 --loso --resume-loso \
    --loso-force-speakers M01,F01
```

### Evaluation Options
```bash
# Beam search decoding
python run_pipeline.py --run-name experiment_v1 --skip-train \
    --beam-search --beam-width 25

# With bigram LM shallow fusion
python run_pipeline.py --run-name experiment_v1 --skip-train \
    --beam-search --beam-width 25 --lm-weight 0.3

# With per-utterance explainability output
python run_pipeline.py --run-name experiment_v1 --skip-train --explain

# With MC-Dropout uncertainty estimation (20 forward passes)
python run_pipeline.py --run-name experiment_v1 --skip-train --uncertainty \
    --uncertainty-samples 20

# Full eval suite
python run_pipeline.py --run-name experiment_v1 --skip-train \
    --beam-search --beam-width 25 --explain --uncertainty
```

### Feature Cache Warm-Up
```bash
# Pre-compute HuBERT features for all utterances (speeds up first epoch)
python run_pipeline.py --run-name cache_warmup --warm-cache --warm-cache-only
```

---

## Training Dynamics

### Progressive Unfreezing Timeline

| Epoch | Event | VRAM impact |
|---|---|---|
| 0 | Entire HuBERT encoder frozen; only classifier + adapter heads train | ~3.5 GB |
| 1 | **Stage 1:** Unfreeze layers 8–11; Adam state reset for these params | ~4.2 GB |
| 6 | **Stage 2:** Unfreeze layers 6–11 (direct call, not `unfreeze_after_warmup`) | ~5.0 GB |
| 12 | **Stage 3:** Unfreeze layers 4–11 (deepest adaptation) | ~6.5 GB |

### Staged `lambda_blank_kl` Ramp (I2)

| Epoch range | λ_blank_kl | Purpose |
|---|---|---|
| 0–9 | 0.10 | Gentle push; prevents early CTC collapse |
| 10–19 | 0.15 | Moderate push |
| ≥ 20 | 0.20 | Full target |

### Adam Momentum Reset on Unfreeze (T-04 fix)

When encoder layers are unfrozen, `_reset_hubert_lr_warmup()` clears Adam first/second moment estimates (`optimizer.state[p].clear()`) for all parameters in the `hubert_encoder` param group. This gives newly-active parameters the closest equivalent to a fresh start. **Caveat:** OneCycleLR's step counter is not reset, so newly-unfrozen parameters enter at the current (potentially decayed) LR position in the OneCycleLR schedule. A full fix requires per-group `CosineAnnealingWarmRestarts` (planned post-submission).

---

## Monitoring

### MLflow
```bash
# Launch MLflow UI
mlflow ui --backend-store-uri file://$(pwd)/mlruns
# Open http://127.0.0.1:5000
```

All experiments log to the `DysarthriaNSR` experiment. Filter by `run_name` tag.

### Key Metrics to Watch

| MLflow key | Healthy range | Interpretation |
|---|---|---|
| `val/per` | Decreasing over epochs | Primary quality metric; triggers checkpointing |
| `train/blank_prob_mean` | Should converge toward 0.75 | CTC insertion bias diagnostic; too high → deletions dominate; too low → insertions dominate |
| `val/constraint_row_entropy` | Moderate (not near 0 or log(V)) | Constraint matrix health; near 0 → degenerate peaked rows; near log(47) → uniform/useless |
| `val/constraint_kl_from_prior` | < 1.0 | Drift from symbolic prior; > 1.0 indicates `lambda_symbolic_kl` may need increasing |
| `train/avg_beta` | ~0.05–0.25 | Adaptive β; should be higher for dysarthric batches |

---

## Checkpoints and Results

### Directory Layout
```
checkpoints/{run_name}/
├── epoch=07-val_per=0.721.ckpt    # Scored checkpoint (low val_per wins)
├── epoch=28-val_per=0.505.ckpt    # Best checkpoint
├── last.ckpt                       # Most recent checkpoint
└── config.yaml                    # Exact config for reproducibility

results/{run_name}/
├── evaluation_results.json        # Primary results artifact
├── config.yaml                    # Config snapshot
├── confusion_matrix.png           # Top-30 phoneme confusion heatmap
├── per_by_length.png              # PER by utterance length × status
├── clinical_gap.png               # Dysarthric vs. control PER bar
├── rule_impact.png                # Symbolic activation analysis / C matrix heatmap
├── blank_probability_histogram.png # Blank prob distribution (target line at 0.75)
├── per_phoneme_per.png            # Per-phoneme PER breakdown (top-30)
├── articulatory_confusion.png     # Articulatory feature confusion matrix
├── severity_vs_per.png            # Scatter: severity vs. per-speaker PER
├── per_by_speaker.png             # Per-speaker PER bar (severity-sorted)
└── explanations.json              # Per-utterance explainability (--explain only)
```

### `evaluation_results.json` Schema

Key fields:
```json
{
  "avg_per": 0.137,
  "overall": {
    "per_macro_speaker": 0.137,
    "per_sample_mean": 0.141,
    "ci": [0.115, 0.162],
    "n_samples": 1200,
    "n_speakers": 3
  },
  "symbolic_impact": {
    "per_neural": 0.145,
    "per_constrained": 0.137,
    "delta_per": -0.008,
    "p_value_neural_vs_constrained": 0.0,
    "ci_95_delta_per": [-0.015, -0.001]
  },
  "articulatory_accuracy": { "manner": 0.786, "place": 0.791, "voice": 0.924 },
  "stratified": {
    "dysarthric": { "per_speaker": 0.189, "n": 600 },
    "control":    { "per_speaker": 0.085, "n": 600 }
  },
  "per_speaker": { "M01": { "per": 0.248, "ci": [0.21, 0.29], "n_samples": 312, "status": 1 } },
  "error_analysis": {
    "error_counts": { "substitutions": 13821, "deletions": 4338, "insertions": 3752, "correct": 42000 }
  },
  "uncertainty": { "computed": false, "n_samples": null, "entropy_mean": null }
}
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| CUDA OOM during training | Batch too large for available VRAM | Reduce `batch_size` to 4–8 and increase `gradient_accumulation_steps` proportionally; or set `max_audio_length=4.0` |
| NaN loss after epoch 12 | Gradient explosion on Stage 3 unfreeze | Enable anomaly detection: `--detect-anomaly`; ensure `gradient_clip_val=1.0` in config |
| `FileNotFoundError: Manifest not found` | Manifest not generated | Run `python src/data/manifest.py` |
| HuBERT download fails (offline/CI) | No internet access | Set `config.model.hubert_model_revision = None` and set `HF_HUB_OFFLINE=1` to use cached model |
| `Checkpoint directory not found` for eval-only | Training not run yet | Run training first: `python run_pipeline.py --run-name NAME --skip-eval` |
| `val/per` not improving after epoch 20 | Learning rate too low post-Stage-3 unfreeze | Increase `encoder_warmup_epochs` to reduce time before first unfreeze; or lower `lambda_ce` to 0.05 |
| Constraint matrix entropy diverging (> 3.0) | SymbolicKLLoss too weak | Increase `lambda_symbolic_kl` from 0.5 to 1.0 |
| LOSO fold crashes with `Tried to step X times` | OneCycleLR scheduler state exhausted in checkpoint | Re-run with `--resume-loso`; pipeline auto-detects exhaustion and does weights-only resume |