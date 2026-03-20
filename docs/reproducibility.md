# DysarthriaNSR — Reproducibility Guide

> Precision standard: this guide is written so that someone unfamiliar with the codebase
> can reproduce main results without asking a single question.
>
> Cross-reference: [docs/pipeline.md](pipeline.md) for data flow,
> [docs/dataset.md](dataset.md) for dataset setup details.

---

## Hardware Requirements

| Requirement | Minimum | Tested Configuration |
|-------------|---------|---------------------|
| GPU | NVIDIA GPU (8+ GB VRAM) | RTX 4060 8GB |
| CUDA | 11.8+ | CUDA 12.x |
| CPU RAM | 16 GB | 32 GB recommended |
| Storage | 50 GB free | SSD strongly recommended (NVMe for HuBERT cache) |

The default configuration (`batch_size=8`, BF16 mixed precision, `use_temporal_downsample=True`) is tuned for 8 GB VRAM. CPU-only training is supported but extremely slow.

**BF16 support check:**
```python
import torch
print(torch.cuda.is_bf16_supported())  # Should print True for RTX 30xx+
```
If BF16 is unavailable, the config automatically falls back to FP16-mixed.

---

## Software Environment

**Python version:** 3.10+ (tested on 3.11)

**All dependencies (pinned):**
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

---

## Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd DysarthriaNSR

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# 3. Install all pinned dependencies
pip install -r requirements.txt

# 4. Download NLTK data required by g2p_en
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

**HuBERT model pre-download (optional but recommended before training):**
```bash
# Downloads ~360MB model from HuggingFace Hub
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

## Dataset Preparation

See [docs/dataset.md](dataset.md) for TORGO corpus access and detailed structure.

### Step 1 — Download Audio Files

```bash
python src/data/download.py
```

Downloads TORGO audio from HuggingFace `abnerh/TORGO-database` and saves to `data/raw/audio/`.
File naming convention: `{speaker_id}_{md5_hash}_{original_name}.wav`

Expected output:
```
data/raw/audio/
├── F01_a1b2c3d4_session1_mic1_001.wav
├── M01_e5f6a7b8_session2_mic2_042.wav
...
```

### Step 2 — Generate Manifest

```bash
python src/data/manifest.py
```

This script:
1. Loads TORGO metadata from HuggingFace (metadata only, not audio)
2. Matches metadata to local `.wav` files via MD5 hash
3. Runs G2P transcription (g2p_en) on transcripts
4. Generates per-phoneme articulatory annotations (manner/place/voice)
5. Filters utterances: duration > 0.1s, phoneme count ≥ 2, rate ≤ 20 phonemes/s
6. Writes: `data/processed/torgo_neuro_symbolic_manifest.csv`

Expected output:
```
MANIFEST GENERATION COMPLETE
Output: data/processed/torgo_neuro_symbolic_manifest.csv
Total Samples: ~16531
Total Hours:   ~20.xx hrs
```

### Step 3 — (Optional) Warm Feature Cache

Pre-computing HuBERT processor outputs saves repeated I/O during training:

```bash
python run_pipeline.py --run-name cache_warmup --warm-cache --warm-cache-only
```

Cache files are stored at `data/processed/feature_cache/{namespace}/{sha1}.pt`. This step is optional but reduces the first epoch from ~2× longer to normal speed.

---

## Configuration

All hyperparameters are defined in `src/utils/config.py` (single source of truth). Key parameters:

### ModelConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hubert_model_id` | `facebook/hubert-base-ls960` | HuBERT variant |
| `hubert_model_revision` | `dba3bb02fda4248b6e082697eee756de8fe8aa8a` | Pinned hub revision |
| `freeze_feature_extractor` | `True` | CNN always frozen |
| `freeze_encoder_layers` | `[0,1,2,3]` | Permanently frozen transformer layers |
| `hidden_dim` | `512` | PhonemeClassifier hidden size |
| `classifier_dropout` | `0.1` | Dropout in classifier + adapter |
| `constraint_weight_init` | `0.05` | Initial β for symbolic blending |
| `use_learnable_constraint` | `True` | Enable `LearnableConstraintMatrix` |
| `use_severity_adapter` | `True` | Enable `SeverityAdapter` |
| `severity_adapter_dim` | `64` | Severity projection bottleneck |
| `use_temporal_downsample` | `True` | Enable stride-2 Conv1d downsampler |
| `use_spec_augment` | `True` | Enable SpecAugment on hidden states |

### TrainingConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `precision` | `bf16-mixed` | Auto-falls back to `16-mixed` |
| `learning_rate` | `3e-5` | Peak LR for classifier/adapter |
| `batch_size` | `8` | Per-GPU batch (effective batch = 32 with grad accum) |
| `gradient_accumulation_steps` | `4` | Effective batch = 8 × 4 = 32 |
| `max_epochs` | `40` | Maximum epochs |
| `lambda_ctc` | `0.80` | CTC loss weight |
| `lambda_ce` | `0.10` | Frame-CE loss weight |
| `lambda_articulatory` | `0.08` | Articulatory CE weight |
| `lambda_ordinal` | `0.05` | Ordinal contrastive weight |
| `lambda_blank_kl` | `0.20` | Blank-prior KL weight (final stage) |
| `lambda_symbolic_kl` | `0.50` | Symbolic KL anchor weight |
| `blank_target_prob` | `0.75` | Target mean blank probability |
| `early_stopping_patience` | `8` | Epochs without val/per improvement |

### DataConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling_rate` | `16000` | Target audio sample rate |
| `max_audio_length` | `6.0` | Maximum utterance length in seconds |
| `split_strategy` | `speaker_stratified` | Speaker-level train/val/test split |
| `train_split` | `0.70` | Fraction of speakers for training |

### To save/load a custom config:

```python
from src.utils.config import Config

# Save current config
config = Config()
config.save(Path("my_run/config.yaml"))

# Load for reproduction
config = Config.load(Path("my_run/config.yaml"))
```

---

## Training

### Standard Single-Run Training

```bash
python run_pipeline.py --run-name experiment_v1
```

This runs the full pipeline: training → evaluation.

Expected console output at startup:
```
🧠 Loading HuBERT: facebook/hubert-base-ls960
   📌 HuBERT revision pinned: dba3bb02...
   ✅ Gradient checkpointing enabled
   🧊 Froze feature extractor (CNN)
   🧊 Froze encoder layers [0, 1, 2, 3]
   ✅ SeverityAdapter enabled (cross-attention, continuous severity)
   ✅ SpecAugment enabled (time/freq masking on HuBERT hidden states)
   ✅ TemporalDownsampler enabled (~50 Hz → ~25 Hz, stride-2 Conv1d)
   ✅ Articulatory auxiliary heads enabled (manner/place/voice)
   ✅ LearnableConstraintMatrix enabled (Proposal P2)
✅ NeuroSymbolicASR ready: <N> total / <M> trainable params (X.X%)
```

**Checkpoint location:** `checkpoints/experiment_v1/`  
**Best checkpoint:** `checkpoints/experiment_v1/epoch=XX-val_per=Y.YYY.ckpt`

### Train Only (Skip Evaluation)

```bash
python run_pipeline.py --run-name experiment_v1 --skip-eval
```

### Resume Training

```bash
# Not directly supported — restart from last.ckpt manually:
python run_pipeline.py --run-name experiment_v1 --skip-eval
# (PyTorch Lightning will detect the existing checkpoint directory)
```

### Neural-Only Ablation (Fastest Baseline)

```bash
python run_pipeline.py --run-name neural_only_v1 --ablation neural_only
```

---

## Evaluation

### Evaluate a Saved Checkpoint

```bash
# Greedy decoding (default)
python run_pipeline.py --run-name experiment_v1 --skip-train

# Beam search (beam width 10)
python run_pipeline.py --run-name experiment_v1 --skip-train \
    --beam-search --beam-width 10

# With explainability output
python run_pipeline.py --run-name experiment_v1 --skip-train \
    --beam-search --beam-width 10 --explain

# With MC-Dropout uncertainty (20 forward passes)
python run_pipeline.py --run-name experiment_v1 --skip-train \
    --uncertainty --uncertainty-samples 20
```

**Results location:** `results/experiment_v1/`  
**Primary output:** `results/experiment_v1/evaluation_results.json`

### Key fields in `evaluation_results.json`

```json
{
  "avg_per": 0.137,
  "overall": {
    "per_macro_speaker": 0.137,
    "per_sample_mean":   0.141,
    "ci": [0.115, 0.162],
    "n_samples": 1200,
    "n_speakers": 3
  },
  "symbolic_impact": {
    "per_neural": 0.145,
    "per_constrained": 0.137,
    "delta_per": -0.008,
    "p_value_neural_vs_constrained": 0.0
  },
  "stratified": {
    "dysarthric": {"per_speaker": 0.189, "n": 600},
    "control":    {"per_speaker": 0.085, "n": 600}
  },
  "per_speaker": {
    "M01": {"per": 0.248, "n_samples": 312}
  }
}
```

---

## Full LOSO-CV

### Running LOSO (Required for Publication)

```bash
# Full 15-fold sweep
python run_pipeline.py --run-name loso_v1 --loso

# Resume from last completed fold
python run_pipeline.py --run-name loso_v1 --loso --resume-loso
```

Progress is saved after each fold to `results/loso_v1_loso_progress.json`. Per-fold checkpoints are in `checkpoints/loso_v1_loso_{speaker}/`.

Latest completed aggregate is in `results/loso_v1_loso_summary.json`:
- macro PER: 0.2848 (95% CI: [0.1921, 0.3801])
- weighted PER: 0.2299
- macro WER: 0.3362
- weighted WER: 0.2631

### Aggregating LOSO Results

```python
import json, pathlib, numpy as np

per_scores = []
for fold_dir in sorted(pathlib.Path("results").glob("loso_v1_loso_*")):
    results_file = fold_dir / "evaluation_results.json"
    if not results_file.exists():
        continue
    with open(results_file) as f:
        data = json.load(f)
    per_scores.append(data["overall"]["per_macro_speaker"])

per_scores = np.array(per_scores)
macro_per = per_scores.mean()
# Bootstrap 95% CI
boot = [np.random.choice(per_scores, len(per_scores)).mean() for _ in range(5000)]
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
print(f"Macro PER = {macro_per:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]  n={len(per_scores)}")
```

---

## Expected Outputs

### What Correct Training Output Looks Like

After epoch 1 (Stage 1 unfreezing):
```
🔥 Stage 1: Unfroze HuBERT layers 8-11 at epoch 1
Epoch 1: val/per=0.65 (expected range for early training)
```

After epoch ~20+:
```
Epoch 20: val/per≈0.36–0.40  (typical range on recent LOSO folds)
train/blank_prob_mean ≈ 0.75  (target met by blank KL loss)
train/avg_beta ≈ 0.05–0.25
```

### Checkpoint Files

```
checkpoints/experiment_v1/
├── epoch=07-val_per=0.721.ckpt
├── epoch=28-val_per=0.505.ckpt   ← best checkpoint (lowest val_per wins)
└── last.ckpt
```

### Results Directory

```
results/experiment_v1/
├── config.yaml                    ← exact run configuration
├── evaluation_results.json        ← primary results
├── confusion_matrix.png
├── per_by_length.png
├── per_by_speaker.png
├── articulatory_confusion.png
├── severity_vs_per.png
├── rule_impact.png
├── blank_probability_histogram.png
├── clinical_gap.png
└── explanations.json              ← only with --explain
```

---

## Determinism & Reproducibility

### Seed Settings

| Source | Seed | Where set |
|--------|------|-----------|
| Global seed | 42 | `pl.seed_everything(42, workers=True)` in `run_pipeline.py` |
| Data split | 42 | `np.random.seed(config.experiment.seed)` before speaker shuffle |
| SpecAugment RNG | Synced to torch initial seed | `_rng.manual_seed(torch.initial_seed())` in `SpecAugmentLayer` |

Run `pl.seed_everything(42, workers=True)` before any data loading or model initialization.

### Non-Deterministic Operations

- CUDA operations are not guaranteed deterministic by default
- To enable full determinism (slower): set `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` environment variable
- The `ExperimentConfig.deterministic=True` field is available for future use but is not currently forwarded to the Lightning `Trainer`

### HuBERT Revision Pinning

The HuBERT checkpoint is pinned to commit `dba3bb02fda4248b6e082697eee756de8fe8aa8a` via `ModelConfig.hubert_model_revision`. This prevents silent upstream model changes from affecting results. Override to `None` only for offline/CI environments:

```python
config.model.hubert_model_revision = None  # development only
```

---

## Smoke Test (Sanity Check)

Verify installation and pipeline integrity without a full training run:

```bash
python run_pipeline.py --run-name smoke --smoke
# Forces: max_epochs=1, limit_train_batches=5
# Expected: completes in ~2–5 minutes; no errors; val/per reported
```

Individual unit tests:
```bash
python -m pytest tests/ -v
```

Automated smoke scripts:
```bash
python scripts/smoke_test.py
# Expected: 7/7 tests pass
```

---

## Troubleshooting

### CUDA Out-of-Memory

```bash
# Reduce batch size (will require more gradient accumulation steps to maintain effective batch):
# In config.py: batch_size=4, gradient_accumulation_steps=8
python run_pipeline.py --run-name experiment_v1
```

Or disable the TemporalDownsampler and reduce max audio length:
```python
config.model.use_temporal_downsample = False
config.data.max_audio_length = 4.0   # 4s instead of 6s
```

### Manifest Not Found

```
FileNotFoundError: Manifest not found at data/processed/torgo_neuro_symbolic_manifest.csv.
Run 'python src/data/manifest.py' to generate it.
```
Run: `python src/data/manifest.py`

### HuBERT Download Fails (Offline / Revision Not Found)

```python
config.model.hubert_model_revision = None
# or set HF_HUB_OFFLINE=1 env var to use cached version
```

### Checkpoint Not Found for Eval-Only Run

```
FileNotFoundError: Checkpoint directory not found: checkpoints/experiment_v1
```
Train first: `python run_pipeline.py --run-name experiment_v1 --skip-eval`

### NaN Loss During Training

Enable anomaly detection for gradient diagnostics:
```bash
python run_pipeline.py --run-name debug_v1 --detect-anomaly
```
Warning: this is 5–10× slower; use only for debugging.

### Zero Matched Samples in Manifest Generation

```
Zero matches. Debugging path strings...
```
The download and manifest scripts use MD5 hashing of HF path strings to match metadata to local files. If the download naming convention changes, the hash won't match. Inspect `src/data/manifest.py` L184 and `src/data/download.py` for the naming format.
