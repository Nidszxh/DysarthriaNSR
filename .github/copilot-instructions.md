# DysarthriaNSR — AI Coding Assistant Instructions

## Project Overview

**Neuro-symbolic self-supervised speech recognition for dysarthric speech with phoneme-level explainability.**

This is a research project combining:
- **Neural**: Self-supervised learning (SSL) models (e.g., wav2vec 2.0) for acoustic-phonetic representations
- **Symbolic**: Phoneme-level reasoning with articulatory constraints and dysarthric substitution rules
- **Clinical**: TORGO dataset with dysarthric (impaired) and control (normal) speech samples

**Key Goal**: Build an explainable ASR system that handles dysarthric speech and provides phoneme-level error attribution.

## Architecture & Data Flow

1. **Data**: TORGO database from HuggingFace (`abnerh/TORGO-database`)
   - Dysarthric speakers: F01, F03, F04, M01-M05
   - Control speakers: FC01-03, MC01-04 (prefixed with 'C')
   - Multiple recording sessions and microphone types (arrayMic, headMic)

2. **Preprocessing Pipeline** ([manifest.py](manifest.py)):
   - Loads HF dataset → Extracts phonemes via `g2p_en` → Maps to articulatory classes → Computes RMS energy → Generates CSV manifest
   - **Critical fields**: `sample_id`, `speaker`, `status`, `label` (1=dysarthric), `transcript`, `phonemes`, `articulatory_classes`, `phn_count`, `duration`, `rms_energy`
   - **Robustness enhancements**: Articulatory feature mapping (stops, fricatives, nasals, etc.) and signal quality assessment
   - Manifest enables reliable speaker-prefixed sample matching for HF dataset

3. **Neural Dataset & Dataloader** ([src/data/dataloader.py](src/data/dataloader.py)):
   - `TorgoNeuroSymbolicDataset`: Streams audio from HF dataset matched by `speaker_filename` key
   - HuBERT feature extraction (16kHz, zero-mean-unit-variance normalized)
   - **Audio truncation**: Long utterances capped at `max_audio_length` (default 8 seconds) to reduce GPU memory
   - Peak normalization to handle TORGO's variable breath support
  - **Class balancing**: Inverse-frequency phoneme weights computed from manifest for CE loss; weighted sampler balances dysarthric vs control
   - Returns: audio features, phoneme IDs, and articulatory metadata
   - **CTC-ready**: Vocabulary includes `<BLANK>` (ID 0), `<PAD>` (ID 1), `<UNK>` (ID 2), then phonemes (ID 3+)

4. **Collation & Batching** ([dataloader.py](dataloader.py)):
   - `NeuroSymbolicCollator`: Pads audio and phoneme sequences with proper attention masks
   - Audio padding value: 0.0 (silence)
   - Label padding value: -100 (ignored by CTC loss)
   - Computes attention mask manually (1 where signal exists, 0 for padding)

5. **Neural Component** ([src/models/model.py](src/models/model.py)):
   - HuBERT encoder (facebook/hubert-base-ls960) extracts 768-dim self-supervised representations
   - **Gradient checkpointing**: Enabled to reduce activation memory during backprop (~40% reduction, ~20% slower)
   - Projection layer reduces to `hidden_dim` (512), followed by phoneme classifier
   - CTC loss aligns variable-length audio to phoneme sequences
   - Supports selective layer freezing for VRAM optimization (freeze first 8 encoder layers by default)

6. **Symbolic Constraint Layer** ([src/models/model.py](src/models/model.py)):
   - `SymbolicConstraintMatrix`: Encodes articulatory similarity via distance between phoneme features
   - `ConstraintAggregation`: Applies learnable weighted blend to neural logits: `logits_constrained = β * logits_neural + (1-β) * (C @ logits_neural)`
   - Dysarthria-aware weighting: Higher dysarthric severity → stronger constraint influence (adaptive β)
  - **Current init**: `constraint_weight_init=0.5` for stronger early symbolic guidance
   - Covers stops, fricatives, nasals, liquids, glides, vowels, diphthongs with phonetic properties (manner, place, voicing)

7. **Training Pipeline** ([train.py](train.py)):
   - Multi-task learning: CTC loss (primary phoneme alignment) + CE loss (frame-level auxiliary)
   - Adaptive neuro-symbolic weighting via `beta`: Neural logits blended with symbolic constraints based on speaker severity
  - **Loss balancing**: `lambda_ctc=0.7`, `lambda_ce=0.3`; CE uses dataset inverse-frequency weights plus boosted BLANK/PAD
  - **Length safety**: Guard to ensure input_lengths ≥ label_lengths before CTC to avoid inf losses on short clips
   - MLflow tracking with safe parameter flattening for nested configs
   - EarlyStopping, ModelCheckpoint, LearningRateMonitor callbacks
  - **Memory optimizations**: batch_size=2, gradient_accumulation=8 (effective 16), gradient_checkpointing=True, max_audio_length=8s, fp16 mixed precision
   - Learning rate: 5e-5, OneCycleLR scheduler with 10% warmup, label smoothing 0.1
   - CUDA memory fragmentation mitigation: `PYTORCH_ALLOC_CONF=expandable_segments:True`

## Critical Developer Workflows

### 1. Data Download & Setup
```bash
python download.py  # Downloads TORGO dataset to ./data/
```
- Sets `HF_HOME` to `./data` to keep HuggingFace cache local
- Uses `Audio(decode=False)` when accessing paths to avoid torchcodec decoder errors
- Safely extracts speaker IDs from filename parsing

### 2. Generate Neuro-Symbolic Manifest
```bash
python src/data/manifest.py --data-dir ./data --out ./data/processed/torgo_neuro_symbolic_manifest.csv
```
- **Requires**: `pip install g2p_en pandas tqdm librosa datasets nltk`
- Automatically downloads NLTK tagger if needed
- **Output columns**: `sample_id`, `hf_index`, `path`, `speaker`, `status`, `label`, `transcript`, `phonemes`, `articulatory_classes`, `phn_count`, `duration`, `rms_energy`, `gender`
- **Robustness features**:
  - Articulatory class mapping (stops, fricatives, nasals, liquids, glides, vowels, diphthongs)
  - RMS energy calculation for signal quality assessment
  - Speaker-prefixed sample IDs for reliable HF dataset matching

### 3. Initialize DataLoader for Training
```python
from src.data.dataloader import TorgoNeuroSymbolicDataset, NeuroSymbolicCollator
from torch.utils.data import DataLoader

dataset = TorgoNeuroSymbolicDataset(
    manifest_path="./data/processed/torgo_neuro_symbolic_manifest.csv",
    processor_id="facebook/hubert-base-ls960",
    sampling_rate=16000,
    max_audio_length=8.0  # Truncate long utterances to 8 seconds
)

collator = NeuroSymbolicCollator(dataset.processor)

loader = DataLoader(
    dataset,
    batch_size=1,  # Small batch for 8GB VRAM
    shuffle=True,
    collate_fn=collator
)

# Each batch returns:
# - input_values: [batch_size, max_time] - HuBERT features
# - attention_mask: [batch_size, max_time] - 1 where valid, 0 for padding
# - labels: [batch_size, max_phonemes] - Phoneme IDs (-100 for padding, ignored by CTC loss)
# - status: [batch_size] - Dysarthric (1) vs Control (0)
# - speakers: [batch_size] - Speaker IDs for analysis
```

### 4. Start Training
```bash
# Run with default config
python train.py

# Or with custom run name
python train.py --run-name my_experiment_v1
```
- **Config system**: Uses Python dataclasses (`ModelConfig`, `TrainingConfig`, `SymbolicConfig`) in `src/utils/config.py`
- **Checkpoints saved to**: `./checkpoints/{run_name}/` (triggered by validation metrics, keeps top 3 models)
- **Results saved to**: `./results/{run_name}/` (evaluation artifacts, confusion matrices, rule hit-rates)
- **MLflow runs logged to**: `mlruns/` with safe parameter flattening (handles nested dict keys)
- **Early stopping**: After 10 validation steps without improvement (metric: `val/per`)
- **Monitoring**: `train/loss`, `train/loss_ctc`, `train/loss_ce`, `train/avg_beta` tracked in MLflow

### 5. View Results
```bash
# Inspect evaluation metrics
cat ./results/{run_name}/evaluation_results.json | python -m json.tool

# View visualizations
ls -la ./results/{run_name}/*.png
```

### 6. Evaluate Trained Model
```python
from train import DysarthriaASRLightning
from evaluate import compute_per, compute_wer, evaluate_model

# Load checkpoint (PyTorch Lightning automatically handles state_dict)
model = DysarthriaASRLightning.load_from_checkpoint("checkpoints/epoch=X-step=Y.ckpt")

# Run comprehensive evaluation with visualizations
evaluate_model(
    model=model.model,  # Extract the underlying NeuroSymbolicASR from Lightning wrapper
    dataloader=test_loader,
    device="cuda",
    phn_to_id=dataset.phn_to_id,
    id_to_phn=dataset.id_to_phn,
    results_dir="results/"
)
```
- Metrics computed per speaker, per articulatory class, per dysarthria status
- Outputs confusion matrices, error analysis, phoneme alignment visualizations
- Clinical diagnostic plots saved to `results/figures/`

### Environment Requirements
- Python 3.8+
- Core: `datasets`, `pandas`, `tqdm`, `librosa`, `g2p_en`, `nltk`
- ML: `torch>=2.0.0`, `transformers>=4.30.0`, `torchaudio`
- Training: `pytorch-lightning>=2.0.0`, `mlflow>=2.5.0`
- Evaluation: `jiwer>=3.0.0`, `editdistance>=0.6.0`

## Project-Specific Conventions

### Data Handling Patterns
- **HuggingFace Audio Access**: Use `Audio(decode=False)` when only accessing paths (during manifest creation)
- **Cache directory**: Set via `cache_dir=` or `HF_HOME` env var to keep data local to project
- **Speaker ID extraction**: Parse from filename (e.g., `F01_Session1_0001.wav` → `F01`)
- **Sample matching**: Use speaker-prefixed keys (`f"{speaker}_{filename}"`) to avoid collisions when multiple speakers have same filename

### Phoneme & Articulatory Processing
- Use `g2p_en` for grapheme-to-phoneme conversion (American English, ARPABET)
- **Filter punctuation** from phoneme sequences: `.`, `,`, `?`, `!`
- Store phonemes as **space-separated strings** in manifest (e.g., `"DH IH S IH Z"`)
- **Articulatory classes**: Map phonemes to manner of articulation for explainability
  - **Stops**: P, B, T, D, K, G — often substituted in dysarthria
  - **Fricatives**: F, V, S, Z, SH, ZH, TH, DH, HH — require precise airflow control
  - **Nasals**: M, N, NG — velopharyngeal control
  - **Liquids**: L, R — tongue control
  - **Glides**: W, Y
  - **Vowels & Diphthongs**: Separate classes for vowel analysis

### Symbolic Constraint Matrix & Articulatory Features
- **Matrix construction**: Built in `SymbolicConstraintMatrix` using euclidean distance between articulatory feature vectors
  - Features: `manner` (stop, fricative, nasal, etc.), `place` (bilabial, alveolar, etc.), `voice` (voiced/voiceless)
  - Distance-based: Similar phonemes (e.g., /p/ and /b/, stops with same place) → lower cost
- **Application**: In `ConstraintAggregation`, weighted combination of neural logits and symbolic guidance:
  ```
  logits_constrained = α * logits_neural + (1-α) * (C @ logits_neural)
  α = severity-aware interpolation weight (0 for severe dysarthria → rely more on rules)
  ```
- **Clinical insight**: Dysarthric substitutions often preserve articulatory place/manner (e.g., velars → alveolars, not fricatives)
  - Use this in error analysis: Higher confusion between articulatorily-similar phonemes is clinically expected

### Training Configuration & Multi-Task Learning
- **Config file**: [src/utils/config.py](src/utils/config.py) with dataclasses for `ModelConfig`, `TrainingConfig`, `SymbolicConfig`
- **Loss weighting**: 
  - CTC loss (phoneme alignment): `lambda_ctc=0.7`
  - CE loss (frame-level auxiliary): `lambda_ce=0.3`
  - **Combined loss**: `loss = lambda_ctc * loss_ctc + lambda_ce * loss_ce`
- **VRAM optimization**: 
  - Batch size 2, gradient accumulation 8 steps = effective batch 16
  - Freeze HuBERT encoder layers 0-7 by default; unfreeze after warmup epochs
  - Layer dropout 0.05 for regularization
- **Learning rate schedule**: Cosine annealing with 500-step warmup
- **Checkpoint strategy**: Save best 3 models (early stopping metric: validation PER)
- **Vocabulary architecture**:
  ```python
  <BLANK> = 0      # CTC alignment state (not a phoneme)
  <PAD> = 1        # Batching padding (ignored in loss)
  <UNK> = 2        # Unknown/OOV phonemes
  Phonemes = 3+    # Actual target labels
  ```
- **Loss computation**: 
  - CTC loss: `torch.nn.CTCLoss` on constrained logits (handles variable-length alignment)
  - CE loss: Frame-level cross-entropy on predictions
  - CE class weights: inverse-frequency from manifest; `<BLANK>`/`<PAD>` up-weighted
  - Label padding value `-100` (automatically ignored by both loss functions)
  - Input lengths from logits time dimension (stride), clamped to at least label lengths to avoid CTC inf

### TORGO-Specific Handling
- **Peak normalization**: Dysarthric speakers have variable breath support; normalize waveform by peak amplitude
- **Duration metadata**: Critical for understanding slow speech characteristics
- **RMS energy**: Use for distinguishing signal quality issues from articulation errors
- **Speaker-level splits**: Small dataset (~15 speakers); use speaker-independent train/test splits for valid evaluation
 - **Weighted sampling**: Train split uses WeightedRandomSampler to balance dysarthric vs control

### Code Style
- Use **emoji prefixes** in print statements: 📦 (loading), 🧠 (processing), ✅ (success), ⚠️ (warnings), 📥 (downloading)
- Type hints for function signatures (`Path`, `pd.DataFrame`, `torch.Tensor`)
- Descriptive variable names aligned with research concepts (`is_dysarthric`, `phonemes`, `articulatory_classes`, `rms_energy`)

## Hidden Implementation Patterns (Critical for Modifications)

### 1. Forward Pass & Neural-Symbolic Integration

In [train.py](train.py), the `forward()` method passes **batch status as speaker severity proxy** to activate adaptive beta:
```python
def forward(self, batch: Dict) -> Dict:
    # 'status' = 0 (control) or 1 (dysarthric) → scale to 0-5 severity range
    severity = batch['status'].float() * 5.0
    return self.model(
        input_values=batch['input_values'],
        attention_mask=batch['attention_mask'],
        speaker_severity=severity  # Triggers ConstraintAggregation weighting
    )
```
**Why**: Beta interpolation weight is severity-aware; dysarthric speakers (status=1) get stronger symbolic constraint influence.

### 2. MLflow Parameter Flattening

The `flatten_config_for_mlflow()` function converts nested dataclass configs to flat dicts before logging:
- **Problem**: MLflow rejects tuple keys and nested dicts
- **Solution**: Recursively flatten with "/" separators; skip complex nested structures
- **Impact**: Without this, MLflow logging crashes on `config.symbolic.__dict__` if it contains tuples (e.g., weights)

### 3. Attention Mask vs Label Padding

Both require **different sentinel values**:
- **attention_mask**: Binary mask (1 for valid, 0 for padding) — used by HuBERT encoder
- **labels**: Phoneme IDs, with **-100 for padding** — automatically ignored by both CTC and CE loss functions
```python
# In compute_loss():
label_lengths = (labels != -100).sum(dim=1)  # Counts actual phoneme tokens
input_lengths = torch.full(..., logits.size(1), ...)  # Assumes full sequence valid
```
**Gotcha**: If labels don't use -100, CTC loss treats padding as "<PAD>" token (ID 1) → wrong alignment.

### 4. Phoneme Vocabulary Building

In [dataloader.py](dataloader.py), vocabulary is built **once** at dataset initialization:
```python
self.phn_to_id = {
    '<BLANK>': 0,  # CTC blank (never a true label)
    '<PAD>': 1,    # Padding in batches
    '<UNK>': 2,    # Unknown phonemes
}
# Phonemes start at ID 3
```
**Critical**: The model's output logits shape is `[batch, time, num_phonemes]` where `num_phonemes = len(phn_to_id)`. If manifest has unobserved phonemes, they're mapped to `<UNK>` (ID 2).

### 5. Symbolic Constraint Matrix Weight Initialization

In [model.py](model.py), `ConstraintAggregation` uses learnable blending weight `alpha`:
```python
# logits_constrained = alpha * logits_neural + (1 - alpha) * (constraint_matrix @ logits_neural)
self.alpha = nn.Parameter(torch.tensor(constraint_weight_init))  # e.g., 0.3
```
- **Initialization**: `constraint_weight_init` from config (default 0.3 = favor neural)
- **Learning**: Alpha is updated during training; monitor via `train/avg_beta` in MLflow
- **Interpretation**: Alpha ≈ 0 → rely on symbolic rules; Alpha ≈ 1 → pure neural

### 6. HuBERT Encoder Layer Freezing

For VRAM optimization, [model.py](model.py) freezes early layers by default:
```python
freeze_encoder_layers = [0, 1, 2, 3, 4, 5]  # Freeze first 6 layers (of 12)
for layer_idx in freeze_encoder_layers:
    for param in encoder.layers[layer_idx].parameters():
        param.requires_grad = False
```
**Why**: Lower layers learn generic acoustic features; dysarthria-specific refinement happens in unfrozen layers.

## Key Files & Directories

- [Overview.md](Overview.md): Detailed system architecture and research motivation
- [download.py](download.py): Dataset download with safe path extraction
- [manifest.py](manifest.py): Neuro-symbolic manifest generation with articulatory and robustness features
- [dataloader.py](dataloader.py): PyTorch dataset and dataloader for CTC training with HF streaming
- [train.py](train.py): PyTorch Lightning training orchestrator with multi-task learning
- [model.py](model.py): NeuroSymbolicASR core model (HuBERT + phoneme classifier + symbolic constraints)
- [evaluate.py](evaluate.py): Comprehensive evaluation with PER, WER, confusion matrices, clinical visualizations
- `./data/`: HuggingFace cache and manifest CSV
- `./data/torgo_neuro_symbolic_manifest.csv`: Generated manifest (16K+ samples with metadata)
- `src/utils/config.py`: Configuration dataclasses (ModelConfig, TrainingConfig, SymbolicConfig)

## What's Implemented

- ✅ Data pipeline: TORGO dataset download, neuro-symbolic manifest with articulatory metadata
- ✅ Neural dataset & dataloader: HuBERT feature extraction, CTC-compatible batching, inverse-frequency class weights
- ✅ NeuroSymbolicASR model: HuBERT encoder + phoneme classifier + symbolic constraint layer (β=0.5 init)
- ✅ Training infrastructure: PyTorch Lightning with multi-task learning, weighted sampler, MLflow logging, callbacks
- ✅ Evaluation metrics: PER, WER, phoneme alignment, confusion matrices, per-speaker analysis
- ✅ **Baseline model (baseline_v1)**: Test PER 0.567 on 3,553 samples (3 TORGO speakers)
  - Dysarthric PER: 0.541 | Control PER: 0.575
  - 94.8M params (416K trainable), trained on 10 speakers, best epoch 17/50
  - **Known issue**: High insertion rate (21,290 insertions vs. 376 deletions)

## Next Steps: What's NOT Built Yet

- **Insertion bias mitigation**: High insertion rate (21,290 insertions) needs investigation
  - Analyze CTC blank probabilities and emission patterns
  - Adjust CE loss `<BLANK>` weight dynamically during training
  - Add insertion-specific regularization (e.g., penalize consecutive non-blank predictions)
- **Phoneme-length stratification**: Analyze PER by utterance length to understand dysarthric vs. control performance gap
- **Explainability dashboard**: Interactive visualization of phoneme confusion matrices, rule activations, per-speaker error patterns
- **Model checkpoints**: Publish pretrained baseline & fine-tuned models for TORGO dysarthric/control cohorts
- **Symbolic rule discovery**: Auto-extract dysarthric substitution rules from confusion matrices (e.g., /p/ → /b/ frequency thresholds)
- **Clinical interface**: ONNX export, streaming inference, real-time feedback for speech therapy
- **Ablation studies**: Quantify contribution of neural vs. symbolic components via systematic evaluation

## Common Pitfalls & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `torchcodec.AudioDecoder not subscriptable` | Trying to access audio dict when decoding is enabled | Use `Audio(decode=False)` for path-only access |
| Index mismatch between manifest and HF dataset | Skipped samples (empty transcripts) in manifest | Use speaker-prefixed `sample_id` key for matching |
| Missing attention_mask in batch | Using `processor.pad()` without unpacking correctly | Compute attention_mask manually from sequence lengths |
| Silence in batch due to failed audio loading | Speaker-filename collision or file not found | Verify `sample_id` format matches: `f"{speaker}_{filename}"` |
| Poor training convergence | Audio normalization not aligned | Ensure peak normalization in `_load_audio()` and Wav2Vec2 feature extraction both active |

## Research Context

When implementing new components:
- **Prioritize explainability**: Phoneme-level outputs over word-level black boxes
- **Clinical relevance**: Features should relate to dysarthria characteristics (articulation errors, timing)
- **Small dataset awareness**: TORGO has ~15 speakers — use speaker-independent splits, consider transfer learning
- **Modular design**: Neural and symbolic components should be separable for ablation studies

## External Dependencies

- **TORGO Dataset**: Clinical dysarthric speech corpus (restricted use, research only)
- **g2p_en**: Carnegie Mellon phoneme dictionary for American English
- **HuggingFace Datasets**: Standard interface for audio dataset loading
