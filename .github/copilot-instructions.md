# Copilot Instructions - DysarthriaNSR (February 2026)

## Project Overview
DysarthriaNSR is a neuro-symbolic ASR system for dysarthric speech recognition. It combines HuBERT (self-supervised) neural representations with articulatory-based symbolic constraints for explainable phoneme-level recognition on the TORGO dataset.

**Status**: ✅ Fully operational end-to-end (download → manifest → training → evaluation)  
**Latest baseline**: Test PER 0.567 ± 0.365 on 3,553 samples (3 TORGO speakers)  
**Known issue**: High insertion rate (21K insertions) indicates CTC blank suppression; mitigation in progress.

---

## Architecture Map (Data & Code Flow)

```
src/data/download.py                    ← HF Dataset API
    ↓
data/raw/audio/{speaker}_{hash}_{name}.wav
    ↓
src/data/manifest.py (g2p_en + phoneme mapping)
    ↓
data/processed/torgo_neuro_symbolic_manifest.csv (16.5k rows)
    ↓
src/data/dataloader.py (TorgoNeuroSymbolicDataset + NeuroSymbolicCollator)
    ↓
train.py (DysarthriaASRLightning + NeuroSymbolicASR model)
    ├─ src/models/model.py (HuBERT + phoneme classifier + symbolic layer)
    ├─ src/utils/config.py (all hyperparameters + ProjectPaths)
    └─ MLflow logging → mlruns/{experiment}/{run_id}/
    ↓ (after training completes)
evaluate.py (PER/WER + confusion matrices + bootstrap CI)
    ↓
results/{run_name}/evaluation_results.json + confusion_matrix.png
```

---

## Implementation Status (What's Done, Not Done)

### ✅ IMPLEMENTED
- **Data pipeline**: download.py → manifest.py → dataloader.py (full chain)
- **Model architecture**: HuBERT encoder (12 frozen/fine-tuned layers) + phoneme classifier + symbolic constraint layer
- **Training**: PyTorch Lightning with MLflow integration; multi-task learning (CTC + CE + articulatory)
- **Evaluation**: PER/WER with bootstrap CI, confusion matrices, per-speaker/per-phoneme breakdown
- **Beam search decoding**: CTC prefix beam search (width=10) for accurate phoneme sequences
- **Configuration management**: All hyperparams in src/utils/config.py (dataclass-based; RTX 4060 optimized)

### ⚠️ KNOWN ISSUES
| Issue | Symptom | Severity | Status |
|-------|---------|----------|--------|
| CTC insertion bias | 21K insertions vs. 376 deletions | HIGH | Diagnostic plan ready; blank_priority_weight testing pending |
| Dysarthric ≤ Control PER | Dysarthric 0.541 vs Control 0.575 (counter-intuitive) | MEDIUM | Need speaker-stratified analysis; test set only 3 speakers |
| Constraint weight (β) initialization | β converges ~0.5; unclear if optimal | LOW | Ablation study (β ∈ {0.0, 0.3, 0.5, 0.7, 1.0}) recommended |

### 🚀 NOT IMPLEMENTED (Future)
- Insertion-bias mitigation (in design phase)
- Multi-speaker adaptation (speaker embedding + speaker-specific β)
- Real-time streaming inference (ONNX export)
- Clinician-facing diagnostic dashboard
- Domain adaptation to non-TORGO dysarthria datasets

---

## Critical Coding Conventions

### 1. Phoneme Handling
**ALWAYS normalize stress**: `normalize_phoneme(phoneme)` before any comparison or vocab building
```python
from src.utils.config import normalize_phoneme
normalize_phoneme("AH0") → "AH"  # Remove stress digit
normalize_phoneme("IY1") → "IY"
```
Why: Manifest uses ARPABET (phonemes with 0/1/2 stress), but model vocab is stress-agnostic



### 2. Vocabulary IDs (NEVER change these)
```python
<BLANK> = 0    # CTC blank token (alignment only, never a label)
<PAD>   = 1    # Padding token (for variable-length batching)
<UNK>   = 2    # Unknown/OOV phonemes (fallback)
IY, AE, ...    = 3–46  # Actual ARPABET phonemes
```
These are built once in `TorgoNeuroSymbolicDataset._build_vocabularies()` and used consistently throughout.

### 3. Label Padding Sentinel
Labels use `-100` for padding; automatically ignored by CTC/CE loss:
```python
# In NeuroSymbolicCollator:
labels = torch.nn.functional.pad(labels, (0, max_len - labels.size(1)), value=-100)

# In compute_loss():
label_lengths = (labels != -100).sum(dim=1)  # Count actual phoneme tokens
```
**Never use 0 or 1 for padding labels** — those are valid token IDs!

### 4. Input Lengths for CTC
CTC loss requires `input_lengths` = number of frames (at ~50 Hz HuBERT output rate):
```python
# Feature extraction: HuBERT uses 320-sample stride (20ms @ 16kHz)
input_lengths = logits.size(1)  # Direct from model logits time dimension

# Validation: input_lengths ≥ label_lengths (CTC requirement)
if (input_lengths < label_lengths).any():
    print("⚠️ CTC error: input too short for labels; dropping sample")
    # Drop sample to avoid CTC loss = inf
```

### 5. Audio Normalization Pipeline
```python
# In TorgoNeuroSymbolicDataset._load_audio():
1. Load WAV @ 16 kHz
2. Peak-normalize: audio /= np.abs(audio).max()  # Critical for dysarthric variability
3. Truncate/pad to max_audio_length (6.0s default)
4. HuBERT processor handles feature extraction (not manual mel-spec)
```

### 6. Speaker Severity Scaling
```python
# In DysarthriaASRLightning.forward():
status = batch['status']  # 0 (control) or 1 (dysarthric)
severity = status.float() * 5.0  # Scale 0→5.0 for adaptive beta
# Passed to model for severity-aware constraint weighting
```

---

## Config File Structure (Single Source of Truth)

All hyperparameters live in **`src/utils/config.py`**. Never hardcode values in train.py or dataloader.py.

### ProjectPaths (auto-created)
```python
root / data / {raw, processed, external}
    / checkpoints / {run_name} / last.ckpt
    / results / {run_name} / evaluation_results.json
    / mlruns / {exp_id} / {run_id}
```

### ModelConfig
```python
freeze_encoder_layers: List[int] = [0, 1, ..., 9]  # First 10/12 frozen (RTX 4060 VRAM)
constraint_weight_init: float = 0.05  # β initialization (learnable, clamped <0.8)
```

### TrainingConfig  
```python
precision: str = "bf16-mixed"  # BF16 for Ada (stable); FP16 for older cards
batch_size: int = 4
gradient_accumulation_steps: int = 8  # Effective batch = 32
learning_rate: float = 3e-5  # OneCycleLR with 5% warmup
lambda_ctc: float = 0.8
lambda_ce: float = 0.2
lambda_articulatory: float = 0.1
blank_priority_weight: float = 1.5  # Blank class weight (for insertion mitigation)
max_epochs: int = 30
encoder_warmup_epochs: int = 3  # Unfreeze after 3 epochs
```

### DataConfig
```python
max_audio_length: float = 6.0  # seconds (99% coverage)
split_strategy: str = "speaker_stratified"  # No speaker overlap between train/val/test
```

### SymbolicConfig
```python
substitution_rules: Dict[Tuple[str,str], float] = {
    ('B', 'P'): 0.85,  # Devoicing
    ('D', 'T'): 0.82,
    # ... 20 total rules (hard-coded evidence-based)
}
manner_weight: float = 0.4
place_weight: float = 0.35
voice_weight: float = 0.25
```

---

## Model Behavior & Training Dynamics

### Forward Pass
```python
def forward(self, batch: Dict) -> torch.Tensor:
    # Audio normalization
    audio = batch['input_values']  # [batch, num_frames]
    
    # HuBERT encoding (first 10 layers frozen for VRAM)
    encoder_out = self.hubert(input_values=audio, ...)  # [batch, frames, 768]
    
    # Phoneme classification
    logits = self.phoneme_classifier(encoder_out)  # [batch, frames, 44]
    
    # Symbolic constraint layer
    severity = batch['status'].float() * 5.0  # 0 or 5.0
    logits_final = self.constraint_layer(logits, severity)  # Blends neural + symbolic
    
    # CTC output (logits_final used for loss)
    return {'logits': logits_final, 'encoder_out': encoder_out}
```

### Multi-Task Loss Computation
```python
loss_ctc = CTCLoss(logits, labels, input_lengths, label_lengths)
loss_ce = CrossEntropyLoss(logits, labels, weight=phoneme_weights, ignore_index=-100)
loss_articulatory = (CE(manner_logits, manner_labels) + 
                     CE(place_logits, place_labels) + 
                     CE(voicing_logits, voicing_labels)) / 3
loss_total = (0.8 * loss_ctc + 0.2 * loss_ce + 0.1 * loss_articulatory)
```

### Class Weighting
- **Phoneme weights**: `inverse_frequency` (rare phonemes weighted higher)
- **Blank weight**: `blank_priority_weight = 1.5` × base weight
- **Speaker balance**: `WeightedRandomSampler` with equal speaker probability

### Encoder Freezing Schedule
- **Epochs 0–2**: First 10 layers frozen; fast convergence
- **Epoch 3+**: Unfreeze all layers; continue training with reduced LR
- **Rationale**: VRAM constraint (5.5 GB vs. 8.2 GB unfrozen); lower layers learn generic acoustic features

---

## Debugging Checklist

### Audio Loading Issues
```python
# Verify audio paths exist
import pandas as pd
df = pd.read_csv("data/processed/torgo_neuro_symbolic_manifest.csv")
assert df['path'].apply(lambda p: Path(p).exists()).all(), "Missing audio files"

# Check peak normalization
from src.data.dataloader import TorgoNeuroSymbolicDataset
ds = TorgoNeuroSymbolicDataset("data/processed/torgo_neuro_symbolic_manifest.csv")
sample = ds[0]
print(f"Max input value: {sample['input_values'].max():.3f}")  # Should be ≈ max HuBERT output
```

### Label/Logits Mismatch
```python
# Verify padding works correctly
batch = next(iter(dataloader))
print(f"Input shape: {batch['input_values'].shape}")  # [batch, frames]
print(f"Logits shape: {model_out['logits'].shape}")  # [batch, frames, 44]
print(f"Labels shape: {batch['labels'].shape}")  # [batch, max_seq_len]
print(f"Unique label values: {batch['labels'].unique()}")  # Should include -100 for padding
```

### CTC Loss NaN
```python
# If CTC loss = NaN, likely input_lengths < label_lengths
batch = next(iter(dataloader))
logits = model(batch)
input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long)
label_lengths = (batch['labels'] != -100).sum(dim=1)
assert (input_lengths >= label_lengths * 0.5).all(), "Inputs too short for labels"
```

### Phoneme Vocab Size Mismatch
```python
# Model output should match vocab size
print(f"Vocab size: {len(ds.phn_to_id)}")  # Should be 44+3=47
print(f"Logits shape: {logits.shape}")  # [..., 47]
assert logits.shape[-1] == len(ds.phn_to_id), "Vocab size mismatch"
```

---

## MLflow Logging & Results

Training automatically:
1. **Logs hyperparameters** (flattened from Config dataclasses)
2. **Logs metrics** (train/loss, val/per, train/beta per epoch)
3. **Saves artifacts** (confusion matrix PNG, evaluation JSON)
4. **Checkpoints** best 2 models to `checkpoints/{run_name}/`

View results:
```bash
# MLflow UI
mlflow ui -p 5000  # Open http://localhost:5000

# Results JSON
cat results/{run_name}/evaluation_results.json | python -m json.tool
```

---

## Common Development Patterns

### Adding a new metric
1. Implement standalone function in `evaluate.py`
2. Test with sample predictions
3. Integrate into `evaluate_model()` function
4. Log to MLflow

### Changing a hyperparameter
1. Edit `src/utils/config.py` (not train.py)
2. Use `Config()` to load; let dataclasses handle defaults
3. Parameter auto-logged to MLflow

### Debugging a training divergence
```bash
# Check MLflow logs
tensorboard --logdir mlruns

# Inspect checkpoint
python -c "import torch; ckpt = torch.load('checkpoints/{run}/last.ckpt'); print(ckpt.keys())"

# Verify dataloader
python -c "from src.data.dataloader import TorgoNeuroSymbolicDataset; ds = TorgoNeuroSymbolicDataset(...); print(ds[0].keys())"
```

---

## File Locations Quick Reference

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `src/data/download.py` | HF dataset download → local audio files | `TorgoManager`, `setup_environment()` |
| `src/data/manifest.py` | Phoneme extraction + articulatory mapping | `SymbolicProcessor`, `build_manifest()` |
| `src/data/dataloader.py` | Dataset + collator + vocabulary building | `TorgoNeuroSymbolicDataset`, `NeuroSymbolicCollator` |
| `src/models/model.py` | HuBERT encoder + phoneme classifier + symbolic layer | `NeuroSymbolicASR`, `SymbolicConstraintLayer`, `ArticulatoryFeatureEncoder` |
| `src/utils/config.py` | All hyperparameters (single source of truth) | `Config`, `ProjectPaths`, `ModelConfig`, `TrainingConfig`, `DataConfig`, `SymbolicConfig` |
| `train.py` | PyTorch Lightning training orchestrator | `DysarthriaASRLightning`, `flatten_config_for_mlflow()` |
| `evaluate.py` | Evaluation metrics + beam search decoder | `evaluate_model()`, `BeamSearchDecoder`, `compute_per()` |

---

## Research Context & Clinical Motivation

**Dysarthria characteristics modeled in symbolic layer**:
- **Devoicing**: Voiced consonants → voiceless (e.g., B→P, D→T, G→K)
- **Fronting**: Velar → alveolar/dental (e.g., K→T, NG→N)
- **Liquid gliding**: R/L → W (reduced motor control)
- **Vowel centralization**: Reduced vowel space (IY→IH, AH centralization)

**Why neuro-symbolic?**
- Pure neural models treat dysarthric errors as noise (black-box)
- Symbolic rules encode clinical phonology (SLPs understand constraints)
- Hybrid approach: neural flexibility + symbolic interpretability

---

## Before Committing Code

- ✅ All functions have docstrings (Args/Returns/Raises)
- ✅ Type hints on function signatures (avoid `Any`)
- ✅ No hardcoded paths (use `ProjectPaths` from config.py)
- ✅ Phonemes normalized via `normalize_phoneme()`
- ✅ MLflow logging for new metrics
- ✅ Tested on RTX 4060 (8GB VRAM target)
- ✅ Labels use `-100` sentinel for padding (not 0 or 1)

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

- [ROADMAP.md](ROADMAP.md): Detailed system architecture and research motivation
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


## Project Map (What Talks to What)
- Data flow: `download.py` (HF TORGO -> data/raw/audio) -> `src/data/manifest.py` (creates `data/processed/torgo_neuro_symbolic_manifest.csv`) -> `src/data/dataloader.py` (`TorgoNeuroSymbolicDataset`, `NeuroSymbolicCollator`) -> `train.py` (`DysarthriaASRLightning`) -> `evaluate.py` (PER/WER + analyses) -> `results/` and `mlruns/`.
- Model core lives in `src/models/model.py` (`NeuroSymbolicASR`, `SymbolicConstraintLayer`), which fuses HuBERT logits with articulatory rules; `normalize_phoneme()` in `src/utils/config.py` is used everywhere for stress-stripping.
- Central config + paths are in `src/utils/config.py` (`ProjectPaths`, `Config`, `ModelConfig`, `TrainingConfig`, etc). Use it as the single source of truth for defaults.

## Critical Workflows (Commands)
- Download TORGO and extract audio: `python download.py` (writes under `data/raw/` and `data/processed/raw_extraction_map.csv`).
- Build manifest with phonemes/articulatory classes: `python src/data/manifest.py` -> `data/processed/torgo_neuro_symbolic_manifest.csv`.
- Train: `python train.py` (PyTorch Lightning + MLflow logging to `mlruns/`).
- Evaluate: `python evaluate.py` (writes to `results/`, supports beam search in `evaluate.py`).

## Project-Specific Conventions
- Phonemes are ARPABET and stress-agnostic: always pass through `normalize_phoneme()` before comparisons or vocab building.
- Special tokens are fixed: `<BLANK>` id 0, `<PAD>` id 1, `<UNK>` id 2 (see `src/data/dataloader.py`).
- VRAM-aware defaults (8GB target) are encoded in `src/utils/config.py` (mixed precision, short max audio length, gradient accumulation). Keep changes aligned with those constraints.
- Training uses CTC + frame-level CE (+ optional articulatory CE) and guards against invalid CTC lengths in `train.py`.

## Integration Points / External Dependencies
- Hugging Face datasets: `abnerh/TORGO-database` in `download.py` and `src/data/manifest.py`.
- HuBERT encoder from `transformers` (`facebook/hubert-base-ls960`) in `src/models/model.py`.
- MLflow tracking in `train.py` with filesystem URI from `ProjectPaths().mlruns_dir`.

## Examples to Follow
- For adding model behavior, mirror the neuro-symbolic fusion path in `src/models/model.py` and keep `SymbolicConstraintLayer` semantics intact.
- For new metrics or analyses, follow the structure in `evaluate.py` (bootstrap CI, per-speaker breakdowns, beam search).
- For data quality or plots, pattern after `src/visualization/diagnostics.py`.

