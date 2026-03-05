# Copilot Instructions - DysarthriaNSR (March 2026)

## Project Overview
DysarthriaNSR is a neuro-symbolic ASR system for dysarthric speech recognition. It combines HuBERT (self-supervised) neural representations with articulatory-based symbolic constraints for explainable phoneme-level recognition on the TORGO dataset.

**Status**: All B1–B12 bugs fixed; smoke tests passing; `baseline_v5` trained and evaluated (March 6, 2026). LOSO-CV not yet run.
**Latest baseline**: baseline_v5 (March 6, 2026) — beam-search test PER **0.4750**, val/per 0.505 (epoch 28/30), 98.9M total / 66.4M trainable, I/D=0.9×. **Critical finding**: symbolic constraints still hurt (`per_neural=0.305` vs `per_constrained=0.475`, Δ=−0.170). Articulatory accuracy: manner 78.3%, place 79.3%, voice 92.3%. Identical performance to v4 — symbolic layer is the bottleneck.
**Previous baseline**: baseline_v4 (March 5, 2026) — beam-search PER 0.4748, val/per 0.504 (epoch 28/40), 66.4M trainable (67.1%), insertion bias resolved (I/D=0.87×)
**Earlier baseline**: baseline_v3 (March 4, 2026) — val/per 0.574, first valid speaker split; no test eval run
**Reference only**: baseline_v2 — greedy PER 0.215, val/per 0.204 (⚠️ B12 unresolved; data leakage; results inflated)
**Superseded**: baseline_v1 — Test PER 0.567 ± 0.365 (B3 attention mask bug was active)
**Orchestrator**: `run_pipeline.py` is the canonical entry point — use it instead of calling `train.py` / `evaluate.py` directly.

---

## Architecture Map (Data & Code Flow)

```
src/data/download.py                    ← HF Dataset API
    ↓
data/raw/audio/train/unknown_{hash}_{SPEAKER}_{session}_{mic}_{n}.wav
    ↓
src/data/manifest.py (g2p_en + phoneme mapping + articulatory labels)
    speaker = path.name.split('_')[2]   ← TORGO ID; [0] is literal 'unknown'
    ↓
data/processed/torgo_neuro_symbolic_manifest.csv (16,531 rows)
    ↓
src/data/dataloader.py (TorgoNeuroSymbolicDataset + NeuroSymbolicCollator)
    ↓
run_pipeline.py                         ← CANONICAL ENTRY POINT
    ├─ run_training()  →  train.py (DysarthriaASRLightning + NeuroSymbolicASR model)
    │     ├─ src/models/model.py
    │     │    LearnableConstraintMatrix (Proposal P2)
    │     │    SeverityAdapter (Proposal P3, cross-attention)
    │     │    PhonemeClassifier (768→512→|V|)
    │     │    SymbolicConstraintLayer (β·P_constrained + (1-β)·P_neural)
    │     ├─ src/models/losses.py (OrdinalContrastiveLoss, BlankPriorKLLoss, SymbolicKLLoss)
    │     └─ src/utils/config.py (all hyperparameters + TORGO_SEVERITY_MAP)
    └─ run_evaluation() →  evaluate.py (macro-PER, WER, bootstrap CI, stats tests, explainability)
    ↓
results/{run_name}/evaluation_results.json + plots + explanations.json

scripts/generate_figures.py             ← publication-quality figure suite (6 plots)
scripts/smoke_test.py                   ← 7 automated tests (all passing)
```

---

## Implementation Status (Current, March 2026)

### ✅ FULLY IMPLEMENTED (all phases complete in code)
- **Data pipeline**: download.py → manifest.py → dataloader.py (full chain)
- **Model**: HuBERT + SeverityAdapter (P3) + PhonemeClassifier + LearnableConstraintMatrix (P2) + SymbolicConstraintLayer with severity-adaptive β
- **Losses**: CTC + frame-CE (on neural logits) + articulatory CE + BlankPriorKL + OrdinalContrastive + SymbolicKL anchor
- **Training**: PyTorch Lightning, differential LR, LOSO CV infrastructure, ablation CLI (`--ablation`)
- **Evaluation**: macro-speaker PER, bootstrap CI, Welch t-test, Wilcoxon, Holm-Bonferroni, severity correlation, per-speaker breakdown
- **Explainability**: PhonemeAttributor, ExplainableOutputFormatter, ArticulatoryConfusionAnalyzer — wired and producing `explanations.json`
- **Uncertainty**: UncertaintyAwareDecoder (MC Dropout) — wired into `evaluate_model` via `--uncertainty` / `compute_uncertainty=True`
- **SymbolicRuleTracker**: instantiated in `SymbolicConstraintLayer._track_activations`; currently returns 0 activations (logging issue, not model issue)
- **run_pipeline.py**: end-to-end orchestrator; owns training + evaluation lifecycle
- **scripts/smoke_test.py**: 7 automated tests — all passing
- **scripts/generate_figures.py**: 6 publication-quality diagnostic plots
- **src/visualization/experiment_plots.py**: visualization library
- **Severity**: TORGO_SEVERITY_MAP provides continuous [0,5] scores per speaker; `get_speaker_severity()` used in both train and eval forward passes

### ✅ ALL CONFIRMED BUGS FIXED (B1–B11 + B12)

| ID | File | Bug | Status |
|----|------|-----|--------|
| B1 | evaluate.py | `generate_explanations` missing from `evaluate_model` signature | ✅ Fixed |
| B2 | train.py / config.py | `Config.load()` classmethod did not exist | ✅ Fixed |
| B3 | train.py `training_step` | Attention mask stride was `batch_size` (4) instead of CTC stride (320) | ✅ Fixed |
| B4 | train.py `training_step` | `outputs_filtered` dict constructed twice; first silently discarded | ✅ Fixed |
| B5 | train.py `validation_step` + `test_step` | `logits_neural` / `hidden_states` not passed to `compute_loss` | ✅ Fixed |
| B6 | train.py `run_loso` | `config.experiment.run_name` mutated cumulatively each fold | ✅ Fixed |
| B7 | losses.py `SymbolicKLLoss` | KL direction reversed: `F.kl_div(log_learned, log_prior)` | ✅ Fixed |
| B8 | losses.py `OrdinalContrastiveLoss` | `.mean()` on empty tensor → NaN when valid batch = 1 | ✅ Fixed |
| B9 | src/explainability/* | `SymbolicRuleTracker.log_rule_activation` never called by model | ✅ Fixed (wired) |
| B10 | evaluate.py | `wer=0.0` hardcoded in `formatter.format_utterance(...)` | ✅ Fixed |
| B11 | dataloader.py | Double normalization note | ✅ Accepted (intentional for dysarthric variability) |
| B12 | manifest.py line 233 | `path.name.split('_')[0]` → always `'unknown'`; speaker at `[2]` | ✅ Fixed & **manifest regenerated** (March 4, 2026) |

---

## KNOWN ARCHITECTURAL NOTE — Frame-CE Label Alignment

The frame-level CE loss (`_compute_ce_loss`) pads phoneme labels [B, L] to [B, T] with -100 and applies NLLLoss. Since CTC does not provide forced alignment, frame positions 0..L do not correspond to actual phoneme boundaries. This is the root cause of the insertion bias. `BlankPriorKLLoss` (lambda_blank_kl=0.05) is the mitigation, but the attention mask passed to it uses the wrong stride (B3 above).

---
| Dysarthric ≤ Control PER | Dysarthric 0.541 vs Control 0.575 (counter-intuitive) | MEDIUM | Need speaker-stratified analysis; test set only 3 speakers |
| Constraint weight (β) initialization | β converges ~0.5; unclear if optimal | LOW | Ablation study (β ∈ {0.0, 0.3, 0.5, 0.7, 1.0}) recommended |

### ⚠️ CURRENT OPEN ISSUES (March 2026)

| Issue | Impact | Status |
|-------|--------|--------|
| **Symbolic constraints hurt PER** (CRITICAL) | per_neural=0.305 vs per_constrained=0.475 (+57% relative); reproduced in v4 **and** v5 | Run `--ablation neural_only`; set `use_learnable_constraint=False` in config and rerun |
| **LOSO-CV not run** (HIGH) | n=3 test speakers; severity correlation p=0.347 (n.s.) — no statistically valid macro-PER estimate | `python run_pipeline.py --run-name loso_v1 --loso` (~15–22h); fix symbolic constraint first |
| Substitution/Deletion dominance | 13,821 subs + 4,338 del vs 3,752 ins in v5 | Insertion bias resolved, substitution dominance persists |
| `SymbolicRuleTracker` low confidence | `avg_confidence=0.131`; most activations are X→`<BLANK>` deletions | Rules not matching confusion patterns; linked to symbolic constraint issue |
| `PhonemeAttributor.attention_attribution` disabled | Requires CTC forced alignment | Future work |
| `ablation_mode='neural_only'` incomplete | Doesn't disable SeverityAdapter / LearnableConstraintMatrix forward passes | Fix before running ablation |

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
| `run_pipeline.py` | **Canonical entry point** — train + evaluate orchestration | `run_training()`, `run_evaluation()`, `run_auto()` |
| `src/data/download.py` | HF dataset download → local audio files | `TorgoManager`, `setup_environment()` |
| `src/data/manifest.py` | Phoneme extraction + articulatory mapping | `SymbolicProcessor`, `build_manifest()`; speaker at `split('_')[2]` |
| `src/data/dataloader.py` | Dataset + collator + vocabulary building | `TorgoNeuroSymbolicDataset`, `NeuroSymbolicCollator` |
| `src/models/model.py` | HuBERT encoder + phoneme classifier + symbolic layer | `NeuroSymbolicASR`, `SymbolicConstraintLayer`, `SeverityAdapter` |
| `src/models/losses.py` | Multi-task loss functions | `OrdinalContrastiveLoss`, `BlankPriorKLLoss`, `SymbolicKLLoss` |
| `src/models/uncertainty.py` | MC-Dropout uncertainty estimation | `UncertaintyAwareDecoder` |
| `src/utils/config.py` | All hyperparameters (single source of truth) | `Config`, `ProjectPaths`, `TORGO_SEVERITY_MAP`, `normalize_phoneme()` |
| `train.py` | PyTorch Lightning training (no evaluation) | `DysarthriaASRLightning`, `train()`, `flatten_config_for_mlflow()` |
| `evaluate.py` | Evaluation metrics + beam search + explanations | `evaluate_model()`, `BeamSearchDecoder`, `compute_per()` |
| `src/explainability/output_format.py` | Per-utterance explanation JSON | `ExplainableOutputFormatter`, `format_utterance()` |
| `src/explainability/rule_tracker.py` | Symbolic rule activation logging | `SymbolicRuleTracker` |
| `src/visualization/experiment_plots.py` | Publication-quality plot library | `generate_all_plots()` and 6 individual plot functions |
| `scripts/generate_figures.py` | CLI for figure generation | `main()` |
| `scripts/smoke_test.py` | 7 automated end-to-end tests | all passing |
| `RESEARCH_BRIEF.md` | Compressed project context (current state) | — |

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

- [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md): **Compressed project context** — start here for any new session
- [ROADMAP.md](ROADMAP.md): Detailed system architecture and research motivation
- [run_pipeline.py](run_pipeline.py): Canonical entry point (train + evaluate orchestrator)
- [train.py](train.py): PyTorch Lightning training — no evaluation; called by run_pipeline
- [evaluate.py](evaluate.py): Comprehensive evaluation with PER, beam search, explainability, uncertainty
- [src/data/manifest.py](src/data/manifest.py): Manifest generation — speaker from `split('_')[2]`
- [src/data/dataloader.py](src/data/dataloader.py): Dataset + collator; CTC-compatible batching
- [src/models/model.py](src/models/model.py): NeuroSymbolicASR core model
- [src/utils/config.py](src/utils/config.py): Single source of truth for all hyperparameters
- [scripts/generate_figures.py](scripts/generate_figures.py): Publication-quality figure suite
- [scripts/smoke_test.py](scripts/smoke_test.py): 7 automated tests (all passing)
| `data/processed/torgo_neuro_symbolic_manifest.csv`: Generated manifest (16,531 rows; **regenerated March 4, 2026** with correct speaker IDs)

## What's Implemented

- ✅ Data pipeline: TORGO dataset download, neuro-symbolic manifest with articulatory metadata
- ✅ Neural dataset & dataloader: HuBERT feature extraction, CTC-compatible batching, inverse-frequency class weights
- ✅ NeuroSymbolicASR model: HuBERT + SeverityAdapter + LearnableConstraintMatrix + SymbolicConstraintLayer
- ✅ Training infrastructure: PyTorch Lightning with multi-task learning, weighted sampler, MLflow logging, callbacks
- ✅ Evaluation metrics: PER, WER, phoneme alignment, confusion matrices, per-speaker analysis, bootstrap CI
- ✅ Explainability: `explanations.json` per-utterance phoneme error analysis (with `--explain`)
- ✅ Uncertainty estimation: MC-Dropout via `UncertaintyAwareDecoder` (with `--uncertainty`)
- ✅ Orchestrator: `run_pipeline.py` — single entry point for train + evaluate
- ✅ Visualization suite: `scripts/generate_figures.py` — 6 publication-quality plots
- ✅ Automated tests: `scripts/smoke_test.py` — 7/7 passing
- ✅ **Manifest regenerated** (March 4, 2026): B12 fully resolved; speaker IDs correct; confirmed in dataloader batches
- ✅ **Baseline model (baseline_v5)**: beam PER **0.4750**, val/per 0.505 (epoch 28/30); 98.9M params (66.4M trainable); I/D=0.9×; articulatory manner 78.3% / place 79.3% / voice 92.3%. Identical to v4 — symbolic layer confirmed bottleneck.
- ✅ **Baseline model (baseline_v4)**: beam PER 0.4748, val/per 0.504 (epoch 28/40); staged unfreezing; insertion bias resolved (I/D=0.87×); articulatory manner 78.6% / place 79.1% / voice 92.4%
- ✅ **Baseline model (baseline_v3)**: val/per 0.574 (epoch 26), 30 epochs; first run with valid speaker-independent splits
- ⚠️ **Baseline model (baseline_v2)**: greedy PER 0.215, beam PER 0.243, val/per 0.204; **invalid** — B12 caused data leakage. Preserved as historical reference.

## Next Steps: What Remains

### Immediate (blocking LOSO)
1. **Fix `ablation_mode='neural_only'`**: Must fully disable SeverityAdapter + LearnableConstraintMatrix forward passes before ablation results are meaningful
2. **Neural-only ablation** (CRITICAL): `python run_pipeline.py --run-name ablation_neural_only --ablation neural_only` — confirm neural floor (`per_neural≈0.305`)
3. **Disable learnable constraint**: Set `use_learnable_constraint=False` in `src/utils/config.py` and validate on a short run

### Next priority
4. **LOSO-CV** (~15–22h): `python run_pipeline.py --run-name loso_v1 --loso` — run only after symbolic constraint is fixed/ablated; produces statistically valid macro-PER (n=15) and severity correlation
5. **Regenerate figures** for baseline_v5: `python scripts/generate_figures.py --run-name baseline_v5`
6. **Additional ablations**: `--ablation no_art_heads`, `--ablation no_constraint_matrix`

### Long-term
- **Inspect `LearnableConstraintMatrix`**: Audit weight initialization; compare data-driven confusion patterns vs. hard-coded symbolic rules
- **CTC forced alignment**: enables `PhonemeAttributor.attention_attribution`
- **Clinician dashboard**: ONNX export, streaming inference
- **Domain adaptation**: non-TORGO dysarthria datasets

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
- Data flow: `src/data/download.py` (HF TORGO → data/raw/audio) → `src/data/manifest.py` (creates `data/processed/torgo_neuro_symbolic_manifest.csv`; speaker from `split('_')[2]`) → `src/data/dataloader.py` (`TorgoNeuroSymbolicDataset`, `NeuroSymbolicCollator`) → `run_pipeline.py` → `train.py` / `evaluate.py` → `results/` and `mlruns/`.
- **`run_pipeline.py` is the canonical entry point.** `train.py` no longer calls `evaluate_model()` internally; `run_pipeline.py` owns that call.
- Model core lives in `src/models/model.py` (`NeuroSymbolicASR`, `SymbolicConstraintLayer`), which fuses HuBERT logits with articulatory rules; `normalize_phoneme()` in `src/utils/config.py` is used everywhere for stress-stripping.
- Central config + paths are in `src/utils/config.py` (`ProjectPaths`, `Config`, `ModelConfig`, `TrainingConfig`, etc). Use it as the single source of truth for defaults.
- Visualization: `scripts/generate_figures.py` calls `src/visualization/experiment_plots.py` which reads `results/{run}/evaluation_results.json` + `explanations.json`.

## Critical Workflows (Commands)
- Download TORGO and extract audio: `python src/data/download.py` (writes under `data/raw/` and `data/processed/raw_extraction_map.csv`).
- Build manifest with phonemes/articulatory classes: `python src/data/manifest.py` → `data/processed/torgo_neuro_symbolic_manifest.csv`. **Regenerated March 4, 2026** — B12 fully resolved.
- **Full pipeline (canonical)**:
  ```bash
  python run_pipeline.py --run-name experiment_v5
  ```
- **Evaluation-only (reuse checkpoint)**:
  ```bash
  # Evaluate baseline_v4 with beam search, explainability, uncertainty
  python run_pipeline.py --run-name baseline_v4 --skip-train --explain --uncertainty --beam-search --beam-width 25
  ```
- **Neural-only ablation (critical — run next)**:
  ```bash
  python run_pipeline.py --run-name ablation_neural_only --ablation neural_only
  ```
- **Smoke test (5 batches)**:
  ```bash
  python run_pipeline.py --run-name smoke_check --smoke-test
  ```
- **Generate publication figures**:
  ```bash
  python scripts/generate_figures.py --run-name baseline_v4
  # Compare multiple runs:
  python scripts/generate_figures.py --run-name baseline_v4 --compare ablation_neural_only no_art_heads
  ```
- **Run unit tests**:
  ```bash
  python scripts/smoke_test.py
  ```
- Train directly (bypasses orchestrator): `python train.py --run-name ...` (evaluation NOT run automatically).
- Evaluate directly: `python evaluate.py` (stub only; use `run_pipeline.py` instead).

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
- For new plots, add a function to `src/visualization/experiment_plots.py` following the house-style helper pattern, and call it from `generate_all_plots()`.
- For data quality or bulk diagnostics, pattern after `src/visualization/diagnostics.py`.

