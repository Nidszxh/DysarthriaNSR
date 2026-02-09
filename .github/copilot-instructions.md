# Copilot Instructions - DysarthriaNSR

## Project Map (What Talks to What)
- Data flow: `download.py` (HF TORGO -> data/raw/audio) -> `src/data/manifest.py` (creates `data/processed/torgo_neuro_symbolic_manifest.csv`) -> `src/data/dataloader.py` -> `train.py` -> `evaluate.py` (library functions) -> `results/` and `mlruns/`.
- Model core: `src/models/model.py` (HuBERT + phoneme head + symbolic constraint layer). Configs live in `src/utils/config.py`.

## Non-Obvious Conventions
- Phonemes are ARPABET and stress-agnostic: always pass through `normalize_phoneme()` before comparison or vocab building.
- Special tokens are fixed: `<BLANK>`=0, `<PAD>`=1, `<UNK>`=2; label padding uses `-100` (ignored by CTC/CE).
- `NeuroSymbolicCollator` computes `input_lengths` as `len(audio)//320` (CTC stride) and creates `attention_mask` manually.
- `status` is a dysarthria flag (0/1) and is scaled to severity 0-5 for adaptive beta in `DysarthriaASRLightning.forward()`.

## Model and Training Behavior
- HuBERT base (`facebook/hubert-base-ls960`) with gradient checkpointing enabled.
- Freezing: feature extractor always frozen; encoder layers from `ModelConfig.freeze_encoder_layers` stay frozen after warmup.
- Symbolic constraints: `SymbolicConstraintLayer` blends neural probs with a constraint matrix built from articulatory similarity and explicit substitution rules; beta is learnable and clamped to 0.8.
- Losses: CTC + frame-level CE + articulatory CE (manner/place/voice). Weights are in `TrainingConfig`.
- Class balancing: inverse-frequency phoneme weights for CE + `WeightedRandomSampler` to balance dysarthric vs control.

## Critical Workflows
- Download and extract: `python download.py` (writes under `data/raw/` and `data/processed/raw_extraction_map.csv`).
- Build manifest: `python src/data/manifest.py` (no CLI args; outputs `data/processed/torgo_neuro_symbolic_manifest.csv`).
- Train: `python train.py` or `python train.py --run-name my_run` (MLflow to `mlruns/`, checkpoints to `checkpoints/{run_name}`).
- Evaluation runs inside `train.py` after fitting via `evaluate_model()`; `evaluate.py` itself is for utilities/tests.

## Integration Points
- HF dataset: `abnerh/TORGO-database` (use `Audio(decode=False)` in `manifest.py` for path-only access).
- Audio paths are matched by `{speaker}_{hash}_{original_name}` (see `download.py` + `manifest.py`).
- Results and diagnostics: `evaluate.py` + `src/visualization/diagnostics.py`.
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

