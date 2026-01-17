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

3. **Neural Dataset & Dataloader** ([dataloader.py](dataloader.py)):
   - `TorgoNeuroSymbolicDataset`: Streams audio from HF dataset matched by `speaker_filename` key
   - Wav2Vec2 feature extraction (16kHz, zero-mean-unit-variance normalized)
   - Peak normalization to handle TORGO's variable breath support
   - Returns: audio features, phoneme IDs, and articulatory metadata
   - **CTC-ready**: Vocabulary includes `<BLANK>` (ID 0), `<PAD>` (ID 1), `<UNK>` (ID 2), then phonemes (ID 3+)

4. **Collation & Batching** ([dataloader.py](dataloader.py)):
   - `NeuroSymbolicCollator`: Pads audio and phoneme sequences with proper attention masks
   - Audio padding value: 0.0 (silence)
   - Label padding value: -100 (ignored by CTC loss)
   - Computes attention mask manually (1 where signal exists, 0 for padding)

5. **Neural Component** ([model.py](model.py)):
   - HuBERT encoder (facebook/hubert-base-ls960) extracts 768-dim self-supervised representations
   - Projection layer reduces to `hidden_dim` (512), followed by phoneme classifier
   - CTC loss aligns variable-length audio to phoneme sequences
   - Supports selective layer freezing for VRAM optimization (freeze first 6 encoder layers by default)

6. **Symbolic Constraint Layer** ([model.py](model.py)):
   - `SymbolicConstraintMatrix`: Encodes articulatory similarity via distance between phoneme features
   - `ConstraintAggregation`: Applies learned weighted matrix to neural logits: `logits_constrained = α * logits_neural + (1-α) * C @ logits_neural`
   - Dysarthria-aware weighting: Higher dysarthric severity → stronger constraint influence
   - Covers stops, fricatives, nasals, liquids, glides, vowels, diphthongs with phonetic properties (manner, place, voicing)

7. **Training Pipeline** ([train.py](train.py)):
   - Multi-task learning: CTC loss (phoneme) + focal loss (dysarthria classification) + KL constraint loss
   - MLflow tracking with safe parameter flattening for nested configs
   - EarlyStopping, ModelCheckpoint, LearningRateMonitor callbacks
   - Gradient accumulation (batch_size=2, accumulation_steps=12) for RTX 4060 8GB VRAM constraint
   - Learning rate: 5e-5, cosine scheduler with warmup, label smoothing 0.1

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
python manifest.py --data-dir ./data --out ./data/torgo_neuro_symbolic_manifest.csv
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
from dataloader import TorgoNeuroSymbolicDataset, NeuroSymbolicCollator
from torch.utils.data import DataLoader

dataset = TorgoNeuroSymbolicDataset(
    manifest_path="./data/torgo_neuro_symbolic_manifest.csv",
    processor_id="facebook/wav2vec2-base-960h",
    sampling_rate=16000,
    use_hf_dataset=True,
    hf_cache_dir="./data"
)

collator = NeuroSymbolicCollator(dataset.processor)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collator
)

# Each batch returns:
# - input_values: [batch_size, max_time] - Wav2Vec2 features
# - attention_mask: [batch_size, max_time] - 1 where valid, 0 for padding
# - labels: [batch_size, max_phonemes] - Phoneme IDs (-100 for padding, ignored by CTC loss)
# - status: [batch_size] - Dysarthric (1) vs Control (0)
# - speakers: [batch_size] - Speaker IDs for analysis
```

### 4. Start Training
```bash
python train.py --config config.yaml  # Uses MLflow for experiment tracking
# Or with overrides:
python train.py --config config.yaml --trainer.max_epochs 10 --training.batch_size 4
```
- Checkpoints saved to `./checkpoints/` (triggered by validation metrics)
- MLflow runs logged to `mlruns/` with safe parameter flattening
- Early stopping after 5 validation epochs without improvement

### 5. Evaluate Trained Model
```python
from train import TrainedModel
from evaluate import compute_per, compute_wer, visualize_confusion_matrix

# Load checkpoint and compute metrics per speaker, articulatory class, dysarthria status
model = TrainedModel.load_from_checkpoint("checkpoints/best_model.ckpt")
per_scores, confusion_mats = model.evaluate(test_loader)
visualize_confusion_matrix(confusion_mats, save_path="results/")
```

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
- **Config file**: [config.py](config.py) with dataclasses for ModelConfig, TrainingConfig, SymbolicConfig
- **Loss weighting**: 
  - CTC loss (phoneme alignment): 1.0
  - Focal loss (dysarthria classification): 0.1 (handles class imbalance)
  - KL constraint loss (symbolic guidance): 0.05
- **VRAM optimization**: 
  - Batch size 2, gradient accumulation 12 steps = effective batch 24
  - Freeze HuBERT encoder layers 0-5 (reduce parameters, VRAM)
  - Layer dropout 0.05 for regularization
- **Learning rate schedule**: Cosine annealing with 500-step warmup
- **Checkpoint strategy**: Save best model (early stopping metric: validation PER)
- **Vocabulary architecture**:
  ```python
  <BLANK> = 0      # CTC alignment state (not a phoneme)
  <PAD> = 1        # Batching padding (ignored in loss)
  <UNK> = 2        # Unknown/OOV phonemes
  Phonemes = 3+    # Actual target labels
  ```
- **Loss computation**: Use `torch.nn.CTCLoss` with reduction on frames (not samples)
- **Label padding**: Use `-100` for CTC-compatible loss functions to ignore padding
- **Input length**: Computed from attention_mask; output length from label sequence length

### TORGO-Specific Handling
- **Peak normalization**: Dysarthric speakers have variable breath support; normalize waveform by peak amplitude
- **Duration metadata**: Critical for understanding slow speech characteristics
- **RMS energy**: Use for distinguishing signal quality issues from articulation errors
- **Speaker-level splits**: Small dataset (~15 speakers); use speaker-independent train/test splits for valid evaluation

### Code Style
- Use **emoji prefixes** in print statements: 📦 (loading), 🧠 (processing), ✅ (success), ⚠️ (warnings), 📥 (downloading)
- Type hints for function signatures (`Path`, `pd.DataFrame`, `torch.Tensor`)
- Descriptive variable names aligned with research concepts (`is_dysarthric`, `phonemes`, `articulatory_classes`, `rms_energy`)

## Key Files & Directories

- [Overview.md](Overview.md): Detailed system architecture and research motivation
- [download.py](download.py): Dataset download with safe path extraction
- [manifest.py](manifest.py): Neuro-symbolic manifest generation with articulatory and robustness features
- [dataloader.py](dataloader.py): PyTorch dataset and dataloader for CTC training with HF streaming
- `./data/`: HuggingFace cache and manifest CSV
- `./data/torgo_neuro_symbolic_manifest.csv`: Generated manifest (16K+ samples with metadata)

## What's Implemented

- ✅ Data pipeline: TORGO dataset download, neuro-symbolic manifest with articulatory metadata
- ✅ Neural dataset & dataloader: HuBERT feature extraction, CTC-compatible batching
- ✅ NeuroSymbolicASR model: HuBERT encoder + phoneme classifier + symbolic constraint layer
- ✅ Training infrastructure: PyTorch Lightning with multi-task learning, MLflow logging, callbacks
- ✅ Evaluation metrics: PER (phoneme error rate), WER, phoneme alignment, confusion matrices

## Next Steps: What's NOT Built Yet

- **Explainability dashboard**: Interactive visualization of phoneme confusion matrices, rule activations, per-speaker error patterns
- **Model checkpoints**: Pretrained baseline & fine-tuned models for TORGO dysarthric/control cohorts
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
