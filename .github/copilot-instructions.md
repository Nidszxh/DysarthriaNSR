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

5. **Neural Component** (TBD):
   - SSL encoder (wav2vec 2.0 base, facebook/wav2vec2-base-960h) processes audio → CTC phoneme predictions
   - Self-supervised learning reduces need for large labeled datasets

6. **Symbolic Constraint Layer** (TBD):
   - Rules for common dysarthric substitutions (e.g., /p/ → /b/, /t/ → /d/)
   - Articulatory similarity constraints guide neural predictions
   - Leverage `articulatory_classes` column for fine-grained error analysis

7. **Explainability Module** (TBD):
   - Phoneme-level error attribution with articulatory class breakdown
   - Rule activation tracking for clinical interpretability

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

### Environment Requirements
- Python 3.8+
- Core: `datasets`, `pandas`, `tqdm`, `librosa`, `g2p_en`, `nltk`
- ML: `torch`, `transformers`, `torchaudio`
- Evaluation: `jiwer` (for WER/PER metrics)

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

### CTC Training Specifics (Critical)
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

## Next Steps: What's NOT Built Yet

- **Training script**: PyTorch Lightning or HuggingFace Trainer with CTC loss, learning rate scheduling, checkpoint management
- **Symbolic constraint layer**: Rule-based post-processing to enforce dysarthric substitution patterns
- **Evaluation metrics**: WER, PER, confidence-based filtering, per-speaker and per-articulatory-class analysis
- **Explainability dashboard**: Phoneme confusion matrices, rule activation heatmaps, per-speaker error patterns
- **Model checkpoints**: Pretrained baselines and fine-tuned models for clinical deployment

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
