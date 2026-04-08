# docs/data.md — Data Pipeline

> Merges and supersedes the former `docs/dataset.md` and `docs/pipeline.md`. Cross-references: [training.md](training.md) for how data flows into the model.

---

## TORGO Corpus

### Origin and Access

TORGO is a dysarthric speech corpus from the University of Toronto (Rudzicz et al., 2012), containing speech from individuals with cerebral palsy and ALS paired with matching controls. Access is available via HuggingFace Hub (`abnerh/TORGO-database`) for research use.

### Speaker Table

| Speaker | Type | Severity score | Gender |
|---|---|---|---|
| F01 | Dysarthric | 4.90 | F |
| F03 | Dysarthric | 4.65 | F |
| F04 | Dysarthric | 3.05 | F |
| M01 | Dysarthric | 4.90 | M |
| M02 | Dysarthric | 4.78 | M |
| M03 | Dysarthric | 4.59 | M |
| M04 | Dysarthric | 2.85 | M |
| M05 | Dysarthric | 2.10 | M |
| FC01 | Control | 0.00 | F |
| FC02 | Control | 0.00 | F |
| FC03 | Control | 0.00 | F |
| MC01 | Control | 0.00 | M |
| MC02 | Control | 0.00 | M |
| MC03 | Control | 0.00 | M |
| MC04 | Control | 0.00 | M |

The complete `TORGO_SEVERITY_MAP` dict is defined in `src/utils/config.py` and is the single source of truth for all severity scores. Formula: `severity = (1 - intelligibility/100) * 5.0`. Control severity is locked at 0.0; the continuous per-speaker map is used for dysarthric speakers (C7 fix — replaced coarse binary status × 5.0).

### Known Dataset Issues

| Issue | Detail |
|---|---|
| Pilot / duplicate utterances | TORGO includes repeated prompts across sessions. No de-duplication applied; repeated transcripts are distinct utterances from different recording conditions. |
| FC02 reduced recordings | FC02 may have fewer recordings than other control speakers. Verify with `df[df['speaker']=='FC02']` after manifest generation. |
| Inverted dysarthric/control PER on small splits | Single-split test sets of ~2 speakers produce unreliable stratified statistics. Control PER > dysarthric PER is a sampling artifact, not a data issue. |
| Control severity tied at 0.0 | All 7 control speakers receive severity=0.0, forming a single bucket in ordinal contrastive loss. Mitigated by using continuous `TORGO_SEVERITY_MAP` scores for dysarthric speakers. |

---

## Setup Commands

```bash
# Step 1: Download TORGO audio from HuggingFace
python src/data/download.py
# Output: data/raw/audio/{split}/{speaker}_{md5hash}_{original_name}.wav
#         data/processed/raw_extraction_map.csv

# Step 2: Generate manifest with G2P + articulatory labels
python src/data/manifest.py
# Output: data/processed/torgo_neuro_symbolic_manifest.csv (expected: ~16,531 rows)
```

---

## Download Pipeline (`src/data/download.py`)

`TorgoManager` in `src/data/download.py` loads TORGO metadata from `abnerh/TORGO-database` via HuggingFace `datasets`, downloads and resamples audio to 16 kHz, and writes files to `data/raw/audio/` with the naming convention:

```
{speaker_id}_{md5_hash_of_hf_path}_{original_filename}.wav
```

The MD5 hash is computed as `hashlib.md5(hf_path.encode()).hexdigest()[:8]` from the HuggingFace path string. This hash-based naming is required for manifest matching: the TORGO HuggingFace dataset has no consistent `speaker_id` metadata field, so the manifest code reconstructs speaker IDs from filenames using `path.name.split('_')[2]` (B12 fix — position [0] returns the literal string `'unknown'`). The output `data/processed/raw_extraction_map.csv` records speaker, filename, path, original sample rate, and target sample rate for every extracted file.

Output layout:
```
data/raw/audio/
└── train/
    ├── M01_abc12345_s1_mic1_001.wav
    ├── FC03_def67890_s2_mic2_042.wav
    └── ...
```

---

## Manifest Generation (`src/data/manifest.py`)

`SymbolicProcessor` in `src/data/manifest.py` performs: (1) G2P transcription via `g2p_en.G2p()` (NLTK-backed), (2) stress stripping via `rstrip('012')`, (3) articulatory class annotation from shared constants in `src/utils/constants.py`, (4) audio metric extraction (duration, RMS energy via `librosa`), and (5) quality filtering.

**B12 fix (speaker extraction):** TORGO HuggingFace filenames follow `unknown_{hash}_{SPEAKER}_{session}_{mic}_{n}.wav`. The speaker ID is at position `[2]` after `split('_')`, not `[0]` (which returns `'unknown'`). Fix: `speaker_id = meta.get("speaker_id", path.name.split('_')[2])`. The manifest was **regenerated on March 4, 2026** with correct TORGO IDs.

`get_features()` uses `@lru_cache(maxsize=2048)` to avoid re-running G2P for repeated transcripts across speakers. Noise tokens (`[SILENCE]`, `[NOISE]`) are stripped before G2P processing.

**Quality filters:**

| Filter | Threshold | Reason |
|---|---|---|
| Minimum phoneme count | ≥ 2 (`MIN_PHONEME_COUNT = 2`) | Single-phoneme utterances unreliable for CTC |
| Maximum phoneme rate | ≤ 20 phonemes/sec (`MAX_PHONEMES_PER_SEC = 20`) | Filters transcription errors or corrupt metadata |
| Minimum duration | > 0.1s | Sub-threshold frames produce degenerate CTC inputs |
| Non-empty G2P output | Required | `g2p_en` must produce ≥ 1 phoneme |

**Expected output:** approximately 16,531 rows, ~20 hours total audio.

**B23 articulatory label corrections** (now centralized in `src/utils/constants.py` and imported by both `manifest.py` and `model.py`):

| Phoneme(s) | Before | After | Reason |
|---|---|---|---|
| SH, ZH, CH, JH | `palatal` (place) | `postalveolar` | SH/ZH are laminal-alveolar/postalveolar, not palatal; CH/JH are postalveolar affricates |
| R | `palatal` (place) | `alveolar` | American English /r/ is a retroflex approximant (alveolar/post-alveolar) |
| W | `bilabial` (place) | `labio-velar` | /w/ is labialized velar (labio-velar approximant) |

---

## Manifest Schema

| Column | Type | Description | Example |
|---|---|---|---|
| `sample_id` | str | Unique identifier derived from speaker + path | `M01_s1_mic1_001.wav` |
| `path` | str | Absolute path to `.wav` file | `/home/user/.../M01_abc12345_s1_mic1_001.wav` |
| `speaker` | str | TORGO speaker ID | `M01` |
| `status` | int | HF `speech_status` field | `1` (dysarthria) or `0` (normal) |
| `label` | int | Alias for `status` | `1` |
| `transcript` | str | Raw text prompt (uppercased) | `PUT THE BOOK ON THE TABLE` |
| `phonemes` | str | Space-separated stress-agnostic ARPABET | `P UH T DH AH B UH K AH N` |
| `articulatory_classes` | str | Legacy alias for `manner_classes` | `stop vowel stop fricative vowel` |
| `manner_classes` | str | Space-separated manner-of-articulation labels | `stop vowel stop fricative vowel` |
| `place_classes` | str | Space-separated place-of-articulation labels | `bilabial central alveolar dental central` |
| `voice_classes` | str | Space-separated voicing labels | `voiceless vowel voiceless voiced vowel` |
| `phn_count` | int | Number of phonemes in the sequence | `17` |
| `phonemes_per_sec` | float | Phoneme rate (phn_count / duration) | `4.2` |
| `duration` | float | Audio duration in seconds | `4.05` |
| `rms_energy` | float | Mean RMS energy from `librosa.feature.rms` | `0.034` |
| `gender` | str | `F` or `M`, inferred from speaker ID | `M` |

---

## Vocabulary System

### Special Tokens (Immutable — Never Change These IDs)

```python
<BLANK> = 0    # CTC blank token — alignment separator; never a label target
<PAD>   = 1    # Padding for variable-length batching
<UNK>   = 2    # Unknown / OOV phonemes (fallback)
# IDs 3–46: actual ARPABET phonemes (stress-stripped, sorted)
```

These IDs are built first in `TorgoNeuroSymbolicDataset._build_vocabularies()` and assumed throughout training, evaluation, and decoding. Changing these assignments silently breaks CTC alignment, CE loss, and all downstream decoding.

**Total vocabulary size: 47 tokens** (44 ARPABET phonemes + 3 special tokens).

### `normalize_phoneme()` — Must Be Called Everywhere

```python
from src.utils.config import normalize_phoneme
normalize_phoneme("AH0")  # → "AH"
normalize_phoneme("IY1")  # → "IY"
normalize_phoneme("UW2")  # → "UW"
```

TORGO manifest uses ARPABET with stress markers (0/1/2). The model vocabulary is stress-agnostic. `normalize_phoneme()` strips stress digits via `str(phoneme).rstrip('012')`. It **must** be called at: manifest build time, dataset initialization (`_build_vocabularies()`), and decode time (`greedy_decode()`, `BeamSearchDecoder.decode()`). Never compare raw ARPABET strings directly.

### `_build_vocabularies()` — Vocabulary Construction Order

The vocabulary is built from `sorted(set(...))` over all phonemes in the manifest after normalization. The `sort()` call is critical for reproducibility: `df['speaker'].unique()` returns first-occurrence order which changes if the manifest is regenerated; sorted order is manifest-independent.

### Articulatory Classes

**Manner of articulation** (8 classes, used by `manner_head`):
```
stop, fricative, affricate, nasal, liquid, glide, vowel, diphthong
```

**Place of articulation** (12 classes, used by `place_head`):
```
bilabial, labiodental, dental, alveolar, postalveolar, palatal, velar,
glottal, labio-velar, front, back, central
```

**Voicing** (3 classes, used by `voice_head`):
```
voiced, voiceless, vowel
```

---

## `TorgoNeuroSymbolicDataset` Internals

### `_load_audio()` — No Peak Normalization (C2 fix)

```
1. Load WAV at native sample rate via torchaudio.load()
2. Convert stereo → mono: torch.mean(waveform, dim=0)
3. Resample to 16,000 Hz via torchaudio.functional.resample() if needed
4. Truncate to max_audio_length = 6.0s = 96,000 samples (~99% TORGO coverage)
5. NO peak normalization
```

`facebook/hubert-base-ls960` was pretrained with HuBERT processor mean/variance normalization. Manual peak-normalization before the processor causes double normalization, deviating from the pretraining distribution. If a file fails to load, the method returns silence (zeros) to avoid crashing training.

### Feature Cache

Processed `input_values` tensors are cached to an LRU disk cache (`data/processed/feature_cache/{namespace}/{sha1}.pt`) and an in-process LRU memory cache (default 2048 entries, LRU eviction via `OrderedDict`). The cache namespace is a 12-character SHA-1 hash of: `processor_id + sampling_rate + max_audio_samples + manifest_name`. This ensures cache invalidation when any of these parameters change. Corrupt cache files are treated as misses and silently overwritten. Cache writes use atomic `.tmp` rename with a unique suffix (`pid.worker_id.thread_id`) to prevent multi-worker collisions.

### `_build_sequence_cache()` + Tensor Materialization

All label and articulatory class sequences are precomputed once at dataset initialization and then materialized into preallocated tensors (`_labels_tensor`, `_manner_tensor`, `_place_tensor`, `_voice_tensor`) with per-sample lengths tracked in `_seq_lengths`. This avoids repeated per-batch string splitting and reduces Python object overhead from per-sample tensor allocations. If a phoneme count mismatches its articulatory class count, both are truncated to the minimum length with a warning.

### Label Padding Sentinel

Labels use `-100` for padding. This is automatically ignored by `nn.CTCLoss` and `nn.CrossEntropyLoss(ignore_index=-100)`. Never use `0` (CTC blank) or `1` (PAD token) as padding — both are valid token IDs that would be treated as real phonemes.

---

## `NeuroSymbolicCollator` Output Tensors

`NeuroSymbolicCollator.__call__()` in `src/data/dataloader.py` L585:

| Tensor | Shape | Padding value | Notes |
|---|---|---|---|
| `input_values` | [B, T_max] | `0.0` (silence) | Pads audio to batch-max length |
| `attention_mask` | [B, T_max] | `0` | 1=valid audio frame, 0=padding |
| `labels` | [B, L_max] | `-100` | Phoneme ID sequences |
| `articulatory_labels` (manner/place/voice) | [B, L_max] | `-100` | Per-phoneme articulatory class IDs |
| `input_lengths` | [B] | — | Approximate: `len(audio) // ctc_stride`; **overridden by model** |
| `label_lengths` | [B] | — | Exact phoneme sequence lengths |
| `status` | [B] | — | 0=control, 1=dysarthric |
| `speakers` | List[str] | — | Speaker ID strings |

**`input_lengths` approximation caveat (L7 fix):** The collator computes `input_lengths = len(audio_samples) // ctc_stride` (default `ctc_stride=320`). HuBERT's actual `_get_feat_extract_output_lengths()` may differ by 0–3 frames. `train.py` overrides `input_lengths` with `outputs['output_lengths']` from the live model forward pass before passing to `CTCLoss`. The `TemporalDownsampler` further halves output lengths: `output_lengths = (output_lengths + 1) // 2`.

---

## Data Splits

### Speaker-Stratified Split (Single-Run)

`create_dataloaders()` in `train.py` performs speaker-level stratification:

1. `df['speaker'].unique()` extracts unique speaker IDs
2. `np.random.seed(config.experiment.seed)` then `np.random.shuffle(speakers)` for deterministic assignment
3. Train: first 70%, Val: next 15%, Test: remaining 15%
4. **C3 round-robin fallback:** If any partition is empty (can happen with ≤ 3 speakers), speakers are assigned round-robin — last speaker → test, second-to-last → val, all others → train. This prevents silent data leakage.

### LOSO Split (`create_loso_splits()`)

1. One speaker held out as the test set (all their utterances)
2. `max(1, int(14 * 0.15)) = 2` speakers drawn from remaining 14 for validation
3. Remaining 12 speakers form the training set
4. Speaker ordering is deterministic: sorted lexicographically by prefix group and numeric suffix (FC01 < FC02 < M01 < M02, etc.) via `_split_speaker_id()` prefix+number sort

### `WeightedRandomSampler` (B20 fix)

Training uses `WeightedRandomSampler` with speaker-level inverse-frequency weights:
```python
speaker_counts = train_df['speaker'].value_counts().to_dict()
train_weights = [1.0 / speaker_counts[dataset.df.iloc[i]['speaker']] for i in train_idx]
sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
```

This ensures each speaker contributes equally per epoch regardless of utterance count. B20 fixed the original bug that used class-level (dysarthric/control) weights instead of speaker-level weights.

---

## Feature Cache Warm-Up

`warm_feature_cache()` in `src/data/warm_feature_cache.py`:

```bash
# Warm all features before training (speeds up first epoch significantly)
python run_pipeline.py --run-name my_run --warm-cache --warm-cache-only

# With custom worker count and batch size
python run_pipeline.py --run-name my_run --warm-cache --warm-cache-only \
    --warm-cache-workers 8 --warm-cache-batch-size 1
```

Expected speedup: eliminates per-epoch HuBERT processor overhead on cache misses. Subsequent training epochs run entirely from disk/memory cache. The `--warm-cache-only` flag exits after warm-up without training or evaluation.
