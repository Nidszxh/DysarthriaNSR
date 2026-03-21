# docs/data.md — Data Pipeline

> Merges and supersedes the former `docs/dataset.md` and `docs/pipeline.md`.
> Cross-references: [docs/training.md](training.md) for how data flows into the model.

---

## TORGO Corpus

### Origin and Access

TORGO is a dysarthric speech corpus developed at the University of Toronto (Rudzicz et al., 2012). It contains speech from individuals with cerebral palsy and ALS, paired with matching controls. Access is available via HuggingFace Hub (`abnerh/TORGO-database`) for research use.

### Speaker List

| Speaker | Type | Status code | Severity score |
|---|---|---|---|
| F01 | Dysarthric | 1 | 4.90 |
| F03 | Dysarthric | 1 | 4.65 |
| F04 | Dysarthric | 1 | 3.05 |
| M01 | Dysarthric | 1 | 4.90 |
| M02 | Dysarthric | 1 | 4.78 |
| M03 | Dysarthric | 1 | 4.59 |
| M04 | Dysarthric | 1 | 2.85 |
| M05 | Dysarthric | 1 | 2.10 |
| FC01 | Control | 0 | 0.00 |
| FC02 | Control | 0 | 0.00 |
| FC03 | Control | 0 | 0.00 |
| MC01 | Control | 0 | 0.00 |
| MC02 | Control | 0 | 0.00 |
| MC03 | Control | 0 | 0.00 |
| MC04 | Control | 0 | 0.00 |

Severity formula: `severity = (1 - intelligibility/100) * 5.0`, producing scores in [0.0, 5.0]. The full `TORGO_SEVERITY_MAP` dict is defined in `src/utils/config.py`.

### Known Dataset Issues

| Issue | Detail |
|---|---|
| Pilot / duplicate utterances | TORGO includes repeated prompts across sessions. No de-duplication applied; repeated transcripts are treated as distinct utterances from different recording conditions. |
| FC02 reduced recordings | FC02 may have fewer recordings than other control speakers. Verify row count with `df[df['speaker']=='FC02']` after manifest generation. |
| Inverted dysarthric/control PER | Single-split test sets of ~2 speakers produce unreliable stratified statistics. Control PER > dysarthric PER observed in some splits is a sampling artifact, not a data issue. |
| Control severity = 0.0 | All 7 control speakers receive `severity=0.0`. This means all controls form a single severity bucket in ordinal contrastive loss. Mitigated by using continuous `TORGO_SEVERITY_MAP` scores for dysarthric speakers. |

---

## Download and Manifest Generation

### Step 1: Download Audio Files
```bash
python src/data/download.py
```

`TorgoManager` in `src/data/download.py` loads TORGO metadata from `abnerh/TORGO-database` via HuggingFace `datasets`, downloads and resamples audio to 16 kHz, and writes files to `data/raw/audio/` with the naming convention:
```
{speaker_id}_{md5_hash_of_hf_path}_{original_filename}.wav
```

The MD5 hash is computed from the HuggingFace path string: `hashlib.md5(hf_path.encode()).hexdigest()[:8]`. This hash is later used by `manifest.py` to match metadata to local files without relying on speaker metadata fields in the HF dataset (which may be absent or inconsistent).

### Step 2: Generate Manifest
```bash
python src/data/manifest.py
# Output: data/processed/torgo_neuro_symbolic_manifest.csv
```

`SymbolicProcessor` in `src/data/manifest.py` performs: (1) G2P transcription via `g2p_en.G2p()` (NLTK-backed), (2) stress stripping via `rstrip('012')`, (3) articulatory class annotation from `PHONEME_DETAILS`, (4) audio metric extraction (duration, RMS energy via `librosa`), and (5) quality filtering.

**B12 fix (speaker extraction):** The TORGO HuggingFace dataset encodes filenames as `unknown_{hash}_{SPEAKER}_{session}_{mic}_{n}.wav`. The speaker ID is at position `[2]` after `split('_')`, not position `[0]` (which returns the literal string `'unknown'`). The fix was applied in the manifest code as `speaker_id = meta.get("speaker_id", path.name.split('_')[2])`, and the manifest was **regenerated on March 4, 2026** with correct TORGO IDs.

**Quality filters applied:**

| Filter | Threshold | Reason |
|---|---|---|
| Minimum phoneme count | ≥ 2 | Single-phoneme utterances unreliable for CTC |
| Maximum phoneme rate | ≤ 20 phonemes/sec | Filters transcription errors or corrupt metadata |
| Minimum duration | > 0.1s | Sub-threshold frames produce degenerate CTC inputs |
| Non-empty G2P output | Required | `g2p_en` must produce ≥ 1 phoneme |

**Expected output:** approximately 16,531 rows, ~20 hours total audio.

---

## Manifest Schema

The manifest at `data/processed/torgo_neuro_symbolic_manifest.csv` contains one row per utterance.

| Column | Type | Description |
|---|---|---|
| `sample_id` | str | Unique identifier: `{speaker}_{session}_{mic}_{n}` |
| `path` | str | Absolute path to `.wav` file |
| `speaker` | str | TORGO speaker ID (e.g., `M01`, `FC03`) |
| `status` | int | 1=dysarthric, 0=control (from HF `speech_status` field) |
| `label` | int | Alias for `status` |
| `transcript` | str | Raw text prompt (uppercased) |
| `phonemes` | str | Space-separated stress-agnostic ARPABET sequence |
| `articulatory_classes` | str | Alias for `manner_classes` (legacy column) |
| `manner_classes` | str | Space-separated manner-of-articulation labels per phoneme |
| `place_classes` | str | Space-separated place-of-articulation labels per phoneme |
| `voice_classes` | str | Space-separated voicing labels per phoneme |
| `phn_count` | int | Number of phonemes in the sequence |
| `phonemes_per_sec` | float | Phoneme rate (phn_count / duration) |
| `duration` | float | Audio duration in seconds |
| `rms_energy` | float | Mean RMS energy from `librosa.feature.rms` |
| `gender` | str | `F` or `M`, inferred from speaker ID |

**Example row:**
```
sample_id: M01_s1_mic1_001
path: /home/user/DysarthriaNSR/data/raw/audio/M01_abc12345_s1_mic1_001.wav
speaker: M01
status: 1
label: 1
transcript: PUT THE BOOK ON THE TABLE
phonemes: P UH T DH AH B UH K AH N DH AH T EY B AH L
manner_classes: stop vowel stop fricative vowel stop vowel stop vowel nasal fricative vowel stop diphthong stop vowel liquid
place_classes: bilabial central alveolar dental central bilabial central velar central alveolar dental central alveolar front bilabial central alveolar
voice_classes: voiceless vowel voiceless voiced vowel voiceless vowel voiceless vowel voiced voiced vowel voiceless voiced voiceless vowel voiced
phn_count: 17
phonemes_per_sec: 4.2
duration: 4.05
rms_energy: 0.034
gender: M
```

---

## Vocabulary

### Special Tokens
```python
<BLANK> = 0    # CTC blank token — alignment separator, never a label target
<PAD>   = 1    # Padding token for variable-length batching
<UNK>   = 2    # Unknown / OOV phonemes (fallback)
```

These IDs are **immutable**. They are built first in `TorgoNeuroSymbolicDataset._build_vocabularies()` and assumed throughout training, evaluation, and decoding. **Never change these assignments.**

### 44 ARPABET Phonemes (IDs 3–46)

Vocabulary is built from sorted unique phonemes in the manifest after `normalize_phoneme()` stress-stripping. Total vocabulary size: **47 tokens** (44 phonemes + 3 special tokens).

`normalize_phoneme(phoneme)` strips stress digits: `AH0 → AH`, `IY1 → IY`. This function is defined in `src/utils/config.py` and must be called at every point phonemes are compared, built into vocabularies, or decoded: manifest build time, dataset initialization, and decode time. **Never compare raw ARPABET strings (with stress) directly.**

### Articulatory Classes

**Manner of articulation** (used by `manner_head`, 8 classes):
```
stop, fricative, affricate, nasal, liquid, glide, vowel, diphthong
```

**Place of articulation** (used by `place_head`, 12 classes):
```
bilabial, labiodental, dental, alveolar, postalveolar, palatal, velar,
glottal, labio-velar, front, back, central
```

**Voicing** (used by `voice_head`, 3 classes):
```
voiced, voiceless, vowel
```

**B23 fix — articulatory label corrections:** Three label assignments were corrected to align with IPA standards. These corrections were applied to both `PHONEME_DETAILS` in `src/data/manifest.py` and `PHONEME_FEATURES` in `src/models/model.py`:

| Phoneme(s) | Before | After | Reason |
|---|---|---|---|
| SH, ZH, CH, JH | `palatal` (place) | `postalveolar` | SH/ZH are laminal-alveolar/postalveolar, not palatal |
| R | `palatal` (place) | `alveolar` | American English /r/ is a retroflex approximant |
| W | `bilabial` (place) | `labio-velar` | /w/ is labio-velar (labialized velar approximant) |

---

## Data Pipeline Internals

### Audio Loading

`TorgoNeuroSymbolicDataset._load_audio()` in `src/data/dataloader.py` L310:

1. Load WAV at native sample rate via `torchaudio.load()`
2. Convert stereo → mono: `torch.mean(waveform, dim=0)`
3. Resample to 16,000 Hz via `torchaudio.functional.resample()` if needed
4. Truncate to `max_audio_length = 6.0s` = 96,000 samples (~99% TORGO coverage)
5. **No peak normalization** (C2 fix): `facebook/hubert-base-ls960` was pretrained with HuBERT processor normalization only. Manual peak-normalization before the processor causes double normalization, deviating from the pretraining distribution.

### HuBERT Processor and Feature Cache

`AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")` applies mean/variance normalization and outputs `input_values` (1-D float tensor at 16 kHz — the CNN inside HuBERT handles convolution).

Processed `input_values` tensors are cached to an LRU disk cache (`data/processed/feature_cache/{namespace}/{sha1}.pt`) and an in-process LRU memory cache (default 2048 entries, LRU eviction via `OrderedDict`). The cache namespace is keyed by processor ID + sample rate + max_samples + manifest name (SHA-1), ensuring cache invalidation when any of these change. Corrupt cache files are treated as misses and silently overwritten.

### Label Padding Sentinel

Labels use `-100` for padding. This sentinel is automatically ignored by both `nn.CTCLoss` and `nn.CrossEntropyLoss(ignore_index=-100)`.
```python
# In NeuroSymbolicCollator:
labels = pad_sequence(labels, batch_first=True, padding_value=-100)
# Compute actual label lengths:
label_lengths = (labels != -100).sum(dim=1)
```

**Never use 0 or 1 as padding values** — these are valid token IDs (`<BLANK>` and `<PAD>` respectively). Using them as padding would cause CTC and CE losses to treat padding as real phoneme tokens.

### NeuroSymbolicCollator Output Tensors

`NeuroSymbolicCollator.__call__()` in `src/data/dataloader.py` L585:

| Tensor | Shape | Padding value | Notes |
|---|---|---|---|
| `input_values` | [B, T_max] | `0.0` (silence) | Pads audio to batch-max length |
| `attention_mask` | [B, T_max] | `0` | 1=valid audio frame, 0=padding |
| `labels` | [B, L_max] | `-100` | Phoneme ID sequences |
| `articulatory_labels` (manner/place/voice) | [B, L_max] | `-100` | Per-phoneme articulatory class IDs |
| `input_lengths` | [B] | — | Approximate: `len(audio) // 320`; overridden by model's `output_lengths` in train.py |
| `label_lengths` | [B] | — | Exact phoneme sequence lengths |
| `status` | [B] | — | 0=control, 1=dysarthric |
| `speakers` | List[str] | — | Speaker ID strings |

**Note on `input_lengths`:** The collator computes `input_lengths = len(audio_samples) // 320` (CTC stride). This approximation can be off by 0–3 frames compared to HuBERT's actual `_get_feat_extract_output_lengths()`. `train.py` overrides this with `outputs['output_lengths']` from the live model forward pass before passing lengths to `CTCLoss`.

---

## Splits

### Speaker-Stratified Split (Single-Run)

`create_dataloaders()` in `train.py` performs speaker-level stratification:

1. `df['speaker'].unique()` extracts unique speaker IDs
2. `np.random.seed(config.experiment.seed)` then `np.random.shuffle(speakers)` for deterministic assignment
3. Train: first 70% of speakers; Val: next 15%; Test: remaining 15%
4. **C3 round-robin fallback:** If the ratio produces any empty partition (can happen with ≤ 3 speakers or unusual ratios), speakers are assigned round-robin — last speaker → test, second-to-last → val, all others → train. This prevents silent data leakage.

### LOSO Split

`create_loso_splits()` in `train.py`:

1. One speaker held out as the test set (all their utterances)
2. `max(1, int(14 * 0.15)) = 2` speakers drawn from remaining 14 for validation
3. Remaining 12 speakers form the training set
4. Speaker ordering is deterministic: sorted lexicographically by prefix group and numeric suffix (e.g., FC01 < FC02 < M01 < M02)

### WeightedRandomSampler (B20 fix)

Training uses `WeightedRandomSampler` with speaker-level inverse-frequency weights:
```python
speaker_counts = train_df['speaker'].value_counts().to_dict()
train_weights = [1.0 / speaker_counts[speaker] for speaker in train_speakers_per_sample]
sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
```

This ensures each speaker contributes equally per epoch, regardless of utterance count. The original B20 bug used class-level (dysarthric/control) weights instead of speaker-level weights, causing high-utterance speakers to dominate gradient signal.