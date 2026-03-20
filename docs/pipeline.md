# DysarthriaNSR — End-to-End Pipeline

> Cross-reference: [docs/architecture.md](architecture.md) for component internals,
> [docs/reproducibility.md](reproducibility.md) for exact reproduction commands.

---

## Pipeline Overview

```
TORGO corpus audio (.wav files)
         │
         ▼
src/data/download.py
  ← downloads from HuggingFace: abnerh/TORGO-database
  ← writes: data/raw/audio/{speaker}_{hash}_{name}.wav
         │
         ▼
src/data/manifest.py
  ← G2P transcription via g2p_en
  ← articulatory annotation (manner/place/voice)
  ← writes: data/processed/torgo_neuro_symbolic_manifest.csv
         │
         ▼
TorgoNeuroSymbolicDataset  (src/data/dataloader.py)
  ← loads manifest CSV
  ← builds phoneme vocabulary (47 tokens)
  ← provides disk + memory feature cache
  ← __getitem__: waveform → HuBERT processor → input_values
         │
         ▼
NeuroSymbolicCollator  (src/data/dataloader.py)
  ← pads audio to batch-max length (value: 0.0)
  ← pads labels to batch-max length (value: -100)
  ← creates attention_mask (1=valid, 0=pad)
         │
         ▼
NeuroSymbolicASR  (src/models/model.py)   ←─ see architecture.md
  ← HuBERT → SpecAugment → SeverityAdapter → TemporalDownsampler
  ← PhonemeClassifier → SymbolicConstraintLayer
  → log_probs_constrained, logits_neural, output_lengths, ...
         │
         ▼
DysarthriaASRLightning  (train.py)
  ← 6-component multi-task loss
  ← 3-stage progressive HuBERT unfreezing
  ← OneCycleLR with differential LR groups
         │
         ▼
evaluate.py  / run_pipeline.py
  ← greedy or beam-search CTC decoding
  ← PER computation (editdistance)
  ← evaluation_results.json + 10 figures
```

---

## Stage 1 — Audio Loading

**Component:** `TorgoNeuroSymbolicDataset._load_audio()` (`dataloader.py` L310)

| Property | Value |
|----------|-------|
| Input format | WAV files (any sample rate; mono or stereo) |
| Target sample rate | 16,000 Hz (`DataConfig.sampling_rate=16000`) |
| Stereo → mono | Average channels: `torch.mean(waveform, dim=0)` |
| Resampling | `torchaudio.functional.resample()` if source SR ≠ 16000 |
| Max length | 6.0 s = 96,000 samples (`DataConfig.max_audio_length=6.0`) |
| Normalization | **None** — HuBERT processor applies its own mean/variance normalization |
| Failure handling | Returns 1-second silence tensor; warns via `warnings.warn` |

**Output:** 1-D waveform tensor [T] where T ≤ 96,000.

**Why no peak normalization?** `facebook/hubert-base-ls960` was pretrained with processor normalization only. Double-normalizing deviates from the pretraining distribution (C2 fix documented in codebase).

---

## Stage 2 — HuBERT Preprocessing

**Component:** `AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")`

The HuBERT `AutoFeatureExtractor`:
- Applies mean/variance normalization to the waveform
- Outputs `input_values` (still a 1-D float tensor at 16 kHz — the CNN inside HuBERT handles convolution)
- `return_tensors="pt"` → squeeze to [T]

**Feature cache:** Processed `input_values` tensors are cached to disk (`data/processed/feature_cache/{namespace}/{sha1_hash}.pt`) and an in-process LRU cache (default size 2048). Cache namespace is keyed by processor ID + sample rate + max_samples + manifest name (SHA-1). Corrupt cache files are treated as misses and silently overwritten.

---

## Stage 3 — Manifest & G2P Annotation

**Component:** `src/data/manifest.py` (run once, offline)

| Step | Tool | Output |
|------|------|--------|
| Load TORGO metadata | HuggingFace `datasets` (`abnerh/TORGO-database`) | `ds` dict with transcript, speaker_id, speech_status |
| Match metadata↔local files | MD5 hash of HF path (matches `download.py` naming scheme) | matched_samples |
| G2P transcription | `g2p_en.G2p` (NLTK-backed) | ARPABET phoneme sequences (stress stripped by `rstrip('012')`) |
| Articulatory annotation | `PHONEME_DETAILS` dict | manner/place/voice class sequences |
| Audio metrics | `librosa.load` + `librosa.feature.rms` | duration, RMS energy |
| Quality filters | `MIN_PHONEME_COUNT=2`, `MAX_PHONEMES_PER_SEC=20`, `duration > 0.1s` | Invalid rows removed |

**Output CSV columns:** `sample_id`, `path`, `speaker`, `status`, `label` (0/1), `transcript`, `phonemes`, `manner_classes`, `place_classes`, `voice_classes`, `phn_count`, `phonemes_per_sec`, `duration`, `rms_energy`, `gender`

**Speaker ID extraction:** `speaker_id = meta.get("speaker_id", path.name.split('_')[2])` — the HF metadata field is preferred; position 2 in the filename is the fallback (B12 fix).

---

## Stage 4 — Batch Construction

**Component:** `NeuroSymbolicCollator.__call__()` (`dataloader.py` L585)

**Input:** List of `__getitem__` dicts (one per sample)

**Padding strategy:**

| Tensor | Padding value | Method |
|--------|--------------|--------|
| `input_values` [B, T_max] | `0.0` (silence) | `pad_sequence(batch_first=True)` |
| `labels` [B, L_max] | `-100` (ignored by CTC/CE) | `pad_sequence(batch_first=True)` |
| `articulatory_labels` (manner/place/voice) | `-100` | `pad_sequence(batch_first=True)` |
| `attention_mask` [B, T_max] | computed (1=valid, 0=pad) | Loop over actual lengths |

**`input_lengths`:** Approximation computed as `len(input_values) // 320` (CTC stride). **Not used directly for CTC loss.** `train.py` overrides with `outputs['output_lengths']` from the live model (which uses `_get_feat_extract_output_lengths()`), which can differ by ±3 frames from the approximation.

**`label_lengths`:** `torch.tensor([len(x) for x in labels])` — exact phoneme sequence lengths.

---

## Stage 5 — HuBERT Feature Extraction

**Component:** `NeuroSymbolicASR.forward()` (`model.py` L827)

```python
hubert_outputs = self.hubert(
    input_values,            # [B, T_audio]
    attention_mask=attention_mask,
    output_hidden_states=False,   # only last_hidden_state; saves ~120MB/batch
    output_attentions=output_attentions,  # True only for explainability
    return_dict=True,
)
hidden_states = hubert_outputs.last_hidden_state  # [B, T', 768]
```

**Output size:** `T' = _get_feat_extract_output_lengths(T_audio_samples)`  
For the default `max_audio_length=6.0s` at 16kHz (96,000 samples): T' ≈ 299 frames (~50 Hz).

**Layer selection:** The final transformer layer's output is used exclusively. HuBERT is configured with `attn_implementation="eager"` (SDPA does not support `output_attentions=True`).

---

## Stage 6 — SpecAugment

Applied to `hidden_states [B, T', 768]` during training only. Parameters:

| Parameter | Default | Config key |
|-----------|---------|-----------|
| Time mask probability | 0.05 | `spec_time_mask_prob` |
| Max time mask length | 10 frames | `spec_time_mask_length` |
| Freq mask probability | 0.05 | `spec_freq_mask_prob` |
| Max freq mask length | 8 dims | `spec_freq_mask_length` |
| Number of time masks per sample | `max(1, int(T*0.05))` | derived |
| Number of freq masks per sample | `max(1, int(D*0.05))` | derived |

Masked values are set to `0.0`. Applied before SeverityAdapter to preserve the adapter's severity signal from being masked.

---

## Stage 7 — SeverityAdapter

See [docs/architecture.md — SeverityAdapter](architecture.md#severityadapter).

**Direction of operation:**
```
severity [B]  →  [B, 1, 768] severity_ctx
hidden_states [B, T', 768]  →  cross-attn  →  layernorm(hidden + attn_out) [B, T', 768]
```

**How severity is sourced:**
- From `TORGO_SEVERITY_MAP` (in `config.py`) when speaker IDs are available in batch metadata
- Fallback: `status.float() * 5.0` (binary: 0.0 for control, 5.0 for dysarthric)

The `TORGO_SEVERITY_MAP` encodes continuous severity scores derived from Rudzicz et al. (2012) intelligibility ratings: `severity = (1 - intelligibility/100) * 5.0`.

---

## Stage 8 — TemporalDownsampler

```
hidden_states [B, T', 768]
    → transpose(1,2)      [B, 768, T']
    → Conv1d(768, 768, kernel=3, stride=2, padding=1)
    → transpose(1,2)      [B, ceil(T'/2), 768]
    → LayerNorm
    → GELU
    → Dropout(0.1)
    = [B, T, 768]  where T = (T' + 1) // 2
```

`output_lengths` must be adjusted by the model:
```python
if _downsample_applied:
    output_lengths = (output_lengths + 1) // 2
output_lengths = output_lengths.clamp(max=hidden_states.size(1))
```

---

## Stage 9 — PhonemeClassifier

```
hidden_states [B, T, 768]

→ Linear(768 → 512)         [B, T, 512]   = shared_features
→ LayerNorm(512)
→ GELU
→ Dropout(0.1)
→ Linear(512 → |V|)         [B, T, |V|]  = logits_neural

Articulatory heads (from shared_features.mean(dim=1)):
    pooled_for_art [B, 512]     ← global average pool over time
    → manner_head: Linear(512 → |manner|)   [B, |manner|]
    → place_head:  Linear(512 → |place|)    [B, |place|]
    → voice_head:  Linear(512 → |voice|)    [B, |voice|]
```

`|V|` = 47 at runtime (built from manifest vocabulary). `|manner|`, `|place|`, `|voice|` built from `PHONEME_DETAILS` constants.

---

## Stage 10 — SymbolicConstraintLayer

Complete tensor shapes:

```
logits_neural            [B, T, V=47]
P_neural = softmax(logits_neural)   [B, T, 47]
C = softmax(logit_C, dim=-1)        [47, 47]  (row-stochastic)
P_constrained = P_neural @ C        [B, T, 47]
P_constrained /= P_constrained.sum(-1)

β_adaptive               [B, 1, 1]
P_final = β * P_constrained + (1-β) * P_neural   [B, T, 47]
# Blank-frame masking: P_final = P_neural where P_neural[blank] >= 0.5
log_probs = log(P_final.clamp_min(1e-6))          [B, T, 47]
```

Return dict keys: `log_probs`, `beta`, `P_neural`, `P_constrained`, `rule_shift`

---

## Stage 11 — CTC Decoding

**Primary (training):** `nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)`
- Input: `log_probs_constrained` transposed to `[T, B, V]`
- `input_lengths`: exact `output_lengths` from model (not collator approximation)
- `label_lengths`: actual phoneme sequence lengths (excluding `-100` padding)
- Invalid samples (`label_length > input_length`) are masked out

**Evaluation (greedy):** `greedy_decode()` (`evaluate.py` L294)
- Applies CTC collapse rules: blank resets, consecutive duplicate collapse
- `output_lengths` passed to truncate padding frames before decoding

**Evaluation (beam search, optional):** `BeamSearchDecoder` (`evaluate.py` L137)
- Beam width: 10 (default), configurable via `--beam-width`
- Length normalisation `acoustic / len^α` with `α=0.6`, applied to acoustic score only
- Optional BigramLM shallow fusion (`λ=0.0` by default = disabled)
- LM uses add-k smoothing (`k=0.5`) to avoid `log(0)`

---

## Stage 12 — PER Computation

**Function:** `compute_per()` (`evaluate.py` L399)

```python
PER = editdistance.eval(prediction_list, reference_list) / len(reference_list)
```

- Denominator: number of phonemes in reference
- Returns `1.0` for empty reference with non-empty prediction; `0.0` for both empty
- Uses `editdistance` C extension (O-5: `rapidfuzz` fast path attempted first in alignment functions)

**Aggregation:** Macro-speaker PER — group utterance PER scores by speaker, compute per-speaker mean, then mean over speakers. This treats each speaker equally regardless of utterance count.

---

## Stage 13 — LOSO-CV Loop

**Function:** `run_loso()` (`train.py`)

```
For each test_speaker in sorted(all_speakers):
    train_speakers = all_speakers - {test_speaker}
    val_speakers = sample 2 from train_speakers
    
    train_loader ← utterances from train_speakers
    val_loader   ← utterances from val_speakers
    test_loader  ← utterances from {test_speaker}
    
    model = NeuroSymbolicASR(...)
    DysarthriaASRLightning.fit(train_loader, val_loader)
    evaluate(test_loader) → fold_per, fold_wer
    
    save fold result to loso_progress.json
    save checkpoint to checkpoints/{run_name}_loso_{speaker}/
```

Resume support: if `--resume-loso` is set, completed folds (present in `loso_progress.json`) are skipped. Aggregated results (`macro_avg_per`, `per_95ci`, `weighted_avg_per`, `macro_avg_wer`) are printed after all folds complete.

**Total folds:** 15 (one per TORGO speaker). Estimated runtime: ~32h on RTX 4060.

---

## Full Tensor Shape Summary

| Stage | Tensor | Shape |
|-------|--------|-------|
| Audio input | `input_values` | [B, T_audio] |
| HuBERT output | `last_hidden_state` | [B, T', 768] where T'≈T_audio/320 |
| After SpecAugment | `hidden_states` | [B, T', 768] (training only, shapes unchanged) |
| After SeverityAdapter | `hidden_states` | [B, T', 768] |
| After TemporalDownsampler | `hidden_states` | [B, T=ceil(T'/2), 768] |
| PhonemeClassifier | `logits_neural` | [B, T, 47] |
| Articulatory (pooled) | `logits_manner/place/voice` | [B, \|manner/place/voice\|] |
| Neural softmax | `P_neural` | [B, T, 47] |
| Constraint matrix | `C` | [47, 47] |
| Constrained distribution | `P_constrained` | [B, T, 47] |
| Final log-probs | `log_probs_constrained` | [B, T, 47] |
| CTC input | transposed | [T, B, 47] |
| Decoded output | phoneme list | List[List[str]], e.g. `[['P', 'AH', 'T'], ...]` |
