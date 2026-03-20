# DysarthriaNSR — Model Architecture

> Cross-reference: [docs/pipeline.md](pipeline.md) for tensor shapes at each stage,
> [docs/experiments.md](experiments.md) for ablation results per component.

---

## Architecture Overview

```
Input Audio [B, T_audio]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  HuBERT Encoder (facebook/hubert-base-ls960)                       │
│  CNN feature extractor (frozen) + 12 Transformer layers            │
│  Freeze schedule: layers 0–3 permanent; 4–11 progressive           │
│                                                       [B, T', 768] │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼ (training only)
┌─────────────────────────────────────────────────────────────────────┐
│  SpecAugmentLayer                                                   │
│  Per-sample time masking + frequency masking on hidden states       │
│                                                       [B, T', 768] │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼ (when use_severity_adapter=True)
┌─────────────────────────────────────────────────────────────────────┐
│  SeverityAdapter                                                    │
│  severity [B] → Linear(1→64) → SiLU → Linear(64→768) → [B,1,768]  │
│  Cross-attention: Q=hidden_states, K=V=severity_ctx                │
│  Residual + LayerNorm                                 [B, T', 768] │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼ (when use_temporal_downsample=True)
┌─────────────────────────────────────────────────────────────────────┐
│  TemporalDownsampler                                                │
│  Conv1d(768, 768, kernel=3, stride=2, padding=1) + LayerNorm+GELU  │
│  ~50 Hz → ~25 Hz (T' → T'//2)                     [B, T'//2, 768] │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PhonemeClassifier                                                  │
│  Linear(768→512) + LayerNorm + GELU + Dropout(0.1) + Linear(512→V) │
│                                              logits_neural [B,T,V] │
│                                                                     │
│  Articulatory heads (from shared 512-dim features via GAP):         │
│  - manner_head: Linear(512 → |manner|)              [B, |manner|]  │
│  - place_head:  Linear(512 → |place|)               [B, |place|]   │
│  - voice_head:  Linear(512 → |voice|)               [B, |voice|]   │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SymbolicConstraintLayer                                            │
│                                                                     │
│  P_neural = softmax(logits_neural)              [B, T, V]          │
│                                                                     │
│  LearnableConstraintMatrix:                                         │
│    logit_C [V, V] = log(C_static + ε) / 0.5  (temperature init)   │
│    C = softmax(logit_C, dim=-1)               [V, V] row-stochastic│
│    P_constrained = P_neural @ C                [B, T, V]           │
│                                                                     │
│  Adaptive β:                                                        │
│    β_adaptive = clamp(β_base + 0.2 * severity/5, 0.0, 0.8)        │
│    β_base is a learnable parameter (init 0.05)                      │
│                                                                     │
│  Fusion:                                                            │
│    P_final = β_adaptive * P_constrained + (1-β_adaptive) * P_neural│
│                                                                     │
│  Blank-frame masking:                                               │
│    P_final = P_neural  where P_neural[:,blank] >= 0.5              │
│    (constraint only applied where model is non-blank-dominant)      │
│                                                                     │
│    log_probs = log(P_final + 1e-6)    [B, T, V]                    │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
CTC Decoding (greedy or beam search)
```

---

## HuBERT Backbone

| Property | Value | Source |
|----------|-------|--------|
| Model ID | `facebook/hubert-base-ls960` | `config.py` L88 |
| Revision (pinned) | `dba3bb02fda4248b6e082697eee756de8fe8aa8a` | `config.py` L92 |
| Total parameters | ~94.7M | Sum from `NeuroSymbolicASR.__init__` print |
| CNN feature extractor | Frozen permanently | `freeze_feature_extractor=True` |
| Attention implementation | `eager` (not SDPA) | Required for `output_attentions=True` |
| Gradient checkpointing | Enabled | Reduces VRAM at cost of ~10% speed |
| Output | `last_hidden_state` [B, T', 768] | T' ≈ T_audio_samples / 320 |

**Freeze schedule (3-stage progressive unfreezing):**

| Stage | Epoch | Layers unfrozen | Code reference |
|-------|-------|-----------------|----------------|
| Warmup | 0 | None (entire encoder frozen) | `model.freeze_encoder()` at init |
| Stage 1 | ≥1 | 8, 9, 10, 11 | `train.py` L590 |
| Stage 2 | ≥6 | 6, 7, 8, 9, 10, 11 | `train.py` L601 |
| Stage 3 | ≥12 | 4, 5, 6, 7, 8, 9, 10, 11 | `train.py` L610 |

Layers 0–3 remain frozen throughout training (`freeze_encoder_layers=[0,1,2,3]`). These lower layers encode generic acoustic features. After Adam momentum reset is applied on each unfreeze event (`_reset_hubert_lr_warmup()`).

---

## SeverityAdapter

**Class:** `SeverityAdapter` (`model.py` L432)  
**Enabled when:** `ModelConfig.use_severity_adapter=True` (default: True)

Injects a **continuous** severity signal `[0.0, 5.0]` into HuBERT hidden states via cross-attention.

**Architecture:**
```
severity [B] 
    → view(-1, 1, 1)              # [B, 1, 1]
    → Linear(1, 64) + SiLU        # [B, 1, 64]
    → Linear(64, 768)              # [B, 1, 768]  = severity_ctx
    
hidden_states [B, T, 768]  (Q)
severity_ctx  [B, 1, 768]  (K, V)
    → nn.MultiheadAttention(768, heads=8, batch_first=True)
    → Dropout(0.1)
    → LayerNorm(hidden_states + attn_out)   # residual
```

**Design rationale:** Replaces the coarse scalar β-adjustment with spatially-aware, frame-level conditioning. Control speakers (severity=0) receive a learned constant offset (bias), not zeros — the adapter is informative for both populations. This allows the model to selectively amplify or suppress dysarthria-relevant acoustic features at each time step.

**Hyperparameters:**

| Parameter | Value | Config key |
|-----------|-------|-----------|
| `hidden_dim` | 768 | Fixed (matches HuBERT) |
| `adapter_dim` | 64 | `ModelConfig.severity_adapter_dim` |
| `n_heads` | 8 | Hardcoded in `SeverityAdapter.__init__` |
| `dropout` | 0.1 | `ModelConfig.classifier_dropout` |

---

## LearnableConstraintMatrix

**Class:** `LearnableConstraintMatrix` (`model.py` L138)  
**Enabled when:** `ModelConfig.use_learnable_constraint=True` (default: True)

**Parameterization:**
```
logit_C [V, V]  ← learnable parameter (nn.Parameter)
C = softmax(logit_C, dim=-1)   → row-stochastic [V, V]
```

**Initialization (temperature-sharpened):**
```
logit_C = log(C_static + 1e-8) / 0.5
```
Plain `log(C_static)` passed through `softmax` produces a much flatter distribution than `C_static` because softmax re-normalises. Dividing by `init_temperature=0.5` preserves the peakedness of the prior, so `softmax(logit_C)` ≈ `C_static` at epoch 0.

**Forward:**
```python
P_constrained = P_neural @ C       # [B, T, V]
P_constrained /= P_constrained.sum(-1, keepdim=True).clamp_min(1e-8)
```

**Training anchor:** `SymbolicKLLoss` (`losses.py` L176) computes `KL(C_learned || C_static)` with `λ=0.5` to prevent arbitrary drift.

**C_static (the symbolic prior)** is built by `_build_static_matrix()` from:
1. Hard-coded `SymbolicConfig.substitution_rules` (20 dysarthric phoneme confusion priors, e.g., `('B','P')=0.85`)
2. Articulatory distance fallback: `exp(-3.0 * √(w_manner*(m₁≠m₂) + w_place*(p₁≠p₂) + w_voice*(v₁≠v₂)))`
3. Identity diagonal, normalised to be row-stochastic

**C_static weights:** `manner=0.4`, `place=0.35`, `voice=0.25` (from `SymbolicConfig`).

---

## ArticulatoryFeatureEncoder

**Class:** `ArticulatoryFeatureEncoder` (`model.py` L43) — *not* a `nn.Module`; used as a static computation class.

Encodes 44 ARPABET phonemes into articulatory feature vectors based on:
- **Manner:** stop, fricative, affricate, nasal, liquid, glide, vowel, diphthong
- **Place:** bilabial, alveolar, velar, labiodental, dental, postalveolar, glottal, palatal, labio-velar, front/back/central (vowels)
- **Voicing:** voiced / voiceless

Used exclusively to build `C_static` in `SymbolicConstraintLayer._build_static_matrix()`.

**Similarity formula:**
```
distance = √(w_manner²·(m₁≠m₂) + w_place²·(p₁≠p₂) + w_voice²·(v₁≠v₂))
similarity = exp(-decay_factor · distance)   where decay_factor=3.0
```

Note: `PHONEME_FEATURES` in `model.py` and `PHONEME_DETAILS` in `manifest.py` define articulatory labels independently for the constraint matrix and G2P annotation pipeline respectively. They are currently in sync (B23 fix) but must be kept manually synchronized.

---

## SymbolicConstraintLayer

**Class:** `SymbolicConstraintLayer` (`model.py` L188)

**Full forward pass:**
```python
P_neural = softmax(logits)                               # [B, T, V]
P_constrained = P_neural @ C                             # via LearnableConstraintMatrix or static matrix
P_constrained /= P_constrained.sum(-1, keepdim=True).clamp_min(1e-8)

β_adaptive = clamp(β_base + 0.2 * severity/5, 0.0, 0.8)  # shape [B, 1, 1]

P_final = β_adaptive * P_constrained + (1 - β_adaptive) * P_neural

# Blank-frame masking (Phase 3 Fix B):
non_blank_mask = P_neural[:, :, 0] < 0.5                # [B, T, 1]
P_final = where(non_blank_mask, P_final, P_neural)       # bypass constraint on blank-dominant frames

log_probs = log(P_final.clamp_min(1e-6))                # [B, T, V]
```

**Adaptive β:**
- `β_base` is a learnable `nn.Parameter`, initialized to `constraint_weight_init=0.05`
- `severity_beta_slope=0.2` sets the rate of increase per normalized severity unit
- Control speakers (severity=0): β ≈ 0.05; severe dysarthric (severity=5): β ≈ 0.25
- Clamped to `[0.0, 0.8]`

**CTC interaction:** The `SymbolicConstraintLayer` outputs `log_probs_constrained`, which are passed directly to `nn.CTCLoss`. The blank-frame masking ensures that ~85% of blank-dominant CTC frames pass through the neural distribution unchanged (the constraint concentrates effect on non-blank frames where phoneme identities matter).

**Gradient flow:** Gradients from CTC and blank-KL flow through `log_probs → P_final → (β, P_constrained, P_neural) → logit_C → HuBERT`. The `SymbolicKLLoss` adds a gradient path from `logit_C` directly.

---

## TemporalDownsampler

**Class:** `TemporalDownsampler` (`model.py` L580)  
**Enabled when:** `ModelConfig.use_temporal_downsample=True` (default: True)

Stride-2 Conv1d bottleneck between HuBERT encoder output and phoneme classifier.

```python
Conv1d(768, 768, kernel_size=3, stride=2, padding=1)  →  LayerNorm  →  GELU  →  Dropout
```

**Output length formula:** `T_out = (T_in + 1) // 2`  (ceiling division)

**Output lengths must be adjusted:**
```python
output_lengths = (output_lengths + 1) // 2   # applied in NeuroSymbolicASR.forward()
```

**Motivation:** Dysarthric speech has elongated, slurred phonemes spanning many frames. Stride-2 forces the model to aggregate context from 3 neighbouring frames before predicting each phoneme (~50→~25 Hz), making CTC alignment substantially more stable with a mostly-frozen encoder.

---

## SpecAugmentLayer

**Class:** `SpecAugmentLayer` (`model.py` L503)  
**Applied:** Training only (`if not self.training: return x`)  
**Applied to:** HuBERT hidden states [B, T, 768] (before SeverityAdapter)

```
Time masking:  n_masks = max(1, int(T * 0.05)),  mask_length ~ Uniform[1, 10]
Freq masking:  n_masks = max(1, int(D * 0.05)),  mask_length ~ Uniform[1, 8]
Masked values: set to 0.0 (close to layer-normalised mean)
```

Each sample in the batch receives independent mask positions (per-sample loop; B13 fix). A dedicated `torch.Generator` makes mask sampling reproducible independent of Python random state (`set_seed()`). SpecAugment is applied before SeverityAdapter to prevent severity signal corruption on masked frames.

---

## Parameter Counts

Approximate counts (exact values printed at model initialization):

| Module | Parameters (approx.) | Trainable |
|--------|----------------------|-----------|
| HuBERT feature extractor (CNN) | ~3.4M | ❌ (frozen) |
| HuBERT transformer layers 0–3 | ~28.2M | ❌ (frozen) |
| HuBERT transformer layers 4–11 | ~56.4M | ✅ (after Stage 3 unfreezing) |
| SeverityAdapter | ~1.2M | ✅ |
| TemporalDownsampler | ~2.4M | ✅ |
| PhonemeClassifier | ~0.8M | ✅ |
| Articulatory heads | ~0.08M | ✅ |
| SymbolicConstraintLayer (logit_C + β) | ~2.2K (47×47+1) | ✅ |

The model prints exact total/trainable counts at initialization. 

---

## Ablation Modes

Six ablation modes are supported via `--ablation` flag (maps to `ModelConfig.ablation_mode`):

| Mode | SeverityAdapter | SymbolicConstraintLayer | ArtHeads | Notes |
|------|-----------------|------------------------|----------|-------|
| `full` | ✅ | ✅ (learnable C) | ✅ | Default production mode |
| `neural_only` | ❌ | ❌ bypassed | ❌ | Pure HuBERT+classifier; **best single-split PER (0.135)** |
| `no_constraint_matrix` | ✅ | ❌ (uses log-softmax of neural logits) | ✅ | Tests SeverityAdapter contribution in isolation |
| `no_art_heads` | ✅ | ✅ | ❌ | Tests articulatory head contribution |
| `no_spec_augment` | ✅ (no spec aug) | ✅ | ✅ | Tests SpecAugment contribution |
| `no_temporal_ds` | ✅ (no downsampler) | ✅ | ✅ | Tests TemporalDownsampler contribution |
| `symbolic_only` | ✅ | ✅ | ✅ | CTC/CE disabled (λ=0); tests pure symbolic signal |
