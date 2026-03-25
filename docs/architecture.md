# docs/architecture.md — Model Architecture

> Cross-references: [../README.md#final-system-figure](../README.md#final-system-figure) for the final rendered architecture figure, [experiments.md](experiments.md) for ablation results, [data.md](data.md) for vocabulary and input tensor shapes, [training.md](training.md) for freeze schedule operational details.

---

## Forward Pass Diagram

```mermaid
graph LR
    A["Input Audio\n[B, T_audio]"] --> B["HuBERT Encoder\n[B, T', 768]\nT' ≈ T_audio/320"]
    B --> C["SpecAugmentLayer\n[B, T', 768]\ntraining only"]
    C --> D["SeverityAdapter\n[B, T', 768]\ncross-attention"]
    D --> E["TemporalDownsampler\n[B, T=⌈T'/2⌉, 768]\nstride-2 Conv1d"]
    E --> F["PhonemeClassifier\n[B, T, 47]\nlogits_neural"]
    F --> G["SymbolicConstraintLayer\n[B, T, 47]\nlog_probs_constrained"]
    G --> H["CTC Decode\nList[List[str]]"]
```

### Tensor Shape Table

| Stage | Tensor | Shape | Notes |
|---|---|---|---|
| Audio input | `input_values` | [B, T_audio] | T_audio ≤ 96,000 (6.0s @ 16 kHz) |
| HuBERT output | `last_hidden_state` | [B, T', 768] | T' ≈ T_audio / 320 (~50 Hz) |
| After SpecAugment | `hidden_states` | [B, T', 768] | Training only; shapes unchanged |
| After SeverityAdapter | `hidden_states` | [B, T', 768] | Cross-attention residual; control speakers still receive learned offset |
| After TemporalDownsampler | `hidden_states` | [B, T, 768] | T = (T' + 1) // 2 (~25 Hz) |
| PhonemeClassifier shared features | `shared_features` | [B, T, 512] | Pre-logits hidden dim |
| PhonemeClassifier output | `logits_neural` | [B, T, 47] | Raw logits over V=47 |
| Articulatory heads (pooled) | `logits_manner/place/voice` | [B, \|class\|] | Global average pool over time (I5 fix) |
| Neural softmax | `P_neural` | [B, T, 47] | softmax(logits_neural) |
| Constraint matrix | `C` | [47, 47] | Row-stochastic, learnable |
| Constrained distribution | `P_constrained` | [B, T, 47] | P_neural @ C, renormalized |
| Blank-masked fusion | `P_final` | [B, T, 47] | β·P_constrained + (1-β)·P_neural; frames where P_neural\[blank\] ≥ 0.5 retain P_neural |
| Log-probs for CTC | `log_probs_constrained` | [B, T, 47] | log(P_final.clamp_min(1e-6)) |
| CTC input | transposed | [T, B, 47] | Required by `nn.CTCLoss` |
| Output lengths | `output_lengths` | [B] | Exact valid CTC frame count; used to mask padding in decode |

---

## Components

### HuBERT Backbone

**Purpose:** Extracts contextual acoustic representations from raw 16 kHz waveforms using 960-hour LibriSpeech self-supervised pretraining.

**Key design decision:** The CNN feature extractor is permanently frozen (`freeze_feature_extractor=True`) because it encodes low-level acoustic primitives that do not benefit from dysarthric fine-tuning. Upper transformer layers are progressively unfrozen over three stages (epochs 1, 6, 12) to balance VRAM constraints against adaptation quality on the small TORGO corpus.

**Class/file:** `HubertModel` from `transformers`, instantiated in `NeuroSymbolicASR.__init__()` (`src/models/model.py` L707). The model is pinned to revision `dba3bb02fda4248b6e082697eee756de8fe8aa8a` for exact reproducibility.

**Performance note (PERF refactor):** When all HuBERT parameters are frozen (epochs 0 through warmup), the encoder is run inside `eval()` + `torch.no_grad()`. This eliminates intermediate activation graph construction for frozen parameters (~15–25% VRAM reduction) and preserves LayerNorm running statistics. After the frozen pass, the original training mode is restored so downstream learnable modules receive gradients normally.

| Parameter | Default | Description |
|---|---|---|
| `hubert_model_id` | `facebook/hubert-base-ls960` | HuBERT checkpoint ID |
| `hubert_model_revision` | `dba3bb02fda4248b6e082697eee756de8fe8aa8a` | Pinned hub revision |
| `freeze_feature_extractor` | `True` | CNN always frozen |
| `freeze_encoder_layers` | `[0, 1, 2, 3]` | Permanently frozen transformer layers |
| `use_gradient_checkpointing` | `True` | Reduces VRAM ~15–25% at ~10% speed cost |

**Forward example:** Input `[2, 16000]` → HuBERT output `[2, 49, 768]`.

---

### SeverityAdapter

**Purpose:** Injects a continuous severity signal [0, 5] into HuBERT hidden states via cross-attention, providing spatially-aware, frame-level dysarthria conditioning.

**Key design decision (Proposal P3):** Cross-attention (rather than a scalar bias) allows the adapter to selectively amplify or suppress dysarthria-relevant acoustic features at each time step. Importantly, control speakers (severity=0) still receive a learned constant offset from the bias terms — unlike a pure gating mechanism that would be a no-op at severity=0.

**Class/file:** `SeverityAdapter` in `src/models/model.py` L432.

| Parameter | Default | Description |
|---|---|---|
| `use_severity_adapter` | `True` | Enable/disable adapter |
| `severity_adapter_dim` | `64` | Bottleneck dim for severity projection (Linear(1,64)→SiLU→Linear(64,768)) |
| `hidden_dim` (cross-attn) | `768` | Matches HuBERT output dim |
| `n_heads` | `8` | Multi-head attention heads |
| `classifier_dropout` | `0.1` | Dropout on attention output |

**Forward example:** Input `[2, 49, 768]` + severity `[2]` → output `[2, 49, 768]` (residual: `LayerNorm(hidden + dropout(cross_attn_out))`).

---

### SpecAugmentLayer

**Purpose:** Applies independent time and frequency masks to HuBERT hidden states during training to prevent overfitting on the small TORGO dataset (~16,531 utterances).

**Key design decision (B13 fix):** Masks are applied per-sample independently. The original bug applied the same random mask to the entire batch. SpecAugment is placed **before** `SeverityAdapter` (B3 fix) to avoid masking the severity conditioning signal — augmentation must operate on clean encoder features before severity injection.

**Class/file:** `SpecAugmentLayer` in `src/models/model.py` L503. No-op at eval/inference time.

| Parameter | Default | Description |
|---|---|---|
| `use_spec_augment` | `True` | Enable/disable |
| `spec_time_mask_prob` | `0.05` | Expected fraction of time steps to mask per sample |
| `spec_time_mask_length` | `10` | Max consecutive frames per time mask |
| `spec_freq_mask_prob` | `0.05` | Expected fraction of feature dims to mask per sample |
| `spec_freq_mask_length` | `8` | Max consecutive dims per frequency mask |

**Forward example:** Input `[2, 49, 768]` → output `[2, 49, 768]` with random contiguous regions zeroed.

---

### TemporalDownsampler

**Purpose:** Halves the frame rate from ~50 Hz to ~25 Hz using a stride-2 Conv1d, forcing the model to aggregate context from neighboring frames before predicting phonemes.

**Key design decision:** Dysarthric speech has elongated, slurred phonemes spanning many frames. With a mostly-frozen encoder, direct per-frame prediction is unstable. Stride-2 downsampling makes CTC alignment substantially more robust. Effective CTC stride: 320 (HuBERT CNN) × 2 (TemporalDownsampler) = 640 effective samples per output frame. Output lengths: `output_lengths = (output_lengths + 1) // 2`.

**Class/file:** `TemporalDownsampler` in `src/models/model.py` L580.

| Parameter | Default | Description |
|---|---|---|
| `use_temporal_downsample` | `True` | Enable/disable |
| `hidden_dim` | `768` | Input/output channels |
| `classifier_dropout` | `0.1` | Post-activation dropout |

**Architecture:** `Conv1d(768, 768, kernel_size=3, stride=2, padding=1)` → `LayerNorm` → `GELU` → `Dropout`. Output length formula: T_out = (T_in + 1) // 2.

**Forward example:** Input `[2, 49, 768]` → output `[2, 25, 768]`.

---

### PhonemeClassifier

**Purpose:** Maps HuBERT hidden states to phoneme logits and articulatory class logits via a two-layer MLP.

**Key design decision (I5 fix):** Articulatory auxiliary heads (manner, place, voice) use global average pooling over time before classification. CTC provides no forced alignment, so frame-level articulatory supervision assigns a single phoneme label to every output frame with incorrect temporal correspondence. Utterance-level mode labels are used as supervision targets instead.

**Class/file:** `PhonemeClassifier` in `src/models/model.py` L634.

| Parameter | Default | Description |
|---|---|---|
| `hidden_dim` | `512` | Projection bottleneck width |
| `classifier_dropout` | `0.1` | Dropout between layers |
| `num_phonemes` | `47` (runtime: `len(phn_to_id)`) | Set to exact vocab size at init |

**Architecture:** `Linear(768, 512)` → `LayerNorm` → `GELU` → `Dropout` → `Linear(512, 47)`.

**Forward example:** Input `[2, 25, 768]` → `logits_neural` `[2, 25, 47]`, `shared_features` `[2, 25, 512]`.

---

### LearnableConstraintMatrix

**Purpose:** An end-to-end trainable phoneme confusion matrix initialized from articulatory priors and anchored by `SymbolicKLLoss` to prevent arbitrary drift.

**Key design decision (B21 fix — Proposal P2):** Plain `log(C_static)` passed through softmax produces a flat distribution because softmax renormalizes. Temperature-sharpened initialization (`logit_C = log(C_static + ε) / init_temperature`, default T=0.5) preserves the diagonal peakedness of the prior at epoch 0. `SymbolicKLLoss` with λ=0.5 (raised from 0.05 per B22) prevents the learned matrix from drifting toward blank-dominated rows.

**KL direction:** `KL(C_learned || C_static)`, computed as `(1/V) Σ_i KL(C_learned[i] || C_static[i])` using explicit `sum/V` normalization (not `batchmean` — B22 fix rationale: `batchmean` with V=47 gives per-row weight ≈0.001, too weak to anchor the matrix).

**Class/file:** `LearnableConstraintMatrix` in `src/models/model.py` L138.

| Parameter | Default | Description |
|---|---|---|
| `use_learnable_constraint` | `True` | Enable learnable vs. static matrix |
| `init_temperature` | `0.5` | Sharpens initialization: `logit_C = log(C + ε) / 0.5` |
| `lambda_symbolic_kl` | `0.5` | KL anchor loss weight |

**Forward example:** Input `P_neural` `[2, 25, 47]` → `P_constrained` `[2, 25, 47]` via `P_neural @ softmax(logit_C)`, row-renormalized.

---

### SymbolicConstraintLayer

**Purpose:** Fuses neural phoneme posteriors with phonologically-motivated priors in a severity-adaptive manner, producing the final log-probabilities passed to CTC.

**This is the core research contribution.** The complete forward-pass equation is:

```
P_constrained = P_neural @ C                    # constraint matrix application
P_fused       = β(s)·P_constrained + (1-β(s))·P_neural  # severity-adaptive blend
β(s)          = clamp(β_base + 0.2·s/5, 0.0, 0.8)       # s ∈ [0, 5]

# Blank-frame masking: bypass constraint where blank is dominant
mask = (P_neural[:,:,0] < blank_constraint_threshold)     # threshold = 0.5
P_final = mask * P_fused + ~mask * P_neural

log_probs_constrained = log(P_final.clamp_min(1e-6))
```

**Blank-frame masking:** ~85% of CTC frames are blank-dominant. Applying the constraint to those frames amplifies blank posteriors through the blank row of C, degrading PER. Only non-blank-dominant frames receive symbolic correction.

**Adaptive β:** Control speakers (severity=0.0) get β≈0.05; severe dysarthric speakers (severity≈4.9) get β≈0.25. The maximum is clamped to 0.8 to always retain some neural contribution.

**Gradient flow:** `logit_C` in `LearnableConstraintMatrix` receives gradients through the full CTC + CE + symbolic KL path. `beta` is an `nn.Parameter` and receives gradients through the blend arithmetic.

**Class/file:** `SymbolicConstraintLayer` in `src/models/model.py` L188.

| Parameter | Default | Description |
|---|---|---|
| `constraint_weight_init` | `0.05` | Initial β_base (learnable, clamped < 0.8) |
| `severity_beta_slope` | `0.2` | Rate of β increase per normalized severity unit |
| `blank_constraint_threshold` | `0.5` | Blank-dominance gate threshold |
| `min_rule_confidence` | `0.05` | Minimum β for rule activation logging (C5 fix) |

**`_track_activations()` (B4b fix):** Wrapped in `torch.no_grad()` to prevent building an unused gradient graph during inference. Capped at 100 indices per call; `SymbolicRuleTracker` caps total activations at `_MAX_ACTIVATIONS = 50,000` (H-5 fix).

---

## Ablation Modes

Controlled via `--ablation` flag (stored in `config.training.ablation_mode`):

| Mode | SeverityAdapter | SymbolicConstraintLayer | Art Heads | Best single-split PER | Notes |
|---|---|---|---|---|---|
| `full` | ✅ | ✅ learnable C | ✅ | 0.1372 | Default production mode |
| `neural_only` | ❌ | ❌ bypassed entirely | ❌ | **0.1346** | Pure HuBERT+classifier; **globally best single-split** |
| `no_constraint_matrix` | ✅ | ❌ log-softmax of neural logits | ✅ | 0.1444 | Isolates SeverityAdapter contribution |
| `no_art_heads` | ✅ | ✅ | ❌ | — | Tests articulatory head contribution |
| `no_spec_augment` | ✅ (no aug) | ✅ | ✅ | — | Tests SpecAugment contribution |
| `no_temporal_ds` | ✅ (no DS) | ✅ | ✅ | — | Tests TemporalDownsampler contribution |
| `symbolic_only` | ✅ | ✅ | ✅ | — | CTC/CE disabled (λ=0); tests pure symbolic signal |

**Note:** `neural_only` achieves the best single-split PER (0.1346) across all evaluated configurations. See [experiments.md](experiments.md) for full analysis.

---

## HuBERT Freeze Schedule

Implemented in `DysarthriaASRLightning.on_train_epoch_start()` in `train.py` L566. The epoch counter is the **effective epoch** = `current_epoch + resume_epoch_offset`, ensuring resumed folds immediately enter the correct stage.

| Stage | Epoch trigger | Layers unfrozen | Code reference | Notes |
|---|---|---|---|---|
| Warmup | 0 | None (entire encoder frozen) | `model.freeze_encoder()` at init | Only classifier + adapter heads train |
| Stage 1 | ≥ `encoder_warmup_epochs` (default **1**) | 8, 9, 10, 11 | `train.py` L590 | Adam state reset for newly-active params |
| Stage 2 | ≥ `encoder_second_unfreeze_epoch` (default **6**) | 6, 7, 8, 9, 10, 11 | `train.py` L601 | Direct `unfreeze_encoder(layers=[6,7,8,9,10,11])` call — not `unfreeze_after_warmup()` (B14 fix) |
| Stage 3 | ≥ `encoder_third_unfreeze_epoch` (default **12**) | 4, 5, 6, 7, 8, 9, 10, 11 | `train.py` L610 | Deepest dysarthric adaptation; layers 0–3 remain frozen throughout |

Layers 0–3 are permanently frozen (`freeze_encoder_layers=[0,1,2,3]`). After each unfreeze event, `_reset_hubert_lr_warmup()` clears Adam first/second moment estimates for the newly active `hubert_encoder` parameter group. The entire per-parameter state dict entry is cleared (`optimizer.state[p].clear()`), not individual keys, to avoid a `KeyError('exp_avg')` on the next `optimizer.step()`. OneCycleLR's step counter is not reset (known limitation T-04; full fix is O-2, per-group schedulers).

---

## Multi-Task Loss

Total loss: `loss = λ_ctc·CTC + λ_ce·CE + λ_art·Art + λ_ord·Ordinal + λ_bkl·BlankKL + λ_skl·SymKL`

| Loss | Formula | λ (current) | File | Notes |
|---|---|---|---|---|
| `CTCLoss` | CTC(log_probs_constrained, labels) | **0.80** | `train.py` | `zero_infinity=True`; uses exact `output_lengths` from model, not approximate `input_lengths` from collator |
| Frame-CE | CrossEntropyLoss(logits_neural, aligned_labels) | **0.10** | `train.py` | Applied to neural logits, not constrained log-probs (C1 fix); reduced from 0.35 to mitigate frame-alignment noise |
| Articulatory CE | (CE(manner)+CE(place)+CE(voice))/3 | **0.08** | `train.py` | Utterance-level via GAP + mode label (I5 fix); reduced from 0.15 |
| `OrdinalContrastiveLoss` | Pairwise cosine hinge with ordinal margin | **0.05** | `src/models/losses.py` | Continuous TORGO severity scores; zero-gradient guard for batch_size=1 (B8 fix) |
| `BlankPriorKLLoss` | KL(Bernoulli(p̄_blank) ∥ Bernoulli(0.75)) | staged **0.10→0.15→0.20** | `src/models/losses.py` | target_prob=0.75 (B2/T-03 fix: 0.85 overshoots); staged warmup (I2) |
| `SymbolicKLLoss` | (1/V) Σ_i KL(C_learned[i] ∥ C_static[i]) | **0.50** | `src/models/losses.py` | Anchors C to symbolic prior; raised from 0.05 (B22 fix); sum/V normalization (not batchmean) |

**B22 rationale for λ_symbolic_kl=0.5:** The old value of 0.05 with `batchmean` reduction over V=47 rows gave an effective per-row weight of ≈0.001 — too weak to prevent the constraint matrix from drifting toward blank-dominated rows. At 0.5 with sum/V normalization, effective per-row weight ≈0.01.

**Staged `lambda_blank_kl` schedule (I2):**

| Epoch range | λ_blank_kl | Purpose |
|---|---|---|
| 0–9 | 0.10 | Gentle push; prevents early CTC collapse while phoneme head learns boundaries |
| 10–19 | 0.15 | Moderate push |
| ≥ 20 | 0.20 | Full target |

---

## Training Configuration Reference

### `ModelConfig` (`src/utils/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `hubert_model_id` | `facebook/hubert-base-ls960` | HuBERT variant |
| `hubert_model_revision` | `dba3bb02fda4248b6e082697eee756de8fe8aa8a` | Pinned hub revision |
| `freeze_feature_extractor` | `True` | CNN always frozen |
| `use_gradient_checkpointing` | `True` | Reduces VRAM ~15–25% |
| `freeze_encoder_layers` | `[0, 1, 2, 3]` | Permanently frozen transformer layers |
| `hidden_dim` | `512` | PhonemeClassifier hidden size |
| `num_phonemes` | `47` (runtime) | Set to `len(phn_to_id)` at init; config default is informational only |
| `classifier_dropout` | `0.1` | Dropout in classifier and adapter |
| `constraint_weight_init` | `0.05` | Initial β_base for symbolic blending |
| `constraint_learnable` | `True` | Enable learnable β parameter |
| `use_articulatory_distance` | `True` | Use articulatory distance for `C_static` fallback |
| `use_learnable_constraint` | `True` | Enable `LearnableConstraintMatrix` (Proposal P2) |
| `use_severity_adapter` | `True` | Enable cross-attention `SeverityAdapter` (Proposal P3) |
| `severity_adapter_dim` | `64` | Severity projection bottleneck |
| `use_temporal_downsample` | `True` | Enable stride-2 Conv1d downsampler |
| `use_spec_augment` | `True` | Enable SpecAugment on hidden states |
| `spec_time_mask_prob` | `0.05` | Fraction of frames to mask per sample |
| `spec_time_mask_length` | `10` | Max consecutive frames per time mask |
| `spec_freq_mask_prob` | `0.05` | Fraction of feature dims to mask per sample |
| `spec_freq_mask_length` | `8` | Max consecutive dims per frequency mask |

### `TrainingConfig` (`src/utils/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `precision` | `bf16-mixed` | BF16 on Ampere+ GPUs |
| `learning_rate` | `3e-5` | Peak LR for classifier/adapter group |
| `weight_decay` | `0.01` | AdamW weight decay |
| `optimizer` | `AdamW` | Optimizer type |
| `lr_scheduler` | `onecycle` | OneCycleLR with cosine annealing |
| `warmup_ratio` | `0.05` | Fraction of total steps for LR warmup |
| `batch_size` | `12` | Per-GPU batch size |
| `gradient_accumulation_steps` | `3` | Effective batch = 12 × 3 = 36 |
| `max_epochs` | `40` | Maximum training epochs |
| `encoder_warmup_epochs` | `1` | Stage 1 unfreeze trigger epoch |
| `encoder_second_unfreeze_epoch` | `6` | Stage 2 unfreeze trigger epoch |
| `encoder_third_unfreeze_epoch` | `12` | Stage 3 unfreeze trigger epoch |
| `gradient_clip_val` | `1.0` | Gradient norm clipping |
| `lambda_ctc` | `0.8` | CTC loss weight |
| `lambda_ce` | `0.10` | Frame-CE loss weight |
| `lambda_articulatory` | `0.08` | Articulatory CE weight |
| `lambda_ordinal` | `0.05` | Ordinal contrastive weight |
| `lambda_blank_kl` | `0.20` | Blank-prior KL weight (final stage) |
| `blank_target_prob` | `0.75` | Target mean blank probability |
| `lambda_symbolic_kl` | `0.50` | Symbolic KL anchor weight |
| `early_stopping_patience` | `8` | Epochs without improvement before stopping (paper full-system runs use 6; ablations use 8) |

**Optimizer LR groups (paper setting):** HuBERT encoder uses `0.1×` peak LR, classifier+adapter uses `1.0×`, and the constraint layer uses `0.5×` under the same OneCycle schedule.
| `ablation_mode` | `full` | Active ablation mode |

---

## Model Parameter Counts (Approximate)

| Component | Approx. Parameters | Notes |
|---|---|---|
| HuBERT Backbone | ~94.7M total; ~4.6M trainable at warmup, ~66.4M at Stage 3 | 12 transformer layers + CNN extractor |
| PhonemeClassifier | ~400K | 768→512 projection + 512→47 head |
| SeverityAdapter | ~600K | Linear(1,64)+SiLU+Linear(64,768) + MultiheadAttention(768, 8 heads) |
| TemporalDownsampler | ~2.1M | Conv1d(768,768,3) + LayerNorm |
| LearnableConstraintMatrix | ~2.2K | 47×47 logit_C parameter |
| SymbolicConstraintLayer | ~2.2K + 1 (β) | Includes logit_C via LearnableConstraintMatrix |
| Articulatory Heads | ~33K | Three Linear(512, class_count) heads |
| **Total** | **~99M** | At Stage 3: ~66.4M trainable (67.1%) |
