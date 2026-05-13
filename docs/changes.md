# Change Reference — DysarthriaNSR v0.4.0

> This document provides a structured reference for all changes implemented in v0.4.0. For a high-level summary, see [CHANGELOG.md](../CHANGELOG.md). For inline code documentation, see the `# [FIX-N]` comment blocks in the source files.

---

## Overview

This version incorporates 13 changes identified during a technical audit of the training and evaluation pipelines. The primary goals were to:

1. Increase symbolic constraint activation by lowering the blank-frame bypass threshold
2. Improve training stability by gating frame-CE loss and resetting OneCycleLR after unfreeze
3. Fix data leakage in BigramLM and improve evaluation diagnostics

Collectively, these changes are expected to improve PER by reducing symbolic layer neutralization and improving convergence during the HuBERT unfreeze phase.

---

## Change Index

| ID | Priority | File(s) | Config field(s) | Category |
|----|----------|---------|-----------------|----------|
| FIX-1 | Critical | src/models/model.py, src/utils/config.py | blank_constraint_threshold | Training quality |
| FIX-2 | Critical | train.py, src/utils/config.py | lambda_ce, frame_ce_start_epoch | Training stability |
| FIX-3 | Critical | train.py | (scheduler reset) | Training stability |
| FIX-4 | Critical | src/utils/config.py, run_pipeline.py | loso_early_stopping_patience | Training quality |
| FIX-5 | High | src/models/losses.py | (blank-mass penalty) | Training quality |
| FIX-6 | High | train.py | (attention mask calculation) | Evaluation correctness |
| FIX-7 | High | src/models/model.py | (SpecAugment gate) | Training quality |
| FIX-8 | High | train.py | (diagnostic logging) | Diagnostics |
| FIX-9 | Medium | run_pipeline.py, src/models/model.py | no_severity_adapter | Ablation support |
| FIX-10 | Medium | evaluate.py | spearman_valid | Metrics |
| FIX-11 | Medium | src/visualization/experiment_plots.py | plot_per_by_manner | Visualization |
| FIX-12 | Medium | run_pipeline.py | (BigramLM data leak) | Data correctness |
| FIX-13 | Medium | evaluate.py | (gradient checkpointing) | Evaluation efficiency |

---

## Detailed Change Notes

### FIX-1 — Blank-frame bypass threshold

**File:** `src/models/model.py`, `SymbolicConstraintLayer.forward()`  
**Config:** `src/utils/config.py`, `SymbolicConfig.blank_constraint_threshold`

**Before:** `0.5`  
**After:** `0.25`

**Motivation:** The previous threshold of 0.5 caused ~85% of CTC frames to bypass the symbolic constraint entirely, since most CTC frames have blank probability > 0.5. This effectively neutralized the symbolic layer's influence on most predictions. Lowering to 0.25 allows the constraint to act on ~35% of frames (ambiguous transition frames) while still protecting blank-dominant frames.

**Interactions:** Works in conjunction with FIX-5 (blank-mass penalty) to address constraint matrix drift. The lowered threshold increases constraint activation, which requires the KL anchor to be stronger to prevent the matrix from drifting toward blank-dominated rows.

**Verification:** Check `val/constraint_kl_from_prior` remains < 1.0. The constraint matrix should show increased activation on non-blank frames.

---

### FIX-2 — Frame-CE loss gating

**File:** `train.py::compute_loss`  
**Config:** `src/utils/config.py`, `TrainingConfig.lambda_ce`, `TrainingConfig.frame_ce_start_epoch`

**Before:** `lambda_ce=0.35`, always active  
**After:** `lambda_ce=0.10`, gated until `epoch >= 15`

**Motivation:** Frame-CE uses proportional nearest-neighbor interpolation to map phoneme labels to logit time dimensions. CTC does not provide forced alignment, so this assignment is approximate and often incorrect. During early training, the model should focus on CTC's sequence-level alignment before receiving potentially conflicting frame-level supervision.

**Interactions:** Lower lambda_ce requires `lambda_symbolic_kl=0.5` to remain strong enough to anchor the constraint matrix. This combination was validated in baseline_v6.

**Verification:** Check training log shows `frame_ce_enabled: false` for epochs 0-14. After epoch 15, verify `train/loss_ce` contributes meaningfully to total loss.

---

### FIX-3 — OneCycleLR warm restart after unfreeze

**File:** `train.py::_reset_hubert_lr_warmup()`  
**Config:** No config change (implementation fix)

**Before:** Adam state cleared but OneCycleLR scheduler continued from current position  
**After:** Adam state cleared + scheduler step counter reset for warm restart

**Motivation:** Previously, newly unfrozen HuBERT parameters entered at the decayed LR position rather than the original peak. This caused convergence slowdown after each unfreeze stage (epochs 1, 6, 12). The warm restart gives newly unfrozen layers a fresh warmup period.

**Interactions:** Addresses T-04 (OneCycleLR + unfreeze interaction). The full O-2 fix (per-group schedulers) is still pending.

**Verification:** Check training curve shows sustained improvement after each unfreeze stage, not plateau. LR should ramp up again at epochs 1, 6, 12.

---

### FIX-4 — LOSO early stopping patience

**File:** `src/utils/config.py`, `TrainingConfig`  
**Config:** `TrainingConfig.losso_early_stopping_patience`

**Before:** `8` (same as single-split)  
**After:** `22`

**Motivation:** Full-system runs (with SymbolicConstraintLayer, SeverityAdapter, and articulatory heads) require longer convergence than ablation runs. The additional loss components need more epochs to stabilize. The original patience of 8 was too aggressive, causing early termination before convergence.

**Interactions:** Single-split runs continue to use `early_stopping_patience=8` as documented.

**Verification:** LOSO runs should complete at least 20+ epochs on most folds without early stopping.

---

### FIX-5 — Blank-mass penalty in SymbolicKLLoss

**File:** `src/models/losses.py`, `SymbolicKLLoss`  
**Config:** No config change (implementation fix)

**Before:** No explicit blank-row penalty  
**After:** Added explicit penalty for blank-dominated rows in constraint matrix

**Motivation:** The constraint matrix rows corresponding to blank (index 0) can drift toward uniform distribution due to the nature of CTC blanks. This drift reduces the constraint's discriminative power. Adding a blank-mass penalty keeps blank rows constrained.

**Interactions:** Works in conjunction with FIX-1 (lowered blank_constraint_threshold). Both changes address different aspects of constraint matrix health: FIX-1 increases activation, FIX-5 prevents degradation.

**Verification:** Check `val/constraint_row_entropy` stays in moderate range (not approaching 0 or log(47)).

---

### FIX-6 — Attention mask downsampling

**File:** `train.py::_downsample_attn_mask()`  
**Config:** No config change (implementation fix)

**Before:** Implicit stride calculation could produce incorrect mask lengths  
**After:** Explicit two-step stride calculation with proper padding handling

**Motivation:** The TemporalDownsampler uses stride-2 Conv1d which affects attention mask alignment. The previous implicit calculation sometimes produced masks that didn't match the actual downsampled sequence length, causing shape mismatches during evaluation.

**Interactions:** Works with TemporalDownsampler to ensure mask alignment is correct.

**Verification:** Run evaluation and verify no shape mismatch errors. Check that mask lengths match downsampled feature lengths.

---

### FIX-7 — SpecAugment gate decoupling

**File:** `src/models/model.py`, `SpecAugmentLayer`  
**Config:** No config change (implementation fix)

**Before:** SpecAugment skipped when `_hubert_is_frozen=True`  
**After:** SpecAugment uses `self.training` only

**Motivation:** Previously, SpecAugment was incorrectly skipped whenever any HuBERT layer was frozen, even though downstream modules (PhonemeClassifier, SeverityAdapter, SymbolicConstraintLayer) were actively training. The decision should be based on overall model training state, not encoder freeze state.

**Interactions:** Ensures SpecAugment is active during all training epochs, including warmup (when HuBERT is frozen but other modules are training).

**Verification:** Check that augmentation is applied during warmup epochs (0+) in training logs.

---

### FIX-8 — SeverityAdapter diagnostic logging

**File:** `train.py::validation_step()`  
**Config:** No config change (diagnostics addition)

**Before:** No output norm monitoring for SeverityAdapter  
**After:** Added output norm logging for debugging adapter behavior

**Motivation:** SeverityAdapter can produce degenerate outputs (NaN, all-zero, extreme values) when severity projection or cross-attention encounters numerical issues. Adding diagnostic logging helps identify these issues early.

**Interactions:** Helps diagnose issues with SeverityAdapter contributions to final loss.

**Verification:** Check validation logs for `val/severity_adapter_output_norm`. Should be non-zero and in reasonable range.

---

### FIX-9 — no_severity_adapter ablation mode

**File:** `run_pipeline.py`, `src/models/model.py`  
**Config:** `ablation_mode` option

**Before:** No ablation mode to disable SeverityAdapter only  
**After:** Added `no_severity_adapter` mode

**Motivation:** Allows researchers to isolate the contribution of SeverityAdapter without requiring the full neural-only baseline. This is useful for understanding the relative importance of severity conditioning vs. symbolic constraint.

**Interactions:** Different from `neural_only` which bypasses both SeverityAdapter and SymbolicConstraintLayer.

**Verification:** Run with `--ablation no_severity_adapter` and verify model trains with symbolic constraint but without severity adapter.

---

### FIX-10 — spearman_valid flag

**File:** `evaluate.py`  
**Config:** No config change (metrics addition)

**Before:** No validity flag for correlation metrics  
**After:** Added `spearman_valid` and `correlation_valid` flags in evaluation stats

**Motivation:** Correlation metrics (Pearson r, Spearman ρ) are only statistically meaningful with n >= 5 speakers. With fewer speakers, the confidence interval spans nearly [-1, 1]. The validity flag signals when these metrics can be trusted.

**Interactions:** Related to B15 (macro-speaker PER) — both address statistical validity of metrics.

**Verification:** Check `evaluation_results.json['stats']['correlation_valid']` is True only for LOSO (15 speakers), False for single-split (~2 speakers).

---

### FIX-11 — plot_per_by_manner visualization

**File:** `src/visualization/experiment_plots.py`, `scripts/generate_figures.py`  
**Config:** No config change (visualization addition)

**Before:** No articulatory-stratified PER visualization  
**After:** Added PER breakdown by manner of articulation

**Motivation:** Provides error analysis stratified by articulatory features (stop, fricative, nasal, liquid, vowel, etc.). Helps identify which phoneme categories are most challenging for the model.

**Interactions:** Related to articulatory auxiliary heads (manner/place/voice classification).

**Verification:** Check that `per_by_manner.json` is generated and `per_by_manner.png` appears in results directory.

---

### FIX-12 — BigramLM data leak fix

**File:** `run_pipeline.py`, `BigramLMScorer`  
**Config:** No config change (data correctness fix)

**Before:** BigramLM built from all phoneme sequences in manifest  
**After:** BigramLM built from training speakers only

**Motivation:** Including validation/test speaker sequences in LM training causes optimistic perplexity estimates and potential data leakage. The LM should only see training data to avoid leakage.

**Interactions:** Affects LM-based decoding metrics (beam search with LM). Make sure to rebuild LM when changing data splits.

**Verification:** Check LM training speaker list matches training set only. Run evaluation with `--lm-weight 0.3` and verify perplexity is realistic.

---

### FIX-13 — Gradient checkpointing in evaluation

**File:** `evaluate.py`  
**Config:** No config change (evaluation efficiency fix)

**Before:** Gradient checkpointing enabled during evaluation  
**After:** Gradient checkpointing disabled during evaluation

**Motivation:** Gradient checkpointing saves memory at the cost of recomputing some activations during the forward pass. During evaluation, memory savings are not needed, and the recomputation overhead slows down inference. Disabling it improves evaluation throughput.

**Interactions:** No impact on model outputs. Only affects evaluation speed and memory usage.

**Verification:** Evaluate should run faster and use slightly more memory (but still within limits).

---

## What NOT to Change

The following were evaluated but intentionally unchanged:

- `SymbolicKLLoss` `sum/V` normalization — correctly anchors constraint matrix
- Peak normalization absence in `_load_audio` — avoids double normalization with HuBERT
- `-100` label padding sentinel — correct for CTC/CE losses
- `init_temperature=0.5` in `LearnableConstraintMatrix` — preserves diagonal peakedness
- `blank_priority_weight=1.0` — appropriate blank prioritization
- `WeightedRandomSampler` speaker-level weights — correctly equalizes speaker contribution
- `output_lengths = (output_lengths + 1) // 2` formula — correct downsampling

If any of these appear in code context near a documented change, they are intentionally unchanged. See the audit notes for rationale.

---

## Rollback Notes

If issues arise after these changes:

| Change | Quick Revert |
|--------|--------------|
| FIX-1 | Set `blank_constraint_threshold = 0.5` in config |
| FIX-2 | Set `lambda_ce = 0.35`, remove `frame_ce_start_epoch` gating |
| FIX-3 | Comment out scheduler reset in `_reset_hubert_lr_warmup` |
| FIX-4 | Set `loso_early_stopping_patience = 8` in config |
| FIX-5 | Remove blank-mass penalty from `SymbolicKLLoss` |
| FIX-7 | Restore `_hubert_is_frozen` check in SpecAugment |
| FIX-12 | Remove training-speaker filter from BigramLM builder |