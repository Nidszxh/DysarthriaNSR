# DysarthriaNSR — Action Items & Implementation Plan

> **Last updated:** March 12, 2026
> **Status:** 11/11 March audit items implemented. Critical open issue: symbolic constraint actively hurts PER (+0.170). Must be resolved before SPCOM 2026 submission.

---

## Table of Contents

1. [Execution Order Summary](#1-execution-order-summary)
2. [Critical — Submission-Blocking](#2-critical--submission-blocking)
3. [High Impact — Strongly Recommended](#3-high-impact--strongly-recommended)
4. [Medium Impact — Recommended Before Camera-Ready](#4-medium-impact--recommended-before-camera-ready)
5. [Optional — Post-Submission / Future Work](#5-optional--post-submission--future-work)
6. [Risk Register](#6-risk-register)
7. [SPCOM 2026 Paper Contributions Checklist](#7-spcom-2026-paper-contributions-checklist)

---

## 1. Execution Order Summary

```
Phase 0 — Quick Code Fixes (< 2h total, do first — most already done):
  ✅ H-1  del trainer/lm between LOSO folds
  ✅ H-3  blank histogram target 0.82 → 0.75
  ✅ H-4  EarlyStopping monitor → 'val_per'
  ✅ H-5  Cap SymbolicRuleTracker._activations
  ✅ H-6  Reuse per-sample PER in stratified_per
  ✅ M-5  Document conformal tau as heuristic
  ✅ M-6  Sort per-speaker plot by severity

Phase 1 — Training Experiments (sequential, order matters):
  C-1  Retrain baseline_v6 with lambda_ce=0.10     [~4h GPU] ← config already set
  C-2  Run neural-only ablation                    [~4h GPU]
  C-3  LOSO-CV loso_v1 (with best config)          [~18h GPU]

Phase 2 — Code Fixes & Figures (after Phase 1 data exists):
  ✅ C-4  Severity-PER scatter plot
  ✅ C-5  Rule-pair confusion bar chart
  ✅ H-2  Cache phoneme alignments (4× speedup)
  N3   Add paired significance test (neural vs constrained)    [1h code]
  N4   BigramLMScorer Laplace smoothing                        [30min code]

Phase 3 — Test Coverage & Medium Items:
  M-1  T-05 proportional label alignment + retrain [2h code + 4h GPU]
  M-2  Integration test (training_step)            [1h]
  M-3  Integration test (evaluate_model)           [1h]
  M-4  TORGO clinical severity buckets             [30min]

Phase 4 — Optional (post-submission):
  O-1  CTC forced alignment
  O-2  Per-group CosineAnnealingWarmRestarts
  O-3  Configurable blank threshold
  O-4  SpecAugment torch.Generator
  O-5  rapidfuzz alignment speedup
  O-6  ONNX export / streaming
  O-7  Non-TORGO generalization
```

---

## 2. Critical — Submission-Blocking

These items block submission. Without them, the paper's central claim ("neuro-symbolic integration improves dysarthric ASR") cannot be made.

---

### C-1 · Retrain `baseline_v6` with `lambda_ce=0.10`

**Difficulty:** Moderate (config already set; ~4h training run)
**Impact:** Submission-blocking — most likely single fix for `per_constrained > per_neural`

**Why it's critical:** The frame-CE loss is applied to positionally-misaligned phoneme labels (pad/truncate, not proportional). At `lambda_ce=0.35`, CE provided 30% of the total loss signal with near-random frame→phoneme associations, which may train the phoneme classifier in a way that conflicts with the symbolic layer's distribution-shifting. Reducing CE weight removes this confounding signal.

The config change is already applied (`lambda_ce=0.10`). Steps:

```bash
python run_pipeline.py --run-name baseline_v6
```

Compare `per_constrained` vs `per_neural` in `results/baseline_v6/evaluation_results.json`.

**Contingency — if constraint still hurts:** Set `lambda_ce=0.00` and retrain `baseline_v6_noce`. Also consider reducing `constraint_weight_init` to 0.01 or clamping β=0 at eval time to obtain a clean neural-only published baseline while reporting the symbolic layer as an analysis tool.

---

### C-2 · Run Neural-Only Ablation

**Difficulty:** Hard (~4h training run)
**Impact:** Submission-blocking — the current `per_neural=0.305` is the greedy decode of the neural sub-path inside the jointly-trained full model, not an independent ablation. A true neural-only baseline trains without the symbolic layer entirely.

```bash
python run_pipeline.py --run-name ablation_neural_v1 --ablation neural_only \
    --beam-search --beam-width 25
```

**Expected outcome table for paper:**

| Model | Beam PER | Δ vs Neural |
|-------|----------|------------|
| `ablation_neural_v1` (neural-only) | ? | — |
| `baseline_v6` (full model) | ? | target: negative (improvement) |
| `baseline_v5` (full model, prev) | 0.4750 | +0.170 (constraint hurts) |

---

### C-3 · Run LOSO-CV (`loso_v1`)

**Difficulty:** Very Hard (~15–22h GPU; run after C-1 confirms constraint fix)
**Impact:** Submission-blocking — n=3 test speakers is statistically invalid; no valid macro-PER or severity correlation possible

**Prerequisites:**
- C-1 must complete first (run LOSO with best-performing config)
- H-1 (`del trainer, lm` between folds) already applied

```bash
python run_pipeline.py --run-name loso_v1 --loso
```

**Deliverables from LOSO:**
- Macro-speaker PER ± 95% CI (n=15 speakers — statistically valid)
- Spearman ρ (severity vs PER) with p-value (n=15)
- Per-fold PER table for supplementary material

---

### N3 · Add Paired Significance Test (Neural vs Constrained)

**Difficulty:** Easy (1h code)
**Impact:** Required for SPCOM — per\_neural and per\_constrained are reported side-by-side with no p-value

Add `bootstrap_paired_per_test(per_scores_neural, per_scores_constrained, n_bootstrap=10000)` to `evaluate_model` and include `p_value_neural_vs_constrained` and `ci_95_delta_per` in `evaluation_results.json`.

---

### N4 · Fix `BigramLMScorer` Zero-Count Transitions

**Difficulty:** Easy (30min)
**Impact:** High — unseen phoneme bigrams return log(0) = −∞, permanently killing beam hypotheses. Critical for dysarthric speech which produces non-standard phoneme sequences.

In `evaluate.py::BigramLMScorer._lm_score`, replace:
```python
count = self.bigrams.get((prev, cur), 0)
```
with:
```python
count = self.bigrams.get((prev, cur), 0) + 1  # Laplace smoothing
# Also add V to the denominator normalization
```

---

## 3. High Impact — Strongly Recommended

These prevent GPU OOM during LOSO, improve figure quality, and eliminate misleading annotations. Most are already implemented (see §8 of Completed Work).

---

### H-1 · GPU OOM Fix in LOSO ✅ DONE

Added `del trainer, lm; torch.cuda.empty_cache()` between folds in `train.py::run_loso()`.

---

### H-2 · Cache Phoneme Alignments in `evaluate_model` ✅ DONE

`phoneme_alignment()` was called 4× per prediction pair. `_all_alignments` and `_neural_alignments` are now pre-computed once and passed to all four analysis functions. Estimated 3–4× speedup for evaluation.

---

### H-3 · Fix `plot_blank_histogram` Target Line ✅ DONE

Was hardcoded at 0.82; now reads from `config.training.blank_target_prob` (0.75).

---

### H-4 · Unify `EarlyStopping` Monitor ✅ DONE

Changed from `val/per` to `val_per` in `run_loso()` to match `ModelCheckpoint`.

---

### H-5 · Cap `SymbolicRuleTracker._activations` ✅ DONE

List capped at 50,000 entries. `flush()` method computes summary stats then clears.

---

### H-6 · Reuse Per-Sample PER in `_compute_stratified_per` ✅ DONE

Eliminates double `compute_per()` call per validation utterance.

---

## 4. Medium Impact — Recommended Before Camera-Ready

---

### M-1 · T-05: Proportional Label Alignment in CE Loss

**Difficulty:** Moderate (2h code + ~4h GPU for validation)
**Impact:** High potential — likely root cause of `per_constrained > per_neural`; fixes frame→phoneme CE misalignment for dysarthric speakers with high duration variance

Replace pad/truncate in `src/utils/sequence_utils.py::align_labels_to_logits()`:

```python
def align_labels_to_logits(labels: torch.Tensor, time_steps: int) -> torch.Tensor:
    """Proportional nearest-neighbour interpolation instead of pad/truncate."""
    B, L = labels.shape
    if L == time_steps:
        return labels
    indices = (torch.arange(time_steps, device=labels.device).float()
               * (L / time_steps)).long().clamp(0, L - 1)
    aligned = labels[:, indices]
    # Preserve -100 padding
    pad_mask = (labels == -100)
    if pad_mask.any():
        pad_fraction = pad_mask.float().mean(dim=1)
        n_pad = (pad_fraction * time_steps).long()
        for b in range(B):
            aligned[b, time_steps - n_pad[b]:] = -100
    return aligned
```

Retrain as `baseline_v7_t05` and compare `per_constrained` delta.

---

### M-2 · Integration Test: `training_step`

**Difficulty:** Easy (1h)
**Impact:** No test currently exercises the full forward → loss → backward → optimizer path

```python
# tests/test_training_step.py
def test_one_training_step():
    config = Config()
    config.training.max_epochs = 1
    config.model.freeze_encoder_layers = list(range(12))  # fully frozen = fast
    # Build synthetic batch of 2 samples
    # Assert loss.item() is finite and > 0
```

---

### M-3 · Integration Test: `evaluate_model`

**Difficulty:** Easy (1h)
**Impact:** `evaluate_model` is 890+ lines with complex branching; no end-to-end test exists; a breakage would only surface at end of a training run

---

### M-4 · Severity Stratification: Use TORGO Clinical Buckets

**Difficulty:** Easy (30min)
**Impact:** Current bucket scheme (mild=[0,2) / moderate=[2,4) / severe=[4,5]) collapses to dysarthric vs control; use Rudzicz 2012 labels (normal/mild/moderate/severe/profound) for clinically meaningful groups

---

### M-5 · Document `conformal_phoneme_sets` τ as APS-Heuristic ✅ DONE

Docstring updated in `src/models/uncertainty.py` to clarify this is not calibrated conformal prediction.

---

### M-6 · `plot_per_by_speaker` Sort by Severity ✅ DONE

Sort order changed from PER-descending to severity-ascending (controls first).

---

### N1 · Replace Binary Severity with Continuous in `OrdinalContrastiveLoss`

**Difficulty:** Low (add `severity_continuous` column to manifest)
**Impact:** Eliminates zero-margin control–control pairs that contribute no gradient

Add `severity_continuous` column to manifest (each speaker's individual `TORGO_SEVERITY_MAP` score). Update `TorgoNeuroSymbolicDataset.__getitem__` to expose `batch['severity_continuous']`. Use in `OrdinalContrastiveLoss` instead of `batch['status'].float() * 5.0`.

---

### N2 · Fix `_PROBABLE_CAUSE_MAP` Key Format in `attribution.py`

**Difficulty:** Low
**Impact:** Currently 4-tuple keys are never matched; all causes return "Unknown articulatory substitution"

Change lookup to use 2-tuple `(manner_ref, manner_pred)` or `(voice_ref, voice_pred)` keys to match what `_infer_probable_cause` actually constructs.

---

### N5 · Canonicalize `PHONEME_DETAILS` / `PHONEME_FEATURES` to Single Source

**Difficulty:** Medium (refactor)
**Impact:** Currently defined independently in `manifest.py` and `model.py`; must be manually kept in sync

Move to `src/utils/constants.py` and import in both files.

---

### N6 · Update `BlankPriorKLLoss` Default and Test Fixture to `target_prob=0.75`

**Difficulty:** Trivial
**Impact:** Class default (0.85) and test fixture (0.85) diverge from training config (0.75); no correctness impact but misleading

---

### N7 · Wire `rule_precision()` into `evaluate_model` Output

**Difficulty:** Low
**Impact:** Clinically meaningful metric (fraction of rule activations that produced correct final prediction); currently implemented but never called

---

## 5. Optional — Post-Submission / Future Work

---

### O-1 · CTC Forced Alignment via `torchaudio.functional.forced_align`

**Difficulty:** Moderate-Hard (4–8h + validation)
**Impact:** High (post-submission) — enables `PhonemeAttributor.attention_attribution` (currently disabled); provides real frame→phoneme boundaries for CE supervision; directly fixes T-05 at the source

```python
from torchaudio.functional import forced_align
frame_labels = forced_align(log_probs.cpu(), labels.cpu(), blank=0)
# Use frame_labels instead of align_labels_to_logits() output for CE loss
```

---

### O-2 · Per-Group `CosineAnnealingWarmRestarts` for Unfreezing Schedule

**Difficulty:** Moderate (3h refactor of `configure_optimizers`)
**Impact:** Medium — newly-unfrozen param groups enter OneCycleLR at potentially-decayed LR positions; per-group restart eliminates this edge case

---

### O-3 · Configurable Blank Masking Threshold in `SymbolicConstraintLayer`

**Difficulty:** Trivial (add config field)
**Impact:** Medium — currently hardcoded at 0.5; enables ablation over constraint aggressiveness

Add `blank_frame_threshold: float = 0.5` to `ModelConfig`, pass to `SymbolicConstraintLayer.__init__`.

---

### O-4 · SpecAugment `torch.Generator` for Reproducible Masking

**Difficulty:** Easy (20min)
**Impact:** Low — improves reproducibility in distributed/multi-worker settings

---

### O-5 · `rapidfuzz` for Phoneme Alignment (10–20× Speedup)

**Difficulty:** Easy (30min)
**Impact:** Low (evaluation speed only) — useful if running many ablations

```python
from rapidfuzz.distance import Levenshtein as RL
ops = RL.editops(ref_seq, pred_seq)
```

Add `rapidfuzz==3.x` to `requirements.txt`.

---

### O-6 · ONNX Export + Streaming Inference

**Difficulty:** Hard (8–16h)
**Impact:** Medium (clinical deployment) — prerequisite for SLP dashboard demo; not needed for SPCOM paper

---

### O-7 · Non-TORGO Dataset Evaluation (Generalization Claim)

**Difficulty:** Very Hard (new dataset pipeline + training)
**Impact:** High (journal version) — SPCOM 4-page format cannot accommodate; target journal extension

---

## 6. Risk Register

| Risk | Probability | Mitigation |
|------|-------------|------------|
| C-1 (`lambda_ce=0.10`) does not fix constraint degradation | Medium | Run `lambda_ce=0.00` ablation; fall back to reporting neural-only as primary result with honest analysis of constraint failure |
| C-3 LOSO GPU OOM | Low (H-1 already applied) | H-1 is applied; monitor VRAM after first fold |
| LOSO results show high variance (Δ\_PER unstable across folds) | Low-Medium | Report macro ± SD; use Wilcoxon signed-rank on per-fold Δ |
| `align_labels_to_logits` proportional fix (M-1) worsens performance | Low | Keep current pad/truncate as fallback; test on held-out val first |
| SPCOM deadline before LOSO completes | Medium | Prioritize C-1 → C-2 → quick-fix items; LOSO can be in revision if reviewers request it |

---

## 7. SPCOM 2026 Paper Contributions Checklist

### Minimum Requirements

| Contribution Claim | Supported? | Blocking Item |
|-------------------|-----------|---------------|
| Neuro-symbolic integration improves over neural-only baseline | ❌ Not yet | C-1 + C-2 |
| Speaker-independent LOSO evaluation (n=15) | ❌ Not yet | C-3 |
| Severity correlates with PER (statistically, n≥5) | ❌ Not yet | C-3 |
| Articulatory accuracy (manner 78.3%, place 79.3%, voice 92.3%) | ✅ baseline\_v5 | — |
| Bootstrap CI on macro-speaker PER | ✅ Implemented | Needs n=15 from LOSO |
| Per-phoneme error analysis + clinical error taxonomy | ✅ Available | — |
| Explainability (per-utterance JSON) | ✅ explanations.json | — |
| Paired significance test neural vs constrained | ❌ Not yet | N3 |
| Beam search with LM fusion (no −∞ hypothesis killing) | ❌ Not yet | N4 Laplace smoothing |

### Architecture Decisions Still Required

1. **Primary result metric:** If the symbolic constraint remains harmful after baseline\_v6, the paper's primary metric must be `per_neural_ablation` (independent, beam-decoded). The constraint layer would be reported as an analysis/explainability tool, not an accuracy-improving component. This reframes the contribution as "neuro-symbolic analysis of dysarthric phoneme errors."

2. **Severity representation:** Binary 0/5 vs continuous 0–5 per speaker. For `OrdinalContrastiveLoss` and `SeverityAdapter` fairness, continuous scores from `TORGO_SEVERITY_MAP` should be the standard.

3. **LOSO vs stratified split:** LOSO is mandatory for speaker-independent claims. The stratified-split results (v3–v5) are development experiments, not main results.
