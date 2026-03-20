# DysarthriaNSR — Research Audit & Technical Risks

> **Audit dates:** March 5–20, 2026 | **Baselines evaluated:** `baseline_v4`, `baseline_v5`, `baseline_v6`, `ablation_neural_only_v7`, `ablation_no_constraint_matrix_v6`, `loso_v1`
> This document consolidates findings from the Senior Research Scientist Audit, the Comprehensive Implementation Audit (IMPLEMENT.md), and the Full Repository Audit Report. It is the reference for research decisions and paper preparation.

---

## Table of Contents

1. [Executive Summary & Scores](#1-executive-summary--scores)
2. [Critical Research Blockers](#2-critical-research-blockers)
3. [Architectural & Training Risks](#3-architectural--training-risks)
4. [Statistical & Scientific Validity Issues](#4-statistical--scientific-validity-issues)
5. [Data Pipeline Assessment](#5-data-pipeline-assessment)
6. [Loss Function Analysis](#6-loss-function-analysis)
7. [Evaluation Methodology Issues](#7-evaluation-methodology-issues)
8. [Explainability & Visualization Issues](#8-explainability--visualization-issues)
9. [Test Coverage Assessment](#9-test-coverage-assessment)
10. [Research Enhancement Proposals](#10-research-enhancement-proposals)
11. [Verification Tests](#11-verification-tests)
12. [Issue Severity Index](#12-issue-severity-index)
13. [Time-Critical Next Steps (March 17)](#13-time-critical-next-steps-march-17)

---

## 1. Executive Summary & Scores

The DysarthriaNSR codebase is architecturally complete and functionally runnable. All 23 historical bugs (B1–B23) and all March audit fixes (H-1 through H-6, M-5, M-6, C-1, C-4, C-5) are implemented. Several previously open evaluation issues (LM smoothing, paired constrained-vs-neural significance test, severity plot n annotation, and probable-cause key mismatch) are now fixed in code. The remaining publication blockers are primarily experimental: LOSO-CV completion and demonstrating a robust symbolic advantage. Two low-risk code hygiene issues identified in the prior audit have now also been closed: `create_single_dataloader` uses speaker-balanced sampling, and `ModelConfig.num_phonemes` now matches the runtime vocabulary size.

### Quality Scores

| Dimension | Score | Notes |
| --------- | ----- | ----- |
| Code correctness | 8.5/10 | B1–B23 fixed; `_PROBABLE_CAUSE_MAP` mismatch fixed; residual CE alignment caveat remains |
| Architecture design | 9/10 | Clean layering, single config source, ablation support |
| Performance efficiency | 7/10 | Alignment caching applied (H-2); LOSO memory leak fixed (H-1) |
| Statistical rigor | 8/10 | Bootstrap CI + paired constrained-vs-neural delta test implemented; LOSO completed (15 folds) |
| Visualization quality | 8/10 | Severity scatter (C-4) and rule confusion bar (C-5) added; publication-ready |
| Test coverage | 8/10 | 7 smoke + 9 pytest, including synthetic integration tests for `training_step` and `evaluate_model` |
| Research validity | 6.5/10 | Symbolic evidence mixed in v6/ablations; frame-CE misalignment still partially mitigated |

**Overall: 7.9/10 — Solid research codebase; short-term risk is now mostly experimental validation, not missing core implementation.**

---

## 2. Critical Research Blockers

### 2.1 Symbolic Constraint Layer Actively Hurts PER

**Status:** ⚠️ Open (reframed) — baseline\_v6 and ablations completed; symbolic story is mixed

**Current evidence (latest runs):**

```text
baseline_v6:
  avg_per: 0.1372
  per_neural: 0.1451
  per_constrained: 0.1372
  p_value_neural_vs_constrained: 0.0
  helpful/neutral/harmful: 9.16% / 87.06% / 3.78%

ablation_neural_only_v7:
  avg_per: 0.1346  (best overall among the three)

ablation_no_constraint_matrix_v6:
  avg_per: 0.1444  (worse than baseline_v6)
```

Interpretation: the constrained path now improves over the model's internal neural sub-path in `baseline_v6`, but a fully neural ablation still achieves the best global PER/WER. This supports a "symbolic helps in-system but not yet SOTA overall" conclusion.

**Why the apparent degradation is real:** B21 (temperature sharpening) and B22 (lambda\_symbolic\_kl 0.05→0.5) were applied before v5 and results are unchanged, confirming the issue is not the constraint matrix initialization.

**Root cause analysis (in estimated likelihood order):**

1. **Frame-CE trains the classifier using incorrect frame→phoneme mapping (most likely).** At `lambda_ce=0.35`, CE contributed 30% of the total loss signal based on positionally-wrong phoneme→frame assignments. The phoneme classifier learns to front-load predictions in a way that conflicts with the symbolic layer's distribution-shifting. Mitigation applied: `lambda_ce` reduced to 0.10 (C-1). Full fix requires C-1 retraining.

2. **`LearnableConstraintMatrix` may still drift toward blank-dominated rows.** Despite `lambda_symbolic_kl=0.5`, the constraint matrix may gradually assign highest weight to `<BLANK>` for most phonemes (since BLANK is the most common CTC prediction). The `val/constraint_kl_from_prior` metric should be monitored — values >1.0 indicate drift.

3. **β underestimation for control speakers.** Control speakers receive `severity=0.0` → `β≈0.05`. At β=0.05, the symbolic constraint has only 5% influence. For control speakers whose neural PER is low (~0.10), even this tiny nudge can degrade performance.

4. **Static substitution rules mismatch TORGO's actual confusion patterns.** The 20 hard-coded substitution rules encode general dysarthric phonology. If the 8 TORGO dysarthric speakers follow different patterns, `C_static` encodes wrong priors.

5. **Blank threshold gate is self-defeating.** The constraint only applies where `P_neural[BLANK] < 0.5` — precisely the frames where the neural model's prediction is most meaningful and most likely correct.

**Recommended next steps:**

- Keep `baseline_v6` as the symbolic reference checkpoint (already run)
- Run a focused symbolic-strength sweep (`constraint_weight_init`, LM weight, and rule-confidence scaling) under one fixed seed/split protocol
- Promote symbolic as primary only if non-inferior to neural-only globally and better on targeted dysarthric strata

---

### 2.2 Test Split Statistical Validity (LOSO)

**Status:** ✅ Resolved — LOSO-CV completed (`loso_v1`, 15/15 folds)

TORGO has 15 speakers. A 70/15/15 random-stratified split yields approximately 10 train / 2 val / 2 test speakers. Every published test PER figure is computed on ≈2 speakers.

- Severity correlation: Pearson r=-0.85, p=0.353 (n=3) — not significant by construction
- Dysarthric vs. control comparison: n=1 vs. n=2 — not a valid comparison
- Inverted severity ordering (Control PER > Dysarthric PER) is almost certainly an artefact

Post-LOSO summary (`results/loso_v1_loso_summary.json`):
- macro PER: 0.2848 (95% CI: 0.1921–0.3801)
- weighted PER: 0.2299
- macro WER: 0.3362
- weighted WER: 0.2631

---

## 3. Architectural & Training Risks

### 3.1 SpecAugment Batch-Uniform Masking ✅ Fixed (B13/v5)

All samples in a batch received identical mask positions. Fixed: per-sample independent masking in `model.py`.

---

### 3.2 Stage 2 Progressive Unfreezing Made Stage 3 a No-Op ✅ Fixed (B14/v5)

Stage 2 called `unfreeze_after_warmup()` which unfroze ALL layers. Fixed: Stage 2 now calls `unfreeze_encoder(layers=[6..11])` directly.

---

### 3.3 Frame-CE Loss Applied to Unaligned Labels (T-05)

**Status:** ⚠️ Partially mitigated — `lambda_ce=0.10`; full fix requires CTC forced alignment

CTC does not provide forced alignment. The frame-CE loss pads phoneme labels `[B, L]` to `[B, T]` with `-100`, assigning ground-truth labels to the first L frames. For a 3-phoneme utterance decoded as 150 frames, CE loss supervises only frames 0–2 with meaningless positional assignments.

**Impact:** The CE loss gradient assigns frame-level errors to whichever phoneme label falls in that position after pad/truncate, biasing the classifier to front-load predictions. This is the most likely cause of `per_constrained > per_neural`.

**Short-term mitigation:** `lambda_ce=0.10` (C-1, applied). **Long-term fix:** CTC segmentation via `torchaudio.functional.forced_align` or removing frame-CE entirely.

---

### 3.4 Checkpoint Monitoring Metric ≠ Publication Metric ✅ Fixed (B15/v5)

`val/per` in `on_validation_epoch_end` used batch-mean-of-means. Fixed: Now accumulates `(pred, ref, speaker)` triples and computes per-speaker PER then macro-average.

---

### 3.5 Attention Mask Stride Ignores TemporalDownsampler ✅ Fixed (B16/v5)

`ctc_stride = 320` but effective stride with TemporalDownsampler is 640. Fixed: `ctc_stride = 320 * 2`.

---

### 3.6 `LearnableConstraintMatrix` Initialization ✅ Partially Fixed (B21/v5)

`softmax(log(C_static))` flattens high-confidence diagonal entries. Fix: initialize with temperature=0.5 (`logit_C = log(C_static + ε) / 0.5`). Applied in v5. Full effectiveness not yet confirmed (see §2.1).

---

### 3.7 Beam Search Length Normalisation ✅ Fixed (B19/v5)

Length norm was dividing `(acoustic + lm_bonus)` by `length^alpha`. LM scores scale naturally with length, causing double-penalization of longer hypotheses. Fixed: length norm applied to acoustic score only.

---

### 3.8 `OrdinalContrastiveLoss` Zero-Margin for Control Pairs

**Status:** ✅ Fixed in training/evaluation forward path (fallback remains binary if speaker IDs unavailable)

All 7 control speakers have `severity=0.0` (binary proxy). Any within-batch control–control pair has `margin=0`, reducing the loss to `ReLU(-cosine_sim)` ≈ 0 (since mean-pooled HuBERT features are nearly always non-negatively aligned). Control–control pairs contribute no gradient — approximately half of all pairs in a balanced batch.

**Fix implemented:** `train.py` and `evaluate.py` now use `get_speaker_severity(speaker_id)` when speaker IDs are present, with fallback to `status * 5.0` only when speaker metadata is missing.

---

### 3.9 `SymbolicKLLoss` Effective Weight History

With the old `lambda_symbolic_kl=0.05` and `reduction="batchmean"` (divides by V=47), the effective per-row penalty weight was `0.05/47 ≈ 0.001` — too weak to anchor the constraint matrix. Fixed: raised to 0.50 (effective weight ~0.01). Applied in B22/v5.

---

### 3.10 OneCycleLR / Progressive Unfreezing Interaction (T-04)

**Status:** ⚠️ Partially mitigated

The `_reset_hubert_lr_warmup()` helper correctly resets Adam momentum states when layers are unfrozen. However, OneCycleLR's LR schedule is not reset — newly-unfrozen parameters enter at the current (potentially lower-than-peak) position. For epoch-12 unfreezing (Stage 3), the LR is near the end of the decay phase.

**Partial fix applied:** Adam momentum reset. **Full fix (O-2):** Per-group `CosineAnnealingWarmRestarts(T_mult=2)`. Not blocking for current submission.

---

### 3.11 Vocabulary Consistency with `strict=False` Checkpoint Loading

**Status:** ⚠️ Risk identified

`phn_to_id` is rebuilt from manifest at every load. `on_save_checkpoint` / `on_load_checkpoint` now persist vocab (B18). If the manifest is regenerated and vocab insertion order shifts, mismatches will be caught with a diff warning. Monitor.

---

## 4. Statistical & Scientific Validity Issues

### 4.1 HuBERT Pretraining Alignment Artifact

`facebook/hubert-base-ls960` was pretrained on LibriSpeech 960h. While TORGO speakers are not in LibriSpeech, the CE loss projects HuBERT representations onto the same ARPABET phoneme set used in HuBERT pretraining, creating a domain alignment artifact. This may inflate control speaker performance and partly explain the inverted dysarthric/control PER ordering in current test splits.

---

### 4.2 Metric Aggregation Macro-Average ✅ Fixed (B15/v5)

Was computing sample mean (`avg_per = np.mean(per_scores)`). Fixed: group by speaker, compute per-speaker mean, then macro-average. Used as primary metric throughout.

---

### 4.3 WeightedRandomSampler Balances Classes, Not Speakers ✅ Fixed (B20/v5)

Sampler used class-level inverse-frequency weights. Fixed: speaker-level inverse-frequency weights.

---

### 4.4 Articulatory Ontology Inconsistency ✅ Fixed (B23/v5)

Training labels (`PHONEME_DETAILS` in `manifest.py`) and constraint matrix (`PHONEME_FEATURES` in `model.py`) used different labels for the same phonemes. Fixed: both files now use IPA-aligned labels.

| Phoneme | Before | After |
| ------- | ------ | ----- |
| SH, ZH, CH, JH | palatal | postalveolar |
| R | palatal | alveolar |
| W | bilabial | labio-velar |

---

### 4.5 Severity Stratification Circularity

`compute_severity_stratified_per` uses buckets mild=[0,2) / moderate=[2,4) / severe=[4,5]. Since all control speakers have `severity=0.0`, they all land in "mild." The analysis reduces to dysarthric vs. control. **Recommendation:** Use TORGO Rudzicz 2012 sub-groups (normal/mild/moderate/severe/profound).

---

### 4.6 LOSO Fold Validation Set Size

Each LOSO fold holds out 1 test speaker and uses `max(1, int(14 * 0.15)) = 2` speakers for validation. Val/PER is based on ~1000 utterances (noisy but acceptable for early stopping).

---

## 5. Data Pipeline Assessment

### 5.1 Manifest & Speaker IDs ✅ Correct

B12 fix: speaker extracted from `path.name.split('_')[2]`. Manifest regenerated with 16,531 rows and correct TORGO IDs.

### 5.2 Vocabulary Building ✅ Correct

Vocab built in sort order from manifest unique phonemes. Size fixed at 47 (44 ARPABET + 3 special). Stress-stripped via `normalize_phoneme()` consistently at manifest build, dataset init, and decode time.

### 5.3 Audio Loading ✅ Correct

No peak normalization (C2 fix). Mono conversion, 16 kHz resample, 6.0s max length (99% coverage). Graceful silence fallback.

### 5.4 `PHONEME_DETAILS` / `PHONEME_FEATURES` Duplication

Both files are currently in sync (B23 fix). However, they are defined independently — any future correction to one must be mirrored manually. **Recommendation:** Canonicalize in `src/utils/constants.py`.

### 5.5 `ModelConfig.num_phonemes` Default ✅ Fixed

Default is now 47, matching the runtime vocabulary convention of 44 ARPABET phonemes plus 3 special tokens. Runtime still overwrites this with `len(phn_to_id)` at `NeuroSymbolicASR` init, but the config is no longer misleading.

### 5.6 `create_single_dataloader` Sampler Policy ✅ Fixed

`dataloader.py::create_single_dataloader` now mirrors the training pipeline and prefers speaker-level inverse-frequency weights when speaker metadata is present, with fallback to label balancing only for manifests that do not expose a speaker column.

---

## 6. Loss Function Analysis

### 6.1 `OrdinalContrastiveLoss`

Pairwise cosine-similarity hinge loss with margin proportional to severity distance. NaN-safe for batch size 1. **Issue:** Zero-margin for control–control pairs (§3.8). Effective fix: continuous `TORGO_SEVERITY_MAP` severity scores.

### 6.2 `BlankPriorKLLoss`

Bernoulli KL of mean-frame blank probability toward `target_prob=0.75`. Staged warmup (0.10 → 0.15 → 0.20) prevents CTC collapse in early training. Correct.

**Status update:** Default in `src/models/losses.py` and `tests/test_losses.py` now matches `0.75`.

### 6.3 `SymbolicKLLoss`

`KL(learned||prior)` via `F.kl_div(log_prior, log_learned, reduction='sum') / V`. Direction correct (B7 fix). Normalization by V prevents coupling to vocabulary size. Correct.

### 6.4 Frame-CE Loss Weight

At `lambda_ce=0.35`, CE contributed 30% of total loss with near-random frame→phoneme assignments. Reduced to 0.10 (C-1). Staged `lambda_blank_kl` (0.10→0.20→0.35) is well-designed.

Current training config uses a gentler staged schedule (0.10→0.15→0.20).

---

## 7. Evaluation Methodology Issues

### 7.1 `BeamSearchDecoder` — LM Zero-Count Transitions

**Status:** ✅ Fixed (N4)

Phoneme bigram LM now uses add-k smoothing (`BigramLMScorer(k=0.5)`), avoiding `log(0)` transition collapse.

This materially reduces brittle pruning of rare/non-canonical dysarthric transitions.

### 7.2 No Paired Significance Test: Neural vs Constrained

**Status:** ✅ Fixed (N3)

`evaluate.py` now computes bootstrap paired constrained-minus-neural PER deltas and writes:

- `symbolic_impact.p_value_neural_vs_constrained`
- `symbolic_impact.ci_95_delta_per`
- mirrored entries in `stats`

### 7.3 Bootstrap CI Scope

Bootstrap CI is computed over macro-speaker PER scores (3–15 values), not utterance-level. With n=3 test speakers, CI is meaningless. With n=15 (post-LOSO), bootstrap CI over 15 fold PER values is statistically appropriate.

### 7.4 Confusion Matrix Without Forced Alignment

Confusion matrices are computed from greedy CTC decoder outputs without forced alignment. The identity of which phoneme was misarticulated is ambiguous without frame-level boundaries. Published confusion statistics (D→AH, T→AH, etc.) and the "vowel centralization pathology" claim are not fully defensible.

**Future work:** CTCSegmentation or Montreal Forced Aligner for framewise phoneme boundaries.

---

## 8. Explainability & Visualization Issues

### 8.1 `SymbolicRuleTracker` Activations

H-5 applied (50,000 activation cap). Rule activations are 90%+ `X→<BLANK>` — this reflects the constraint matrix's current degenerate state. Re-evaluate with `--explain` after LOSO-CV (once constraint is fixed or explicitly characterized).

### 8.2 `rule_precision()` Not Called

Implemented in `rule_tracker.py` and still not called directly from `evaluate_model`; however, evaluation now reports an utterance-level proxy (`constraint_precision` and `rule_precision_proxy`).

**Status:** ⚠️ Partially addressed (proxy added; direct tracker precision still needs ground-truth-aware rule-use accounting)

### 8.3 `PhonemeAttributor._PROBABLE_CAUSE_MAP` Key Format Mismatch

**Status:** ✅ Fixed

Lookup map and `_infer_probable_cause` now use consistent key formats (2-tuple for manner, 3-tuple for voice/place where appropriate), and attribution falls back to "Undetermined substitution" rather than a universal unknown path.

### 8.4 `plot_severity_vs_per` Missing n= Annotation

At n=8 dysarthric speakers, Pearson r has wide CIs. Plot should display `f"n={n_speakers}"` and optionally bootstrap CI on the regression slope.

**Status:** ✅ Fixed

`plot_severity_vs_per` now includes `n` in the annotation string and keeps significance labeling explicit.

### 8.5 `conformal_phoneme_sets` τ Tautology

`tau = 1.0 - (1.0 - coverage) = coverage`. Documented as APS-like heuristic (M-5). Not a correctness issue for current research context; clinical deployment would require proper conformal calibration.

---

## 9. Test Coverage Assessment

### 9.1 Smoke Tests (Unit 7/7 Passing)

| # | Test | Status |
| --- | ---- | ------ |
| 1 | `TORGO_SEVERITY_MAP` range [0,5] | ✅ |
| 2 | `LearnableConstraintMatrix` gradient flow | ✅ |
| 3 | `BlankPriorKLLoss` non-negativity + mask sensitivity | ✅ |
| 4 | `OrdinalContrastiveLoss` correctness | ✅ |
| 5 | Explainability formatter output contract | ✅ |
| 6 | LOSO ordering/resume source guards | ✅ |
| 7 | Compact fold progress callback output | ✅ |

Additional profile: `python scripts/smoke_test.py --profile pipeline` runs a tiny train-only CLI integration smoke.

### 9.2 Pytest and Integration Tests

Nine test files now cover: `test_config.py`, `test_dataloader.py`, `test_losses.py`, `test_metrics.py`, `test_constraint_gradient.py`, `test_beam_decoder.py`, `test_symbolic_constraint.py`, `test_training_step.py`, and `test_evaluate_model.py`.

`TestBlankPriorKLLoss.setup_method` now uses `target_prob=0.75`, aligned with production defaults.

### 9.3 Remaining Integration Gaps

No test currently covers:

- `BeamSearchDecoder` output determinism across random seeds
- `SymbolicRuleTracker` log-and-recall round-trip
- Per-fold LOSO resume/progress bookkeeping under interrupted training

Core single-step training and evaluation regressions are now covered by synthetic integration tests, but multi-fold orchestration and explainability bookkeeping still rely on end-to-end runs.

---

## 10. Research Enhancement Proposals

### Proposal 1 — Ordinal Contrastive Loss with Continuous Severity (High Impact)

**Status:** Implemented in main train/eval paths via `get_speaker_severity`; binary fallback retained for missing speaker metadata
**Venue fit:** INTERSPEECH / TASLP
**Expected impact:** 3–5% relative PER reduction for severe dysarthric speakers; eliminates zero-gradient control–control pairs

```python
# Use TORGO_SEVERITY_MAP continuous scores:
severity = batch['severity_continuous']  # [0.0–5.0 per speaker]
```

---

### Proposal 2 — Learnable Constraint Matrix with KL Anchor (Novel Contribution)

**Status:** Implemented; C-1/C-2 completed. Current evidence is mixed (helps internal neural sub-path in `baseline_v6`, but neural-only remains best overall single-split).
**Venue fit:** TASLP / ACL
**Expected impact:** Discovered `C` matrix is a publishable result — dysarthric confusion topology learned from data

---

### Proposal 3 — Cross-Attention Severity Adapter (Medium Impact)

**Status:** Implemented (`use_severity_adapter=True`)
**Venue fit:** INTERSPEECH
**Expected impact:** Replaces binary β scaling with representational severity conditioning; correct for both populations

---

### Proposal 4 — Uncertainty-Aware Decoding (Medium Impact)

**Status:** `UncertaintyAwareDecoder` implemented; requires `--uncertainty` flag
**Venue fit:** INTERSPEECH / ML4H
**Expected impact:** High clinical utility; provides calibrated uncertainty estimates for SLP trust

---

### Proposal 5 — CTC Forced Alignment (High Impact, Long-Term)

**Status:** Not implemented
**Venue fit:** Journal extension
**Expected impact:** Enables `PhonemeAttributor.attention_attribution`; fixes frame-CE supervision at root level

---

## 11. Verification Tests

Run these after any significant code change. All should pass.

```bash
# 1. Smoke test suite
python run_pipeline.py --run-name smoke_all_phases --max_epochs 1 --limit_train_batches 5

# 2. Severity map values in [0, 5]
python -c "
from src.utils.config import TORGO_SEVERITY_MAP
assert all(0.0 <= v <= 5.0 for v in TORGO_SEVERITY_MAP.values())
print('PASS: severity map values in [0, 5]')
"

# 3. Learnable constraint matrix gradient check
python -c "
import torch
from src.models.model import SymbolicConstraintLayer
from src.utils.config import get_default_config
cfg = get_default_config()
phn_to_id = {'<BLANK>':0,'<PAD>':1,'<UNK>':2,'P':3,'B':4}
id_to_phn = {v:k for k,v in phn_to_id.items()}
layer = SymbolicConstraintLayer(5, phn_to_id, id_to_phn, cfg.symbolic, learnable=True)
logits = torch.randn(2, 10, 5)
out = layer(logits)
out['log_probs'].mean().backward()
assert layer.logit_C.grad is not None
print('PASS: learnable constraint matrix gradients flow')
"

# 4. BlankPriorKLLoss sanity check
python -c "
import torch, torch.nn.functional as F
from src.models.losses import BlankPriorKLLoss
loss_fn = BlankPriorKLLoss(blank_id=0, target_prob=0.75)
logits = torch.zeros(2, 50, 5)
logits[:,:,0] = 1.0
log_probs = F.log_softmax(logits, dim=-1)
loss = loss_fn(log_probs, attention_mask=torch.ones(2, 50, dtype=torch.long))
assert loss.item() >= 0.0
print(f'PASS: BlankPriorKLLoss = {loss.item():.4f}')
"

# 5. OrdinalContrastiveLoss sanity check
python -c "
import torch
from src.models.losses import OrdinalContrastiveLoss
loss_fn = OrdinalContrastiveLoss(margin_per_level=0.3)
embeddings = torch.randn(4, 10, 768)
severity = torch.tensor([0.0, 0.0, 5.0, 5.0])
loss = loss_fn(embeddings, severity)
assert loss.item() >= 0.0
print(f'PASS: OrdinalContrastiveLoss = {loss.item():.4f}')
"

# 6. Explainability formatter structure
python -c "
from src.explainability.output_format import ExplainableOutputFormatter
fmt = ExplainableOutputFormatter()
result = fmt.format_utterance('test_001', 'hello world', 'hello', [], {})
assert 'utterance_id' in result and 'phoneme_analysis' in result
print('PASS: explainability formatter output structure valid')
"
```

---

## 12. Issue Severity Index

| ID | Severity | Component | Issue | Status |
| ---- | -------- | --------- | ----- | ------ |
| §2.1 | **CRITICAL** | `model.py`, `train.py` | Symbolic effect is mixed: helps vs internal neural path in v6 but neural-only ablation still best overall | ⚠️ Open (reframed) |
| §3.3 | **CRITICAL** | `sequence_utils.py` | Frame-CE pad/truncate label mismatch (T-05); lambda\_ce=0.10 mitigates | ⚠️ Mitigated |
| §2.2 | **CRITICAL** | Experimental design | Small-split statistical fragility addressed via LOSO 15/15 | ✅ C-3 complete |
| §7.1 | **MAJOR** | `evaluate.py` | `BigramLMScorer` zero-count transitions kill hypotheses with −∞ | ✅ Fixed |
| §7.2 | **MAJOR** | `evaluate.py` | No paired significance test between neural vs constrained PER | ✅ Fixed |
| §3.8 | **MAJOR** | `losses.py` | `OrdinalContrastiveLoss` zero-margin for all control pairs | ✅ Fixed in forward path |
| §8.3 | MINOR | `attribution.py` | `_PROBABLE_CAUSE_MAP` key format mismatch | ✅ Fixed |
| §8.2 | MINOR | `rule_tracker.py` | `rule_precision()` not directly wired in evaluation | ⚠️ Partially addressed |
| §5.4 | MINOR | `manifest.py` / `model.py` | `PHONEME_DETAILS` duplicates `PHONEME_FEATURES`; manual sync required | ❌ N5 pending |
| §5.5 | MINOR | `config.py` | `ModelConfig.num_phonemes` default now matches runtime vocab size | ✅ Fixed |
| §6.2 | MINOR | `losses.py` | `BlankPriorKLLoss` default/test mismatch | ✅ Fixed |
| §8.5 | MINOR | `uncertainty.py` | `conformal_phoneme_sets` τ tautology | ✅ Documented (M-5) |
| §8.4 | MINOR | `experiment_plots.py` | `plot_severity_vs_per` missing n= annotation | ✅ Fixed |
| §9.3 | MINOR | `tests/` | Remaining gaps are beam determinism, rule-tracker round-trip, and LOSO resume orchestration tests | ⚠️ Partially addressed |
| §3.10 | LOW | `train.py` | OneCycleLR not reset when layers unfrozen (T-04 partially mitigated) | ⚠️ Mitigated |
| §5.6 | LOW | `dataloader.py` | `create_single_dataloader` now mirrors speaker-level weighting | ✅ Fixed |

---

## 13. Time-Critical Next Steps (March 17)

Given limited time, prioritize only items that materially affect publication claims.

### Priority 1 (must run)

1. Run LOSO-CV with resume enabled (publication validity blocker):

```bash
python run_pipeline.py --run-name loso_v1 --loso --resume-loso
```

1. After LOSO completes, compute and report:

- macro-speaker PER mean and 95% CI across folds
- paired constrained-vs-neural delta PER with p-value
- dysarthric vs control fold-level summary

### Priority 2 (symbolic claim triage)

1. Keep `baseline_v6` as symbolic reference.

1. Run a compact symbolic sweep (3-4 runs max) varying only:

- `constraint_weight_init` (for example: `0.01`, `0.03`, `0.05`)
- optional LM fusion weight (`beam_lm_weight`) if using beam-search decoding

1. Use one acceptance rule:

- symbolic model is accepted as primary only if global PER is non-inferior to neural-only and improves at least one dysarthric-focused slice.

### Priority 3 (quick code cleanups, low risk)

1. No additional low-risk code fixes are required for the current paper path; the remaining non-blocking code item is direct `SymbolicRuleTracker.rule_precision()` wiring, which should only be implemented alongside ground-truth-aware activation counting.
2. If LOSO finishes in time, add one focused regression test for LOSO resume/progress JSON so the new interruption-safe path is protected.
3. In paper text, describe `constraint_precision` / `rule_precision_proxy` explicitly as an utterance-level proxy rather than frame-level symbolic precision.
