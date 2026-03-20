# DysarthriaNSR — Completed Work Log

> **Last updated:** March 20, 2026
> All items below are fully resolved, implemented, and verified unless otherwise noted.

---

## Table of Contents

1. [Baselines & Infrastructure](#1-baselines--infrastructure)
2. [Bug Fixes: Critical (B/C-series)](#2-bug-fixes-critical-bc-series)
3. [Bug Fixes: Important (I-series)](#3-bug-fixes-important-i-series)
4. [Bug Fixes: Evaluation (E-series)](#4-bug-fixes-evaluation-e-series)
5. [Bug Fixes: Quality (Q-series)](#5-bug-fixes-quality-q-series)
6. [Bug Fixes: Documentation (D-series)](#6-bug-fixes-documentation-d-series)
7. [Tech Audit Fixes (March 6, 2026)](#7-tech-audit-fixes-march-6-2026)
8. [Action Plan Items Implemented (March 12, 2026)](#8-action-plan-items-implemented-march-12-2026)
9. [Structural Refactoring](#9-structural-refactoring)
10. [Baseline Experiment Results](#10-baseline-experiment-results)

---

## 1. Baselines & Infrastructure

| Item | Description |
|------|-------------|
| B1–B12 | All initial bugs fixed (CTC stride, KL direction, LOSO name mutation, speaker extraction, etc.) |
| `run_pipeline.py` | Single entry-point orchestrator implemented |
| baseline\_v3 | Trained — first valid speaker splits; val/PER 0.574 |
| baseline\_v4 | Trained + evaluated — beam PER 0.4748; insertion bias resolved (I/D=0.87×) |
| baseline\_v5 | Trained + evaluated — beam PER 0.4750; same config as v4 but 30 epochs |
| Manifest regenerated | B12 fix applied; correct TORGO speaker IDs via `split('_')[2]`; 16,531 rows |
| Smoke tests | 7/7 passing |
| Pytest unit tests | 49 passing across 7 test files |
| Explainability JSON | Uncertainty MC-Dropout wired into evaluation pipeline |
| Figure suite | `scripts/generate_figures.py` with 6+ publication-quality plots |
| LOSO infrastructure | `run_loso()` in `train.py`; `--loso` flag in `run_pipeline.py` |
| LOSO run | `loso_v1` completed (15/15 folds): macro PER 0.2848, weighted PER 0.2299 |

---

## 2. Bug Fixes: Critical (B/C-series)

| ID | File | Fix Applied |
|----|------|-------------|
| B1–B12 | Various | Initial bug sweep: CTC stride, KL direction, LOSO name mutation, speaker extraction, manifest speaker IDs, etc. |
| B13 | `model.py` | SpecAugment per-sample independent masking (was batch-uniform) |
| B14 | `train.py` | Stage 2 progressive unfreezing no longer a no-op; calls `unfreeze_encoder(layers=[6..11])` directly |
| B15 | `train.py` | Checkpoint monitoring uses true macro-speaker PER |
| B16 | `train.py` | Attention mask stride accounts for TemporalDownsampler stride-2 (`ctc_stride = 320 * 2`) |
| B17 | `evaluate.py` | `decode_predictions` undefined `config` global removed |
| B18 | `model.py` | `on_save_checkpoint` / `on_load_checkpoint` persist vocabulary; diff warning on mismatch |
| B19 | `evaluate.py` | Beam search length norm applied to acoustic score only, not acoustic+LM |
| B20 | `train.py` | `WeightedRandomSampler` uses speaker-level inverse-frequency weights |
| B21 | `model.py` | `LearnableConstraintMatrix` init uses temperature=0.5 to preserve diagonal peakedness |
| B22 | `config.py` | `lambda_symbolic_kl` raised 0.05 → 0.50 |
| B23 | `manifest.py` / `model.py` | Articulatory ontology unified: postalveolar for SH/ZH/CH/JH, alveolar for R, labio-velar for W |
| C1 | `losses.py` | Raw logits to NLLLoss (negative CE loss) — fixed |
| C2 | `dataloader.py` | Double audio normalization — removed |
| C3 | `train.py` | Data leakage via sample-level fallback split — fixed |
| C4 | `evaluate.py` | Explanations JSON all-empty — fixed |
| C5 | `config.py` | `min_rule_confidence` lowered 0.5 → 0.05; rule tracker activations now flow |
| C6 | `model.py` | `art_ce_losses` changed from plain dict to ModuleDict |
| C7 | `losses.py` | Binary severity in ordinal contrastive loss — replaced with continuous |
| C8 | `requirements.txt` | `statsmodels` added |

---

## 3. Bug Fixes: Important (I-series)

| ID | File | Fix Applied |
|----|------|-------------|
| I1 | `evaluate.py` | WER now computed at corpus level |
| I2 | `train.py` / `losses.py` | Insertion bias resolved: I/D ratio reduced from 4.6× to 0.87× in v4/v5 |
| I3 | `evaluate.py` | Full TORGO (15 speakers) evaluated in baseline\_v4/v5 |
| I4 | `run_pipeline.py` | `UncertaintyAwareDecoder` wired into evaluation pipeline |
| I5 | `model.py` | Articulatory heads label misalignment — fixed |
| I6 | `train.py` | `SymbolicKLLoss` now eagerly constructed in `__init__` (registered as Lightning module attribute) |
| I7 | `config.py` | `Config()` global instantiated at import time — removed |

---

## 4. Bug Fixes: Evaluation (E-series)

| ID | Fix Applied |
|----|-------------|
| E1 | Spearman correlation degenerate with n=3 speakers — guarded with n-check (`correlation_valid` flag) |
| E2 | WER missing from results — added (see I1) |
| E3 | No per-phoneme PER breakdown — added |
| E4 | No learning curve plot saved automatically — added |
| E5 | Articulatory confusion depends on broken explainability — fixed |
| E6 | `plot_rule_impact` skips when top\_rules empty — unblocked by C5 fix |
| E7 | Bootstrap CI reliability issue under tiny split — addressed by completed LOSO-CV (`loso_v1`, 15/15) |

---

## 5. Bug Fixes: Quality (Q-series)

| ID | File | Fix Applied |
|----|------|-------------|
| Q1 | `evaluate.py` / `sequence_utils.py` | `_align_labels_to_logits` deduplicated; moved to `src/utils/sequence_utils.py` |
| Q2 | `dataloader.py` | Conflicting `create_dataloaders` removed; `train.py` version is canonical |
| Q3 | `tests/` | 49 pytest unit tests added across losses, metrics, config, dataloader |
| Q4 | `train.py` | `train/grad_norm` logged every 50 steps; `--detect-anomaly` flag added |
| Q5 | `run_pipeline.py` | Checkpoint filename `/` character fixed |
| Q6 | `losses.py` | SymbolicKLLoss direction comment corrected |
| Q7 | `model.py` | `ablation_mode='neural_only'` now fully bypasses SeverityAdapter + SymbolicConstraintLayer |

---

## 6. Bug Fixes: Documentation (D-series)

| ID | Fix Applied |
|----|-------------|
| D1 | Setup/installation guide added to README |
| D2 | Reproducibility seed documentation added (README) |
| D3 | `requirements.txt` version-pinned (torch==2.9.0, transformers==4.57.1, etc.) |
| D4 | `config.save()` called in `train()` (checkpoints dir) and `run_pipeline.py` (results dir) |
| D5 | `docs/loso_guide.md` written |
| D6 | MLflow tracking URI uses absolute path (`file://...resolve()`) with `MLFLOW_TRACKING_URI` env override |

---

## 7. Tech Audit Fixes (March 6, 2026)

These items were resolved as part of the `tech_audit_march2026` review pass.

| Ref | File | Fix Applied |
|-----|------|-------------|
| §2.3 | `model.py` | SpecAugment now applies per-sample independent masking |
| §2.4 | `train.py` | Stage 2 progressive unfreezing calls `unfreeze_encoder(layers=[6..11])` directly — Stage 3 is no longer a no-op |
| §2.6 | `train.py` | Checkpoint monitoring uses true macro-speaker PER |
| §2.7 | `train.py` | Attention mask stride accounts for TemporalDownsampler stride-2 |
| §3.1 | `evaluate.py` | `decode_predictions` undefined `config` global removed |
| §3.2 | `model.py` | `LearnableConstraintMatrix` init uses temperature=0.5 to preserve diagonal peakedness |
| §3.3 | `evaluate.py` | Beam search length norm applied to acoustic score only, not acoustic+LM |
| §3.4 | `train.py` | `validation_step` and `test_step` pass downsampled `attention_mask` to `compute_loss` |
| §3.5 | `train.py` | `WeightedRandomSampler` uses speaker-level inverse-frequency weights |
| §3.8 | `config.py` | `lambda_symbolic_kl` raised 0.05 → 0.50 |
| §5.3 | `manifest.py` / `model.py` | Articulatory ontology unified (SH/ZH/CH/JH=postalveolar, R=alveolar, W=labio-velar) |

---

## 8. Action Plan Items Implemented (March 12, 2026)

These items from `ACTION_PLAN.md` have been implemented as part of the pre-SPCOM audit pass.

| ID | Description | Location |
|----|-------------|----------|
| H-1 | `del trainer, lm` + `torch.cuda.empty_cache()` between LOSO folds | `train.py::run_loso()` |
| H-2 | Pre-compute `_all_alignments` and `_neural_alignments` once; pass to all four analysis functions | `evaluate.py` |
| H-3 | `plot_blank_histogram` target line reads from `config.training.blank_target_prob` (0.75); was hardcoded 0.82 | `evaluate.py` |
| H-4 | `EarlyStopping` monitor unified to `val_per` (was `val/per`) | `train.py::run_loso()` |
| H-5 | `SymbolicRuleTracker._activations` capped at 50,000 entries; `flush()` method added | `src/explainability/rule_tracker.py` |
| H-6 | `_compute_stratified_per` reuses stored `output['per_scores']` instead of recomputing | `train.py` |
| M-5 | `conformal_phoneme_sets` docstring updated to document APS-like heuristic (not calibrated conformal) | `src/models/uncertainty.py` |
| M-6 | `plot_per_by_speaker` sort order changed to severity-ascending (controls first) | `src/visualization/experiment_plots.py` |
| C-1 | `lambda_ce` reduced 0.35 → 0.10; baseline\_v6 retrained and evaluated | `src/utils/config.py` |
| C-4 | `plot_severity_vs_per` scatter plot implemented and wired into `evaluate_model` | `evaluate.py` + `experiment_plots.py` |
| C-5 | `plot_rule_pair_confusion` horizontal bar chart implemented | `experiment_plots.py` |

---

## 9. Structural Refactoring

- Removed internal `evaluate_model()` call from `train()` — evaluation now exclusively owned by `run_pipeline.py`
- Removed `trainer.test(...)` from `train()` to prevent crashes in smoke mode
- Added `last.ckpt` fallback in `run_pipeline.py` when `best_model_path` is empty
- `train(config, limit_train_batches=None)` — smoke-test parameter threaded to `pl.Trainer`
- `UncertaintyAwareDecoder` (MC Dropout) wired into `evaluate_model` via `compute_uncertainty` flag
- `scripts/smoke_test.py` upgraded to profiles: `unit` (7 checks) + `pipeline` (tiny CLI integration)
- `sequence_utils.py` extracted as shared utility to eliminate duplicate alignment code

---

## 10. Baseline Experiment Results

### baseline\_v5 (Historical pre-LOSO reference)

| Metric | Value |
|--------|-------|
| Beam PER (constrained) | 0.4750 |
| Greedy PER (neural sub-path, internal) | 0.305 ⚠️ not a valid beam comparison |
| Val/PER | 0.505 (epoch 28/30) |
| Articulatory manner accuracy | 78.3% |
| Articulatory place accuracy | 79.3% |
| Articulatory voice accuracy | 92.3% |
| Insertion/Deletion ratio | 0.9× |
| Total parameters | 98.9M |
| Trainable parameters | 66.4M |

> **Note:** `per_neural=0.305` was measured via greedy decode of the model's internal
> neural sub-path, not via an independent neural-only model trained without the symbolic
> layer. This comparison is now available via `ablation_neural_only_v7`.

### baseline\_v4

| Metric | Value |
|--------|-------|
| Beam PER | 0.4748 |
| Insertion / Deletion ratio | 0.87× (resolved from 4.6×) |
| `per_neural` (greedy internal only) | 0.305 ⚠️ not comparable to beam |
| `per_constrained` (beam) | 0.4742 |

### baseline\_v2 (historical; data leakage — invalid)

| Metric | Value |
|--------|-------|
| Greedy test PER | 0.123 |
| Beam PER | 0.243 |
| Insertions / Deletions | 3,143 / 678 |
| Test speakers | 3 (MC02, MC04, M03) |
| Note | All speaker\_id fields show "unknown" due to manifest bug (now fixed) |

### baseline\_v1 (historical)

| Metric | Value |
|--------|-------|
| Test PER | 0.567 ± 0.365 |
| Insertions / Deletions | 21,290 / 376 (56× ratio) |
| Learned β | ~0.50 |

### Available Artifacts (`results/baseline_v2/`)

| File | Contents |
|------|----------|
| `evaluation_results.json` | Aggregate stats, per\_speaker dict, error\_analysis, uncertainty block |
| `explanations.json` | 2,481 per-utterance records with phoneme-level error analysis |
| `confusion_matrix.png` | Phoneme confusion heatmap |
| `per_by_length.png` | PER vs utterance length, dysarthric vs control |
| `articulatory_confusion.png` | Articulatory feature confusion matrix |
| `clinical_gap.png` | Dysarthric vs control PER gap visualization |
| `figures/` | Publication-quality diagnostics |

### Smoke Test Status

| # | Test | Status |
|---|------|--------|
| 1 | Config + severity map sanity | ✅ |
| 2 | `LearnableConstraintMatrix` gradient flow | ✅ |
| 3 | `BlankPriorKLLoss` non-negativity + mask sensitivity | ✅ |
| 4 | `OrdinalContrastiveLoss` correctness (incl. batch size 1) | ✅ |
| 5 | Explainability formatter output contract | ✅ |
| 6 | LOSO ordering/resume source guards | ✅ |
| 7 | Compact fold progress callback output | ✅ |

**Profiles:** `python scripts/smoke_test.py --profile unit` (default) and `--profile pipeline` (tiny train-only integration smoke).
