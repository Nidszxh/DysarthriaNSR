# DysarthriaNSR — Project Roadmap

> **Last updated:** March 20, 2026  
> **Current baseline:** `baseline_v6` — avg_per=0.137, per_neural=0.145, per_constrained=0.137  
> **Neural-only ablation:** `ablation_neural_only_v7` — avg_per=0.135 (best single-split result to date)

---

## Status Legend

- ✅ Completed and verified
- ⚠️ Implemented, partially effective / caveats remain
- ❌ Not yet done — open blocker
- 🔲 Planned but not started

---

## Completed & Verified Components

### Core Architecture

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| `NeuroSymbolicASR` forward pass | `src/models/model.py` | ✅ | Full 6-component pipeline |
| `HuBERT` backbone | `model.py` L707 | ✅ | `facebook/hubert-base-ls960`, revision `dba3bb02` pinned |
| `LearnableConstraintMatrix` | `model.py` L138 | ✅ | Temperature-sharpened init (T=0.5); softmax-parameterised |
| `SymbolicConstraintLayer` | `model.py` L188 | ✅ | Adaptive β, blank-frame masking applied |
| `SeverityAdapter` | `model.py` L432 | ✅ | Cross-attention over [B,1,768] severity context |
| `TemporalDownsampler` | `model.py` L580 | ✅ | stride-2 Conv1d, ~50→~25 Hz |
| `SpecAugmentLayer` | `model.py` L503 | ✅ | Per-sample independent masking (B13 fix) |
| `PhonemeClassifier` | `model.py` L634 | ✅ | 768 → 512 → \|V\| |
| Articulatory heads | `model.py` L755 | ✅ | Utterance-level via GAP (I5 fix) |
| 3-stage HuBERT unfreezing | `train.py` L566 | ✅ | Epochs 1/6/12; layers 8-11/6-11/4-11 |

### Loss Functions

| Loss | File | λ | Status |
|------|------|---|--------|
| `CTCLoss` | `train.py` | 0.80 | ✅ `zero_infinity=True`; uses model `output_lengths` |
| Frame-CE on neural logits | `train.py` | 0.10 | ✅ Applied to `logits_neural`, not constrained (C1 fix) |
| Articulatory CE (manner/place/voice) | `train.py` | 0.08 | ✅ Utterance-mode target (L2 fix) |
| `OrdinalContrastiveLoss` | `losses.py` | 0.05 | ✅ Continuous TORGO severity scores |
| `BlankPriorKLLoss` | `losses.py` | staged 0.10→0.15→0.20 | ✅ Target 0.75; staged warmup (I2) |
| `SymbolicKLLoss` | `losses.py` | 0.50 | ✅ Anchors C to symbolic prior |

### Training Infrastructure

| Item | Status | Notes |
|------|--------|-------|
| PyTorch Lightning `DysarthriaASRLightning` | ✅ | Multi-task training, macro-speaker PER monitoring |
| Differential LR (HuBERT ×0.1, head ×1.0, symbolic ×0.5) | ✅ | `train.py` L995 |
| OneCycleLR with cosine annealing | ✅ | |
| Adam momentum reset on unfreeze (T-04) | ✅ | `_reset_hubert_lr_warmup()` |
| Staged `lambda_blank_kl` warmup (I2) | ✅ | 0.10→0.15→0.20 over epochs 0/10/20 |
| Macro-speaker PER validation metric (B15) | ✅ | Groups by speaker before averaging |
| Checkpoint vocab persistence (B18) | ✅ | `on_save_checkpoint` / `on_load_checkpoint` |
| `WeightedRandomSampler` speaker-level (B20) | ✅ | Inverse-frequency speaker weights |
| Speaker-stratified 70/15/15 split | ✅ | `create_dataloaders()` |
| `run_loso()` with resume support | ✅ | Per-fold checkpoint + progress JSON |

### Data Pipeline

| Item | Status | Notes |
|------|--------|-------|
| TORGO manifest generation (`manifest.py`) | ✅ | 16,531 rows; speaker at `split('_')[2]` (B12) |
| `TorgoNeuroSymbolicDataset` | ✅ | Disk + memory LRU feature cache |
| `NeuroSymbolicCollator` | ✅ | Pads audio to 0.0; labels to -100 |
| Phoneme vocabulary: 47 tokens | ✅ | 44 ARPABET + `<BLANK>`/`<PAD>`/`<UNK>` |
| Articulatory labels (manner/place/voice) | ✅ | Aligned with IPA (B23: postalveolar fixes) |
| No peak normalization (C2) | ✅ | HuBERT processor handles normalization |
| 6.0s max audio length | ✅ | ~99% TORGO coverage |

### Evaluation

| Item | Status | Notes |
|------|--------|-------|
| `compute_per()` using `editdistance` | ✅ | PER = (S+D+I)/N |
| Bootstrap 95% CI (1000 samples) | ✅ | Over macro-speaker PER scores |
| Paired constrained vs. neural bootstrap test | ✅ | `bootstrap_paired_per_delta()` (N3) |
| `BeamSearchDecoder` (CTC prefix beam search) | ✅ | Length norm on acoustic only (B19) |
| `BigramLMScorer` with add-k smoothing | ✅ | k=0.5, prevents log(0)=−∞ (N4) |
| Greedy CTC decoder with padding masking | ✅ | `output_lengths` prevents insertion inflation |
| Stratified PER (dysarthric vs control) | ✅ | Logged separately in evaluation |
| Per-speaker PER | ✅ | `per_speaker` dict in `evaluation_results.json` |
| 10 publication-quality figures | ✅ | See `evaluate.py` and `scripts/generate_figures.py` |
| MLflow tracking | ✅ | All hyperparameters + metrics logged |
| YAML config persistence | ✅ | `results/{run_name}/config.yaml` |

### Fixes Applied (Historical B1–B23)

All 23 historical bug fixes (B1–B23) plus all March audit fixes (H-1 through H-6, C-1 through C-7, I-1 through I-6, L-2, M-5, M-6, N-3, N-4, O-5, T-01 through T-05, B1–B5 from conversation d9d04797) are implemented in the current codebase.

---

## Work In Progress

### LOSO Cross-Validation

| Item | Status | Detail |
|------|--------|--------|
| `run_loso()` implementation | ✅ | Supports `--resume-loso` |
| Full 15-fold sweep | ✅ Completed | `loso_v1` complete (15/15 folds) |
| Aggregate LOSO metrics | ✅ | macro PER 0.2848 (95% CI 0.1921–0.3801), weighted PER 0.2299, macro WER 0.3362 |
| Per-fold PER aggregation | ✅ (code ready) | `macro_avg_per`, `per_95ci`, `weighted_avg_per`, `macro_avg_wer` |

### Symbolic Constraint Characterization

| Item | Status | Detail |
|------|--------|--------|
| `baseline_v6` (constrained) | ✅ | avg_per=0.137, per_neural=0.145, per_constrained=0.137 |
| `ablation_neural_only_v7` | ✅ | avg_per=0.135 (best single-split) |
| `ablation_no_constraint_matrix_v6` | ✅ | avg_per=0.144 (eliminates learnable C, keeps SeverityAdapter) |
| Symbolic sweep | ❌ | Planned: vary `constraint_weight_init ∈ {0.01, 0.03, 0.05}` under fixed seed |
| LOSO-level symbolic stratified analysis | ⚠️ | Next SPCOM positioning priority: verify dysarthric-strata gains vs neural-only reference |

---

## Known Issues & Open Risks

### Critical (Publication-Blocking)

| ID | Component | Issue | Evidence | Mitigation |
|----|-----------|-------|----------|------------|
| C-3 | Experimental design | Small-split statistical fragility addressed via LOSO 15/15 | `loso_v1_loso_summary.json` | ✅ Resolved |
| §2.1 | `model.py`, `train.py` | Neural-only ablation marginally beats full constrained model (0.135 vs 0.137) | `ablation_neural_only_v7` | ⚠️ Reframed: symbolic helps internal neural sub-path but not vs pure HuBERT |

### Major (Research Validity)

| ID | Component | Issue | Status |
|----|-----------|-------|--------|
| T-05 | `sequence_utils.py` | Frame-CE `align_labels_to_logits` pads/truncates without forced alignment | ⚠️ Mitigated: `lambda_ce=0.10` |
| — | `evaluation` | Confusion matrices from greedy CTC without forced alignment — phoneme boundaries unvalidated | ⚠️ Documented limitation |
| §4.5 | Stratification | Severity buckets collapse to dysarthric/control (all controls severity=0.0) | ⚠️ Documented |

### Minor (Non-Blocking)

| ID | Component | Issue | Status |
|----|-----------|-------|--------|
| §5.4 | `manifest.py`, `model.py` | `PHONEME_DETAILS` and `PHONEME_FEATURES` defined independently (manual sync risk) | ❌ Pending: move to `src/utils/constants.py` |
| §5.5 | `config.py` | `ModelConfig.num_phonemes` now aligned with runtime vocab convention | ✅ Fixed |
| §5.6 | `dataloader.py` | `create_single_dataloader` now mirrors speaker-level weighting policy | ✅ Fixed |
| §8.2 | `rule_tracker.py` | `rule_precision()` not wired into `evaluate_model` output | ⚠️ Proxy added |
| §3.10 | `train.py` | OneCycleLR not reset after progressive unfreezing; momentum reset applied as workaround | ⚠️ Partial fix via `_reset_hubert_lr_warmup()` |
| §9.3 | `tests/` | No integration test for `training_step` or `evaluate_model` end-to-end | ❌ Post-paper |

---

## Planned Improvements

### Short-Term (Post-LOSO)

1. **Publish leaderboard artifacts:** export per-fold table + macro/weighted metrics from `loso_v1_loso_summary.json`
2. **Targeted dysarthric optimization sweep:** prioritize M01/M02/M04/M05/F01 failure modes
3. **Compact symbolic sweep:** vary `constraint_weight_init ∈ {0.01, 0.03, 0.05}` under fixed protocol
4. **Acceptance rule:** keep symbolic model primary only if globally non-inferior to neural-only and better on at least one dysarthric strata metric

### Medium-Term (Post-Submission)

1. **CTC forced alignment** via `torchaudio.functional.forced_align` — fixes T-05 at root level; enables phoneme-boundary-accurate confusion matrices
2. **Per-group `CosineAnnealingWarmRestarts`** scheduler — full fix for T-04 OneCycleLR/progressive-unfreeze interaction
3. **Canonicalize phoneme features** `PHONEME_DETAILS` / `PHONEME_FEATURES` into `src/utils/constants.py`
4. **Integration tests** for `training_step` + `evaluate_model` (M-2/M-3 from audit)

### Long-Term Research (Journal Extension)

1. **CTCSegmentation** or Montreal Forced Aligner for framewise boundaries — makes articulatory attribution defensible
2. **Conformal phoneme sets** with proper calibration (replaces τ tautology in `uncertainty.py`)
3. **Per-speaker constraint matrix** — personalized `C` matrices conditioned on severity trajectory

---

## Target Milestones

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| All B1–B23 fixes implemented | Feb 2026 | ✅ Done |
| `baseline_v6` trained | Mar 12, 2026 | ✅ Done |
| Neural-only ablation evaluated | Mar 13, 2026 | ✅ Done |
| Full LOSO-CV sweep | Mar 2026 | ✅ Completed |
| SPCOM 2026 paper submission | TBD 2026 | 🔲 Pending post-LOSO packaging |

---

## MLflow Experiment Tracking

Experiments are logged to `mlruns/` (local MLflow). To view:

```bash
mlflow ui --backend-store-uri file://$(pwd)/mlruns
# then open http://127.0.0.1:5000
```

Key tracked runs: `baseline_v4`, `baseline_v5`, `baseline_v6`, `ablation_neural_only_v7`, `ablation_no_constraint_matrix_v6`, `loso_v1`.
