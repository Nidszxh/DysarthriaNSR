# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-05-13

### Fixed

- **Blank-frame bypass threshold** (`src/models/model.py`, `SymbolicConfig`): lowered `blank_constraint_threshold` from 0.5 to 0.25, increasing constraint activation from ~15% to ~35% of CTC frames.
  - Previous value bypassed the constraint on ~85% of frames, effectively neutralizing the symbolic layer's influence on most predictions.

- **OneCycleLR warm restart** (`train.py`): added warm restart of OneCycleLR scheduler after each HuBERT unfreeze stage.
  - Previously, newly unfrozen parameters entered at the decayed LR position rather than the original peak, causing convergence issues post-unfreeze.

- **LOSO early stopping patience** (`TrainingConfig`): raised `loso_early_stopping_patience` to 22 epochs for full-system LOSO runs.
  - Full-system runs require longer convergence time than ablation runs; 8 epochs was insufficient.

### Changed

- **Frame-CE loss gating** (`train.py::compute_loss`): reduced frame-CE weight and gated behind `frame_ce_start_epoch=15`.
  - Frame-CE uses proportional nearest-neighbor label alignment which is noisy. Disabling it during early CTC-only phase allows the model to learn basic phoneme boundaries before introducing conflicting frame-level supervision.

- **Blank-mass penalty in SymbolicKLLoss** (`src/models/losses.py`): added explicit blank-row penalty to prevent constraint matrix from drifting toward blank-dominated rows.
  - Works in conjunction with the lowered blank_constraint_threshold (FIX-1) to address constraint matrix degradation.

- **SpecAugment gate decoupling** (`src/models/model.py`): decoupled SpecAugment from `_hubert_is_frozen`, now uses `self.training` only.
  - Previously, SpecAugment was incorrectly skipped whenever any HuBERT layer was frozen, even though downstream modules were actively training.

- **Attention mask downsampling** (`train.py::_downsample_attn_mask`): rewritten to use explicit two-step stride calculation.
  - The previous implicit calculation could produce incorrect mask lengths, causing shape mismatches during evaluation.

### Added

- **no_severity_adapter ablation mode** (`run_pipeline.py`, `src/models/model.py`): added ablation mode that disables SeverityAdapter while keeping SymbolicConstraintLayer.
  - Allows isolation of SeverityAdapter contribution without requiring full neural-only baseline.

- **spearman_valid flag** (`evaluate.py`): added to evaluation stats output to indicate statistical validity of correlation metrics.
  - Correlation is only valid when n_speakers >= 5; the flag signals when this threshold is met.

- **plot_per_by_manner visualization** (`src/visualization/experiment_plots.py`, `scripts/generate_figures.py`): added articulatory-stratified PER visualization.
  - Shows PER breakdown by manner of articulation (stop, fricative, nasal, vowel, etc.) for error analysis.

- **Gradient checkpointing toggle** (`evaluate.py`): disabled gradient checkpointing during evaluation passes.
  - Reduces memory overhead and improves evaluation throughput; gradient checkpointing is only beneficial during training.

### Fixed

- **BigramLM data leak** (`run_pipeline.py`): fixed to build language model from training speakers only.
  - Previously included test/validation speaker phoneme sequences, causing optimistic perplexity estimates.

- **SeverityAdapter diagnostic logging** (`train.py::validation_step`): added output norm monitoring for debugging.
  - Helps identify when the adapter is producing degenerate outputs (e.g., NaN, all-zero, or extreme values).