# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.1] — 2026-07-19

### Changed

- **Restored aa10b8d training behavior** (post-hoc): reverted 4 changes from b65e3a6/05628ce that caused performance regression vs v4_final:
  - Removed `label_smoothing=0.1` from CE loss calls (now 0, matching v4_final checkpoint)
  - `_compute_stratified_per` back to per-utterance averaging (not macro-speaker) for dysarthric/control metrics
  - Dataloader BLANK/PAD/UNK weight multipliers restored (1.2/1.5/1.1) — removed in earlier cleanup
  - `decode_predictions` beam_width default restored to 10

### Kept (from b65e3a6/05628ce)

- `pyproject.toml`, new test suite (`test_explainability.py`, `test_utils.py`)
- Config-driven LR multipliers (`encoder_lr_multiplier`, `symbolic_lr_multiplier`)
- Seeded `StratifiedMicroBatchSampler` for reproducibility
- `logger.info/warning` replacing `print` / bare `except: pass` throughout
- Decoder confounding fix (neural path beam search during eval)
- `on_test_epoch_end` test metric aggregation
- VRAM estimate improvements (gradient-checkpointing-aware)

## [0.6.0] — 2026-06-11

### Full Eval Results

- **Canonical result** (v4_final, beam width=25): macro-speaker PER=**0.133** (95% CI: [0.079, 0.200]), WER=0.116, I/D=2.1×. Both paths decoded with beam search — per_neural=0.131, per_constrained=0.133. Symbolic Δ = +0.0015 (p=0.246, not significant).
- **Articulatory accuracy**: manner=81.7%, place=90.5%, voice=95.3%
- **Temperature calibration**: M05 τ=1.25, M01 τ=1.03
- **Ablation chain (core evidence)**: SeverityAdapter alone degrades PER to 0.1444 (+7.3% vs neural-only 0.1346). Adding the constraint recovers all of that loss and surpasses neural-only (0.1326). The constraint's primary role is as a **training-time regularizer** for severity-adaptive fusion, not inference-time per-frame correction.
- **Conclusion**: The symbolic constraint is practically identical to the neural sub-path (Δ = +0.0015, p=0.246). The constraint's value is as a training-time regularizer (ablation chain) plus clinical interpretability.

### Fixed

- **Decoder confounding in evaluation** (`evaluate.py:1734-1746`): neural sub-path now uses beam search when `--beam-search` is set, matching the constrained path's decoder. Previously neural was always greedily decoded regardless of flag. This was responsible for ~90% of the apparent +0.003 gap between neural and constrained (narrowed to +0.0015 with fair comparison).

- **blank_constraint_threshold diagnostic** (`evaluate.py:1581-1592`): eval now reads `blank_constraint_threshold` from `model.symbolic_config` or `model.symbolic_layer.config` — previously checked non-existent `model.config` attribute, always defaulting to 0.25.

- **CTC forced alignment batching** (`train.py::_compute_ce_loss_aligned`): replaced B per-sample `TAF.forced_align` calls with a single batched call. If the batched call fails, all samples gracefully fall back to `align_labels_to_logits`. Inner per-label loop optimized to zero `.item()` calls — uses tensor `prev_i`, `torch.where`, `seq.cpu()` one-time transfer, list instead of dict for label lookup.

- **Gradient norm computation** (`train.py`): changed from `torch.cat` (giant temp tensor ~760 MB) to incremental L2 norm formula. Eliminates OOM risk on 8 GB GPU.

- **BlankPriorKLLoss per-sample targets** (`src/models/losses.py`): changed from batch-mean severity target to per-sample KL targets. Controls get q=0.80, dysarthric get q=0.70 regardless of batch composition.

- **Memory cache size** (`src/data/dataloader.py`): reduced `memory_cache_size` default from 2048 to 256. Per-worker cache drops from ~800 MB to ~100 MB; LRU cache provides near-zero benefit for shuffled train access.

- **B1** (`evaluate.py`:923-924): removed bogus `n_ins` increment on substitution — was inflating insertion counts.

- **B2** (`evaluate.py`:1824-1830): neural PER changed from per-utterance mean to per-speaker macro-mean — `delta_per` now compares apples-to-apples with constrained.

- **B3** (`src/models/losses.py`:67-74): added `torch.nan_to_num(z, nan=0.0)` after `F.normalize` in `OrdinalContrastiveLoss` — prevents NaN from all-padding frames.

- **B4** (`src/data/dataloader.py`:240-243): fixed non-contiguous articulatory vocab IDs — `i+3` changed to `len(vocab)`.

- **N1 — Epsilon values** (`src/models/model.py`, `src/models/losses.py`, `src/models/uncertainty.py`, `train.py`): changed all `1e-8` epsilons to `1e-6` for BF16 numerical safety.

- **Swallowed exception** (`train.py`:1058-1059): `except Exception: pass` changed to `except RuntimeError: logger.warning(..., exc_info=True)`.

- **T1** (`tests/test_training_step.py`:178-220): rewrite fallback test to mock `TAF.forced_align` as raising — original never hit the fallback path.

- **T2** (`tests/test_dataloader.py`:56): `assert != 0` changed to `assert == -100` — was checking wrong condition.

- **T3** (`scripts/smoke_test.py`:243): removed emoji from assertion — fails on non-UTF-8 terminals.

- **Callback output** (`train.py`:1501-1518): `_CompactFoldProgressCallback` now `print()`s in addition to `logger.info()`; emoji removed from log message.

- **Config round-trip safety** (`src/utils/config.py`): `get_default_config()` now returns `copy.deepcopy(_default_config)` — prevents test/mutation leakage.

### Changed

- **High-β diagnostic** (`v4_final_beta_high`): evaluated with β_base=0.3, β_slope=1.5 (M03 β=0.8 vs default 0.23). Dysarthric PER collapsed from 0.079 to 0.804 (10.2× worse), deletions 4.1×. Confirms the constraint matrix has no useful inference-time phoneme-confusion knowledge and must remain weak at inference. The ablation chain is definitive: constraint's value is as a training-time regularizer, not inference-time fusion.

- **`--ablation` default** (`run_pipeline.py`:420,725): changed from `"full" to `None` — only overrides config when explicitly passed. Ablation mode falls back to config file or default.

- **Unknown YAML key warning** (`src/utils/config.py`): added `logger.warning(...)` for unknown keys in `load_config` — no longer silently ignored.

### Removed

- **Dead code — model** (`src/models/model.py`): removed `_unfreeze_all_hubert`, `unfreeze_after_warmup`, `count_parameters`, `set_seed` stub, and unconditional `return_features=False` branch.

- **Dead code — dataloader** (`src/data/dataloader.py`): removed `create_single_dataloader` + `main()` — never called from pipeline or tests.

- **Dead config fields** (`src/utils/config.py`): removed `constraint_learnable`, `log_gradients`, `log_model_architecture`, `save_predictions`, `save_confusion_matrix`, `save_attention_maps`, `temperature_default` — zero external references.

## [0.5.0] — 2026-06-09

### Fixed

- **CTC forced alignment fallback** (`train.py::_compute_ce_loss_aligned`): replaced silent label-drop with proportional interpolation (`align_labels_to_logits`) when `torchaudio.functional.forced_align` produces no valid frames.
  - Previously dropped the sample from CE supervision without notification, silently reducing effective CE supervision.

- **StratifiedMicroBatchSampler zero-length guard** (`train.py`): added `len()` guard against `n_ctrl == 0` (division by zero).
  - Previously crashed when a batch had zero control speakers.

- **Stratified sampler DataLoader kwargs** (`train.py::create_dataloaders`): filtered out `batch_size`, `shuffle`, `sampler`, `drop_last` when `batch_sampler` is provided.
  - Previously raised `ValueError` from PyTorch for mutually exclusive DataLoader arguments.

- **Logging format error** (`train.py::_MetricLoggerCallback.on_validation_epoch_end`): fixed `%8s` → `width` for blank probability metric in epoch summary.

- **Lazy forced_align import** (`train.py:521`): hoisted `from torchaudio.functional import forced_align` from per-batch hot path to module-level `import torchaudio.functional as TAF`.
  - Eliminated ~50ms redundant import overhead on every training step across all epochs.

- **Dead dir() guard** (`run_pipeline.py:544`): removed `'val_loader' in dir()` — unnecessary when `val_loader` is unconditionally assigned from return value.

### Changed

- **All print() → logger.info()** (`train.py`): converted 29 `print()` calls in `_CompactFoldProgressCallback` and `run_loso()` to `logger.info()` for structured log capture.
  - Enables log-level filtering and consistent MLflow/console output formatting.

- **Staged loss weight scheduling** (`train.py::on_train_epoch_start`): added 3-stage warmup for `lambda_ordinal` (0.01→0.03→0.05 across epochs 10/20) and `lambda_symbolic_kl` (0.1→0.3→0.5 across epochs 5/15).
  - Prevents ordinal and symbolic constraints from interfering with basic phoneme discrimination during early training.

- **Weighted loss breakdown logging** (`train.py::_MetricLoggerCallback.on_train_epoch_end`): logs each loss component's raw magnitude × λ with percentage of weighted total.
  - Enables one-shot hyperparameter balance audit without running separate diagnostic runs.

- **Configurable severity constant** (`src/utils/config.py:SymbolicConfig`): added `severity_normalization_constant: float = 5.0`; propagated to `_compute_adaptive_beta`, `_get_batch_severity`, and `evaluate_model` fallback.
  - Allows tuning beta scaling without hardcoded TORGO ceiling assumption.

- **Config roundtrip YAML fix** (`src/utils/config.py::to_dict`): converts `tuple` to `list` for safe YAML serialization.
  - Previously produced `!!python/tuple` tags that broke config reload.

- **Third-party logger suppression** (`run_pipeline.py`): set `huggingface_hub`, `httpx`, `httpcore`, `transformers`, `pytorch_lightning`, `urllib3`, `requests` to `WARNING`; added `warnings.filterwarnings` for HF_TOKEN unauthenticated request warning.

### Added

- **Per-speaker temperature calibration** (`evaluate.py`): added `calibrate_speaker_temperatures` function and `--calibrate-temperature` CLI flag.
  - Enables temperature scaling tuned per held-out speaker for improved beam-search calibration.

- **Row entropy penalty** (`src/models/losses.py:SymbolicKLLoss`): added `constraint_entropy_penalty_weight` (default 0.05) to regularize constraint matrix row entropy toward static prior.
  - Discourages degenerate row distributions in the learnable constraint matrix.

- **Stratified micro-batch sampler** (`train.py`): added `StratifiedMicroBatchSampler` with 3:1 dysarthric/control interleaving, gated by `use_stratified_micro_batch` + `stratified_dysarthric_ratio` in `TrainingConfig`.
  - Ensures every micro-batch sees both dysarthric and control samples for stable severity adaptation.

### Removed

- **Dead code**: removed duplicate `lambda_ce: float = 0.05` (overridden by `0.15`); removed unused `import pandas as pd` from `evaluate.py`; removed unused `NeuroSymbolicCollator` import from `run_pipeline.py`; removed 4 lazy imports inside hot loops (hoisted to module level).

- **Heuristic test runs**: removed 6 temporary loss-audit directories from `checkpoints/` and `results/`.

### Hyperparameter Audit (empirical, epoch 3)

Raw loss magnitudes confirmed balanced: CTC=3.60 (79.9%), CE=3.76 (15.6%), Art=1.45 (3.2%),
BlankKL=0.06 (0.3%), Ordinal=1.38 (1.9%), SymKL=0.27 (3.7%). No loss term is dominated or silent.
All λ values left at existing defaults.

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