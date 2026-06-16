# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **FastAPI inference service** (`serve.py`): health-check and model-metadata endpoints plus a `/transcribe` route that accepts multipart audio uploads (WAV, FLAC, MP3, OGG, M4A) and returns a phoneme sequence with articulatory classes. The model is loaded once at startup via a lifespan hook; severity can be supplied explicitly or resolved from a TORGO speaker ID. Supports both greedy and beam-search decoding on CUDA or CPU. See `docs/serving.md`.
- **Docker** (`Dockerfile`, `docker-compose.yml`, `.dockerignore`): multi-stage build (`base` → `train` / `serve`) on `nvidia/cuda:12.8.0-runtime-ubuntu22.04`. `docker compose up --build serve` runs GPU inference with a `/health` healthcheck on port 8000. `docker compose run train` runs one-off training jobs.
- **GitHub Actions CI** (`.github/workflows/ci.yml`): lint and test jobs run on push to main and on pull requests. Lint installs `ruff==0.11.0` and `mypy==1.15.0`, then runs `ruff check .` and `mypy serve.py --follow-imports=silent`. Test installs CPU-only `torch==2.9.0`/`torchaudio==2.9.0`, runs the full pytest suite, the smoke-test unit profile, and an `import serve` gate.
- **Lint cleanup**: repo-wide `ruff check .` passes clean. Config extended with rules `N` (pep8-naming), `UP` (pyupgrade), `S324` (hashlib), `E402` (module-level import). ~126 issues auto-fixed; manual fixes applied to `train.py`, `scripts/smoke_test.py`, and `serve.py`.

## [0.6.1] — 2026-06-16

### Changed

- **Restored v4_final training behavior** (post-hoc revert of b65e3a6): four changes from that commit caused a measurable regression against the canonical checkpoint and were rolled back:
  - `label_smoothing=0.1` removed from CE loss (restored to 0, matching v4_final)
  - `_compute_stratified_per` restored to per-utterance averaging for dysarthric/control metrics
  - Dataloader BLANK/PAD/UNK weight multipliers restored to 1.2 / 1.5 / 1.1
  - `decode_predictions` `beam_width` default restored to 10
- **Preserved from b65e3a6/05628ce**: new `pyproject.toml` with ruff/mypy configs; test suite additions (`test_explainability.py`, `test_utils.py`); config-driven LR multipliers (`encoder_lr_multiplier`, `symbolic_lr_multiplier`); seeded `StratifiedMicroBatchSampler`; logger output replacing bare `print` / `except: pass`; decoder confounding fix; `on_test_epoch_end` aggregation; gradient-checkpointing-aware VRAM estimates.

## [0.6.0] — 2026-06-11

### Full Eval Results

- **Canonical result** (v4_final, beam width=25): macro-speaker PER=**0.133** (95% CI [0.079, 0.200]), WER=0.116, I/D=2.1×. Both neural and constrained paths decoded with beam search gives per_neural=0.131, per_constrained=0.133. Symbolic Δ = +0.0015 (p=0.246, not significant).
- **Articulatory accuracy**: manner=81.7%, place=90.5%, voice=95.3%.
- **Temperature calibration**: M05 τ=1.25, M01 τ=1.03.
- **Ablation chain**: neural-only baseline (no symbolic) = 0.1346. Adding SeverityAdapter alone degrades to 0.1444 (+7.3%). Adding the constraint matrix recovers to 0.1326 — better than neural-only. The constraint's primary value is as a **training-time regularizer** for severity-adaptive fusion, not as an inference-time per-frame correction.

### Fixed

- **Decoder confounding** (`evaluate.py:1734-1746`): neural sub-path now uses beam search when `--beam-search` is set, matching the constrained path. Previously neural was always greedy, inflating the apparent gap by ~0.003.
- **blank_constraint_threshold read path** (`evaluate.py:1581-1592`): now reads from `model.symbolic_config` or `model.symbolic_layer.config` instead of a non-existent `model.config`, which previously forced a hard-coded default of 0.25.
- **CTC forced-alignment batching** (`train.py::_compute_ce_loss_aligned`): replaced per-sample `TAF.forced_align` calls with a single batched call. Gracefully falls back to `align_labels_to_logits` on failure. Inner loop uses tensor indices and `torch.where` to eliminate repeated `.item()` calls.
- **Gradient norm computation** (`train.py`): replaced `torch.cat` of per-layer norms (produced a ~760 MB temporary tensor) with an incremental L2 formula. Eliminates OOM risk at batch=12 on 8 GB GPUs.
- **BlankPriorKLLoss targets** (`src/models/losses.py`): per-sample KL targets instead of batch-mean. Controls get q=0.80, dysarthric get q=0.70, independent of batch composition.
- **Memory cache size** (`src/data/dataloader.py`): reduced `memory_cache_size` from 2048 to 256 entries. Per-worker cache drops from ~800 MB to ~100 MB; shuffled train access makes the larger cache ineffective.
- **B1** (`evaluate.py:923-924`): removed spurious `n_ins` increment on substitution errors — was inflating insertion counts.
- **B2** (`evaluate.py:1824-1830`): neural PER now uses per-speaker macro-mean (`delta_per` compares apples-to-apples with constrained path).
- **B3** (`src/models/losses.py:67-74`): added `torch.nan_to_num(z, nan=0.0)` after `F.normalize` in `OrdinalContrastiveLoss` to handle all-padding frames.
- **B4** (`src/data/dataloader.py:240-243`): articulatory vocab IDs corrected from `i+3` to `len(vocab)` to handle non-contiguous phoneme sets.
- **N1 — Epsilon values** (`src/models/model.py`, `src/models/losses.py`, `src/models/uncertainty.py`, `train.py`): changed all `1e-8` epsilons to `1e-6` for BF16 numerical stability.
- **Swallowed exception** (`train.py:1058-1059`): `except Exception: pass` replaced with `except RuntimeError: logger.warning(..., exc_info=True)`.
- **T1** (`tests/test_training_step.py:178-220`): fallback test now mocks `TAF.forced_align` raising — original test never exercised the fallback path.
- **T2** (`tests/test_dataloader.py:56`): `assert != 0` corrected to `assert == -100` — was checking the wrong sentinel.
- **T3** (`scripts/smoke_test.py:243`): removed emoji from assertion message — fails on non-UTF-8 terminals.
- **Callback output** (`train.py:1501-1518`): `_CompactFoldProgressCallback` now `print()`s in addition to `logger.info()`; emoji removed.
- **Config deep-copy safety** (`src/utils/config.py`): `get_default_config()` returns `copy.deepcopy(_default_config)` — prevents cross-test mutation leakage.

### Changed

- **High-β diagnostic** (`v4_final_beta_high`): evaluated with β_base=0.3 and β_slope=1.5 (M03 β=0.8 vs default 0.23). Dysarthric PER collapsed from 0.079 to 0.804 (10.2× worse); deletions rose 4.1×. Confirms the constraint matrix has no useful inference-time phoneme-confusion knowledge. The ablation chain is definitive: the constraint's value is as a training-time regularizer, not an inference-time correction.
- **`--ablation` CLI default** (`run_pipeline.py:420,725`): changed from `"full"` to `None` — only overrides config when explicitly passed; falls back to the config file or its own default.
- **Unknown YAML key warning** (`src/utils/config.py`): added `logger.warning(...)` for unrecognized keys in `load_config` — no longer silently dropped.

### Removed

- **Dead code — model** (`src/models/model.py`): removed `_unfreeze_all_hubert`, `unfreeze_after_warmup`, `count_parameters`, `set_seed` stub, and the unconditional `return_features=False` branch.
- **Dead code — dataloader** (`src/data/dataloader.py`): removed `create_single_dataloader` and `main()` — neither was called from the pipeline or tests.
- **Dead config fields** (`src/utils/config.py`): removed `constraint_learnable`, `log_gradients`, `log_model_architecture`, `save_predictions`, `save_confusion_matrix`, `save_attention_maps`, `temperature_default` — zero external references.

## [0.5.0] — 2026-06-09

### Fixed

- **CTC forced-alignment fallback** (`train.py::_compute_ce_loss_aligned`): replaced silent sample-drop with proportional interpolation (`align_labels_to_logits`) when `torchaudio.functional.forced_align` returns no valid frames.
- **StratifiedMicroBatchSampler zero-control guard** (`train.py`): added `len()` check before dividing by `n_ctrl`. Previously crashed on batches with no control speakers.
- **Stratified sampler DataLoader kwargs** (`train.py::create_dataloaders`): filters out `batch_size`, `shuffle`, `sampler`, `drop_last` when `batch_sampler` is provided. Previously raised `ValueError` for mutually exclusive arguments.
- **Logging format placeholder** (`train.py::_MetricLoggerCallback.on_validation_epoch_end`): `%8s` replaced with `width` for the blank-probability metric to avoid `TypeError`.
- **Lazy forced_align import** (`train.py:521`): hoisted `from torchaudio.functional import forced_align` from per-batch hot path to module-level `import torchaudio.functional as TAF`. Eliminates ~50 ms redundant import overhead per training step.
- **Dead dir() guard** (`run_pipeline.py:544`): removed `'val_loader' in dir()` — `val_loader` is unconditionally assigned from the return value.

### Changed

- **print() → logger.info()** (`train.py`): converted 29 `print()` calls in `_CompactFoldProgressCallback` and `run_loso()` to structured logging. Enables log-level filtering and consistent MLflow/console output.
- **Staged loss scheduling** (`train.py::on_train_epoch_start`): three-stage warmup for `lambda_ordinal` (0.01→0.03→0.05 at epochs 10/20) and `lambda_symbolic_kl` (0.1→0.3→0.5 at epochs 5/15). Prevents constraints from destabilising basic phoneme discrimination in early epochs.
- **Loss breakdown logging** (`train.py::_MetricLoggerCallback.on_train_epoch_end`): each loss component's raw magnitude × λ is logged with its percentage of the weighted total. Enables one-shot hyperparameter audits without separate diagnostic runs.
- **Configurable severity ceiling** (`src/utils/config.py:SymbolicConfig`): `severity_normalization_constant: float = 5.0` propagated to `_compute_adaptive_beta`, `_get_batch_severity`, and `evaluate_model`. Removes the hardcoded TORGO ceiling assumption.
- **YAML config roundtrip** (`src/utils/config.py::to_dict`): converts `tuple` to `list` for safe YAML serialization. Previously emitted `!!python/object/apply:builtins.tuple` tags that broke config reload.
- **Third-party logger suppression** (`run_pipeline.py`): set `huggingface_hub`, `httpx`, `httpcore`, `transformers`, `pytorch_lightning`, `urllib3`, `requests` to `WARNING`; added `warnings.filterwarnings` for unauthenticated HF Hub requests.

### Added

- **Per-speaker temperature calibration** (`evaluate.py`): `calibrate_speaker_temperatures` function and `--calibrate-temperature` CLI flag. Scales logits per held-out speaker to anchor blank probability at the target.
- **Row entropy penalty** (`src/models/losses.py:SymbolicKLLoss`): `constraint_entropy_penalty_weight` (default 0.05) regularises constraint matrix row entropy toward the static prior, discouraging degenerate distributions.
- **Stratified micro-batch sampler** (`train.py`): `StratifiedMicroBatchSampler` with 3:1 dysarthric/control interleaving, gated by `use_stratified_micro_batch` and `stratified_dysarthric_ratio` in `TrainingConfig`. Guarantees each micro-batch contains both dysarthric and control samples for stable severity adaptation.

### Removed

- **Dead code**: duplicate `lambda_ce: float = 0.05` field (overridden by `0.15`); unused `import pandas as pd` from `evaluate.py`; unused `NeuroSymbolicCollator` import from `run_pipeline.py`; four lazy imports inside hot loops hoisted to module level.
- **Heuristic test artifacts**: six temporary loss-audit directories removed from `checkpoints/` and `results/`.

### Hyperparameter Audit (epoch 3)

Raw loss magnitudes confirm balance: CTC=3.60 (79.9%), CE=3.76 (15.6%), Art=1.45 (3.2%),
BlankKL=0.06 (0.3%), Ordinal=1.38 (1.9%), SymKL=0.27 (3.7%). No term is dominated or silent.
All λ values left at existing defaults.

## [0.4.0] — 2026-05-13

### Fixed

- **Blank-frame bypass threshold** (`src/models/model.py`, `SymbolicConfig`): lowered `blank_constraint_threshold` from 0.5 to 0.25, increasing constraint activation from ~15% to ~35% of CTC frames.
- **OneCycleLR warm restart** (`train.py`): scheduler now restarts at its peak LR after each HuBERT unfreeze stage. Previously resumed at the decayed position, slowing convergence post-unfreeze.
- **LOSO early stopping patience** (`TrainingConfig`): raised `loso_early_stopping_patience` to 22 epochs. Full-system runs converge slower than ablation runs; the previous value of 8 was premature.

### Changed

- **Frame-CE gating** (`train.py::compute_loss`): weight reduced and gated behind `frame_ce_start_epoch=15`. Noisy nearest-neighbour label alignment is deferred until the model has learned basic phoneme boundaries.
- **Blank-mass penalty** (`src/models/losses.py`): added explicit penalty on constraint matrix rows dominated by the blank token. Works with the lowered bypass threshold to prevent matrix degradation.
- **SpecAugment gate** (`src/models/model.py`): decoupled from `_hubert_is_frozen`; now keyed on `self.training` only. Previously skipped whenever any encoder layer was frozen, even with actively training downstream modules.
- **Attention mask downsampling** (`train.py::_downsample_attn_mask`): rewritten with explicit two-step stride calculation. The implicit formula could produce incorrect lengths on odd-sized inputs, causing shape mismatches at evaluation.

### Added

- **`no_severity_adapter` ablation mode** (`run_pipeline.py`, `src/models/model.py`): disables `SeverityAdapter` while keeping `SymbolicConstraintLayer`. Isolates the adapter's contribution without requiring a full neural-only baseline.
- **`spearman_valid` flag** (`evaluate.py`): signals whether the Spearman correlation is statistically valid (n_speakers ≥ 5).
- **`plot_per_by_manner`** (`src/visualization/experiment_plots.py`, `scripts/generate_figures.py`): articulatory-stratified PER heatmap broken down by manner of articulation.
- **Gradient checkpointing toggle** (`evaluate.py`): disabled during evaluation to reduce memory overhead and improve throughput.

### Fixed

- **BigramLM data leak** (`run_pipeline.py`): language model now built from training speakers only. Previously included test/validation phoneme sequences, producing optimistic perplexity estimates.
- **SeverityAdapter diagnostic logging** (`train.py::validation_step`): added output-norm monitoring to catch degenerate outputs (NaN, all-zero, extreme values) early.