# DysarthriaNSR — Technical Audit TODO

**Audit date:** March 2026  
**Auditor basis:** Full codebase read + output analysis (evaluation_results.json, current_output.txt, explanations.json)  
**Baseline:** `baseline_v2` — Macro-speaker PER 0.123, I/D ratio 4.6×, 3 speakers, 3 548 test samples

---

## 1. Critical Issues (Must Fix)

These are confirmed bugs with direct evidence from code or outputs. They corrupt results, produce silent failures, or create invalid experimental conditions.

---

### C1 — Raw logits fed to NLLLoss (negative CE loss, corrupted gradients)

**Problem:**  
`_compute_ce_loss` in [train.py](train.py) receives `outputs.get('logits_neural', ...)`, which is the raw linear output of `PhonemeClassifier` (no softmax/log-softmax applied). `nn.NLLLoss` requires **log-probabilities**. Passing raw logits causes the loss to compute `-logit_value_at_target`, which becomes negative when the model is confident (logit > 0 for correct class). This is directly confirmed by `val/loss` going negative from epoch 5 onwards in `current_output.txt` (−0.213 → −5.858).

**Evidence:** `train/loss` turns negative from epoch 5; `_compute_ce_loss` passes `outputs.get('logits_neural', ...)` to `self.ce_loss` (an `NLLLoss`); `logits_neural` comes from `PhonemeClassifier.classifier` which is a bare `nn.Linear` with no activation.

**Suggested Fix:**  
In `_compute_ce_loss`, replace:
```python
logits_flat = log_probs.reshape(-1, num_classes)
return self.ce_loss(logits_flat, labels_flat)
```
With:
```python
log_probs_flat = F.log_softmax(log_probs.reshape(-1, num_classes), dim=-1)
return self.ce_loss(log_probs_flat, labels_flat)
```
Alternatively, change `self.ce_loss` from `nn.NLLLoss` to `nn.CrossEntropyLoss` and pass raw logits directly.

**Priority: HIGH**

---

### C2 — Double audio normalization corrupts HuBERT input distribution

**Problem:**  
`TorgoNeuroSymbolicDataset._load_audio()` peak-normalizes the waveform (`waveform = waveform / max_val`), then `__getitem__` passes the result to `self.processor(waveform, ...)`. `AutoFeatureExtractor` for `facebook/hubert-base-ls960` applies its own zero-mean unit-variance normalization by default. The two normalizations are not composable: the HuBERT processor's normalizer assumes input in the raw waveform amplitude range, not pre-normalized. This deviates from HuBERT's pre-training conditions.

**Evidence:** Bug B11 in the project audit memo. `_load_audio` lines `if max_val > 0: waveform = waveform / max_val` followed by `audio_features = self.processor(waveform, ...)` in `__getitem__`.

**Suggested Fix:**  
Either (a) remove the explicit peak-normalization in `_load_audio` and rely entirely on the HuBERT processor, OR (b) instantiate the processor with `do_normalize=False` and keep only the peak normalization — then audit that CTC training still converges. Option (a) is closer to HuBERT's pre-training setup.

**Priority: HIGH**

---

### C3 — Data leakage: speaker-level split silently falls back to sample-level mixing

**Problem:**  
`create_dataloaders` in [train.py](train.py) attempts a speaker-stratified split but falls back to a random sample-level split when any split is empty. The logs for `baseline_v2` show: *"Speaker-level split produced empty split(s). Falling back to sample-level split for this run."* This means utterances from the same speakers appear in both training and test sets, violating the speaker-independent evaluation assumption and inflating all reported metrics.

**Evidence:** `current_output.txt` line: "⚠️ Speaker-level split produced empty split(s). Falling back to sample-level split." The root cause is using only 3 speakers with a 70/15/15 ratio — 3 × 0.15 < 1 → zero val/test speakers.

**Suggested Fix:**  
Guard against the fallback by raising an error unless explicitly overridden. For low-speaker-count runs, replace the random fallback with a deterministic round-robin speaker assignment: always put at least one speaker in each split. For `baseline_v2` (3 speakers): 1 test speaker, 1 val speaker, 1 train speaker. Log a warning but do not silently fall back to a sample-level split that breaks evaluation validity.

```python
if len(test_speakers) == 0:
    # Round-robin: one speaker per split regardless of requested ratio
    test_speakers  = [speakers[-1]]
    val_speakers   = [speakers[-2]] if len(speakers) > 1 else test_speakers
    train_speakers = [s for s in speakers if s not in val_speakers and s not in test_speakers]
    if not train_speakers:
        raise ValueError("Need at least 3 speakers for speaker-stratified splits.")
```

**Priority: HIGH**

---

### C4 — Explanations JSON produces all-empty records (silent explainability failure)

**Problem:**  
`results/baseline_v2/explanations.json` contains 2 481 entries all equal to `{}`. The `ExplainableOutputFormatter.add()` method stores its argument and `save_explanations()` writes them to disk. The problem is that `format_utterance()` may produce a valid dict that gets stored, but the file shows empty dicts, indicating that `formatter.add(explanation)` is appending `{}` instead of the formatted dict. Looking at the code flow in `evaluate.py`, `explanation = formatter.format_utterance(...)` followed by `formatter.add(explanation)` — the result of `format_utterance` is a complete schema dict, so the issue is in how `save_explanations` serializes the list.

**Evidence:** All 2 481 entries in `explanations.json` are `{}`. The explainability pipeline does not raise any exception (the `try/except` block in `evaluate_model` prints "✅ Explainability artifacts generated").

**Suggested Fix:**  
1. Read `src/explainability/output_format.py`'s `save_explanations` method and verify it iterates over `self._explanations` correctly.  
2. Add an assertion after `formatter.add(explanation)`: `assert explanation, f"Empty explanation for utt_{i}"` so silent failures become visible.  
3. Write a unit test: `formatter.format_utterance(...)` should return a non-empty dict, and `save_explanations` should write non-empty records.

**Priority: HIGH**

---

### C5 — Symbolic rule tracker min_confidence blocks all activations

**Problem:**  
`SymbolicRuleTracker` is initialized with `min_confidence=symbolic_config.min_rule_confidence` (default: 0.5). The β weight is initialized to `constraint_weight_init=0.05` and is clamped to a maximum of 0.8. When `_track_activations` calls `tracker.log_rule_activation(..., confidence=beta_value)`, the confidence is β ≈ 0.05–0.2 throughout training — always below the 0.5 threshold. Consequently, `tracker._activations` remains empty and `generate_explanation()` always returns `total_activations: 0`.

**Evidence:** `evaluation_results.json`: `"rule_impact": {"total_activations": 0, "activation_rate": 0.0}`. β convergence is ~0.05–0.2 based on training logs.

**Suggested Fix:**  
Either (a) lower `min_rule_confidence` to 0.1 (matching the expected working range of β), or (b) track all activations regardless of confidence and apply the threshold only when reporting "high_confidence_corrections". Option (b) preserves diagnostic value at all confidence levels.

**Priority: HIGH**

---

### C6 — `art_ce_losses` stored as plain `dict`, not `nn.ModuleDict`

**Problem:**  
`self.art_ce_losses = {}` in `_init_loss_functions` stores `nn.CrossEntropyLoss` instances in a plain Python dict. PyTorch Lightning does not register dict members for device management — their internal weight tensors are not automatically moved to GPU. The workaround in `on_fit_start` manually iterates and moves `.weight`, but this is brittle (breaks if a new key is added, or if accessed before `on_fit_start` fires).

**Evidence:** `train.py` line `self.art_ce_losses = {}` with `nn.CrossEntropyLoss` values; `on_fit_start` has manual `.to(self.device)` calls for these weights.

**Suggested Fix:**  
Replace with `self.art_ce_losses = nn.ModuleDict({...})` so Lightning registers and moves them automatically. Remove the manual device-move code in `on_fit_start`.

**Priority: HIGH**

---

### C7 — `TORGO_SEVERITY_MAP` not used in `compute_loss` severity calculation

**Problem:**  
`DysarthriaASRLightning.forward()` correctly uses per-speaker severity scores from `TORGO_SEVERITY_MAP` when speaker IDs are available. However, `training_step` computes `severity = batch['status'].float() * 5.0`, which always maps control→0.0 and dysarthric→5.0 (binary, not continuous). This binary severity is passed to `OrdinalContrastiveLoss`, which is designed for continuous severity gradients — making the ordinal part of the loss equivalent to a standard contrastive loss with a fixed large margin.

**Evidence:** `train.py` `training_step`: `severity = batch['status'].float() * 5.0` rather than `[get_speaker_severity(s) for s in batch['speakers']]`.

**Suggested Fix:**  
Use the same per-speaker lookup from `forward()` in `training_step`:
```python
speakers = batch.get('speakers', [])
if speakers and isinstance(speakers[0], str):
    severity = torch.tensor([get_speaker_severity(s) for s in speakers],
                             dtype=torch.float32, device=device)
else:
    severity = batch['status'].float() * 5.0
```

**Priority: HIGH**

---

### C8 — `statsmodels` not listed in `requirements.txt`

**Problem:**  
`evaluate.py` imports `from statsmodels.stats.multitest import multipletests` for Holm-Bonferroni correction, but `statsmodels` is absent from `requirements.txt`. This causes a silent fallback to a Bonferroni correction (which is more conservative) on systems without `statsmodels`. Reported statistical results are inconsistent depending on the environment.

**Evidence:** `requirements.txt` does not contain `statsmodels`; `evaluate.py` has the import in a `try/except` block.

**Suggested Fix:**  
Add `statsmodels>=0.14.0` to `requirements.txt` and remove the `try/except` fallback. If portability is required, at minimum document that statsmodels is needed for accurate p-value corrections.

**Priority: HIGH**

---

## 2. Important Improvements

These are gaps that significantly limit system capabilities but do not crash the pipeline.

---

### I1 — WER is never computed at the corpus level

**Problem:**  
`jiwer` is imported in `evaluate.py` and `compute_wer_texts()` is implemented, but `evaluate_model()` never calls it. The `results` dict has no `wer` key. The `wer` field in `explanations.json` is set to `utt_per` (a proxy), not actual WER. This makes it impossible to report standard ASR metrics for publication.

**Suggested Fix:**  
In `evaluate_model`, after decoding all predictions, join phoneme lists into space-separated strings and call `compute_wer_texts(predictions_text, references_text)`. Add `'wer'` to the results dict and print it in the summary.

**Priority: HIGH**

---

### I2 — Insertion bias (4.6× I/D ratio) not resolved

**Problem:**  
`baseline_v2` still shows 3 143 insertions vs. 678 deletions (4.6× ratio, target: <3×). The `blank_prob_mean` is persistently ~0.91 throughout all 30 epochs while the target is 0.30. The `lambda_blank_kl=0.05` is insufficient to push the blank distribution toward the target. The bug C1 (raw logits to NLLLoss causing negative CE) further masks the problem by making the total loss negative, which distorts gradient signals.

**Suggested Fix:**  
1. Fix C1 first — the negative CE loss directly undermines blank regularization.  
2. After C1 fix, progressively increase `lambda_blank_kl` from 0.05 → 0.15 → 0.30 across training stages.  
3. Consider a per-utterance blank budget: at inference time, if the blank fraction exceeds 0.7, apply a length-penalty during beam search.  
4. Add a diagnostic plot: blank probability histogram per epoch to the results artifacts.

**Priority: HIGH**

---

### I3 — No full-TORGO evaluation (3 speakers is insufficient)

**Problem:**  
All reported results use only 3 of the ~15 available TORGO speakers (MC02, MC04, M03). With n=3 speakers, macro-PER CI is extremely wide ([0.101, 0.159]), Spearman correlation is degenerate (r=0.0, p=1.0), and the dysarthric-vs-control comparison is based on 1 dysarthric speaker vs 2 control speakers. No generalizable conclusions can be drawn.

**Suggested Fix:**  
Run evaluation with all available TORGO speakers using LOSO (`--loso` flag). The `run_loso` function and `create_loso_splits` are already implemented. Ensure the manifest covers all speakers by verifying `data/processed/torgo_neuro_symbolic_manifest.csv` contains all 15 speakers.

**Priority: HIGH**

---

### I4 — `UncertaintyAwareDecoder` is implemented but never activated

**Problem:**  
`src/models/uncertainty.py` provides `UncertaintyAwareDecoder` with MC-Dropout, but `evaluation_results.json` shows `"uncertainty": {"computed": false}`. `compute_uncertainty=False` is the default in `evaluate_model`. The `ExperimentConfig.compute_uncertainty` flag exists but is never read in `run_pipeline.py` to override the default.

**Suggested Fix:**  
In `run_pipeline.py`'s evaluation call, pass `compute_uncertainty=config.experiment.compute_uncertainty` and `uncertainty_n_samples=config.experiment.uncertainty_n_samples` to `evaluate_model`.

**Priority: MEDIUM**

---

### I5 — Articulatory head accuracy (manner 34%, place 30%) extremely low

**Problem:**  
The articulatory auxiliary heads have very low accuracy (manner=0.34, place=0.30), which suggests either the label alignment is wrong or the heads are not learning. Since labels are phoneme-level (one label per phoneme, ~5–25 labels per utterance) but logits are frame-level (~300 frames per utterance), the `_align_labels_to_logits` function pads labels to frame length with -100. This means only the first L frames (one per phoneme) are trained — a strong implicit forced-alignment assumption that is almost certainly wrong for CTC models.

**Suggested Fix:**  
Either (a) use a dedicated temporal pooling module to produce one prediction per phoneme region using CTC attention, or (b) supervise articulatory heads at the utterance level using a simple pool/average over all frames (predicting the set of manner/place/voice features present rather than per-frame). Option (b) is simpler and avoids the frame-alignment issue entirely.

**Priority: MEDIUM**

---

### I6 — `SymbolicKLLoss` lazily initialized but potentially on wrong device if re-used

**Problem:**  
`self._symbolic_kl_loss` is set to `None` in `_init_loss_functions` and initialized on first use in `compute_loss` via `.to(self.device)`. If the model is moved to a different device between calls (e.g., during checkpoint loading for evaluation), `_symbolic_kl_loss` will be on the old device. Additionally, because it's not a registered module, it won't appear in `state_dict`.

**Suggested Fix:**  
Initialize `SymbolicKLLoss` eagerly in `_init_loss_functions` after the model is initialized (the static matrix is accessible via `model.symbolic_layer.static_constraint_matrix`). Register it as `self.symbolic_kl_loss = SymbolicKLLoss(static_C)` so it participates in Lightning's device management.

**Priority: MEDIUM**

---

### I7 — `cfg = Config()` instantiated at module import time

**Problem:**  
`src/utils/config.py` runs `cfg = Config()` at module load time. `Config.__init__` calls `torch.cuda.is_available()`, `torch.cuda.get_device_properties()`, prints VRAM estimates to stdout, and creates disk directories. Importing `from src.utils.config import normalize_phoneme` triggers all of this as a side effect — VRAM output printed during every import, directories created unexpectedly.

**Evidence:** VRAM status is printed to stdout even during `evaluate.py` or test runs that import config.

**Suggested Fix:**  
Remove `cfg = Config()` from module level. Change `get_default_config()` to construct on first call (lazy singleton):
```python
_default_config: Optional[Config] = None
def get_default_config() -> Config:
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config
```

**Priority: MEDIUM**

---

## 3. Evaluation & Metrics Improvements

---

### E1 — Severity-PER Spearman correlation returns degenerate r=0.0, p=1.0

**Problem:**  
With n=3 speakers having PERs [0.159, 0.101, 0.109] and severities [0.0, 0.0, 4.59], the Spearman rank correlation collapses (two control speakers have tied ranks → degenerate). `scipy.stats.spearmanr` returns r=0.0, p=1.0 in this case, which is a degenerate result, not a valid measurement. The Pearson r=-0.39 with p=0.744 is correctly non-significant, but neither metric is meaningful with n=3.

**Suggested Fix:**  
(a) Annotate in results that correlation requires n≥5 speakers for meaningful inference; suppress the metric when n<5.  
(b) Fix C3/I3 to run over all 15 speakers so genuine severity correlation can be measured.  
(c) Guard the print output: `if len(sev_scores) >= 5: print(f"Severity ↔ PER correlation: r=...")`.

**Priority: MEDIUM**

---

### E2 — Missing overall WER in results (see also I1)

**Problem:**  
`evaluation_results.json` has no `wer` key. WER is the standard metric reported in ASR papers. Without it, the results are not comparable to other dysarthric ASR systems.

**Suggested Fix:**  
Compute corpus-level WER using phoneme sequences joined as words (or actual word transcripts from the `transcripts` field in the manifest if available). Report both PER and WER.

**Priority: HIGH (same as I1)**

---

### E3 — No per-phoneme error breakdown in results

**Problem:**  
`error_analysis.common_confusions` lists top-20 substitution pairs but there is no per-phoneme PER breakdown showing which specific phonemes are most error-prone. This is critical for clinical interpretability (e.g., identifying that fricatives or velars are most affected).

**Suggested Fix:**  
Add a `per_phoneme_per` dict to the results, mapping each phoneme to its individual substitution/deletion/insertion rates. This can be computed during `analyze_phoneme_errors` by counting errors per reference phoneme. Also generate a bar chart `results/{run}/per_phoneme_per.png`.

**Priority: MEDIUM**

---

### E4 — No training curve visualization

**Problem:**  
MLflow logs `val/per` per epoch but no plot is produced in `results/`. The existing `generate_figures.py` in `scripts/` may handle some plots, but there is no learning curve (val/per vs epoch) automatically saved alongside evaluation results. Reviewers cannot assess convergence without manually querying MLflow.

**Suggested Fix:**  
At the end of `train()`, retrieve the run's metric history from MLflow and save `results/{run_name}/learning_curve.png` showing `train/loss` and `val/per` vs epoch.

**Priority: MEDIUM**

---

### E5 — Articulatory confusion analysis never generates `articulatory_confusion.png` when articulatory accuracy is low

**Problem:**  
In `evaluate.py`, `art_analyzer.plot_feature_confusion(results_dir / 'articulatory_confusion.png')` is only called inside the `if generate_explanations:` block. In `baseline_v2`, `generate_explanations=True` was used (explanations.json exists), but the articulatory confusion plot is not in `results/baseline_v2/figures/`. This suggests the `art_analyzer.accumulate_from_errors` call receives no substitution errors (because the explainability pipeline fails silently — see C4).

**Suggested Fix:**  
Move the articulatory confusion analysis outside the explainability block by computing it directly from `error_analysis['common_confusions']` which is always available. This removes the dependency on the (currently broken) explainability pipeline.

**Priority: MEDIUM**

---

### E6 — `plot_rule_impact` silently skips when top_rules is empty

**Problem:**  
`plot_rule_impact(rule_stats, results_dir / 'rule_impact.png')` returns immediately when `top_rules` is empty (`if not stats or not stats.get('top_rules'): return`). With zero rule activations (C5), this plot is never generated. Reviewers see no symbolic impact visualization.

**Suggested Fix:**  
After fixing C5 (lower min_confidence), rule activations will be recorded. Additionally, generate a fallback plot showing the static constraint matrix heatmap (which is always available) to give interpretability even when dynamic activations are zero.

**Priority: LOW**

---

### E7 — Bootstrap CI computed over only 3 macro speaker means

**Problem:**  
`compute_per_with_ci` is called with macro_per_scores = list of 3 values (one per speaker). Bootstrap CI with n=3 is not a reliable confidence interval — the percentile estimates are dominated by which of the 3 values is re-sampled. The CI for baseline_v2 is [0.101, 0.159], which is simply the range of the 3 speaker PERs.

**Suggested Fix:**  
(a) Note in results that CI is unreliable with n<10.  
(b) Complement macro-speaker CI with sample-level CI (already available as `per_sample_mean` with 3548 samples, which gives a more reliable CI).  
(c) Fix I3 to use full TORGO data so meaningful CIs can be computed.

**Priority: LOW**

---

## 4. Codebase Improvements

---

### Q1 — `_align_labels_to_logits` duplicated in train.py and evaluate.py

**Problem:**  
Identical function defined in both `train.py` and `evaluate.py`. Any fix to one must be applied to the other manually.

**Suggested Fix:**  
Move to a shared utility module (e.g., `src/utils/sequence_utils.py`) and import from both files.

**Priority: MEDIUM**

---

### Q2 — `create_dataloaders` function has two conflicting implementations

**Problem:**  
A `create_dataloaders` function exists in both `src/data/dataloader.py` (simple, no splits) and `train.py` (full speaker-stratified splits). The `dataloader.py` version creates only a single `DataLoader` with a sampler and returns it. The `train.py` version returns a `(train, val, test)` tuple. Both are named identically, causing confusion and risk of wrong import.

**Suggested Fix:**  
Rename `dataloader.py::create_dataloaders` to `create_training_dataloader` (or remove it since it's a simplified version), and keep the full `train.py::create_dataloaders` as the canonical implementation.

**Priority: MEDIUM**

---

### Q3 — No unit tests exist anywhere

**Problem:**  
The project has no automated tests. `scripts/smoke_test.py` exists but only tests end-to-end pipeline integrity (and only runs with 5 batches). There are no unit tests for loss functions, metric computations, phoneme alignment, config loading, or data processing. This makes it impossible to detect regressions when fixing the bugs listed in this document.

**Suggested Fix:**  
Create `tests/` directory with at minimum:
- `test_losses.py`: verify `OrdinalContrastiveLoss` returns ≥0, `BlankPriorKLLoss` returns ≥0 (critical given C1 suggests the system operates at negative loss values)
- `test_metrics.py`: verify `compute_per` on known sequences; verify `compute_wer_texts`
- `test_config.py`: verify `Config.load(Config.save(...))` round-trip
- `test_dataloader.py`: verify `NeuroSymbolicCollator` produces correct shapes and -100 label padding

**Priority: MEDIUM**

---

### Q4 — Gradient NaN/Inf detection not enabled

**Problem:**  
With a negative total loss (C1) and high blank probabilities, gradients may occasionally spike or become NaN, especially during the first 3 epochs (frozen encoder). There is no NaN/Inf detection beyond `gradient_clip_val=1.0`. Silent NaN propagation can corrupt model weights without any warning.

**Suggested Fix:**  
Add a `detect_anomaly` flag for debugging runs, and add a gradient norm logging callback:
```python
self.log('train/grad_norm', self._compute_grad_norm(), on_step=True)
```
Consider adding `torch.autograd.set_detect_anomaly(True)` as a CLI debug flag.

**Priority: MEDIUM**

---

### Q5 — Checkpoint filename collides when best model is not saved

**Problem:**  
`ModelCheckpoint` uses `filename='epoch={epoch:02d}-val_per={val/per:.3f}'`. The `val/per` contains a slash which is invalid in filenames on Linux (it creates a subdirectory `val/`). This can silently create `checkpoints/{run}/val/per=0.204.ckpt` instead of a flat file, breaking the `checkpoint_callback.best_model_path` path resolution.

**Suggested Fix:**  
Change the filename template to avoid slashes:
```python
filename='epoch={epoch:02d}-valper={val_per:.3f}'
```
And configure `ModelCheckpoint(..., monitor='val/per', ...)` separately. Alternatively use a custom `ModelCheckpoint` that replaces `/` with `_` in the filename.

**Priority: MEDIUM**

---

### Q6 — SymbolicKLLoss direction comment is misleading

**Problem:**  
`losses.py` contains an extensive comment block explaining the KL direction fix (Bug B7). The comment is correct in its conclusion but the explanation references `F.kl_div(log_q, log_p)` using variable names that don't match PyTorch's API signature (`F.kl_div(input, target)`), which could confuse future developers.

**Suggested Fix:**  
Rewrite the comment to use PyTorch's exact parameter names and cite the PyTorch docs version:
```python
# F.kl_div(input, target, log_target=True) computes sum(exp(target)*(target-input))
# = KL(target_distribution || input_distribution)
# We want KL(learned || prior), so: input=log_prior, target=log_learned ✓
```

**Priority: LOW**

---

### Q7 — `ablation_mode='neural_only'` still runs SeverityAdapter and LearnableConstraintMatrix forward passes

**Problem:**  
When `ablation_mode='neural_only'`, the loss weights for symbolic components are set to 0, but the model still runs `SeverityAdapter`, `SymbolicConstraintLayer`, and `LearnableConstraintMatrix` forward passes in every step. This means neural-only ablation still uses a severity-conditioned representation, making it not a true neural baseline.

**Suggested Fix:**  
Pass `ablation_mode` to the model's forward method and conditionally skip the SeverityAdapter and SymbolicConstraintLayer operations:
```python
if self.model_config.ablation_mode == 'neural_only':
    return F.log_softmax(logits_neural, dim=-1)  # skip symbolic layer
```

**Priority: MEDIUM**

---

## 5. Documentation & Reproducibility

---

### D1 — No setup/installation guide in README

**Problem:**  
`README.md` (if it exists) does not contain a step-by-step setup guide, hardware requirements, or expected training time. New contributors cannot reproduce results without reading multiple source files.

**Suggested Fix:**  
Add a `## Setup` section to README.md with:
```
1. conda create -n dysarthria python=3.10 && conda activate dysarthria
2. pip install -r requirements.txt
3. python src/data/download.py        # downloads TORGO dataset
4. python src/data/manifest.py        # builds manifest CSV
5. python run_pipeline.py --run-name baseline_v3  # full pipeline
```
Include: GPU requirement (≥8 GB VRAM recommended), expected training time (~2 h on RTX 4060 @ 30 epochs), dataset size (16 531 samples).

**Priority: HIGH**

---

### D2 — No reproducibility seed documentation

**Problem:**  
The code uses `pl.seed_everything(42)` and `torch.use_deterministic_algorithms(True, warn_only=True)`. However, HuBERT's attention with `attn_implementation="eager"` is not fully deterministic on all CUDA versions, and `warn_only=True` means non-deterministic ops proceed silently. Results may not be exactly reproducible across machines.

**Suggested Fix:**  
Document in README which operations are non-deterministic and by how much (typical run-to-run variance in PER). Add a reproducibility checklist that notes CUDA version, PyTorch version, and HuBERT checkpoint hash used in the baseline runs.

**Priority: MEDIUM**

---

### D3 — `requirements.txt` missing version pins for critical packages

**Problem:**  
`torch>=2.0.0` accepts PyTorch 2.6 which has breaking API changes (e.g., `torchaudio.functional.resample` signature). Unpinned versions make future installs fragile.

**Suggested Fix:**  
Pin the exact versions used for baseline_v2 in `requirements.txt` and provide a `requirements-lock.txt` or `environment.yml` with exact package versions. At minimum, pin: `torch==2.x.x`, `transformers==4.x.x`, `pytorch-lightning==2.x.x`.

**Priority: MEDIUM**

---

### D4 — Experiment configuration not saved alongside checkpoints

**Problem:**  
Training runs are identified by `run_name`, but the config YAML is not automatically saved to `checkpoints/{run_name}/` or `results/{run_name}/`. Re-running evaluation on a checkpoint requires knowing the exact config that was used.

**Suggested Fix:**  
At the start of `train()`, call `config.save(checkpoint_dir / 'config.yaml')`. In `run_pipeline.py`, also save `config.save(results_dir / 'config.yaml')` after evaluation. This ensures every results directory is fully self-contained.

**Priority: MEDIUM**

---

### D5 — LOSO workflow not demonstrated end-to-end in documentation

**Problem:**  
LOSO cross-validation is implemented but only mentioned briefly in `run_pipeline.py`'s docstring. There is no documentation of what outputs to expect, how long it takes (estimated ~30 hours for 15 folds), or how to interpret fold-level PERs.

**Suggested Fix:**  
Add a `docs/loso_guide.md` with: command to run LOSO, expected output structure (`results/loso_v1_loso_F01/`, etc.), time estimate, and how to aggregate fold results. Include a code snippet showing how to load and plot per-fold PERs.

**Priority: LOW**

---

### D6 — MLflow tracking URI uses relative `file:` URI which breaks across machines

**Problem:**  
`tracking_uri = f"file:{ProjectPaths().mlruns_dir}"` produces a URI like `file:/home/user/Projects/.../mlruns`. On a different machine or with a different user, this path is invalid. MLflow run IDs won't be found.

**Suggested Fix:**  
Use an absolute file URI: `f"file://{ProjectPaths().mlruns_dir.resolve()}"`. Or better, support a configurable `MLFLOW_TRACKING_URI` environment variable that overrides the default, enabling remote tracking (e.g., MLflow server) without code changes.

**Priority: LOW**

---

## Summary Table

| ID | Category | Title | Priority |
|----|----------|-------|----------|
| C1 | Critical | Raw logits to NLLLoss (negative CE loss) | **HIGH** |
| C2 | Critical | Double audio normalization | **HIGH** |
| C3 | Critical | Data leakage: sample-level fallback split | **HIGH** |
| C4 | Critical | Explanations.json all-empty (silent failure) | **HIGH** |
| C5 | Critical | Rule tracker min_confidence blocks all activations | **HIGH** |
| C6 | Critical | `art_ce_losses` plain dict, not ModuleDict | **HIGH** |
| C7 | Critical | Binary severity used in ordinal contrastive loss | **HIGH** |
| C8 | Critical | `statsmodels` missing from requirements.txt | **HIGH** |
| I1 | Important | WER never computed at corpus level | **HIGH** |
| I2 | Important | Insertion bias 4.6× not resolved | **HIGH** |
| I3 | Important | Full TORGO (15 speakers) never evaluated | **HIGH** |
| I4 | Important | UncertaintyAwareDecoder not wired in pipeline | **MEDIUM** |
| I5 | Important | Articulatory heads accuracy 30–34% (label misalignment) | **MEDIUM** |
| I6 | Important | SymbolicKLLoss lazy init fragile w.r.t. device | **MEDIUM** |
| I7 | Important | `Config()` global instantiated at import time | **MEDIUM** |
| E1 | Evaluation | Spearman correlation degenerate with n=3 speakers | **MEDIUM** |
| E2 | Evaluation | WER missing from results (see I1) | **HIGH** |
| E3 | Evaluation | No per-phoneme PER breakdown | **MEDIUM** |
| E4 | Evaluation | No learning curve plot saved automatically | **MEDIUM** |
| E5 | Evaluation | Articulatory confusion plot depends on broken explainability | **MEDIUM** |
| E6 | Evaluation | Rule impact plot never generated | **LOW** |
| E7 | Evaluation | Bootstrap CI unreliable with n=3 speaker means | **LOW** |
| Q1 | Code Quality | `_align_labels_to_logits` duplicated | **MEDIUM** |
| Q2 | Code Quality | Two conflicting `create_dataloaders` implementations | **MEDIUM** |
| Q3 | Code Quality | No unit tests | **MEDIUM** |
| Q4 | Code Quality | No gradient NaN/Inf detection | **MEDIUM** |
| Q5 | Code Quality | Checkpoint filename contains `/` (invalid) | **MEDIUM** |
| Q6 | Code Quality | SymbolicKLLoss direction comment misleading | **LOW** |
| Q7 | Code Quality | `ablation_mode='neural_only'` not fully neural | **MEDIUM** |
| D1 | Docs | No setup/installation guide | **HIGH** |
| D2 | Docs | Reproducibility seed documentation missing | **MEDIUM** |
| D3 | Docs | `requirements.txt` not pinned | **MEDIUM** |
| D4 | Docs | Config not saved alongside checkpoints | **MEDIUM** |
| D5 | Docs | LOSO workflow undocumented | **LOW** |
| D6 | Docs | MLflow relative URI breaks across machines | **LOW** |

---

## Recommended Fix Order

1. **C1** → Fix negative CE loss first (corrupts all subsequent analysis)
2. **C2** → Fix double normalization (improves audio feature quality)
3. **C3** → Fix data leakage (makes metrics valid)
4. **C5, C7** → Fix rule tracking and severity handling (enables symbolic analysis)
5. **I1, E2** → Add WER (required for publication-quality results)
6. **I3** → Run full TORGO LOSO (generates meaningful statistics)
7. **C4** → Fix explainability JSON (enables interpretability analysis)
8. **C6, I6** → Fix loss module registration and device management
9. **I2** → Address insertion bias with corrected loss
10. **Q1–Q5, D1–D4** → Code quality and documentation cleanup
