# DysarthriaNSR — Technical Audit TODO

**Audit date:** March 5, 2026 (re-audited; original audit March 2026)
**Auditor basis:** Full codebase re-read: train.py, evaluate.py, dataloader.py, config.py, losses.py, output_format.py, rule_tracker.py, run_pipeline.py
**Baseline (current):** `baseline_v3` — val/PER 0.574 (epoch 26), 30 epochs, RTX 4060, correct speaker-stratified splits; **test eval not yet run**
**Previous baseline (invalid, preserved):** `baseline_v2` — PER 0.215, data leakage (B12: speaker always `'unknown'`)

---

## Quick Resolution Status

| ID | Title | Status |
|----|-------|--------|
| C1 | Raw logits to NLLLoss (negative CE loss) | ✅ FIXED |
| C2 | Double audio normalization | ✅ FIXED |
| C3 | Data leakage: sample-level fallback split | ✅ FIXED |
| C4 | Explanations JSON all-empty | ✅ CODE FIXED (unverified — test eval pending) |
| C5 | Rule tracker min_confidence blocks all activations | ❌ OPEN |
| C6 | `art_ce_losses` plain dict, not ModuleDict | ✅ FIXED |
| C7 | Binary severity in ordinal contrastive loss | ✅ FIXED |
| C8 | `statsmodels` missing from requirements.txt | ✅ FIXED |
| I1 | WER never computed at corpus level | ✅ FIXED |
| I2 | Insertion bias 4.6× not resolved | ⚠️ SUBSTANTIALLY ADDRESSED |
| I3 | Full TORGO (15 speakers) never evaluated | ❌ OPEN |
| I4 | UncertaintyAwareDecoder not wired in pipeline | ✅ FIXED |
| I5 | Articulatory heads label misalignment | ✅ FIXED |
| I6 | SymbolicKLLoss lazy init fragile w.r.t. device | ⚠️ PARTIALLY FIXED |
| I7 | `Config()` global instantiated at import time | ✅ FIXED |
| E1 | Spearman correlation degenerate with n=3 speakers | ✅ FIXED (guarded) |
| E2 | Missing WER in results (see I1) | ✅ FIXED |
| E3 | No per-phoneme PER breakdown | ✅ FIXED |
| E4 | No learning curve plot saved automatically | ✅ FIXED |
| E5 | Articulatory confusion depends on broken explainability | ✅ FIXED |
| E6 | `plot_rule_impact` skips when top_rules empty | ⚠️ UNBLOCKED once C5 fixed |
| E7 | Bootstrap CI unreliable with n=3 speaker means | ❌ OPEN (needs LOSO) |
| Q1 | `_align_labels_to_logits` duplicated in two files | ❌ OPEN |
| Q2 | Two conflicting `create_dataloaders` implementations | ❌ OPEN |
| Q3 | No unit tests | ⚠️ PARTIALLY ADDRESSED (smoke_test.py) |
| Q4 | Gradient NaN/Inf detection not enabled | ❌ OPEN |
| Q5 | Checkpoint filename contains `/` (invalid on Linux) | ✅ FIXED |
| Q6 | SymbolicKLLoss direction comment misleading | ✅ FIXED |
| Q7 | `ablation_mode='neural_only'` not fully neural | ❌ OPEN |
| D1 | No setup/installation guide in README | ✅ FIXED |
| D2 | Reproducibility seed documentation missing | ❌ OPEN |
| D3 | `requirements.txt` not version-pinned | ❌ OPEN |
| D4 | Config not saved alongside checkpoints | ❌ OPEN |
| D5 | LOSO workflow undocumented end-to-end | ❌ OPEN |
| D6 | MLflow relative URI breaks across machines | ❌ OPEN |

---

## 1. Open Critical Issues

### C5 — SymbolicRuleTracker `min_confidence` blocks all activations (root cause of zero activations)

**Status: OPEN**

**Problem:**
`SymbolicRuleTracker.__init__` sets `self.min_confidence = 0.5` (sourced from `symbolic_config.min_rule_confidence`).
`log_rule_activation` only appends to `self._activations` when `confidence >= self.min_confidence`.
The `confidence` argument passed by `SymbolicConstraintLayer._track_activations` is the current β blend weight,
which is initialized to `constraint_weight_init=0.05` and in practice converges to ~0.1–0.3.
Since 0.1–0.3 < 0.5 threshold, **every activation is silently discarded**, and
`generate_explanation()` always returns `total_activations: 0`.

This is confirmed by `evaluation_results.json`: `"rule_impact": {"total_activations": 0, "activation_rate": 0.0}`.

**Fix:**
Option A (recommended): Lower `min_rule_confidence` to 0.05 in `SymbolicConfig`:
```python
min_rule_confidence: float = 0.05
```
Option B: Remove the threshold from `log_rule_activation` entirely; apply it only in
`generate_explanation()` when reporting `high_confidence_corrections`.

**Priority: HIGH** (blocks symbolic interpretability entirely)

---

## 2. Open Important Improvements

### I2 — Insertion bias not fully resolved (SUBSTANTIALLY ADDRESSED)

**Status: SUBSTANTIALLY ADDRESSED — needs post-baseline_v3-eval confirmation**

**What was done:**
- `on_train_epoch_start` now ramps `lambda_blank_kl` in three stages: 0.10 (ep <5) → 0.20 (ep <15) → 0.35 (final)
- `blank_target_prob` raised to 0.82
- `blank_priority_weight` raised to 2.5

**Remaining gap:**
- baseline_v3 test eval has not been run; `blank_prob_mean` for baseline_v3 is unknown
- baseline_v2 had `blank_prob_mean ≈ 0.857` (target 0.30); the gap may have narrowed but is unconfirmed

**Next steps:**
1. Run `python run_pipeline.py --run-name baseline_v3 --skip-train --beam-search --beam-width 25 --explain --uncertainty`
2. Check `blank_prob_mean` in `evaluation_results.json`
3. If still >0.70, increase `lambda_blank_kl` to 0.50 and re-run 5 epochs

**Priority: HIGH** (blocked by I3 — run test eval first)

---

### I3 — Full TORGO evaluation not yet run

**Status: OPEN — IMMEDIATE NEXT ACTION**

**Problem:**
- baseline_v3 was trained (val/PER 0.574) but test evaluation has never been run
- All metrics from the training run are validation-only; no test-split PER exists
- LOSO cross-validation (15 folds) never run; macro-PER CI is based on 3 speakers in baseline_v2

**Commands:**
```bash
# Step 1: Run baseline_v3 test evaluation
python run_pipeline.py --run-name baseline_v3 --skip-train \
    --beam-search --beam-width 25 --explain --uncertainty

# Step 2: Regenerate figures
python scripts/generate_figures.py --run-name baseline_v3

# Step 3: Run LOSO (long: ~30h on RTX 4060 for 15 folds)
python run_pipeline.py --run-name baseline_v3_loso --loso
```

**Priority: HIGH** (all other evaluation gaps depend on this)

---

### I6 — SymbolicKLLoss lazy init partially fixed

**Status: PARTIALLY FIXED**

**What remains:**
`SymbolicKLLoss` is now initialized in `_init_loss_functions` conditionally (not at forward-time),
but it is stored as `self.symbolic_kl_loss: Optional[SymbolicKLLoss] = None` and assigned via
`self.symbolic_kl_loss = SymbolicKLLoss(static_C).to(self.device)`.
It is still not registered via `nn.Module` properly (not in a `nn.ModuleList`/assigned as an attribute
that PL recognizes at construction time), so device moves during checkpoint loading could still
leave it on the wrong device.

**Fix:**
Assign it directly as a module attribute before `super().__init__()` call is complete, or use
`self.register_module("symbolic_kl_loss", SymbolicKLLoss(static_C))`.

**Priority: LOW** (not causing failures currently)

---

## 3. Open Evaluation Improvements

### E6 — Rule impact plot gates on C5

**Status: UNBLOCKED ONCE C5 FIXED**

`plot_rule_impact` is now called unconditionally with the live model as an argument (improvement over
the original version). However it still returns early when `stats.get('top_rules')` is empty, which
is always the case while C5 is open. No action needed beyond fixing C5.

**Priority: LOW**

---

### E7 — Bootstrap CI unreliable with n=3 speaker means

**Status: OPEN (blocked by I3)**

Bootstrap CI computed over 3 macro-speaker PER values is essentially a range estimate, not a
statistical confidence interval. The CI will be meaningful only after LOSO produces ≥10 fold PERs.

No code change needed; this is resolved automatically by running I3.

**Priority: LOW**

---

## 4. Open Code Quality Issues

### Q1 — `_align_labels_to_logits` duplicated in train.py and evaluate.py

**Status: OPEN**

Identical function definition exists in both files. Any fix must be manually applied twice or
they will silently diverge.

**Fix:**
Move to `src/utils/sequence_utils.py` and import from both:
```python
from src.utils.sequence_utils import align_labels_to_logits
```

**Priority: MEDIUM**

---

### Q2 — Two conflicting `create_dataloaders` implementations

**Status: OPEN**

- `train.py` line ~957: Returns `(train_loader, val_loader, test_loader)` with full speaker-stratified splits and weighted sampling — **canonical version**
- `src/data/dataloader.py` line 515: Creates a single `DataLoader` with a sampler, returns it — **simplified/legacy version**

Both are named `create_dataloaders`. Risk of importing the wrong one.

**Fix:**
Rename `dataloader.py::create_dataloaders` to `create_single_dataloader` or delete it
(it is not imported by any current file based on search). Keep `train.py::create_dataloaders`
as the canonical implementation and eventually move it to `dataloader.py`.

**Priority: MEDIUM**

---

### Q3 — No pytest-based unit tests

**Status: PARTIALLY ADDRESSED**

`scripts/smoke_test.py` provides 7 end-to-end smoke tests (all currently passing).
However, there are no fast isolated unit tests for individual components (losses, metrics, config I/O).
Regressions in individual functions are not caught until a full smoke test run.

**Missing test coverage:**
- `test_losses.py`: verify OrdinalContrastiveLoss ≥0, BlankPriorKLLoss ≥0, SymbolicKLLoss ≥0
- `test_metrics.py`: verify `compute_per` on known phoneme sequences; `compute_wer_texts`
- `test_config.py`: verify `Config.save` + `Config.load` round-trip preserves all fields
- `test_dataloader.py`: verify `NeuroSymbolicCollator` produces correct shapes; labels padded with -100

**Fix:**
```bash
mkdir tests/
# Create tests/test_losses.py, tests/test_metrics.py, etc.
pip install pytest
pytest tests/ -v
```

**Priority: MEDIUM**

---

### Q4 — Gradient NaN/Inf detection not enabled

**Status: OPEN**

`gradient_clip_val=1.0` limits gradient magnitude but does not detect NaN/Inf.
Silent NaN propagation corrupts weights without any warning.
No `torch.autograd.set_detect_anomaly(True)` or gradient norm logging is active.

**Fix:**
Add optional gradient norm logging in `training_step`:
```python
if self.global_step % 50 == 0:
    total_norm = sum(
        p.grad.data.norm(2).item() ** 2
        for p in self.parameters() if p.grad is not None
    ) ** 0.5
    self.log('train/grad_norm', total_norm, on_step=True)
```
Also add a CLI `--detect-anomaly` flag that calls `torch.autograd.set_detect_anomaly(True)`.

**Priority: MEDIUM**

---

### Q7 — `ablation_mode='neural_only'` not fully neural

**Status: OPEN**

When `ablation_mode='neural_only'`:
- Loss weights for symbolic components are set to 0 in `compute_loss` ✅
- But `model.forward()` still runs `SeverityAdapter`, `SymbolicConstraintLayer`, and `LearnableConstraintMatrix`

This means the neural-only ablation still uses severity-conditioned hidden states, making it not
a true neural baseline. Gradient flow still passes through the symbolic layer even though its
loss contribution is zeroed.

**Fix:**
Pass `ablation_mode` to `NeuroSymbolicASR.forward()` and short-circuit:
```python
if self.ablation_mode == 'neural_only':
    log_probs = F.log_softmax(logits_neural, dim=-1)
    return {'logits_constrained': log_probs, 'logits_neural': logits_neural, ...}
```

**Priority: MEDIUM**

---

## 5. Open Documentation & Reproducibility

### D2 — Reproducibility seed documentation missing

**Status: OPEN**

The code uses `pl.seed_everything(42)` and `torch.use_deterministic_algorithms(True, warn_only=True)`.
HuBERT attention (`attn_implementation="eager"`) is not fully deterministic on all CUDA versions
and `warn_only=True` means non-deterministic ops proceed silently. Run-to-run PER variance is unknown.

**Fix:**
Add a reproducibility section to README.md noting: CUDA version, PyTorch version, HuBERT
checkpoint hash (`facebook/hubert-base-ls960` SHA), and measured run-to-run PER std dev.

**Priority: MEDIUM**

---

### D3 — `requirements.txt` version pins underspecified

**Status: OPEN**

`torch>=2.0.0` accepts PyTorch 2.6 which has breaking changes. Other key packages (`transformers`,
`pytorch-lightning`) are similarly loose-pinned.

**Fix:**
Pin the exact versions used for baseline_v3:
```
torch==2.x.x+cu121
transformers==4.x.x
pytorch-lightning==2.x.x
torchaudio==2.x.x
```
Provide an `environment.yml` or `requirements-lock.txt` with full dependency freeze.

**Priority: MEDIUM**

---

### D4 — Config not saved alongside checkpoints

**Status: OPEN**

`Config.save(path)` and `Config.load(path)` classmethods exist in `config.py` but are never called
by `run_pipeline.py` or `train.py`. Without a saved config, reproducing a specific run requires
knowing the exact CLI flags and default values, which may change between code versions.

**Fix:**
In `train()`, after `checkpoint_dir.mkdir(...)`:
```python
config.save(checkpoint_dir / 'config.yaml')
```
In `run_auto()`, after `results_dir` is established:
```python
results_dir.mkdir(parents=True, exist_ok=True)
config.save(results_dir / 'config.yaml')
```

**Priority: MEDIUM**

---

### D5 — LOSO workflow undocumented end-to-end

**Status: OPEN**

LOSO cross-validation is implemented in `run_loso()` but: expected outputs, run time (~30 h on
RTX 4060 for 15 folds), and how to interpret fold-level results are not documented anywhere.
Output paths (`results/{run}_loso_{speaker}/`, summary JSON) are only discoverable from source.

**Fix:**
Add `docs/loso_guide.md`:
- Command to run
- Expected output structure
- Time estimate
- How to aggregate fold PERs and interpret LOSO summary JSON

**Priority: LOW**

---

### D6 — MLflow relative URI breaks across machines

**Status: OPEN**

`tracking_uri = f"file:{ProjectPaths().mlruns_dir}"` produces a path-relative URI.
On a different machine or user account this URI is invalid, and runs cannot be found.

**Fix:**
```python
tracking_uri: str = f"file://{ProjectPaths().mlruns_dir.resolve()}"
```
Or support `MLFLOW_TRACKING_URI` env var override.

**Priority: LOW**

---

## Summary Table (Updated)

| ID | Category | Title | Priority | Status |
|----|----------|-------|----------|--------|
| C1 | Critical | Raw logits to NLLLoss (negative CE loss) | HIGH | ✅ FIXED |
| C2 | Critical | Double audio normalization | HIGH | ✅ FIXED |
| C3 | Critical | Data leakage: sample-level fallback split | HIGH | ✅ FIXED |
| C4 | Critical | Explanations JSON all-empty | HIGH | ✅ CODE FIXED (eval pending) |
| C5 | Critical | Rule tracker min_confidence blocks activations | HIGH | ❌ OPEN |
| C6 | Critical | `art_ce_losses` plain dict | HIGH | ✅ FIXED |
| C7 | Critical | Binary severity in ordinal contrastive loss | HIGH | ✅ FIXED |
| C8 | Critical | `statsmodels` missing from requirements.txt | HIGH | ✅ FIXED |
| I1 | Important | WER never computed at corpus level | HIGH | ✅ FIXED |
| I2 | Important | Insertion bias 4.6× not resolved | HIGH | ⚠️ SUBSTANTIALLY ADDRESSED |
| I3 | Important | Full TORGO (15 speakers) never evaluated | HIGH | ❌ OPEN — IMMEDIATE |
| I4 | Important | UncertaintyAwareDecoder not wired | MEDIUM | ✅ FIXED |
| I5 | Important | Articulatory heads label misalignment | MEDIUM | ✅ FIXED |
| I6 | Important | SymbolicKLLoss lazy init fragile | MEDIUM | ⚠️ PARTIALLY FIXED |
| I7 | Important | `Config()` global instantiated at import | MEDIUM | ✅ FIXED |
| E1 | Evaluation | Spearman correlation degenerate (n=3) | MEDIUM | ✅ FIXED (guarded) |
| E2 | Evaluation | WER missing from results | HIGH | ✅ FIXED |
| E3 | Evaluation | No per-phoneme PER breakdown | MEDIUM | ✅ FIXED |
| E4 | Evaluation | No learning curve plot | MEDIUM | ✅ FIXED |
| E5 | Evaluation | Articulatory confusion depends on broken explainability | MEDIUM | ✅ FIXED |
| E6 | Evaluation | Rule impact plot never generated | LOW | ⚠️ UNBLOCKED AFTER C5 |
| E7 | Evaluation | Bootstrap CI unreliable (n=3) | LOW | ❌ OPEN (needs LOSO) |
| Q1 | Quality | `_align_labels_to_logits` duplicated | MEDIUM | ❌ OPEN |
| Q2 | Quality | Two conflicting `create_dataloaders` | MEDIUM | ❌ OPEN |
| Q3 | Quality | No unit tests | MEDIUM | ⚠️ PARTIAL (smoke_test.py) |
| Q4 | Quality | No gradient NaN/Inf detection | MEDIUM | ❌ OPEN |
| Q5 | Quality | Checkpoint filename contains `/` | MEDIUM | ✅ FIXED |
| Q6 | Quality | SymbolicKLLoss comment misleading | LOW | ✅ FIXED |
| Q7 | Quality | `ablation_mode='neural_only'` not fully neural | MEDIUM | ❌ OPEN |
| D1 | Docs | No setup/installation guide | HIGH | ✅ FIXED |
| D2 | Docs | Reproducibility documentation missing | MEDIUM | ❌ OPEN |
| D3 | Docs | `requirements.txt` not version-pinned | MEDIUM | ❌ OPEN |
| D4 | Docs | Config not saved alongside checkpoints | MEDIUM | ❌ OPEN |
| D5 | Docs | LOSO workflow undocumented | LOW | ❌ OPEN |
| D6 | Docs | MLflow relative URI breaks across machines | LOW | ❌ OPEN |
| N1 | New | baseline_v3 trained 30 epochs vs config 40 | LOW | ⚠️ INVESTIGATE |
| N2 | New | SpecAugment / temporal downsampling not ablated | MEDIUM | ❌ OPEN |

---

## Recommended Fix Order (Remaining Open Issues)

1. **I3** — Run baseline_v3 test evaluation immediately (unlocks all downstream metrics)
2. **C5** — Fix SymbolicRuleTracker min_confidence (unblocks E6, enables symbolic interpretability)
3. **I2 verification** — Check blank_prob_mean in baseline_v3 results; increase lambda_blank_kl if still >0.70
4. **N2** — Run no-augment ablation to isolate B12 vs. SpecAugment PER contribution
5. **D4** — Add config.save() calls in train() and run_auto() (ensures reproducibility)
6. **Q1, Q2** — Consolidate duplicate functions into shared modules
7. **Q7** — Wire ablation_mode to model.forward() for true neural-only ablation
8. **Q3** — Add pytest unit tests for losses, metrics, config I/O
9. **Q4** — Add gradient norm logging + optional anomaly detection flag
10. **D2, D3** — Add reproducibility notes + pin requirements
11. **I3 LOSO** — Run full LOSO-CV to get valid macro-PER CI
12. **D5, D6** — LOSO guide, MLflow URI fix
