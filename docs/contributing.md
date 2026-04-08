# docs/contributing.md — Contributing Guide

> Cross-references: [architecture.md](architecture.md) for adding model components, [evaluation.md](evaluation.md) for adding metrics.

---

## Non-Negotiable Conventions

The following rules are enforced in code review. Each maps to a concrete bug or design decision that demonstrates why the rule exists.

**1. `normalize_phoneme()` on every ARPABET string.**

```python
from src.utils.config import normalize_phoneme
normalize_phoneme("AH0")  # → "AH"
normalize_phoneme("IY1")  # → "IY"
```

TORGO manifest uses ARPABET with stress markers (0/1/2). The model vocabulary is stress-agnostic. Any comparison against raw ARPABET strings will fail or produce UNK matches. Call `normalize_phoneme()` at manifest build time, dataset initialization, decode time, and in any loss computation involving phoneme strings. Never compare raw ARPABET strings directly.

**2. Vocabulary ID immutability (BLANK=0, PAD=1, UNK=2).**

```python
<BLANK> = 0    # CTC blank — alignment separator, never a label target
<PAD>   = 1    # Padding token for variable-length batching
<UNK>   = 2    # Unknown / OOV phonemes
# IDs 3–46: ARPABET phonemes
```

These IDs are built first in `TorgoNeuroSymbolicDataset._build_vocabularies()` and assumed throughout training, evaluation, and decoding. Changing them silently breaks: CTC loss (blank ID is hardcoded as `phn_to_id['<BLANK>']` in `DysarthriaASRLightning._init_loss_functions()`), checkpoint compatibility, and all downstream decoding.

**3. `-100` padding sentinel everywhere — never `0` or `1`.**

```python
# Correct
labels = pad_sequence(labels, batch_first=True, padding_value=-100)
label_lengths = (labels != -100).sum(dim=1)

# Wrong — 0 is BLANK (valid CTC token), 1 is PAD (valid token ID)
labels = pad_sequence(labels, batch_first=True, padding_value=0)
```

`nn.CTCLoss` and `nn.CrossEntropyLoss(ignore_index=-100)` automatically ignore `-100` positions. Using `0` or `1` causes CTC and CE losses to treat padding as real phoneme tokens.

**4. `ProjectPaths` for all file I/O — no hardcoded paths.**

```python
from src.utils.config import get_project_root, ProjectPaths

results_dir = get_project_root() / "results" / config.experiment.run_name
ckpt_dir = get_project_root() / "checkpoints" / config.experiment.run_name
```

**5. Type hints + docstrings on all public functions.** Args/Returns/Raises format:

```python
def compute_per(prediction: List[str], reference: List[str]) -> float:
    """Compute Phoneme Error Rate (PER) using edit distance.
    Args:
        prediction: Predicted phoneme sequence.
        reference:  Reference phoneme sequence.
    Returns:
        PER in [0, ∞). Can exceed 1.0 when insertions > reference length.
    """
```

**6. MLflow logging for every new metric.** Use `self.log('train/my_metric', value, ...)` for Lightning metrics; `mlflow.log_metric()` for evaluation-only metrics.

**7. Tests in `tests/` for correctness invariants; structural checks in `scripts/smoke_test.py`.** New loss functions need a `tests/test_losses.py` test covering non-negativity, scalar output, and NaN-free behavior on edge cases.

---

## Adding a New Model Component

1. **Implement** in `src/models/model.py` following the `SeverityAdapter` / `TemporalDownsampler` pattern. Use `nn.Module` with complete `__init__()` and `forward()` signatures. Log construction to `logger.info()`.

2. **Add config key** to the appropriate dataclass in `src/utils/config.py` (typically `ModelConfig`). Include a default value, a `use_*` boolean flag, and a descriptive comment.

3. **Wire into `NeuroSymbolicASR.__init__()`** using the `use_*` flag pattern: `if model_config.use_my_component: self.my_component = MyComponent(...)`.

4. **Wire into `NeuroSymbolicASR.forward()`** with an `ablation_mode` guard:
```python
if self.my_component is not None and ablation_mode != "no_my_component":
    hidden_states = self.my_component(hidden_states)
```

5. **Add a loss class** (if the component introduces a new loss term) to `src/models/losses.py` as an `nn.Module` (not a plain function — I6 fix requirement so Lightning device-manages the module). Instantiate it in `DysarthriaASRLightning._init_loss_functions()`. Add a `lambda_*` config field to `TrainingConfig`.

6. **Wire into `compute_loss()`** and add MLflow logging in `training_step()`.

7. **Update the ablation mode table** in [architecture.md](architecture.md) with a new row.

8. **Add a smoke test check** in `scripts/smoke_test.py` if the component has critical invariants (gradient flow, output shape, non-negativity).

9. **Add a refactor note** at the top of modified files: `# [CLEAN]`, `# [PERF]`, `# [REPRO]` per the existing log convention.

---

## Adding a New Metric

1. **Implement** as a standalone function in `evaluate.py` with complete type annotations and a docstring. The function should take raw lists/arrays and return a scalar or dict, with no model or dataloader dependencies (testable in isolation).

2. **Test** in `tests/test_metrics.py` covering: edge cases (empty sequences, perfect match, all-wrong), non-negativity, scalar output, and NaN-free behavior.

3. **Integrate** into `evaluate_model()`: call the function in the appropriate place in the evaluation loop and add results to the `results` dict.

4. **Log to MLflow:** add a `mlflow.log_metric()` call.

5. **Document** in the Metrics Reference section of [evaluation.md](evaluation.md): formula, aggregation method, where computed, output field name in `evaluation_results.json`, acceptable range, and validity conditions.

---

## Adding a New Ablation Mode

1. **Add to the choices list** in `run_pipeline.py _build_parser()`:
```python
choices=["full", "neural_only", ..., "my_new_ablation"]
```

2. **Add a guard** in `NeuroSymbolicASR.forward()`:
```python
if ablation_mode == "my_new_ablation":
    # bypass specific component
```

3. **Add a row** to the ablation mode table in [architecture.md](architecture.md).

4. **Run a smoke test** before committing: `python scripts/smoke_test.py --profile unit`.

---

## Fix Naming Convention

| Prefix | Category | Examples |
|---|---|---|
| B | Critical correctness bugs | B1–B23 historical series |
| C | Critical architectural / training bugs | C1 (lambda_ce), C2 (no peak norm) |
| I | Important bugs (correctness or metric validity) | I2 (staged KL warmup), I5 (utterance-level art heads) |
| E | Evaluation-specific bugs | E5 (always generate art confusion), E6 (fallback heatmap) |
| H | High-impact hotfixes | H-1 (VRAM cleanup), H-2 (alignment cache), H-5 (activation cap) |
| T | Technical debt / training-stability issues | T-04 (OneCycleLR + unfreeze interaction) |
| N | New features or enhancements | N3 (bootstrap delta test), N4 (add-k LM), N7 (rule precision proxy) |
| O | Optional / deferred improvements | O-2 (per-group scheduler) |
| M | Medium-priority issues | M-5 (conformal APS heuristic), M-6 (per-speaker severity sort) |
| Q | Code quality and refactoring | Q7 (true neural-only ablation bypass) |

To add a fix: implement it, add a row to `docs/02_COMPLETED_WORK.md` with `ID | File | Fix Applied`, and add a comment at the affected code site.

### Pre-Commit Checklist

Before committing any code change:

- All public functions have docstrings with Args/Returns/Raises
- Type hints on all function signatures (avoid `Any`)
- No hardcoded paths (use `ProjectPaths` from `config.py`)
- Phonemes normalized via `normalize_phoneme()`
- New metrics logged to MLflow
- Tested on RTX 4060 (8 GB VRAM target)
- Labels use `-100` sentinel for padding (not `0` or `1`)
- If bug fixed: add the fix ID and summary to `docs/02_COMPLETED_WORK.md`

---

## Known Codebase Risks

The following non-blocking issues are known from the March 2026 research audit. They do not affect training or evaluation correctness but require awareness when modifying the codebase.

### Articulatory Constants Source of Truth

`PHONEME_DETAILS` and `PHONEME_FEATURES` are now centralized in `src/utils/constants.py` and imported by both `src/data/manifest.py` and `src/models/model.py`.

**If you modify articulatory labels:** Update `src/utils/constants.py` only. Keep `scripts/smoke_test.py` Test 1 passing, which asserts tuple/dict constant consistency.

### `ModelConfig.num_phonemes` Runtime Assignment

`ModelConfig.num_phonemes` defaults to `None`. At runtime, `NeuroSymbolicASR.__init__()` sets it to `len(phn_to_id)` from the manifest vocabulary (typically 47 on TORGO).

**If you encounter this:** Trust `len(dataset.phn_to_id)` as the authoritative vocabulary size, not `model_config.num_phonemes`.

### `rule_precision()` Not Wired (N7)

`SymbolicRuleTracker.rule_precision()` in `src/explainability/rule_tracker.py` is implemented but not called from `evaluate_model()`. An utterance-level proxy (`constraint_precision.rule_precision`) is reported instead.

**If you want true rule precision:** It requires knowing which reference phoneme appeared at each CTC frame — unavailable without `torchaudio.functional.forced_align`. Do not wire `rule_precision()` directly until forced alignment is integrated.

### LOSO Resume Orchestration Test Gap (§9.3)

The `weights_only_resume` and `scheduler_exhausted` paths in `run_loso()` have no automated integration test. If a fold resumes incorrectly, debug by checking:
1. `lm.resume_epoch_offset` — should equal the saved `start_epoch` from `weights_only_resume.pt`
2. The `weights_only_resume.pt` marker file in the fold's checkpoint directory — delete it to force a clean weights-only resume from `last.ckpt`

**Recommended fix:** Add `tests/test_loso_resume.py` with a synthetic 3-fold LOSO run that interrupts and resumes each code path. This is post-paper work.
