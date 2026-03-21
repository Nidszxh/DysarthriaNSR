# docs/contributing.md — Contributing Guide

> Cross-references: [docs/architecture.md](architecture.md) for adding model components, [docs/evaluation.md](evaluation.md) for adding metrics.

---

## Code Conventions

All conventions below are from `.github/copilot-instructions.md` and are enforced in code review.

### 1. Phoneme Normalization

**Always** call `normalize_phoneme()` before any comparison or vocabulary building involving phoneme strings:
```python
from src.utils.config import normalize_phoneme

normalize_phoneme("AH0")  # → "AH"
normalize_phoneme("IY1")  # → "IY"
```

TORGO manifest uses ARPABET with stress markers (0/1/2). The model vocabulary is stress-agnostic. **Never compare raw ARPABET strings directly.** This applies at manifest build time, dataset initialization, decode time, and in any loss computation.

### 2. Vocabulary ID Immutability

These three token IDs are fixed and must never be reassigned:
```python
<BLANK> = 0    # CTC blank — alignment separator
<PAD>   = 1    # Padding for variable-length batches
<UNK>   = 2    # Unknown / OOV phonemes
# IDs 3–46: actual ARPABET phonemes (stress-stripped)
```

Vocabulary is built once in `TorgoNeuroSymbolicDataset._build_vocabularies()`. Changing the order or re-assigning special token IDs will silently break CTC alignment and CE loss.

### 3. Label Padding Sentinel

Use `-100` for all sequence padding, never `0` or `1`:
```python
# Correct
labels = pad_sequence(labels, batch_first=True, padding_value=-100)
label_lengths = (labels != -100).sum(dim=1)

# Wrong — 0 is BLANK, 1 is PAD (valid token IDs)
labels = pad_sequence(labels, batch_first=True, padding_value=0)
```

`nn.CTCLoss` and `nn.CrossEntropyLoss(ignore_index=-100)` automatically ignore `-100` positions.

### 4. File Paths via ProjectPaths

Use `ProjectPaths` from `src/utils/config.py` for all file I/O. **No hardcoded paths.** Every output path must be derived from `config.experiment.run_name`:
```python
from src.utils.config import get_project_root, ProjectPaths

results_dir = get_project_root() / "results" / config.experiment.run_name
ckpt_dir = get_project_root() / "checkpoints" / config.experiment.run_name
```

### 5. Type Hints and Docstrings

All public functions must have complete type hints and docstrings (Args/Returns/Raises format):
```python
def compute_per(prediction: List[str], reference: List[str]) -> float:
    """
    Compute Phoneme Error Rate (PER) using edit distance.

    Args:
        prediction: Predicted phoneme sequence.
        reference:  Reference phoneme sequence.

    Returns:
        PER in [0, ∞). Can exceed 1.0 when insertions > reference length.
    """
```

### 6. MLflow Logging

Log all new metrics to MLflow in the step or epoch where they are computed:
```python
self.log('val/my_new_metric', value, on_epoch=True, prog_bar=False)
# For step-level metrics:
self.log('train/my_metric', value, on_step=True, on_epoch=False)
```

Hyperparameters are logged automatically via `flatten_config_for_mlflow()` at the start of training.

---

## Adding a New Component

1. **Implement** in `src/models/model.py`, following the pattern of existing components (`SeverityAdapter`, `TemporalDownsampler`). Use `nn.Module` with complete `__init__()` and `forward()` signatures.

2. **Add config key** to the appropriate dataclass in `src/utils/config.py` (typically `ModelConfig`). Include a default value and a descriptive comment.

3. **Add ablation hook** in `NeuroSymbolicASR.forward()` — check `ablation_mode` before applying the component and provide a bypass path that returns the unchanged tensor. Update the ablation mode table in [docs/architecture.md](architecture.md).

4. **Add to loss computation** if the component introduces a new loss term: add the loss class to `src/models/losses.py`, instantiate it in `DysarthriaASRLightning._init_loss_functions()`, weight it with a new `lambda_*` parameter in `TrainingConfig`, and add it to `compute_loss()` in `train.py`.

5. **Add MLflow logging** for new loss terms and diagnostic metrics.

6. **Update smoke test** in `scripts/smoke_test.py` if the component has critical invariants (e.g., gradient flow, output shape, non-negativity).

7. **Update docs** in [docs/architecture.md](architecture.md): add a new subsection to "Components" with the purpose, design decision, class/file reference, and constructor argument table.

---

## Adding a New Metric

Follow the pattern established in `evaluate.py`:

1. **Implement** as a standalone function in `evaluate.py` with complete type annotations and docstring. The function should take raw lists/arrays and return a scalar or dict, with no model or dataloader dependencies.

2. **Test** with synthetic predictions: write a pytest test in `tests/test_metrics.py` covering edge cases (empty sequences, perfect match, all-wrong).

3. **Integrate** into `evaluate_model()`: call the function in the appropriate place in the evaluation loop and add results to the `results` dict.

4. **Log to MLflow:** add a `mlflow.log_metric()` call (or use the Lightning `self.log()` if in a training context).

5. **Add to `evaluation_results.json` schema:** document the new key in the schema section of [docs/evaluation.md](evaluation.md).

6. **Document** in the Metrics Reference section of [docs/evaluation.md](evaluation.md): formula, aggregation method, where it is computed, acceptable range.

---

## Bug Fix Conventions

### Naming Scheme

| Prefix | Category |
|---|---|
| B | Critical correctness bugs (e.g., B1–B23 historical series) |
| C | Critical architectural / training bugs |
| I | Important bugs (correctness or metric validity) |
| E | Evaluation-specific bugs |
| Q | Code quality and refactoring |
| H | Hotfixes (high-priority code quality or performance) |
| T | Technical debt or training-stability issues |
| N | New features or enhancements |

### Pre-Commit Checklist

Before committing any code change:

- All public functions have docstrings with Args/Returns/Raises
- Type hints on all function signatures (avoid `Any`)
- No hardcoded paths (use `ProjectPaths` from `config.py`)
- Phonemes normalized via `normalize_phoneme()`
- New metrics logged to MLflow
- Tested on RTX 4060 (8 GB VRAM target)
- Labels use `-100` sentinel for padding (not `0` or `1`)
- If bug fixed: add the fix ID and summary to `docs/02_COMPLETED_WORK.md` (cross-reference only — do not create that file here)

---

## Known Codebase Risks

The following non-blocking issues are known from the March 2026 research audit. They do not affect correctness of training or evaluation but require awareness when modifying the codebase.

### N5 — `PHONEME_DETAILS` / `PHONEME_FEATURES` Duplication

`PHONEME_DETAILS` in `src/data/manifest.py` and `PHONEME_FEATURES` in `src/models/model.py` define articulatory class mappings independently. They are currently in sync (B23 fix applied to both), but any future correction to one must be mirrored manually in the other.

**If you encounter this:** Add the fix to **both** files simultaneously. The long-term fix is to move both to `src/utils/constants.py` and import from there (N5, planned post-submission).

### `ModelConfig.num_phonemes` Misleading Comment

`ModelConfig.num_phonemes = 47` is the correct runtime value (44 ARPABET + 3 special tokens). The config comment may reference 44. At runtime, `NeuroSymbolicASR.__init__()` overwrites this with `len(phn_to_id)` from the manifest vocabulary, so the config default is only a documentation issue.

**If you encounter this:** Trust the runtime value from `len(dataset.phn_to_id)`. The config default is informational only.

### `rule_precision()` Not Wired

`SymbolicRuleTracker.rule_precision()` in `src/explainability/rule_tracker.py` is implemented but not called from `evaluate_model()`. An utterance-level proxy (`constraint_precision.rule_precision`) is reported instead.

**If you encounter this:** Implementing true rule precision requires knowing which reference phoneme appeared at each CTC frame (requires forced alignment). Do not wire `rule_precision()` directly until `torchaudio.functional.forced_align` is integrated.

### LOSO Resume Orchestration Gaps

The LOSO resume logic in `run_loso()` covers three cases: (1) completed folds from progress JSON, (2) weights-only resume from exhausted scheduler checkpoints, (3) normal checkpoint resume. However, there is no integration test covering multi-fold interruption and resume. The resume path is exercised only through end-to-end LOSO runs.

**If you encounter this:** If a fold resumes incorrectly (wrong epoch count, wrong freeze stage), check `lm.resume_epoch_offset` and the `weights_only_resume.pt` marker file in the fold's checkpoint directory. Delete the marker to force a clean weights-only resume from `last.ckpt`.