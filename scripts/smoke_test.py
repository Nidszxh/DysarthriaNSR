"""DysarthriaNSR smoke tests aligned with current training pipeline.

Purpose
-------
Fast, self-contained checks for critical invariants after recent LOSO/resume/
logging refactors. These tests intentionally avoid long training runs.

Usage
-----
    python scripts/smoke_test.py [--tests 1,2,3,4,5,6,7]

Current checks
--------------
1. Config + severity map sanity (range + YAML round-trip)
2. Learnable symbolic constraint gradient flow
3. BlankPriorKLLoss non-negative and mask-aware
4. OrdinalContrastiveLoss non-negative and batch=1-safe
5. Explainability formatter output contract
6. LOSO ordering/resume safeguards are present in source
7. Compact fold progress callback emits one-line epoch summary
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import os
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Callable, Dict, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS: Dict[int, bool] = {}


def run_test(n: int, name: str, fn: Callable[[], None]) -> bool:
    """Run *fn* and record pass/fail in RESULTS[n]."""
    try:
        fn()
        print(f"  [PASS] Test {n}: {name}")
        RESULTS[n] = True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] Test {n}: {name}")
        traceback.print_exc()
        RESULTS[n] = False
    return RESULTS[n]


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

def test1_config_and_severity_map() -> None:
    """Config load round-trip and TORGO severity range sanity."""
    import yaml  # noqa: PLC0415

    from src.utils.config import Config, TORGO_SEVERITY_MAP  # noqa: PLC0415

    from src.utils.config import TORGO_SEVERITY_MAP  # noqa: PLC0415

    invalid = {k: v for k, v in TORGO_SEVERITY_MAP.items() if not (0.0 <= v <= 5.0)}
    assert not invalid, f"Invalid severity values: {invalid}"

    minimal_yaml = {
        "experiment": {"run_name": "smoke_yaml_test"},
        "training": {"max_epochs": 1, "batch_size": 2},
        "model": {"constraint_weight_init": 0.1},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        yaml.dump(minimal_yaml, fh)
        tmp_path = fh.name

    cfg = Config.load(tmp_path)
    assert cfg.experiment.run_name == "smoke_yaml_test"
    assert cfg.training.max_epochs == 1
    assert cfg.training.batch_size == 2


def test2_learnable_constraint_gradient() -> None:
    """Gradient must flow through LearnableConstraintMatrix (logit_C)."""
    import torch  # noqa: PLC0415

    from src.models.model import SymbolicConstraintLayer  # noqa: PLC0415
    from src.utils.config import Config  # noqa: PLC0415

    cfg = Config()
    phn_to_id = {"<BLANK>": 0, "<PAD>": 1, "<UNK>": 2, "P": 3, "B": 4}
    id_to_phn = {v: k for k, v in phn_to_id.items()}

    layer = SymbolicConstraintLayer(
        num_phonemes=len(phn_to_id),
        phn_to_id=phn_to_id,
        id_to_phn=id_to_phn,
        symbolic_config=cfg.symbolic,
        learnable=True,
    )
    logits = torch.randn(2, 10, len(phn_to_id))
    out = layer(logits)
    loss = out["log_probs"].mean()
    loss.backward()

    assert layer.learnable_matrix.logit_C.grad is not None, (
        "logit_C has no gradient — learnable constraint matrix is not in the compute graph"
    )


def test3_blank_prior_kl_loss() -> None:
    """BlankPriorKLLoss must be non-negative and respond to masking."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    from src.models.losses import BlankPriorKLLoss  # noqa: PLC0415

    loss_fn = BlankPriorKLLoss(blank_id=0, target_prob=0.3)

    logits = torch.zeros(2, 50, 5)
    logits[:, :, 0] = 1.0
    # Make tail frames strongly non-blank so masking must matter.
    logits[:, 35:, 1] = 5.0
    log_probs = F.log_softmax(logits, dim=-1)

    full_mask = torch.ones(2, 50, dtype=torch.long)
    trimmed_mask = full_mask.clone()
    trimmed_mask[:, 35:] = 0

    loss_full = loss_fn(log_probs, attention_mask=full_mask)
    loss_trimmed = loss_fn(log_probs, attention_mask=trimmed_mask)

    assert loss_full.item() >= 0.0
    assert loss_trimmed.item() >= 0.0
    assert abs(loss_full.item() - loss_trimmed.item()) > 1e-8, (
        "BlankPriorKLLoss appears insensitive to attention_mask"
    )


def test4_ordinal_contrastive_loss() -> None:
    """OrdinalContrastiveLoss must return a non-negative scalar (including edge case batch=1)."""
    import torch  # noqa: PLC0415

    from src.models.losses import OrdinalContrastiveLoss  # noqa: PLC0415

    loss_fn = OrdinalContrastiveLoss(margin_per_level=0.3)

    # Normal batch (4 items)
    embeddings = torch.randn(4, 10, 768)
    severity = torch.tensor([0.0, 0.0, 5.0, 5.0])
    loss = loss_fn(embeddings, severity)
    assert loss.item() >= 0.0, f"Loss negative on normal batch: {loss.item()}"

    # Edge case: single item — previously returned NaN (Bug B8)
    embeddings_1 = torch.randn(1, 10, 768)
    severity_1 = torch.tensor([5.0])
    loss_1 = loss_fn(embeddings_1, severity_1)
    assert not (loss_1 != loss_1).item(), (  # NaN check: NaN != NaN is True
        f"OrdinalContrastiveLoss returned NaN for batch_size=1 (Bug B8 not fixed)"
    )
    assert loss_1.item() >= 0.0, f"Loss negative on single-item batch: {loss_1.item()}"


def test5_explainability_formatter_structure() -> None:
    """ExplainableOutputFormatter.format_utterance must return required keys."""
    from src.explainability.output_format import ExplainableOutputFormatter  # noqa: PLC0415

    fmt = ExplainableOutputFormatter()
    result = fmt.format_utterance(
        utterance_id="test_001",
        ground_truth="hello world",
        prediction="hello",
        errors=[],
        symbolic_rules_summary={},
    )

    required_keys = {"utterance_id", "phoneme_analysis"}
    missing = required_keys - set(result.keys())
    assert not missing, f"Formatter output missing keys: {missing}"


def test6_loso_source_guards_present() -> None:
    """LOSO source must include deterministic ordering and completed-fold skip logic."""
    import train  # noqa: PLC0415

    src = inspect.getsource(train.run_loso)

    assert "_split_speaker_id" in src, "Deterministic speaker parser missing in run_loso"
    assert "completed_folds" in src and "Skipping completed fold" in src, (
        "Completed-fold skip path missing in run_loso"
    )
    assert "Re-opening completed fold" not in src, (
        "Legacy re-open completed fold path still present in run_loso"
    )


def test7_compact_progress_callback_output() -> None:
    """Compact callback should print one-line epoch metrics without crashing."""
    import torch  # noqa: PLC0415

    from train import _CompactFoldProgressCallback  # noqa: PLC0415

    class DummyTrainer:
        sanity_checking = False
        max_epochs = 25
        callback_metrics = {
            'train/loss_epoch': torch.tensor(1.23),
            'val/loss': torch.tensor(2.34),
            'val/per': torch.tensor(0.56),
            'train/blank_prob_mean_epoch': torch.tensor(0.78),
        }

    class DummyConfig:
        class training:
            max_epochs = 25

    class DummyModule:
        current_epoch = 3
        resume_epoch_offset = 0
        config = DummyConfig()

    cb = _CompactFoldProgressCallback()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        cb.on_validation_epoch_end(DummyTrainer(), DummyModule())

    out = buffer.getvalue()
    assert "📈 Epoch" in out and "val/per" in out and "blank=" in out, (
        f"Unexpected compact callback output: {out!r}"
    )


def test8_pipeline_cli_smoke() -> None:
    """Run a tiny end-to-end CLI smoke stage (train-only)."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_name = f"smoke_cli_{int(time.time())}"
    cmd = [
        sys.executable,
        "run_pipeline.py",
        "--run-name", run_name,
        "--smoke",
        "--skip-eval",
        "--max-epochs", "1",
        "--batch-size", "2",
        "--early-stopping-patience", "1",
        "--no-gradient-checkpointing",
    ]
    proc = subprocess.run(
        cmd,
        cwd=project_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-60:])
        raise AssertionError(
            f"Pipeline smoke failed (exit={proc.returncode}). Output tail:\n{tail}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

UNIT_TESTS: List[tuple] = [
    (1, "Config + severity map sanity",                      test1_config_and_severity_map),
    (2, "Learnable constraint matrix gradient flows",        test2_learnable_constraint_gradient),
    (3, "BlankPriorKLLoss sanity + masking",                 test3_blank_prior_kl_loss),
    (4, "OrdinalContrastiveLoss sanity (incl. batch_size=1)", test4_ordinal_contrastive_loss),
    (5, "ExplainableOutputFormatter output structure",        test5_explainability_formatter_structure),
    (6, "LOSO ordering/resume source guards",                test6_loso_source_guards_present),
    (7, "Compact fold progress callback output",             test7_compact_progress_callback_output),
]

PIPELINE_TESTS: List[tuple] = [
    (8, "CLI pipeline smoke (train-only, --smoke)",          test8_pipeline_cli_smoke),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="DysarthriaNSR smoke tests")
    parser.add_argument(
        "--profile",
        choices=["unit", "pipeline", "all"],
        default="unit",
        help="Smoke profile: fast unit checks, CLI pipeline check, or both.",
    )
    parser.add_argument(
        "--tests",
        default="",
        help="Comma-separated test numbers to run (default: all)",
    )
    args = parser.parse_args()

    if args.profile == "unit":
        available_tests = UNIT_TESTS
    elif args.profile == "pipeline":
        available_tests = PIPELINE_TESTS
    else:
        available_tests = UNIT_TESTS + PIPELINE_TESTS

    if args.tests:
        requested = {int(t.strip()) for t in args.tests.split(",") if t.strip()}
        tests_to_run = [(n, name, fn) for n, name, fn in available_tests if n in requested]
    else:
        tests_to_run = available_tests

    print(f"\n{'='*65}")
    print(f"  DysarthriaNSR Smoke Tests  ({len(tests_to_run)} selected)")
    print(f"{'='*65}")

    # Add project root to sys.path so imports work when called from scripts/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    for n, name, fn in tests_to_run:
        run_test(n, name, fn)

    passed = sum(1 for v in RESULTS.values() if v)
    failed = len(RESULTS) - passed
    print(f"\n{'─'*65}")
    print(f"  {passed}/{len(RESULTS)} passed  |  {failed} failed")
    print(f"{'─'*65}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
