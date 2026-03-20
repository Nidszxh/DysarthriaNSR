#!/usr/bin/env python3
"""
generate_figures.py — Publication-quality visualization suite for DysarthriaNSR.

Reads evaluation_results.json (and optionally explanations.json) for one or
more experiment runs and saves all diagnostic figures to
``results/<run_name>/figures/``.

Usage examples::

    # Single run
    python scripts/generate_figures.py --run-name baseline_v2

    # Single run with comparison overlay (ablations)
    python scripts/generate_figures.py --run-name baseline_v2 \\
        --compare neural_only no_art_heads no_constraint_matrix

    # Override results root if running from a different working directory
    python scripts/generate_figures.py --run-name baseline_v2 \\
        --results-root /data/experiments/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so ``src`` package is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).resolve().parent       # scripts/
_PROJECT_ROOT = _SCRIPT_DIR.parent                   # DysarthriaNSR/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.utils.config import TORGO_SEVERITY_MAP                 # noqa: E402
from src.visualization.experiment_plots import generate_all_plots  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, label: str) -> dict | None:
    """Load a JSON file; returns ``None`` and prints a warning on failure."""
    if not path.exists():
        print(f"  ⚠  {label} not found: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"  ✗  {label} is malformed ({exc}): {path}")
        return None


def _load_run(results_root: Path, run_name: str) -> tuple[dict | None, dict | None, dict | None]:
    """
    Load ``evaluation_results.json``, ``explanations.json``, and
    ``per_phoneme_per.json`` for *run_name*.

    Returns:
        (eval_results, explanations, per_phoneme_per) — any may be ``None``.
    """
    run_dir = results_root / run_name
    eval_results  = _load_json(run_dir / "evaluation_results.json",
                               f"evaluation_results ({run_name})")
    explanations  = _load_json(run_dir / "explanations.json",
                               f"explanations ({run_name})")
    per_phoneme   = _load_json(run_dir / "per_phoneme_per.json",
                               f"per_phoneme_per ({run_name})")
    return eval_results, explanations, per_phoneme


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_REQUIRED_EVAL_KEYS  = {"avg_per", "overall", "error_analysis", "per_speaker",
                        "symbolic_impact", "articulatory_accuracy"}
_REQUIRED_UTT_KEYS   = {"utterance_id", "speaker_id", "severity", "per", "uncertainty"}

def _validate_eval_schema(eval_results: dict, run_name: str) -> list[str]:
    """
    Check that *eval_results* contains the fields needed for visualizations.

    Returns:
        List of missing key paths (empty when schema is sufficient).
    """
    missing: list[str] = []

    for key in _REQUIRED_EVAL_KEYS:
        if key not in eval_results:
            missing.append(f"evaluation_results['{key}']")

    if "error_analysis" in eval_results:
        ec = eval_results["error_analysis"].get("error_counts", {})
        for k in ("substitutions", "deletions", "insertions", "correct"):
            if k not in ec:
                missing.append(f"evaluation_results['error_analysis']['error_counts']['{k}']")

    if "per_speaker" in eval_results:
        for spk, spk_data in eval_results["per_speaker"].items():
            for k in ("per", "ci", "status"):
                if k not in spk_data:
                    missing.append(f"per_speaker['{spk}']['{k}']")
            break  # Only check the first speaker as a sample

    return missing


def _validate_utterance_schema(utterances: list[dict]) -> tuple[list[str], list[str]]:
    """
    Check per-utterance records for required keys.

    Returns:
        (missing_keys, warnings) — lists of strings describing issues.
    """
    if not utterances:
        return [], ["explanations.json has no utterances"]

    sample = utterances[0]
    missing = [k for k in _REQUIRED_UTT_KEYS if k not in sample]
    issues  = []

    # Check for all-null uncertainty
    has_uncertainty = any(u.get("uncertainty") is not None for u in utterances)
    if not has_uncertainty:
        issues.append(
            "All utterance 'uncertainty' values are null — "
            "re-run evaluate.py with --uncertainty to populate calibration plots."
        )

    # Check for speaker collapse (all 'unknown')
    speakers = {u.get("speaker_id", "unknown") for u in utterances}
    if speakers == {"unknown"}:
        issues.append(
            "All utterance 'speaker_id' values are 'unknown' — "
            "manifest speaker extraction bug present; regenerate manifest first."
        )

    return missing, issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate_figures",
        description="Generate publication-quality diagnostic figures for a DysarthriaNSR run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-name",
        required=True,
        metavar="RUN",
        help="Name of the experiment run (subdirectory under --results-root).",
    )
    parser.add_argument(
        "--results-root",
        default=str(_PROJECT_ROOT / "results"),
        metavar="DIR",
        help="Root directory that contains per-run result subdirectories. "
             "Default: <project_root>/results",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=[],
        metavar="RUN",
        help="Additional run names to include in the experiment comparison bar chart.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        metavar="DIR",
        help="Override output directory.  Default: <results-root>/<run-name>/figures/",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip schema validation warnings.",
    )

    args = parser.parse_args()

    results_root = Path(args.results_root)
    run_name     = args.run_name
    save_dir     = Path(args.out_dir) if args.out_dir else (results_root / run_name / "figures")

    print(f"\n{'='*60}")
    print(f"  DysarthriaNSR — Generate Figures")
    print(f"  Run:        {run_name}")
    print(f"  Output:     {save_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Load primary run
    # ------------------------------------------------------------------
    eval_results, explanations, per_phoneme_per = _load_run(results_root, run_name)

    if eval_results is None:
        print(f"\n✗  Cannot find evaluation_results.json for run '{run_name}'.")
        print(f"   Expected path: {results_root / run_name / 'evaluation_results.json'}")
        print("   Did you run evaluation yet?  python run_pipeline.py --run-name <NAME>")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------
    if not args.no_validate:
        missing_eval = _validate_eval_schema(eval_results, run_name)
        if missing_eval:
            print("  ⚠  evaluation_results.json is missing optional fields:")
            for m in missing_eval:
                print(f"       • {m}")
            print()

        if explanations:
            utterances: list[dict] = explanations.get("utterances", [])
            missing_utt, utt_warnings = _validate_utterance_schema(utterances)
            if missing_utt:
                print("  ⚠  Per-utterance records are missing required fields:")
                for m in missing_utt:
                    print(f"       • {m}")
                print()
            for w in utt_warnings:
                print(f"  ⚠  {w}")
            if utt_warnings:
                print()
        else:
            print("  ℹ  No explanations.json found — uncertainty plots will show placeholders.")
            print(f"     To generate: python run_pipeline.py --run-name {run_name} "
                  "--skip-train --explain --uncertainty\n")

    # ------------------------------------------------------------------
    # Load comparison runs (if requested)
    # ------------------------------------------------------------------
    comparison_results: dict[str, dict] | None = None
    if args.compare:
        comparison_results = {}
        for cmp_run in args.compare:
            cmp_eval, _, _ = _load_run(results_root, cmp_run)
            if cmp_eval is not None:
                comparison_results[cmp_run] = cmp_eval
                print(f"  ✓ Loaded comparison run: {cmp_run}")
            else:
                print(f"  ⚠  Comparison run '{cmp_run}' not found — skipped.")
        print()

    # ------------------------------------------------------------------
    # Generate all plots
    # ------------------------------------------------------------------
    print("  Generating figures...")
    saved = generate_all_plots(
        eval_results=eval_results,
        explanations=explanations,
        run_name=run_name,
        save_dir=save_dir,
        severity_map=TORGO_SEVERITY_MAP,
        comparison_results=comparison_results,
        per_phoneme_per=per_phoneme_per,
    )

    print(f"\n{'='*60}")
    print(f"  ✅  {len(saved)} figure(s) saved to:")
    print(f"     {save_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
