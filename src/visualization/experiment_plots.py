"""
experiment_plots.py — Publication-quality diagnostic visualizations for DysarthriaNSR.

Each public function accepts parsed JSON data (dicts / lists of dicts) and a
``save_path`` argument.  Functions return the resolved ``Path`` of the saved
figure so callers can report or chain them.

All plots follow a consistent house style:
    sns.set_context("paper")
    sns.set_style("whitegrid")
    palette = "viridis"

Usage (from generate_figures.py)::

    from src.visualization.experiment_plots import generate_all_plots
    generate_all_plots(eval_results, explanations, run_name="baseline_v2", save_dir=Path("results/baseline_v2/figures"))
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Headless backend; must appear before pyplot import

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------

_PALETTE = "viridis"
_CONTEXT = "paper"
_STYLE = "whitegrid"
_DPI = 300
_FIGSIZE_SINGLE = (5.5, 4.0)   # Single-column figure
_FIGSIZE_WIDE   = (8.0, 4.5)   # Two-column / wider figure

# Cohort colours (accessible, consistent across plots)
_COLOUR_DYSARTHRIC = "#2d7dd2"   # blue
_COLOUR_CONTROL    = "#e63946"   # red


def _apply_style() -> None:
    """Apply the DysarthriaNSR house style to the current matplotlib session."""
    sns.set_context(_CONTEXT)
    sns.set_style(_STYLE)
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "figure.dpi":         _DPI,
        "savefig.dpi":        _DPI,
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })


def _save(fig: plt.Figure, path: Path) -> Path:
    """Save ``fig`` to ``path``, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. Error Type Distribution
# ---------------------------------------------------------------------------

def plot_error_distribution(
    error_counts: Dict[str, int],
    save_path: Path,
) -> Path:
    """
    Bar chart of CTC edit-distance operation counts.

    Args:
        error_counts:  Dict with keys ``substitutions``, ``deletions``,
                       ``insertions``, ``correct`` and integer values.
        save_path:     Destination PNG path.

    Returns:
        Resolved ``save_path``.
    """
    _apply_style()

    labels  = ["Correct", "Substitutions", "Deletions", "Insertions"]
    keys    = ["correct", "substitutions", "deletions", "insertions"]
    counts  = [error_counts.get(k, 0) for k in keys]
    colours = sns.color_palette(_PALETTE, n_colors=4)

    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    bars = ax.bar(labels, counts, color=colours, edgecolor="white", linewidth=0.6, zorder=3)

    total = sum(counts) or 1
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=7,
        )

    ax.set_ylabel("Token Count")
    ax.set_title("Error Type Distribution")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# 2. PER by Speaker
# ---------------------------------------------------------------------------

def plot_per_by_speaker(
    per_speaker: Dict[str, Dict],
    save_path: Path,
) -> Path:
    """
    Horizontal bar chart of per-speaker PER with 95 % bootstrap CI whiskers.

    Args:
        per_speaker:  Dict mapping speaker ID → ``{per, ci, std, n_samples, status}``.
                      ``status`` 0 = control, 1 = dysarthric.
        save_path:    Destination PNG path.

    Returns:
        Resolved ``save_path``.
    """
    _apply_style()

    if not per_speaker:
        warnings.warn("plot_per_by_speaker: empty per_speaker dict — skipping.", stacklevel=2)
        return Path(save_path)

    # Sort by PER descending for readability
    speakers = sorted(per_speaker.keys(), key=lambda s: per_speaker[s]["per"], reverse=True)
    pers     = [per_speaker[s]["per"]    for s in speakers]
    statuses = [per_speaker[s].get("status", -1) for s in speakers]

    ci_lo = [per_speaker[s]["ci"][0] if "ci" in per_speaker[s] else per_speaker[s]["per"] for s in speakers]
    ci_hi = [per_speaker[s]["ci"][1] if "ci" in per_speaker[s] else per_speaker[s]["per"] for s in speakers]
    xerr_lo = [p - lo for p, lo in zip(pers, ci_lo)]
    xerr_hi = [hi - p for p, hi in zip(pers, ci_hi)]

    colours = [_COLOUR_DYSARTHRIC if s == 1 else _COLOUR_CONTROL for s in statuses]

    fig, ax = plt.subplots(figsize=(6.0, max(3.0, 0.55 * len(speakers) + 1.5)))
    y_pos   = np.arange(len(speakers))

    ax.barh(
        y_pos, pers,
        xerr=[xerr_lo, xerr_hi],
        color=colours,
        edgecolor="white",
        linewidth=0.5,
        error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "dimgrey"},
        zorder=3,
    )

    # Mean line
    mean_per = float(np.mean(pers))
    ax.axvline(mean_per, color="dimgrey", linestyle="--", linewidth=1.0, label=f"Mean PER = {mean_per:.3f}")

    # Annotate bars
    for idx, per in enumerate(pers):
        ax.text(per + 0.003, y_pos[idx], f"{per:.3f}", va="center", ha="left", fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Phoneme Error Rate (PER)")
    ax.set_title("PER by Speaker  (error bars: 95 % CI)")

    legend_handles = [
        mpatches.Patch(color=_COLOUR_DYSARTHRIC, label="Dysarthric"),
        mpatches.Patch(color=_COLOUR_CONTROL,    label="Control"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)
    ax.xaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(ci_hi) * 1.18)

    fig.tight_layout()
    return _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# 3. Severity vs PER
# ---------------------------------------------------------------------------

def plot_severity_vs_per(
    per_speaker: Dict[str, Dict],
    severity_map: Dict[str, float],
    save_path: Path,
) -> Path:
    """
    Scatter plot of speaker-level severity score vs PER with OLS regression line.

    Args:
        per_speaker:   Dict mapping speaker ID → ``{per, status, ...}``.
        severity_map:  ``TORGO_SEVERITY_MAP`` dict (str → float).
        save_path:     Destination PNG path.

    Returns:
        Resolved ``save_path``.
    """
    _apply_style()

    speakers   = list(per_speaker.keys())
    severities = [severity_map.get(s.split("_")[0].upper(), 2.5) for s in speakers]
    pers       = [per_speaker[s]["per"] for s in speakers]
    statuses   = [per_speaker[s].get("status", -1) for s in speakers]

    x = np.array(severities, dtype=float)
    y = np.array(pers,       dtype=float)

    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    colours = [_COLOUR_DYSARTHRIC if s == 1 else _COLOUR_CONTROL for s in statuses]

    ax.scatter(x, y, c=colours, s=90, zorder=4, edgecolors="white", linewidths=0.5)

    for spk, xi, yi in zip(speakers, x, y):
        ax.annotate(
            spk, (xi, yi),
            textcoords="offset points", xytext=(5, 3),
            fontsize=6.5, color="dimgrey",
        )

    # OLS regression (only if ≥2 unique x values)
    annotation_text = "r = N/A  (≤ 1 unique severity)"
    if len(set(x)) >= 2:
        slope, intercept, r, p_val, _ = sp_stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="dimgrey", linewidth=1.2, linestyle="--", zorder=3)
        sig_star = "*" if p_val < 0.05 else " (n.s.)"
        annotation_text = f"r = {r:.3f},  p = {p_val:.3f}{sig_star}"

    ax.text(
        0.97, 0.97, annotation_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="lightgrey"),
    )

    legend_handles = [
        mpatches.Patch(color=_COLOUR_DYSARTHRIC, label="Dysarthric"),
        mpatches.Patch(color=_COLOUR_CONTROL,    label="Control"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False)
    ax.set_xlabel("Severity Score  [0 = control, 5 = most severe]")
    ax.set_ylabel("Phoneme Error Rate (PER)")
    ax.set_title("Severity vs PER")
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# 4. Uncertainty vs PER
# ---------------------------------------------------------------------------

def plot_uncertainty_vs_per(
    utterances: List[Dict],
    save_path: Path,
) -> Path:
    """
    Scatter plot of per-utterance predictive entropy vs PER for uncertainty calibration.

    If uncertainty data is unavailable (all ``None``), a placeholder figure is saved
    informing the user to rerun with ``--uncertainty``.

    Args:
        utterances:  List of utterance dicts from ``explanations.json``.
        save_path:   Destination PNG path.

    Returns:
        Resolved ``save_path``.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)

    valid = [(u["uncertainty"], u["per"]) for u in utterances
             if u.get("uncertainty") is not None and u.get("per") is not None]

    if not valid:
        ax.text(
            0.5, 0.5,
            "Uncertainty data not available.\nRe-run evaluation with --uncertainty flag.",
            ha="center", va="center", fontsize=9, transform=ax.transAxes, color="grey",
        )
        ax.set_title("Uncertainty vs PER  [no data]")
        fig.tight_layout()
        return _save(fig, Path(save_path))

    entropies, pers = zip(*valid)
    x = np.array(entropies, dtype=float)
    y = np.array(pers,       dtype=float)

    # Hex-bin for density (readable when n > 500)
    if len(x) > 500:
        hb = ax.hexbin(x, y, gridsize=40, cmap=_PALETTE, mincnt=1, linewidths=0.2)
        fig.colorbar(hb, ax=ax, label="Sample count")
    else:
        ax.scatter(x, y, alpha=0.5, s=12,
                   c=np.array(sns.color_palette(_PALETTE, len(x))), zorder=3)

    # Regression
    if len(set(x)) >= 2:
        slope, intercept, r, p_val, _ = sp_stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, slope * x_line + intercept,
                color="crimson", linewidth=1.3, linestyle="--", label=f"r = {r:.3f}")
        ax.legend(frameon=False)

    ax.set_xlabel("Predictive Entropy  (MC Dropout)")
    ax.set_ylabel("Phoneme Error Rate (PER)")
    ax.set_title(f"Uncertainty Calibration  (n = {len(x):,})")
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# 5. Uncertainty Distribution (dysarthric vs control)
# ---------------------------------------------------------------------------

def plot_uncertainty_distribution(
    utterances: List[Dict],
    severity_map: Dict[str, float],
    save_path: Path,
) -> Path:
    """
    KDE histogram comparing predictive entropy distributions for dysarthric vs control.

    Args:
        utterances:   List of utterance dicts from ``explanations.json``.
        severity_map: ``TORGO_SEVERITY_MAP`` — used to determine cohort from speaker_id.
        save_path:    Destination PNG path.

    Returns:
        Resolved ``save_path``.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)

    valid = [u for u in utterances if u.get("uncertainty") is not None]

    if not valid:
        ax.text(
            0.5, 0.5,
            "Uncertainty data not available.\nRe-run evaluation with --uncertainty flag.",
            ha="center", va="center", fontsize=9, transform=ax.transAxes, color="grey",
        )
        ax.set_title("Uncertainty Distribution  [no data]")
        fig.tight_layout()
        return _save(fig, Path(save_path))

    dysarthric_e = []
    control_e    = []

    for u in valid:
        spk = u.get("speaker_id", "unknown")
        base_id = spk.split("_")[0].upper()
        sev = severity_map.get(base_id, 2.5)
        # Control speakers have severity 0.0; dysarthric > 0.0
        if sev == 0.0:
            control_e.append(u["uncertainty"])
        else:
            dysarthric_e.append(u["uncertainty"])

    datasets = {"Dysarthric": (dysarthric_e, _COLOUR_DYSARTHRIC),
                "Control":    (control_e,    _COLOUR_CONTROL)}

    for label, (data, colour) in datasets.items():
        if not data:
            continue
        arr = np.array(data, dtype=float)

        # Histogram bars
        ax.hist(arr, bins=40, density=True, alpha=0.25, color=colour, edgecolor="none")

        # KDE
        if len(arr) > 5:
            kde = sp_stats.gaussian_kde(arr, bw_method="scott")
            x_line = np.linspace(arr.min(), arr.max(), 300)
            ax.plot(x_line, kde(x_line), color=colour, linewidth=1.8, label=label)

            # 95 % bootstrap CI on mean
            boot_means = [np.mean(np.random.choice(arr, len(arr), replace=True))
                          for _ in range(1000)]
            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
            mn = arr.mean()
            ax.axvline(mn, color=colour, linestyle="--", linewidth=1.0, alpha=0.7)
            ax.axvspan(ci_lo, ci_hi, alpha=0.08, color=colour)

    ax.set_xlabel("Predictive Entropy  (MC Dropout)")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty Distribution — Dysarthric vs Control")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# 6. Experiment Comparison
# ---------------------------------------------------------------------------

def plot_experiment_comparison(
    results: Dict[str, Dict],
    save_path: Path,
    metric: str = "avg_per",
) -> Path:
    """
    Grouped bar chart comparing a scalar metric across multiple experiment runs.

    Args:
        results:   Dict mapping run name → parsed ``evaluation_results.json`` dict.
        save_path: Destination PNG path.
        metric:    Top-level key from ``evaluation_results.json`` to compare.
                   Defaults to ``"avg_per"``.  If the key is nested (e.g. for
                   ablation sub-categories) callers should pre-aggregate.

    Returns:
        Resolved ``save_path``.
    """
    _apply_style()

    if not results:
        warnings.warn("plot_experiment_comparison: empty results dict — skipping.", stacklevel=2)
        return Path(save_path)

    run_names  = list(results.keys())
    values     = []
    ci_errs_lo = []
    ci_errs_hi = []

    for run in run_names:
        r = results[run]
        val = r.get(metric, r.get("overall", {}).get("per_macro_speaker", 0.0))
        values.append(float(val))
        ci = r.get("overall", {}).get("ci", [val, val])
        ci_errs_lo.append(float(val - ci[0]))
        ci_errs_hi.append(float(ci[1] - val))

    palette = sns.color_palette(_PALETTE, n_colors=len(run_names))
    x_pos   = np.arange(len(run_names))

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    bars = ax.bar(
        x_pos, values,
        color=palette,
        edgecolor="white",
        linewidth=0.6,
        yerr=[ci_errs_lo, ci_errs_hi],
        error_kw={"elinewidth": 1.2, "capsize": 4, "ecolor": "dimgrey"},
        zorder=3,
        width=0.55,
    )

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(run_names, rotation=25, ha="right")
    ax.set_ylabel("Macro-Speaker PER")
    ax.set_title(f"Experiment Comparison  [{metric}]  (error bars: 95 % CI)")
    ax.set_ylim(0, max(values) * 1.2)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def generate_all_plots(
    eval_results: Dict,
    explanations: Optional[Dict],
    run_name: str,
    save_dir: Path,
    severity_map: Optional[Dict[str, float]] = None,
    comparison_results: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Path]:
    """
    Generate the full diagnostic visualization suite for one experiment run.

    Args:
        eval_results:         Parsed ``evaluation_results.json``.
        explanations:         Parsed ``explanations.json`` (or ``None`` when absent).
        run_name:             Experiment identifier (used only in log messages).
        save_dir:             Directory where figures are saved.
        severity_map:         ``TORGO_SEVERITY_MAP`` dict.  Defaults to importing
                              from ``src.utils.config`` when ``None``.
        comparison_results:   Optional dict of ``{run_name: eval_results}`` for the
                              experiment comparison plot.  When supplied, the current
                              run is automatically included.

    Returns:
        Dict mapping plot name → saved ``Path``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if severity_map is None:
        try:
            from src.utils.config import TORGO_SEVERITY_MAP  # type: ignore
            severity_map = TORGO_SEVERITY_MAP
        except ImportError:
            severity_map = {}

    utterances: List[Dict] = []
    if explanations and "utterances" in explanations:
        utterances = explanations["utterances"]

    per_speaker: Dict = eval_results.get("per_speaker", {})
    error_counts: Dict = eval_results.get("error_analysis", {}).get("error_counts", {})

    saved: Dict[str, Path] = {}

    # ------------------------------------------------------------------
    # 1. Error Type Distribution
    # ------------------------------------------------------------------
    if error_counts:
        p = plot_error_distribution(error_counts, save_dir / "error_distribution.png")
        saved["error_distribution"] = p
        print(f"  ✓ error_distribution       → {p.name}")
    else:
        print("  ⚠  error_distribution skipped — error_counts missing")

    # ------------------------------------------------------------------
    # 2. PER by Speaker
    # ------------------------------------------------------------------
    if per_speaker:
        p = plot_per_by_speaker(per_speaker, save_dir / "per_by_speaker.png")
        saved["per_by_speaker"] = p
        print(f"  ✓ per_by_speaker            → {p.name}")
    else:
        print("  ⚠  per_by_speaker skipped — per_speaker missing")

    # ------------------------------------------------------------------
    # 3. Severity vs PER
    # ------------------------------------------------------------------
    if per_speaker and severity_map:
        p = plot_severity_vs_per(per_speaker, severity_map, save_dir / "severity_vs_per.png")
        saved["severity_vs_per"] = p
        print(f"  ✓ severity_vs_per           → {p.name}")
    else:
        print("  ⚠  severity_vs_per skipped — per_speaker or severity_map missing")

    # ------------------------------------------------------------------
    # 4. Uncertainty vs PER
    # ------------------------------------------------------------------
    p = plot_uncertainty_vs_per(utterances, save_dir / "uncertainty_vs_per.png")
    saved["uncertainty_vs_per"] = p
    print(f"  ✓ uncertainty_vs_per        → {p.name}")

    # ------------------------------------------------------------------
    # 5. Uncertainty Distribution
    # ------------------------------------------------------------------
    p = plot_uncertainty_distribution(utterances, severity_map or {}, save_dir / "uncertainty_distribution.png")
    saved["uncertainty_distribution"] = p
    print(f"  ✓ uncertainty_distribution  → {p.name}")

    # ------------------------------------------------------------------
    # 6. Experiment Comparison
    # ------------------------------------------------------------------
    if comparison_results is not None:
        merged = {run_name: eval_results, **comparison_results}
        p = plot_experiment_comparison(merged, save_dir / "experiment_comparison.png")
        saved["experiment_comparison"] = p
        print(f"  ✓ experiment_comparison     → {p.name}")
    else:
        # Emit single-bar comparison as baseline reference
        p = plot_experiment_comparison({run_name: eval_results}, save_dir / "experiment_comparison.png")
        saved["experiment_comparison"] = p
        print(f"  ✓ experiment_comparison     → {p.name}  (single run)")

    print(f"\n  All figures saved to: {save_dir}")
    return saved
