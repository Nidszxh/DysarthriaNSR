"""
Visualization module for DysarthriaNSR.

Provides diagnostic plots and clinical visualizations for phoneme-level
speech recognition analysis.
"""

from .diagnostics import run_diagnostics
from .experiment_plots import (
    generate_all_plots,
    plot_error_distribution,
    plot_per_by_speaker,
    plot_severity_vs_per,
    plot_uncertainty_vs_per,
    plot_uncertainty_distribution,
    plot_rule_pair_confusion,
    plot_experiment_comparison,
    plot_neural_vs_constrained,
    plot_neural_vs_constrained_comparison,
    plot_articulatory_accuracy,
    plot_articulatory_accuracy_comparison,
    plot_by_length,
    plot_phoneme_per,
    plot_per_by_manner,
)

__all__ = [
    "run_diagnostics",
    "generate_all_plots",
    "plot_error_distribution",
    "plot_per_by_speaker",
    "plot_severity_vs_per",
    "plot_uncertainty_vs_per",
    "plot_uncertainty_distribution",
    "plot_rule_pair_confusion",
    "plot_experiment_comparison",
    "plot_neural_vs_constrained",
    "plot_neural_vs_constrained_comparison",
    "plot_articulatory_accuracy",
    "plot_articulatory_accuracy_comparison",
    "plot_by_length",
    "plot_phoneme_per",
    "plot_per_by_manner",
]
