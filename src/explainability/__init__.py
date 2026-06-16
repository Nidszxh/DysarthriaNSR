"""
Explainability Package for DysarthriaNSR

Implements ROADMAP §6: Phoneme Attribution Analysis, Symbolic Rule Tracking,
Articulatory Confusion Analysis, and Clinical Explanation Formatting.
"""

from .articulator_analysis import ArticulatoryConfusionAnalyzer
from .attribution import PhonemeAttributor
from .output_format import ExplainableOutputFormatter
from .rule_tracker import SymbolicRuleTracker

__all__ = [
    "PhonemeAttributor",
    "SymbolicRuleTracker",
    "ArticulatoryConfusionAnalyzer",
    "ExplainableOutputFormatter",
]
