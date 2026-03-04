"""
Explainability Package for DysarthriaNSR

Implements ROADMAP §6: Phoneme Attribution Analysis, Symbolic Rule Tracking,
Articulatory Confusion Analysis, and Clinical Explanation Formatting.
"""

from .attribution import PhonemeAttributor
from .rule_tracker import SymbolicRuleTracker
from .articulator_analysis import ArticulatoryConfusionAnalyzer
from .output_format import ExplainableOutputFormatter

__all__ = [
    "PhonemeAttributor",
    "SymbolicRuleTracker",
    "ArticulatoryConfusionAnalyzer",
    "ExplainableOutputFormatter",
]
