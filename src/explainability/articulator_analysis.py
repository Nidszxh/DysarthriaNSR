"""
Articulatory Confusion Analyzer (ROADMAP §6.3)

Builds per-feature (manner / place / voice) confusion matrices from phoneme-level
error alignments and produces publication-quality heatmaps.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class ArticulatoryConfusionAnalyzer:
    """
    Analyzes phoneme substitution errors at the articulatory feature level.

    Instead of raw phoneme confusions (44×44), this produces 3 small matrices:
        - Manner confusion:  e.g. stop → fricative
        - Place confusion:   e.g. bilabial → alveolar  
        - Voice confusion:   e.g. voiced → voiceless

    These are directly interpretable by speech-language pathologists and map
    to known dysarthric error patterns (devoicing, fronting, liquid-gliding).

    Args:
        phoneme_features: Dict mapping phoneme → {manner, place, voice} features.
    """

    def __init__(self, phoneme_features: Optional[Dict] = None):
        if phoneme_features is None:
            try:
                from src.models.model import ArticulatoryFeatureEncoder
                self.phoneme_features = ArticulatoryFeatureEncoder.PHONEME_FEATURES
            except ImportError:
                self.phoneme_features = {}
        else:
            self.phoneme_features = phoneme_features

        # Confusion accumulators: feature_dim → {expected_val → {predicted_val → count}}
        self._confusion: Dict[str, Dict[str, Dict[str, int]]] = {
            "manner": defaultdict(lambda: defaultdict(int)),
            "place": defaultdict(lambda: defaultdict(int)),
            "voice": defaultdict(lambda: defaultdict(int)),
        }
        self._n_substitutions = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Accumulation
    # ──────────────────────────────────────────────────────────────────────────

    def accumulate_from_errors(
        self,
        substitution_errors: List[Tuple[str, str]],
    ) -> None:
        """
        Accumulate articulatory confusion counts from substitution error pairs.

        Args:
            substitution_errors: List of (reference_phoneme, predicted_phoneme) tuples
        """
        for ref_ph, pred_ph in substitution_errors:
            f_ref = self.phoneme_features.get(ref_ph)
            f_pred = self.phoneme_features.get(pred_ph)
            if f_ref is None or f_pred is None:
                continue

            for dim in ("manner", "place", "voice"):
                v_ref = f_ref.get(dim, "unknown")
                v_pred = f_pred.get(dim, "unknown")
                self._confusion[dim][v_ref][v_pred] += 1

            self._n_substitutions += 1

    def reset(self) -> None:
        """Clear all accumulated confusion counts."""
        for dim in self._confusion:
            self._confusion[dim] = defaultdict(lambda: defaultdict(int))
        self._n_substitutions = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Analysis
    # ──────────────────────────────────────────────────────────────────────────

    def get_confusion_dict(self) -> Dict[str, Dict]:
        """Return raw confusion counts keyed by feature dimension."""
        return {dim: dict(self._confusion[dim]) for dim in self._confusion}

    def get_top_confusions(self, dim: str = "manner", top_k: int = 10) -> List[Tuple]:
        """
        Return the most frequent confusion pairs for a given feature dimension.

        Args:
            dim: Feature dimension ("manner", "place", or "voice")
            top_k: Number of pairs to return

        Returns:
            List of ((ref_val, pred_val), count) sorted by count descending.
        """
        pairs = []
        for ref_val, preds in self._confusion[dim].items():
            for pred_val, count in preds.items():
                if ref_val != pred_val:  # Only off-diagonal
                    pairs.append(((ref_val, pred_val), count))
        return sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]

    # ──────────────────────────────────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────────────────────────────────

    def plot_feature_confusion(
        self,
        save_path: Path,
        figsize: Tuple[int, int] = (18, 5),
    ) -> None:
        """
        Plot 3-panel articulatory feature confusion heatmaps.

        One panel per feature dimension (manner / place / voice), row-normalised
        so the diagonal represents per-category accuracy.

        Args:
            save_path: Output PNG path.
            figsize: Figure size (width, height) in inches.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        dims = [("manner", "Manner of Articulation"),
                ("place", "Place of Articulation"),
                ("voice", "Voicing")]

        for ax, (dim, title) in zip(axes, dims):
            conf = self._confusion[dim]
            all_cats = sorted(set(
                k for d in conf.values() for k in list(d.keys()) + [k for k in conf]
            ))
            if not all_cats:
                ax.set_title(f"{title}\n(no data)")
                ax.axis("off")
                continue

            n = len(all_cats)
            cat_idx = {c: i for i, c in enumerate(all_cats)}
            matrix = np.zeros((n, n))

            for ref_val, preds in conf.items():
                if ref_val in cat_idx:
                    for pred_val, count in preds.items():
                        if pred_val in cat_idx:
                            matrix[cat_idx[ref_val], cat_idx[pred_val]] += count

            # Row-normalise
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix_norm = np.divide(matrix, row_sums, where=row_sums != 0)

            sns.heatmap(
                matrix_norm,
                ax=ax,
                xticklabels=all_cats,
                yticklabels=all_cats,
                cmap="YlOrRd",
                vmin=0,
                vmax=1,
                annot=n <= 8,
                fmt=".2f" if n <= 8 else "",
                cbar_kws={"label": "Proportion"},
            )
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("Reference", fontsize=10)
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.tick_params(axis="y", rotation=0, labelsize=8)

        plt.suptitle(
            f"Articulatory Feature Confusion (N={self._n_substitutions} substitutions)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Articulatory confusion plot saved to {save_path}")
