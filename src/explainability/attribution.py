"""
Phoneme Attribution Analysis (ROADMAP §6.1 & §6.2)

Provides two attribution strategies:
  1. Alignment-based: Levenshtein alignment → token-level error attribution
  2. Attention-based: Aggregate HuBERT attention maps to phoneme boundaries
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# Map common articulatory feature substitution patterns to probable clinical causes
_PROBABLE_CAUSE_MAP = {
    ("voice", "voiced", "voiceless"): "Devoicing (voiced → voiceless)",
    ("liquid", "glide"): "Liquid gliding (R/L → W/Y)",
    ("fricative", "stop"): "Stopping (fricative → stop)",
    ("fricative", "affricate"): "Affrication",
    ("vowel", "vowel"): "Vowel centralization",
    ("place", "bilabial", "alveolar"): "Fronting (bilabial → alveolar)",
    ("place", "velar", "alveolar"): "Fronting (velar → alveolar)",
}


class PhonemeAttributor:
    """
    Phoneme-level error attribution for clinical explainability.

    Implements ROADMAP §6.1 (alignment-based) and §6.2 (attention-based)
    attribution methods.

    Args:
        phoneme_features: Dict mapping phoneme string → {manner, place, voice}
            (defaults to ArticulatoryFeatureEncoder.PHONEME_FEATURES if None)
    """

    def __init__(self, phoneme_features: Optional[Dict] = None):
        if phoneme_features is None:
            # Import default features from the model module
            try:
                from src.models.model import ArticulatoryFeatureEncoder
                self.phoneme_features = ArticulatoryFeatureEncoder.PHONEME_FEATURES
            except ImportError:
                self.phoneme_features = {}
        else:
            self.phoneme_features = phoneme_features

    # ──────────────────────────────────────────────────────────────────────────
    # Alignment-Based Attribution (ROADMAP §6.1 Method 1)
    # ──────────────────────────────────────────────────────────────────────────

    def alignment_attribution(
        self,
        predicted: List[str],
        reference: List[str],
    ) -> List[Dict]:
        """
        Attribute prediction errors to specific phoneme positions via
        Levenshtein alignment.

        Args:
            predicted:  Predicted phoneme sequence
            reference:  Ground-truth phoneme sequence

        Returns:
            List of error dicts, each containing:
                - type:                 'substitution' | 'deletion' | 'insertion'
                - position:             Position in reference sequence
                - predicted_phoneme:    Predicted symbol (or None for deletion)
                - expected_phoneme:     Reference symbol (or None for insertion)
                - articulatory_features: Feature dict of reference phoneme
                - feature_differences:  Dict of differing features (substitutions)
                - probable_cause:       Clinical explanation string
                - neural_confidence:    None (filled during evaluation if available)
        """
        alignment = self._levenshtein_alignment(predicted, reference)
        errors = []

        for pos, (op, pred_ph, ref_ph) in enumerate(alignment):
            if op == "correct":
                continue

            feat_ref = self.phoneme_features.get(ref_ph, {}) if ref_ph else {}
            feat_pred = self.phoneme_features.get(pred_ph, {}) if pred_ph else {}

            # Compute feature differences for substitutions
            feature_differences = {}
            if op == "substitution" and feat_ref and feat_pred:
                for feat in ("manner", "place", "voice"):
                    v_ref = feat_ref.get(feat)
                    v_pred = feat_pred.get(feat)
                    if v_ref != v_pred:
                        feature_differences[feat] = f"{v_ref} → {v_pred}"

            probable_cause = self._infer_probable_cause(
                op, feat_ref, feat_pred, feature_differences
            )

            errors.append({
                "type": op,
                "position": pos,
                "predicted_phoneme": pred_ph,
                "expected_phoneme": ref_ph,
                "articulatory_features": feat_ref,
                "feature_differences": feature_differences,
                "probable_cause": probable_cause,
                "neural_confidence": None,
            })

        return errors

    # ──────────────────────────────────────────────────────────────────────────
    # Attention-Based Attribution (ROADMAP §6.1 Method 2)
    # ──────────────────────────────────────────────────────────────────────────

    def attention_attribution(
        self,
        attention_weights: Tuple[torch.Tensor, ...],
        phoneme_boundaries: Optional[List[Tuple[int, int]]] = None,
        n_layers_to_use: int = 4,
    ) -> np.ndarray:
        """
        Aggregate HuBERT attention maps to produce a per-frame importance score.

        Method: Average over the last n_layers_to_use transformer layers and
        all attention heads, then optionally pool over phoneme boundary regions.

        Args:
            attention_weights: Tuple of attention tensors from HuBERT
                (one per layer, each [B, heads, T, T])
            phoneme_boundaries: Optional list of (start_frame, end_frame) per phoneme
            n_layers_to_use: Number of final layers to aggregate (default: 4)

        Returns:
            Per-frame importance scores [T] aggregated over heads and layers,
            or per-phoneme scores [n_phonemes] if boundaries are provided.
        """
        if not attention_weights:
            return np.array([])

        # Take the last n layers
        layers = attention_weights[-n_layers_to_use:]  # list of [B, H, T, T]

        # Stack and average over layers and heads: [T, T]
        stacked = torch.stack([
            layer[0].mean(dim=0)  # [H, T, T] → [T, T]
            for layer in layers
        ]).mean(dim=0)  # [T, T]

        # Column-sum gives attention received per frame (importance)
        importance = stacked.sum(dim=0).cpu().numpy()  # [T]

        if phoneme_boundaries is None:
            return importance

        # Pool over phoneme boundary intervals
        phoneme_scores = []
        for start, end in phoneme_boundaries:
            region = importance[start:end] if end > start else importance[start:start+1]
            phoneme_scores.append(float(region.mean()))

        return np.array(phoneme_scores)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _levenshtein_alignment(
        pred: List[str],
        ref: List[str],
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """
        Compute Levenshtein alignment between predicted and reference sequences.

        Returns:
            List of (operation, predicted_phoneme, reference_phoneme) tuples.
            Operations: 'correct', 'substitution', 'deletion', 'insertion'
        """
        m, n = len(pred), len(ref)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i - 1] == ref[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        # Backtrack
        alignment = []
        i, j = m, n
        while i > 0 or j > 0:
            if i == 0:
                alignment.append(("insertion", None, ref[j - 1]))
                j -= 1
            elif j == 0:
                alignment.append(("deletion", pred[i - 1], None))
                i -= 1
            elif pred[i - 1] == ref[j - 1]:
                alignment.append(("correct", pred[i - 1], ref[j - 1]))
                i -= 1
                j -= 1
            else:
                options = [
                    (dp[i - 1][j - 1], "substitution", i - 1, j - 1),
                    (dp[i - 1][j], "deletion", i - 1, j),
                    (dp[i][j - 1], "insertion", i, j - 1),
                ]
                options.sort()
                _, op, ni, nj = options[0]
                if op == "substitution":
                    alignment.append((op, pred[i - 1], ref[j - 1]))
                elif op == "deletion":
                    alignment.append((op, pred[i - 1], None))
                else:
                    alignment.append((op, None, ref[j - 1]))
                i, j = ni, nj

        return alignment[::-1]

    def _infer_probable_cause(
        self,
        op: str,
        feat_ref: Dict,
        feat_pred: Dict,
        feature_differences: Dict,
    ) -> str:
        """Map articulatory feature differences to clinical cause labels."""
        if op == "deletion":
            manner = feat_ref.get("manner", "")
            return f"{'Cluster simplification' if manner in ('stop','affricate') else 'Phoneme omission'}"
        if op == "insertion":
            return "Intrusion (epenthesis)"

        # Substitution: pattern-match feature differences
        voice_diff = "voice" in feature_differences
        manner_diff = "manner" in feature_differences
        place_diff = "place" in feature_differences

        ref_manner = feat_ref.get("manner", "")
        pred_manner = feat_pred.get("manner", "")
        ref_voice = feat_ref.get("voice", "")
        pred_voice = feat_pred.get("voice", "")
        ref_place = feat_ref.get("place", "")
        pred_place = feat_pred.get("place", "")

        if voice_diff:
            mapped = _PROBABLE_CAUSE_MAP.get(("voice", ref_voice, pred_voice))
            if mapped:
                return mapped

        if voice_diff and not manner_diff and not place_diff:
            if ref_voice == "voiced" and pred_voice == "voiceless":
                return "Devoicing (voiced → voiceless)"
            return "Voicing change"

        mapped = _PROBABLE_CAUSE_MAP.get((ref_manner, pred_manner))
        if mapped:
            return mapped

        if place_diff:
            mapped = _PROBABLE_CAUSE_MAP.get(("place", ref_place, pred_place))
            if mapped:
                return mapped

        if manner_diff:
            if ref_manner == "liquid" and pred_manner == "glide":
                return "Liquid gliding (R/L → W/Y)"
            if ref_manner in ("fricative", "affricate") and pred_manner == "stop":
                return "Stopping (fricative/affricate → stop)"
            if place_diff and not voice_diff:
                return "Fronting (place of articulation shift)"
        if ref_manner == "vowel" and pred_manner == "vowel":
            return "Vowel centralization / reduction"

        return "Undetermined substitution"
