"""
Symbolic Rule Tracker (ROADMAP §6.2)

Logs which symbolic constraint rules fired during inference, enabling SLPs to
understand which dysarthric substitution patterns the model identified.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Maximum number of individual activation events held in memory.
# At 5 % activation rate over 15 LOSO folds this would otherwise reach ~9 M
# entries (≈ hundreds of MB).  Summary statistics are computed on the capped
# list; no information is lost at the aggregate level.
_MAX_ACTIVATIONS: int = 50_000


class SymbolicRuleTracker:
    """
    Tracks symbolic rule activations from the SymbolicConstraintLayer.

    A rule is considered "activated" when the argmax of P_final differs from
    the argmax of P_neural — meaning the symbolic constraint changed the model's
    top phoneme prediction.

    Usage:
        tracker = SymbolicRuleTracker()
        # In SymbolicConstraintLayer._track_activations():
        tracker.log_rule_activation(rule_id, input_ctx, output_correction,
                                    blend_weight=beta, prediction_confidence=prob)
        # After evaluation:
        explanation = tracker.generate_explanation()
    """

    def __init__(self, min_confidence: float = 0.05):
        self.min_confidence = min_confidence
        self._activations: List[Dict[str, Any]] = []
        self._total_frames: int = 0

    # Logging API

    def log_rule_activation(
        self,
        rule_id: str,
        input_phoneme: str,
        output_phoneme: str,
        blend_weight: float,
        prediction_confidence: Optional[float] = None,
        frame_idx: Optional[int] = None,
        speaker_id: Optional[str] = None,
    ) -> None:
        """
        Log a single rule activation event.

        Args:
            rule_id:               Identifier string, e.g. "B->P" or "R->W"
            input_phoneme:         Neural top-1 phoneme (before symbolic correction)
            output_phoneme:        Final top-1 phoneme (after symbolic correction)
            blend_weight:          Constraint blend weight β at this frame (range ≈ 0.05–0.25)
            prediction_confidence: max(P_final) at this frame — actual softmax confidence
                                   of the final prediction (optional)
            frame_idx:             Frame index in the utterance (optional)
            speaker_id:            Speaker identifier (optional, for stratification)
        """
        if blend_weight >= self.min_confidence:
            # H-5: Cap list size to prevent OOM during multi-fold evaluation.
            if len(self._activations) < _MAX_ACTIVATIONS:
                self._activations.append({
                    "rule_id": rule_id,
                    "input": input_phoneme,
                    "output_correction": output_phoneme,
                    "blend_weight": blend_weight,
                    "prediction_confidence": prediction_confidence,
                    "frame_idx": frame_idx,
                    "speaker_id": speaker_id,
                })

    def log_frame_count(self, n_frames: int) -> None:
        """Register the total number of frames processed (for activation rate)."""
        self._total_frames += n_frames

    # Analysis API

    def generate_explanation(self) -> Dict[str, Any]:
        """
        Synthesise a high-level explanation of rule activation patterns.

        Returns:
            {
              total_activations: int,
              activation_rate: float,            # activations / total_frames
              high_confidence_corrections: list, # prediction_confidence > 0.7
              rule_frequency: {rule_id: count},
              top_rules: [(rule_id, count), ...],# Top-10
              per_speaker_activations: {speaker: {rule_id: count}},
              avg_blend_weight: float,           # mean β across activations
              avg_prediction_confidence: float | None,  # mean max(P_final) at changed frames
            }
        """
        if not self._activations:
            return {
                "total_activations": 0,
                "activation_rate": 0.0,
                "high_confidence_corrections": [],
                "rule_frequency": {},
                "top_rules": [],
                "per_speaker_activations": {},
                "avg_blend_weight": 0.0,
                "avg_prediction_confidence": None,
            }

        rule_counts = Counter(a["rule_id"] for a in self._activations)
        # Filter on prediction_confidence (max P_final) rather than blend_weight (β ≈ 0.05–0.25)
        high_conf = [a for a in self._activations if (a.get("prediction_confidence") or 0) > 0.7]
        avg_blend = sum(a["blend_weight"] for a in self._activations) / len(self._activations)
        pred_confs = [a["prediction_confidence"] for a in self._activations
                      if a.get("prediction_confidence") is not None]
        avg_pred_conf = sum(pred_confs) / len(pred_confs) if pred_confs else None

        per_speaker: Dict[str, Counter] = defaultdict(Counter)
        for a in self._activations:
            if a.get("speaker_id"):
                per_speaker[a["speaker_id"]][a["rule_id"]] += 1

        activation_rate = (
            len(self._activations) / self._total_frames
            if self._total_frames > 0
            else 0.0
        )

        return {
            "total_activations": len(self._activations),
            "activation_rate": round(activation_rate, 4),
            "high_confidence_corrections": high_conf[:50],  # cap for JSON size
            "rule_frequency": dict(rule_counts),
            "top_rules": rule_counts.most_common(10),
            "per_speaker_activations": {
                spk: dict(cnt) for spk, cnt in per_speaker.items()
            },
            "avg_blend_weight": round(avg_blend, 4),
            "avg_prediction_confidence": round(avg_pred_conf, 4) if avg_pred_conf is not None else None,
        }

    def rule_precision(
        self,
        correct_rule_uses: int,
        total_rule_uses: Optional[int] = None,
    ) -> float:
        """
        Compute rule precision: fraction of rule activations that resulted in
        a correct phoneme prediction.

        Args:
            correct_rule_uses: Number of activations where the final prediction
                               matched the ground-truth phoneme.
            total_rule_uses:   Total activations (defaults to len(self._activations)).

        Returns:
            Precision in [0, 1].
        """
        total = total_rule_uses or len(self._activations)
        if total == 0:
            return 0.0
        return correct_rule_uses / total

    def clear(self) -> None:
        """Reset all recorded activations (call between evaluation runs)."""
        self._activations.clear()
        self._total_frames = 0
