"""
Explainable Output Formatter (ROADMAP §6.4)

Produces structured, utterance-level explanation JSON matching the ROADMAP §6.4
schema — ready for consumption by clinical dashboards and speech-language
pathologists.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExplainableOutputFormatter:
    """
    Formats per-utterance prediction + error analysis into a structured
    clinical explanation matching the ROADMAP §6.4 JSON schema.

    The schema includes:
        - utterance_id, ground_truth, prediction, wer
        - phoneme_analysis.errors: type, position, expected/predicted phoneme,
          articulatory features, probable cause, symbolic rule, neural confidence
        - symbolic_rules_summary: total fired, top rules, avg confidence
        - attention_visualization: base64-encoded attention heatmap (optional)

    Accumulated explanations can be written to disk as `explanations.json`.
    """

    def __init__(self):
        self._explanations: List[Dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Formatting
    # ──────────────────────────────────────────────────────────────────────────

    def format_utterance(
        self,
        utterance_id: str,
        ground_truth: str,
        prediction: str,
        errors: List[Dict[str, Any]],
        symbolic_rules_summary: Dict[str, Any],
        wer: float = 0.0,
        per: float = 0.0,
        attention_heatmap_b64: Optional[str] = None,
        speaker_id: Optional[str] = None,
        severity: Optional[float] = None,
        uncertainty: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Format a single utterance into the ROADMAP §6.4 explanation schema.

        Args:
            utterance_id:       Unique sample identifier
            ground_truth:       Reference transcription (text)
            prediction:         Model transcription (text)
            errors:             List of error dicts from PhonemeAttributor
            symbolic_rules_summary: Output from SymbolicRuleTracker.generate_explanation()
            wer:                Word Error Rate for this utterance
            per:                Phoneme Error Rate for this utterance
            attention_heatmap_b64: Optional base64-encoded attention PNG
            speaker_id:         Speaker identifier
            severity:           Continuous severity score [0, 5]
            uncertainty:        Predictive entropy from UncertaintyAwareDecoder

        Returns:
            Explanation dict matching the ROADMAP §6.4 schema.
        """
        formatted_errors = []
        for err in errors:
            entry: Dict[str, Any] = {
                "type": err.get("type"),
                "position": err.get("position"),
                "predicted_phoneme": err.get("predicted_phoneme"),
                "expected_phoneme": err.get("expected_phoneme"),
                "articulatory_features": err.get("articulatory_features", {}),
                "feature_differences": err.get("feature_differences", {}),
                "probable_cause": err.get("probable_cause", "Unknown"),
                "neural_confidence": err.get("neural_confidence"),
                "symbolic_correction_applied": err.get("symbolic_correction_applied", False),
                "symbolic_rule_activated": err.get("symbolic_rule_activated"),
            }
            formatted_errors.append(entry)

        explanation = {
            "utterance_id": utterance_id,
            "speaker_id": speaker_id,
            "severity": severity,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "wer": round(float(wer), 4),
            "per": round(float(per), 4),
            "uncertainty": round(float(uncertainty), 4) if uncertainty is not None else None,
            "phoneme_analysis": {
                "n_errors": len(formatted_errors),
                "errors": formatted_errors,
            },
            "symbolic_rules_summary": {
                "total_fired": symbolic_rules_summary.get("total_activations", 0),
                "activation_rate": symbolic_rules_summary.get("activation_rate", 0.0),
                "avg_blend_weight": symbolic_rules_summary.get("avg_blend_weight", 0.0),
                "avg_prediction_confidence": symbolic_rules_summary.get("avg_prediction_confidence"),
                "top_rules": [
                    {"rule_id": rule, "count": count}
                    for rule, count in symbolic_rules_summary.get("top_rules", [])
                ],
            },
        }

        if attention_heatmap_b64:
            explanation["attention_visualization"] = attention_heatmap_b64

        return explanation

    def add(self, explanation: Dict[str, Any]) -> None:
        """Append a formatted utterance explanation to the internal buffer."""
        self._explanations.append(explanation)

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save_explanations(
        self,
        results_dir: Path,
        filename: str = "explanations.json",
    ) -> Path:
        """
        Write all accumulated explanations to a JSON file.

        Args:
            results_dir: Directory to write the output file
            filename:    Output filename (default: explanations.json)

        Returns:
            Path to the written file.
        """
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / filename

        payload = {
            "n_utterances": len(self._explanations),
            "utterances": self._explanations,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

        print(f"✅ Explanations saved to {output_path} ({len(self._explanations)} utterances)")
        return output_path

    def clear(self) -> None:
        """Reset accumulated explanations."""
        self._explanations.clear()

    def __len__(self) -> int:
        return len(self._explanations)
