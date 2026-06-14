"""
Unit tests for the explainability module.

Tests verify:
- ExplainableOutputFormatter produces correct schema
- save_explanations writes valid JSON
- ArticulatoryConfusionAnalyzer accumulates and reports correctly
- Edge cases: empty error lists, missing features, unknown phonemes
"""
import json
import pytest
from pathlib import Path

from src.explainability.output_format import ExplainableOutputFormatter
from src.explainability.articulator_analysis import ArticulatoryConfusionAnalyzer


# ── ExplainableOutputFormatter ──────────────────────────────────────────────

class TestExplainableOutputFormatter:
    def setup_method(self):
        self.formatter = ExplainableOutputFormatter()

    def test_format_utterance_basic_schema(self):
        explanation = self.formatter.format_utterance(
            utterance_id="utt_001",
            ground_truth="AH B C",
            prediction="AH D C",
            errors=[
                {
                    "type": "substitution",
                    "position": 1,
                    "predicted_phoneme": "D",
                    "expected_phoneme": "B",
                    "articulatory_features": {},
                    "feature_differences": {},
                    "probable_cause": "place assimilation",
                    "neural_confidence": 0.45,
                    "symbolic_correction_applied": True,
                    "symbolic_rule_activated": "alveolarisation",
                }
            ],
            symbolic_rules_summary={
                "total_activations": 5,
                "activation_rate": 0.12,
                "avg_blend_weight": 0.3,
                "avg_prediction_confidence": 0.72,
                "top_rules": [("alveolarisation", 3), ("fronting", 2)],
            },
            wer=0.33,
            per=0.33,
            speaker_id="M01",
            severity=4.9,
            uncertainty=0.85,
        )
        assert explanation["utterance_id"] == "utt_001"
        assert explanation["speaker_id"] == "M01"
        assert explanation["severity"] == 4.9
        assert explanation["phoneme_analysis"]["n_errors"] == 1
        assert explanation["symbolic_rules_summary"]["total_fired"] == 5
        assert explanation["uncertainty"] == 0.85

    def test_format_utterance_no_errors(self):
        explanation = self.formatter.format_utterance(
            utterance_id="utt_002",
            ground_truth="AH B C",
            prediction="AH B C",
            errors=[],
            symbolic_rules_summary={},
            wer=0.0,
            per=0.0,
        )
        assert explanation["phoneme_analysis"]["n_errors"] == 0
        assert explanation["symbolic_rules_summary"]["total_fired"] == 0

    def test_format_utterance_no_uncertainty(self):
        explanation = self.formatter.format_utterance(
            utterance_id="utt_003",
            ground_truth="AH B C",
            prediction="AH D C",
            errors=[],
            symbolic_rules_summary={},
            wer=0.33,
            per=0.33,
        )
        assert explanation["uncertainty"] is None

    def test_add_and_clear(self):
        self.formatter.add({"utterance_id": "utt_001"})
        assert len(self.formatter) == 1
        self.formatter.clear()
        assert len(self.formatter) == 0

    def test_save_explanations_writes_valid_json(self, tmp_path):
        self.formatter.add(
            self.formatter.format_utterance(
                utterance_id="utt_001",
                ground_truth="AH B",
                prediction="AH D",
                errors=[],
                symbolic_rules_summary={},
                wer=0.5,
                per=0.5,
            )
        )
        output_path = self.formatter.save_explanations(tmp_path)
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert data["n_utterances"] == 1
        assert len(data["utterances"]) == 1

    def test_save_explanations_empty(self, tmp_path):
        output_path = self.formatter.save_explanations(tmp_path)
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert data["n_utterances"] == 0

    def test_multiple_utterances(self):
        for i in range(5):
            explanation = self.formatter.format_utterance(
                utterance_id=f"utt_{i:03d}",
                ground_truth="AH",
                prediction="AH",
                errors=[],
                symbolic_rules_summary={},
                wer=0.0,
                per=0.0,
            )
            self.formatter.add(explanation)
        assert len(self.formatter) == 5

    def test_error_entry_fields(self):
        errors = [
            {
                "type": "deletion",
                "position": 0,
                "predicted_phoneme": None,
                "expected_phoneme": "B",
                "articulatory_features": {"manner": "stop", "place": "bilabial"},
                "feature_differences": {},
                "probable_cause": "Unknown",
                "neural_confidence": None,
                "symbolic_correction_applied": False,
                "symbolic_rule_activated": None,
            }
        ]
        explanation = self.formatter.format_utterance(
            utterance_id="utt_del",
            ground_truth="B C",
            prediction="C",
            errors=errors,
            symbolic_rules_summary={},
            wer=0.5,
            per=0.5,
        )
        err = explanation["phoneme_analysis"]["errors"][0]
        assert err["type"] == "deletion"
        assert err["predicted_phoneme"] is None

    def test_attention_heatmap_included_when_provided(self):
        explanation = self.formatter.format_utterance(
            utterance_id="utt_attn",
            ground_truth="AH",
            prediction="AH",
            errors=[],
            symbolic_rules_summary={},
            wer=0.0,
            per=0.0,
            attention_heatmap_b64="iVBORw0KGgo=",
        )
        assert "attention_visualization" in explanation

    def test_symbolic_rules_summary_empty_rules(self):
        explanation = self.formatter.format_utterance(
            utterance_id="utt_empty_rules",
            ground_truth="AH",
            prediction="AH",
            errors=[],
            symbolic_rules_summary={"total_activations": 0, "activation_rate": 0.0},
            wer=0.0,
            per=0.0,
        )
        assert explanation["symbolic_rules_summary"]["top_rules"] == []


# ── ArticulatoryConfusionAnalyzer ──────────────────────────────────────────

class TestArticulatoryConfusionAnalyzer:
    def setup_method(self):
        self.analyzer = ArticulatoryConfusionAnalyzer()

    def test_accumulate_single_substitution(self):
        self.analyzer.accumulate_from_errors([("P", "B")])
        confusion = self.analyzer.get_confusion_dict()
        assert confusion["manner"]["stop"]["stop"] == 1
        assert confusion["place"]["bilabial"]["bilabial"] == 1
        assert confusion["voice"]["voiceless"]["voiced"] == 1

    def test_accumulate_multiple_substitutions(self):
        errors = [("P", "B"), ("T", "D"), ("K", "G")]
        self.analyzer.accumulate_from_errors(errors)
        confusion = self.analyzer.get_confusion_dict()
        assert confusion["manner"]["stop"]["stop"] == 3
        assert confusion["voice"]["voiceless"]["voiced"] == 3

    def test_unknown_phoneme_skipped(self):
        self.analyzer.accumulate_from_errors([("ZZZ", "B"), ("P", "YYY")])
        confusion = self.analyzer.get_confusion_dict()
        assert sum(sum(v.values()) for d in confusion["manner"].values() for v in d.values()) == 0

    def test_top_confusions_returns_off_diagonal(self):
        errors = [("P", "B")] * 5 + [("T", "D")] * 3 + [("K", "G")] * 2
        self.analyzer.accumulate_from_errors(errors)
        top_manner = self.analyzer.get_top_confusions("voice", top_k=5)
        assert len(top_manner) == 1
        assert top_manner[0][0] == ("voiceless", "voiced")
        assert top_manner[0][1] == 10

    def test_reset_clears_counts(self):
        self.analyzer.accumulate_from_errors([("P", "B")])
        self.analyzer.reset()
        confusion = self.analyzer.get_confusion_dict()
        assert confusion["manner"] == {}
        assert confusion["place"] == {}
        assert confusion["voice"] == {}

    def test_manner_confusion_example(self):
        """Test a known dysarthric pattern: stop → fricative."""
        self.analyzer.accumulate_from_errors([("T", "S")])
        top_manner = self.analyzer.get_top_confusions("manner")
        assert any(pair[0] == ("stop", "fricative") for pair in top_manner)

    def test_n_substitutions_tracked(self):
        self.analyzer.accumulate_from_errors([("P", "B"), ("T", "D")])
        assert self.analyzer._n_substitutions == 2

    def test_empty_accumulate_no_error(self):
        self.analyzer.accumulate_from_errors([])
        confusion = self.analyzer.get_confusion_dict()
        for dim in ("manner", "place", "voice"):
            assert confusion[dim] == {}
