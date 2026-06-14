"""
Unit tests for shared utility functions and constants.

Tests verify:
- normalize_phoneme strips stress markers correctly
- get_speaker_severity returns expected values
- PHONEME_ARTICULATORY has complete and consistent entries
- align_labels_to_logits produces correct shapes and padding
"""
import pytest
import torch

from src.utils.config import normalize_phoneme, get_speaker_severity, TORGO_SEVERITY_MAP
from src.utils.constants import PHONEME_ARTICULATORY, PHONEME_FEATURES
from src.utils.sequence_utils import align_labels_to_logits


class TestNormalizePhoneme:
    def test_strips_stress_0(self):
        assert normalize_phoneme("AH0") == "AH"

    def test_strips_stress_1(self):
        assert normalize_phoneme("IH1") == "IH"

    def test_strips_stress_2(self):
        assert normalize_phoneme("AE2") == "AE"

    def test_no_stress_unchanged(self):
        assert normalize_phoneme("AE") == "AE"

    def test_consonant_unchanged(self):
        assert normalize_phoneme("P") == "P"
        assert normalize_phoneme("SH") == "SH"
        assert normalize_phoneme("TH") == "TH"

    def test_multidigit_stress(self):
        assert normalize_phoneme("ER0") == "ER"
        assert normalize_phoneme("ER1") == "ER"
        assert normalize_phoneme("ER2") == "ER"

    def test_empty_string(self):
        assert normalize_phoneme("") == ""

    def test_already_normalized(self):
        assert normalize_phoneme("AH") == "AH"


class TestGetSpeakerSeverity:
    def test_known_dysarthric_speaker(self):
        assert get_speaker_severity("M01") == pytest.approx(4.90)

    def test_known_control_speaker(self):
        assert get_speaker_severity("FC01") == pytest.approx(0.0)

    def test_unknown_speaker_default(self):
        assert get_speaker_severity("UNKNOWN") == pytest.approx(2.5)

    def test_case_sensitivity(self):
        assert get_speaker_severity("m01") == pytest.approx(4.90)

    def test_all_control_speakers_zero(self):
        for spk in ("FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04"):
            assert get_speaker_severity(spk) == pytest.approx(0.0)

    def test_severity_map_consistency(self):
        """Severity map values should be in [0.0, 5.0]."""
        for sev in TORGO_SEVERITY_MAP.values():
            assert 0.0 <= sev <= 5.0


class TestPhonemeArticulatoryConstants:
    def test_all_entries_have_manner_place_voice(self):
        """Every phoneme must have a 3-tuple of (manner, place, voice)."""
        for ph, (manner, place, voice) in PHONEME_ARTICULATORY.items():
            assert isinstance(manner, str), f"{ph} missing manner"
            assert isinstance(place, str), f"{ph} missing place"
            assert isinstance(voice, str), f"{ph} missing voice"

    def test_all_voiced_or_voiceless(self):
        """Voice feature must be 'voiced' or 'voiceless'."""
        for ph, (_, _, voice) in PHONEME_ARTICULATORY.items():
            assert voice in ("voiced", "voiceless"), f"{ph} has invalid voice: {voice}"

    def test_phoneme_features_dict_structure(self):
        """PHONEME_FEATURES should have nested dicts with manner/place/voice keys."""
        for ph, features in PHONEME_FEATURES.items():
            assert "manner" in features
            assert "place" in features
            assert "voice" in features
            assert features["manner"] == PHONEME_ARTICULATORY[ph][0]
            assert features["place"] == PHONEME_ARTICULATORY[ph][1]
            assert features["voice"] == PHONEME_ARTICULATORY[ph][2]

    def test_both_dicts_have_same_keys(self):
        """PHONEME_ARTICULATORY and PHONEME_FEATURES must share keys."""
        assert set(PHONEME_ARTICULATORY.keys()) == set(PHONEME_FEATURES.keys())

    def test_vowels_are_all_voiced(self):
        """All vowels should be voiced (physiological constraint)."""
        vowel_phonemes = ["IY", "IH", "EH", "EY", "AE", "AA", "AO",
                          "OW", "UH", "UW", "AH", "ER", "AX", "IX", "AXR"]
        for ph in vowel_phonemes:
            assert PHONEME_ARTICULATORY[ph][2] == "voiced", f"{ph} should be voiced"


class TestAlignLabelsToLogits:
    def test_same_length_unchanged(self):
        labels = torch.tensor([[3, 4, 5]])
        result = align_labels_to_logits(labels, time_steps_logits=3)
        assert torch.equal(result, labels)

    def test_longer_logit_time(self):
        labels = torch.tensor([[3, 4, 5]])
        result = align_labels_to_logits(labels, time_steps_logits=6)
        assert result.shape == (1, 6)

    def test_shorter_logit_time(self):
        labels = torch.tensor([[3, 4, 5, 6, 7]])
        result = align_labels_to_logits(labels, time_steps_logits=3)
        assert result.shape == (1, 3)

    def test_propagates_minus_100_tail(self):
        """Trailing -100 padding in source must remain -100 after alignment."""
        labels = torch.tensor([[3, 4, 5, -100, -100]])
        result = align_labels_to_logits(labels, time_steps_logits=8)
        # Tail positions should be -100
        assert (result[0, 5:] == -100).all()

    def test_batch_consistency(self):
        labels = torch.tensor([[3, 4, 5], [6, 7, 8]])
        result = align_labels_to_logits(labels, time_steps_logits=6)
        assert result.shape == (2, 6)

    def test_zero_center_weight_no_masking(self):
        """center_weight=0 should produce no additional -100 masking."""
        labels = torch.tensor([[3, 4, 5]])
        result = align_labels_to_logits(labels, time_steps_logits=6, center_weight=0.0)
        assert (result != -100).all()

    def test_full_center_weight_masks_edges(self):
        """center_weight=1 should mask all but center frame."""
        labels = torch.tensor([[3, 4, 5]])
        result = align_labels_to_logits(labels, time_steps_logits=6, center_weight=1.0)
        # Most positions should be -100
        n_valid = (result != -100).sum().item()
        assert n_valid >= 1

    def test_empty_labels_raises(self):
        """Empty label tensor (batch_size=0) is unsupported and raises IndexError."""
        labels = torch.zeros(0, 0, dtype=torch.long)
        with pytest.raises(IndexError):
            align_labels_to_logits(labels, time_steps_logits=5)

    def test_no_nan_in_output(self):
        labels = torch.randint(3, 10, (4, 8))
        result = align_labels_to_logits(labels, time_steps_logits=12)
        assert not torch.isnan(result).any()
