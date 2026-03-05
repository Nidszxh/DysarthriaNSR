"""
Unit tests for evaluation metrics.

Tests verify:
- compute_per returns expected values for known inputs
- Edge cases: empty reference, empty prediction, perfect match
- No NaN or negative values
"""
import pytest
from evaluate import compute_per


class TestComputePer:
    def test_perfect_match(self):
        """Identical prediction and reference → PER = 0.0."""
        assert compute_per(["AH", "B", "C"], ["AH", "B", "C"]) == pytest.approx(0.0)

    def test_all_wrong(self):
        """Completely wrong prediction → PER = 1.0 (substitutions == reference length)."""
        assert compute_per(["X", "Y", "Z"], ["A", "B", "C"]) == pytest.approx(1.0)

    def test_one_substitution(self):
        """One substitution in 3 → PER = 1/3."""
        assert compute_per(["AH", "X", "C"], ["AH", "B", "C"]) == pytest.approx(1.0 / 3)

    def test_empty_reference_empty_prediction(self):
        """Both empty → PER = 0.0 (defined by convention)."""
        assert compute_per([], []) == pytest.approx(0.0)

    def test_empty_reference_nonempty_prediction(self):
        """Non-empty prediction with empty reference → PER = 1.0."""
        assert compute_per(["AH"], []) == pytest.approx(1.0)

    def test_empty_prediction_nonempty_reference(self):
        """Deletion of all reference phonemes → PER = 1.0."""
        assert compute_per([], ["AH", "B"]) == pytest.approx(1.0)

    def test_one_insertion(self):
        """One insertion → PER = 1/2 (1 edit / 2 reference tokens)."""
        assert compute_per(["A", "B", "C"], ["A", "C"]) == pytest.approx(1.0 / 2)

    def test_one_deletion(self):
        """One deletion → PER = 1/3 (1 edit / 3 reference tokens)."""
        assert compute_per(["A", "C"], ["A", "B", "C"]) == pytest.approx(1.0 / 3)

    def test_per_can_exceed_one(self):
        """PER can be > 1.0 when many insertions exist relative to short reference."""
        per = compute_per(["A", "B", "C", "D", "E"], ["A"])
        assert per > 1.0

    def test_non_negative(self):
        """PER should never be negative."""
        per = compute_per(["AH", "B"], ["B", "AH"])
        assert per >= 0.0

    def test_single_phoneme_match(self):
        assert compute_per(["AH"], ["AH"]) == pytest.approx(0.0)

    def test_single_phoneme_mismatch(self):
        assert compute_per(["AH"], ["IH"]) == pytest.approx(1.0)
