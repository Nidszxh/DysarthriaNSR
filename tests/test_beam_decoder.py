"""
Unit tests for BeamSearchDecoder.

Tests verify:
- Peaked distributions decode to the expected phoneme sequence
- Output never contains special tokens (BLANK / PAD / UNK)
- Returned score is always a finite number
- All-blank input decodes to an empty sequence
- Beam-width=1 does not crash and returns a list
"""
import numpy as np
import pytest

from evaluate import BeamSearchDecoder


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_vocab() -> dict:
    return {0: '<BLANK>', 1: '<PAD>', 2: '<UNK>', 3: 'P', 4: 'B', 5: 'T'}


def _peaked_log_probs(T: int, V: int, peaked_ids) -> np.ndarray:
    """Build [T, V] log_probs where each frame is a one-hot at peaked_ids[t]."""
    log_probs = np.full((T, V), -100.0, dtype=np.float32)
    for t, idx in enumerate(peaked_ids):
        log_probs[t, idx] = 0.0  # log(1) = 0
    return log_probs


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBeamSearchDecoder:

    def setup_method(self):
        self.decoder    = BeamSearchDecoder(beam_width=5, blank_id=0)
        self.id_to_phn  = _tiny_vocab()

    # -- Correctness on peaked inputs -----------------------------------------

    def test_peaked_single_phoneme(self):
        """[P, P, BLANK] should collapse to ['P']."""
        log_probs = _peaked_log_probs(3, 6, [3, 3, 0])
        phonemes, _ = self.decoder.decode(log_probs, self.id_to_phn)
        assert phonemes == ['P'], f"Expected ['P'], got {phonemes}"

    def test_peaked_two_phoneme_sequence(self):
        """[P, BLANK, B] should decode to ['P', 'B']."""
        log_probs = _peaked_log_probs(3, 6, [3, 0, 4])
        phonemes, _ = self.decoder.decode(log_probs, self.id_to_phn)
        assert phonemes == ['P', 'B'], f"Expected ['P', 'B'], got {phonemes}"

    def test_all_blank_decodes_empty(self):
        """All-blank input must decode to an empty sequence."""
        log_probs = _peaked_log_probs(5, 6, [0, 0, 0, 0, 0])
        phonemes, _ = self.decoder.decode(log_probs, self.id_to_phn)
        assert phonemes == [], f"Expected [], got {phonemes}"

    # -- Output validity -------------------------------------------------------

    def test_no_special_tokens_in_output(self):
        """Output should never contain BLANK, PAD, or UNK."""
        rng = np.random.default_rng(42)
        log_probs = rng.standard_normal((20, 6)).astype(np.float32)
        log_probs -= log_probs.max(axis=-1, keepdims=True)
        phonemes, _ = self.decoder.decode(log_probs, self.id_to_phn)
        special = {'<BLANK>', '<PAD>', '<UNK>'}
        bad = [p for p in phonemes if p in special]
        assert not bad, f"Special tokens found in output: {bad}"

    def test_score_is_finite(self):
        """Returned log-probability score must be finite."""
        rng = np.random.default_rng(7)
        log_probs = rng.standard_normal((10, 6)).astype(np.float32)
        log_probs -= log_probs.max(axis=-1, keepdims=True)
        _, score = self.decoder.decode(log_probs, self.id_to_phn)
        assert np.isfinite(score), f"Expected finite score, got {score}"

    def test_output_is_list(self):
        """decode() must always return a list (even for empty output)."""
        rng = np.random.default_rng(99)
        log_probs = rng.standard_normal((8, 6)).astype(np.float32)
        log_probs -= log_probs.max(axis=-1, keepdims=True)
        phonemes, _ = self.decoder.decode(log_probs, self.id_to_phn)
        assert isinstance(phonemes, list)

    # -- Length normalisation --------------------------------------------------

    def test_length_norm_does_not_crash(self):
        """length_norm_alpha > 0 should not raise an exception."""
        decoder = BeamSearchDecoder(beam_width=5, blank_id=0, length_norm_alpha=0.6)
        rng = np.random.default_rng(11)
        log_probs = rng.standard_normal((10, 6)).astype(np.float32)
        log_probs -= log_probs.max(axis=-1, keepdims=True)
        phonemes, score = decoder.decode(log_probs, self.id_to_phn)
        assert isinstance(phonemes, list)
        assert np.isfinite(score)

    def test_all_blank_with_length_norm(self):
        """All-blank + length_norm_alpha=0.6 must still decode to []."""
        decoder = BeamSearchDecoder(beam_width=5, blank_id=0, length_norm_alpha=0.6)
        log_probs = _peaked_log_probs(5, 6, [0, 0, 0, 0, 0])
        phonemes, _ = decoder.decode(log_probs, self.id_to_phn)
        assert phonemes == []

    # -- Beam width 1 ----------------------------------------------------------

    def test_beam_width_1_returns_list(self):
        """Beam width 1 is valid and should return a list."""
        decoder = BeamSearchDecoder(beam_width=1, blank_id=0)
        log_probs = _peaked_log_probs(4, 6, [3, 0, 4, 0])
        phonemes, _ = decoder.decode(log_probs, self.id_to_phn)
        assert isinstance(phonemes, list)

    def test_pad_token_treated_as_non_emitting(self):
        """PAD-dominant frames should not create emitted tokens in the beam path."""
        # [PAD, P, PAD] should decode to ['P'] when PAD is handled blank-like.
        log_probs = _peaked_log_probs(3, 6, [1, 3, 1])
        phonemes, _ = self.decoder.decode(log_probs, self.id_to_phn)
        assert phonemes == ['P'], f"Expected ['P'], got {phonemes}"

    def test_pad_unk_mass_not_added_to_blank(self):
        """Moderate PAD/UNK probabilities must not inflate blank transitions.

        If PAD/UNK are incorrectly merged into the blank path, this setup tends
        to collapse to an empty prediction (deletion-heavy failure mode).
        """
        probs = np.array([0.35, 0.125, 0.125, 0.40, 1e-8, 1e-8], dtype=np.float64)
        probs = probs / probs.sum()
        log_probs = np.log(np.tile(probs, (6, 1)).astype(np.float32))

        phonemes, _ = self.decoder.decode(log_probs, self.id_to_phn)
        assert phonemes == ['P'], f"Expected ['P'], got {phonemes}"
