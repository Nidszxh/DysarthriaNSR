"""
Unit tests for SymbolicConstraintLayer.

Tests verify:
- blank_constraint_threshold masking: blank-dominant frames bypass the constraint
- P_constrained is not None in full mode
- Output distributions are valid (sum to 1, no NaN)
- Severity-adaptive beta shifts blending weight
"""
import pytest
import torch
import torch.nn.functional as F

from src.models.model import SymbolicConstraintLayer
from src.utils.config import SymbolicConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tiny_vocab():
    """Minimal vocabulary: BLANK=0, PAD=1, UNK=2, P=3, B=4."""
    phn_to_id = {'<BLANK>': 0, '<PAD>': 1, '<UNK>': 2, 'P': 3, 'B': 4}
    id_to_phn  = {v: k for k, v in phn_to_id.items()}
    return phn_to_id, id_to_phn


def _make_constraint_layer(learnable: bool = False) -> SymbolicConstraintLayer:
    phn_to_id, id_to_phn = _make_tiny_vocab()
    cfg = SymbolicConfig()
    return SymbolicConstraintLayer(
        num_phonemes=len(phn_to_id),
        phn_to_id=phn_to_id,
        id_to_phn=id_to_phn,
        symbolic_config=cfg,
        constraint_weight=0.1,
        learnable=learnable,
        use_learnable_matrix=learnable,
    )


# ── Blank-frame masking (H1) ─────────────────────────────────────────────────

class TestBlankFrameMasking:

    def test_blank_dominant_frame_preserved(self):
        """When blank prob > threshold, log-probs at that frame must stay blank-dominant."""
        layer = _make_constraint_layer(learnable=False)
        layer.eval()
        V = 5
        # blank(id=0) gets logit=10 → P_neural[0,0,0] ≈ 1.0 >> threshold=0.5
        logits = torch.full((1, 1, V), -10.0)
        logits[0, 0, 0] = 10.0

        with torch.no_grad():
            out = layer(logits, speaker_severity=torch.tensor([0.0]))

        # After masking, blank frame should still be blank-dominant
        log_probs = out['log_probs']
        assert log_probs[0, 0, 0] > log_probs[0, 0, 3], (
            "Blank-dominant frame should not be flipped by the constraint"
        )

    def test_non_blank_frame_not_blank_dominant(self):
        """When phoneme logit is high, output should be phoneme-dominant."""
        layer = _make_constraint_layer(learnable=False)
        layer.eval()
        V = 5
        logits = torch.full((1, 1, V), -5.0)
        logits[0, 0, 3] = 5.0  # 'P' is dominant (blank stays small)

        with torch.no_grad():
            out = layer(logits, speaker_severity=torch.tensor([0.0]))

        log_probs = out['log_probs']
        assert log_probs[0, 0, 3] > log_probs[0, 0, 0], (
            "Non-blank frame should remain phoneme-dominant after constraint"
        )

    def test_p_constrained_present_in_full_mode(self):
        """In full mode, P_constrained must not be None."""
        layer = _make_constraint_layer(learnable=False)
        layer.eval()
        logits = torch.randn(2, 5, 5)
        with torch.no_grad():
            out = layer(logits, speaker_severity=torch.tensor([0.0, 5.0]))
        assert out['P_constrained'] is not None

    def test_output_sums_to_one(self):
        """exp(log_probs) must sum to 1 per frame (valid probability distribution)."""
        layer = _make_constraint_layer(learnable=False)
        layer.eval()
        logits = torch.randn(2, 10, 5)
        with torch.no_grad():
            out = layer(logits, speaker_severity=torch.tensor([0.0, 5.0]))
        probs = out['log_probs'].exp()
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            f"Row sums deviate from 1: min={row_sums.min():.6f}, max={row_sums.max():.6f}"
        )

    def test_no_nan_in_output(self):
        """log_probs must not contain NaN for any valid input."""
        layer = _make_constraint_layer(learnable=False)
        layer.eval()
        torch.manual_seed(0)
        logits = torch.randn(4, 20, 5)
        with torch.no_grad():
            out = layer(logits, speaker_severity=torch.tensor([0.0, 1.0, 3.0, 5.0]))
        assert not torch.isnan(out['log_probs']).any()


# ── Severity-adaptive beta ────────────────────────────────────────────────────

class TestSeverityAdaptiveBeta:

    def test_high_severity_higher_beta(self):
        """Higher severity should produce a higher effective beta (more symbolic weight)."""
        layer = _make_constraint_layer(learnable=False)
        layer.eval()
        logits = torch.randn(2, 5, 5)
        with torch.no_grad():
            out_low  = layer(logits, speaker_severity=torch.tensor([0.0, 0.0]))
            out_high = layer(logits, speaker_severity=torch.tensor([5.0, 5.0]))

        beta_low  = out_low['beta'].item()
        beta_high = out_high['beta'].item()
        assert beta_high >= beta_low, (
            f"Expected beta_high ({beta_high:.4f}) >= beta_low ({beta_low:.4f})"
        )


# ── Gradient flow ─────────────────────────────────────────────────────────────

class TestConstraintGradientFlow:

    def test_learnable_mode_gradient_to_logit_C(self):
        """After backward, logit_C must accumulate a gradient."""
        layer = _make_constraint_layer(learnable=True)
        layer.train()
        logits = torch.randn(2, 5, 5)
        out = layer(logits, speaker_severity=torch.tensor([0.0, 5.0]))
        out['log_probs'].mean().backward()
        assert layer.learnable_matrix.logit_C.grad is not None, (
            "Expected gradient to flow to learnable_matrix.logit_C"
        )

    def test_beta_receives_gradient(self):
        """After backward, the learnable beta parameter should have a gradient."""
        layer = _make_constraint_layer(learnable=True)
        layer.train()
        logits = torch.randn(2, 5, 5)
        out = layer(logits, speaker_severity=torch.tensor([0.0, 5.0]))
        out['log_probs'].mean().backward()
        assert layer.beta.grad is not None, "Expected gradient to flow to beta"

    def test_no_nan_gradients(self):
        """Gradients should not contain NaN values."""
        layer = _make_constraint_layer(learnable=True)
        layer.train()
        logits = torch.randn(4, 10, 5)
        out = layer(logits, speaker_severity=torch.tensor([0.0, 1.0, 3.0, 5.0]))
        out['log_probs'].mean().backward()
        assert not torch.isnan(layer.learnable_matrix.logit_C.grad).any()
        assert not torch.isnan(layer.beta.grad)
