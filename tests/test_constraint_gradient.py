"""
Unit tests for LearnableConstraintMatrix gradient flow.

Tests verify:
- Gradients reach logit_C in learnable mode
- Static mode has no trainable logit_C
- Output of LearnableConstraintMatrix is always row-stochastic and NaN-free
- SymbolicConstraintLayer in static mode has no learnable_matrix
"""
import pytest
import torch
import torch.nn.functional as F

from src.models.model import LearnableConstraintMatrix, SymbolicConstraintLayer
from src.utils.config import SymbolicConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tiny_vocab():
    phn_to_id = {'<BLANK>': 0, '<PAD>': 1, '<UNK>': 2, 'P': 3, 'B': 4}
    id_to_phn  = {v: k for k, v in phn_to_id.items()}
    return phn_to_id, id_to_phn


def _make_learnable_matrix(V: int = 5) -> LearnableConstraintMatrix:
    init_matrix = torch.eye(V)
    return LearnableConstraintMatrix(num_phonemes=V, init_matrix=init_matrix)


def _make_constraint_layer(learnable: bool) -> SymbolicConstraintLayer:
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


# ── LearnableConstraintMatrix ─────────────────────────────────────────────────

class TestLearnableConstraintMatrix:

    def test_gradient_flows_to_logit_C(self):
        """After forward + backward, logit_C must have a non-None gradient."""
        lcm = _make_learnable_matrix()
        P_neural = F.softmax(torch.randn(2, 8, 5), dim=-1)
        P_constrained = lcm(P_neural)
        P_constrained.mean().backward()
        assert lcm.logit_C.grad is not None, (
            "logit_C should have a gradient after backward()"
        )

    def test_gradient_logit_C_not_nan(self):
        """Gradient of logit_C must not contain NaN."""
        lcm = _make_learnable_matrix()
        P_neural = F.softmax(torch.randn(3, 15, 5), dim=-1)
        lcm(P_neural).mean().backward()
        assert not torch.isnan(lcm.logit_C.grad).any(), (
            "logit_C.grad contains NaN"
        )

    def test_output_row_stochastic(self):
        """Each row of the output must sum to 1 (valid probability distribution)."""
        lcm = _make_learnable_matrix()
        P_neural = F.softmax(torch.randn(2, 10, 5), dim=-1)
        with torch.no_grad():
            out = lcm(P_neural)
        row_sums = out.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            f"Rows don't sum to 1: min={row_sums.min():.6f}, max={row_sums.max():.6f}"
        )

    def test_output_no_nan(self):
        """Output must not contain NaN for random valid inputs."""
        torch.manual_seed(42)
        lcm = _make_learnable_matrix()
        P_neural = F.softmax(torch.randn(4, 20, 5), dim=-1)
        with torch.no_grad():
            out = lcm(P_neural)
        assert not torch.isnan(out).any()

    def test_C_property_is_row_stochastic(self):
        """The .C property (softmax of logit_C) must be row-stochastic."""
        lcm = _make_learnable_matrix()
        C = lcm.C
        row_sums = C.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_init_sharpens_prior(self):
        """Temperature-sharpened init: diagonal of C at init should be > 1/V."""
        V = 5
        lcm = _make_learnable_matrix(V)
        diagonal = lcm.C.diagonal()
        # With temperature=0.5 sharpening, diagonal entries must exceed uniform (1/V)
        assert (diagonal > 1.0 / V).all(), (
            f"Expected diagonal > {1/V:.3f}, got {diagonal.tolist()}"
        )


# ── SymbolicConstraintLayer gradient modes ────────────────────────────────────

class TestSymbolicConstraintLayerGradientModes:

    def test_learnable_mode_gradient_to_logit_C(self):
        """In learnable mode, backward from log_probs must reach logit_C."""
        layer = _make_constraint_layer(learnable=True)
        layer.train()
        logits = torch.randn(2, 5, 5)
        out = layer(logits, speaker_severity=torch.tensor([0.0, 5.0]))
        out['log_probs'].mean().backward()
        assert layer.learnable_matrix.logit_C.grad is not None

    def test_static_mode_has_no_learnable_matrix(self):
        """In static mode, learnable_matrix must be None."""
        layer = _make_constraint_layer(learnable=False)
        assert layer.learnable_matrix is None

    def test_static_mode_no_trainable_params_except_beta(self):
        """Static mode: only beta should be a trainable parameter."""
        layer = _make_constraint_layer(learnable=False)
        trainable = [name for name, p in layer.named_parameters() if p.requires_grad]
        assert trainable == ['beta'], (
            f"Expected only ['beta'] to be trainable, got {trainable}"
        )

    def test_learnable_mode_beta_receives_gradient(self):
        """In learnable mode, beta must receive a gradient after backward."""
        layer = _make_constraint_layer(learnable=True)
        layer.train()
        logits = torch.randn(2, 5, 5)
        out = layer(logits, speaker_severity=torch.tensor([0.0, 5.0]))
        out['log_probs'].mean().backward()
        assert layer.beta.grad is not None
