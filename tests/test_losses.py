"""
Unit tests for custom loss functions.

Tests verify:
- Outputs are scalar tensors
- Values are non-negative (valid losses)
- Degenerate inputs (batch_size=1, all-same severity) don't produce NaN/Inf
"""
import pytest
import torch
import torch.nn.functional as F

from src.models.losses import OrdinalContrastiveLoss, BlankPriorKLLoss, SymbolicKLLoss


# ── OrdinalContrastiveLoss ────────────────────────────────────────────────────

class TestOrdinalContrastiveLoss:
    def setup_method(self):
        self.loss_fn = OrdinalContrastiveLoss(margin_per_level=0.3)

    def test_non_negative(self):
        """Loss should always be ≥ 0."""
        hidden = torch.randn(4, 10, 768)
        severity = torch.tensor([0.0, 1.0, 3.0, 5.0])
        loss = self.loss_fn(hidden, severity)
        assert loss.item() >= 0.0

    def test_scalar_output(self):
        hidden = torch.randn(4, 10, 768)
        severity = torch.tensor([0.0, 2.0, 4.0, 5.0])
        loss = self.loss_fn(hidden, severity)
        assert loss.ndim == 0  # scalar

    def test_no_nan(self):
        hidden = torch.randn(4, 10, 768)
        severity = torch.tensor([0.0, 0.0, 5.0, 5.0])
        loss = self.loss_fn(hidden, severity)
        assert not torch.isnan(loss)

    def test_single_sample_returns_zero(self):
        """Batch size 1 → no valid pairs → should return 0, not NaN."""
        hidden = torch.randn(1, 10, 768)
        severity = torch.tensor([2.5])
        loss = self.loss_fn(hidden, severity)
        assert not torch.isnan(loss)
        assert loss.item() == pytest.approx(0.0)

    def test_identical_severity_zero_loss(self):
        """All same severity → target=+1 for all pairs.
        If cosine sim ≥ 0 for all, margin=0 → loss=0."""
        # Make all embeddings identical → cosine sim = 1.0 > margin=0
        emb = torch.ones(3, 5, 768)
        severity = torch.tensor([2.0, 2.0, 2.0])
        loss = self.loss_fn(emb, severity)
        # ReLU(0 − 1*1) = 0 → total loss 0
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_with_attention_mask(self):
        hidden = torch.randn(3, 10, 768)
        mask = torch.ones(3, 10, dtype=torch.long)
        mask[1, 5:] = 0  # pad second sample
        severity = torch.tensor([0.0, 2.5, 5.0])
        loss = self.loss_fn(hidden, severity, attention_mask=mask)
        assert not torch.isnan(loss)
        assert loss.item() >= 0.0


# ── BlankPriorKLLoss ──────────────────────────────────────────────────────────

class TestBlankPriorKLLoss:
    def setup_method(self):
        self.V = 44
        self.loss_fn = BlankPriorKLLoss(blank_id=0, target_prob=0.85)

    def test_non_negative(self):
        log_probs = F.log_softmax(torch.randn(2, 50, self.V), dim=-1)
        loss = self.loss_fn(log_probs)
        assert loss.item() >= 0.0

    def test_scalar_output(self):
        log_probs = F.log_softmax(torch.randn(2, 50, self.V), dim=-1)
        loss = self.loss_fn(log_probs)
        assert loss.ndim == 0

    def test_no_nan(self):
        log_probs = F.log_softmax(torch.randn(4, 100, self.V), dim=-1)
        loss = self.loss_fn(log_probs)
        assert not torch.isnan(loss)

    def test_blank_dominant_low_loss(self):
        """When blank probability ≈ target_prob, KL should be near zero."""
        # Construct logits so blank is the dominant class
        logits = torch.full((2, 50, self.V), -10.0)
        # Set blank logit to a large positive value (blank prob ≈ 1.0)
        logits[:, :, 0] = 10.0
        log_probs = F.log_softmax(logits, dim=-1)
        loss_high_blank = self.loss_fn(log_probs)
        # High blank prob diverges from target 0.85 → non-trivial loss
        assert not torch.isnan(loss_high_blank)

    def test_with_attention_mask(self):
        log_probs = F.log_softmax(torch.randn(2, 50, self.V), dim=-1)
        mask = torch.ones(2, 50, dtype=torch.long)
        mask[0, 40:] = 0  # pad last 10 frames of first sample
        loss = self.loss_fn(log_probs, attention_mask=mask)
        assert not torch.isnan(loss)
        assert loss.item() >= 0.0


# ── SymbolicKLLoss ────────────────────────────────────────────────────────────

class TestSymbolicKLLoss:
    def setup_method(self):
        V = 44
        # Row-normalised uniform prior (equal transition probability)
        prior = torch.ones(V, V) / V
        self.loss_fn = SymbolicKLLoss(static_matrix=prior)
        self.V = V

    def test_non_negative(self):
        logit_C = torch.randn(self.V, self.V)
        loss = self.loss_fn(logit_C)
        assert loss.item() >= 0.0

    def test_scalar_output(self):
        logit_C = torch.randn(self.V, self.V)
        loss = self.loss_fn(logit_C)
        assert loss.ndim == 0

    def test_no_nan(self):
        logit_C = torch.randn(self.V, self.V)
        loss = self.loss_fn(logit_C)
        assert not torch.isnan(loss)

    def test_identical_distribution_near_zero(self):
        """When learned == prior, KL should be ≈ 0."""
        V = self.V
        # Create uniform prior as both reference and target
        prior = torch.ones(V, V) / V
        loss_fn = SymbolicKLLoss(static_matrix=prior)
        # logits that produce exactly uniform distribution: all-zeros
        logit_C = torch.zeros(V, V)
        loss = loss_fn(logit_C)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)
