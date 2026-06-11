"""
Integration test for one full training step.

Covers: forward -> compute_loss -> backward -> optimizer step on a synthetic batch.
Uses a lightweight dummy model to avoid HuBERT downloads in CI.
"""

# REFACTOR LOG
# [FIX-T05] Added test_ce_loss_aligned_forced_align_fallback to verify
#         fallback path when torchaudio forced_align raises.

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import get_default_config
from src.data.dataloader import NeuroSymbolicCollator
from train import DysarthriaASRLightning


class _DummyHubert(nn.Module):
    """Minimal HuBERT stub with _get_feat_extract_output_lengths method."""
    def __init__(self):
        super().__init__()
        # Mock feature extractor for output length calculation
        self.feature_extractor = nn.Identity()
        # Add a dummy parameter so optimizer has something to track
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def _get_feat_extract_output_lengths(self, input_lengths):
        # HuBERT CNN stride is 320
        return (input_lengths - 1) // 320 + 1


class _DummyNeuroSymbolicASR(nn.Module):
    """Minimal model stub compatible with DysarthriaASRLightning."""

    def __init__(self, num_classes: int, time_steps: int = 6, hidden_dim: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.hidden_dim = hidden_dim
        self.proj_in = nn.Linear(1, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, num_classes)
        self.temporal_downsampler = None
        self.hubert = _DummyHubert()

    def freeze_encoder(self) -> None:
        return

    def unfreeze_encoder(self, layers=None) -> None:
        return

    def forward(self, input_values, attention_mask=None, speaker_severity=None, ablation_mode="full"):
        batch_size = input_values.size(0)
        device = input_values.device

        # Build a small temporal representation from the waveform mean.
        x = input_values.mean(dim=1, keepdim=True)  # [B, 1]
        h0 = self.proj_in(x).unsqueeze(1).repeat(1, self.time_steps, 1)  # [B, T, H]

        if speaker_severity is not None:
            # Inject severity so ordinal loss receives non-constant embeddings.
            h0 = h0 + speaker_severity.view(batch_size, 1, 1) / 10.0

        logits_neural = self.proj_out(torch.tanh(h0))
        log_probs_constrained = F.log_softmax(logits_neural, dim=-1)

        return {
            "logits_constrained": log_probs_constrained,
            "logits_neural": logits_neural,
            "hidden_states": h0,
            "beta": torch.tensor(0.1, device=device),
            "output_lengths": torch.full((batch_size,), self.time_steps, dtype=torch.long, device=device),
            "logits_manner": None,
            "logits_place": None,
            "logits_voice": None,
        }


class TestTrainingStepIntegration:
    def test_one_training_step_backward_and_step(self):
        phn_to_id = {"<BLANK>": 0, "<PAD>": 1, "<UNK>": 2, "P": 3, "B": 4}
        id_to_phn = {v: k for k, v in phn_to_id.items()}

        cfg = get_default_config()
        cfg.training.ablation_mode = "neural_only"
        cfg.model.use_learnable_constraint = False

        model = _DummyNeuroSymbolicASR(num_classes=len(phn_to_id), time_steps=6)
        lm = DysarthriaASRLightning(
            model=model,
            config=cfg,
            phn_to_id=phn_to_id,
            id_to_phn=id_to_phn,
            class_weights=torch.ones(len(phn_to_id), dtype=torch.float32),
            articulatory_weights=None,
        )

        # Avoid requiring a Lightning Trainer for metric logging in this unit test.
        lm.log = lambda *args, **kwargs: None
        lm.train()

        batch = {
            "input_values": torch.randn(2, 1920),
            "attention_mask": torch.ones(2, 1920, dtype=torch.long),
            "labels": torch.tensor([[3, 4, 3, 4], [4, 3, 4, 3]], dtype=torch.long),
            "label_lengths": torch.tensor([4, 4], dtype=torch.long),
            "status": torch.tensor([1, 0], dtype=torch.long),
        }

        loss = lm.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0.0, "Loss should be positive"

        optimizer = torch.optim.Adam(lm.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()

        grads = [p.grad for p in lm.parameters() if p.requires_grad]
        assert any(g is not None for g in grads), "Expected gradients after backward"
        assert all(torch.isfinite(g).all() for g in grads if g is not None), "All grads must be finite"

        optimizer.step()

    def test_reset_hubert_lr_warmup_clears_adam_state_safely(self):
        phn_to_id = {"<BLANK>": 0, "<PAD>": 1, "<UNK>": 2, "P": 3, "B": 4}
        id_to_phn = {v: k for k, v in phn_to_id.items()}

        cfg = get_default_config()
        cfg.training.ablation_mode = "neural_only"
        cfg.model.use_learnable_constraint = False

        model = _DummyNeuroSymbolicASR(num_classes=len(phn_to_id), time_steps=6)
        # Provide a HuBERT-like module so configure_optimizers creates the expected param group.
        model.hubert = _DummyHubert()
        model.phoneme_classifier = model.proj_out
        model.severity_adapter = nn.Identity()
        model.manner_head = None
        model.place_head = None
        model.voice_head = None
        model.symbolic_layer = nn.Identity()

        lm = DysarthriaASRLightning(
            model=model,
            config=cfg,
            phn_to_id=phn_to_id,
            id_to_phn=id_to_phn,
            class_weights=torch.ones(len(phn_to_id), dtype=torch.float32),
            articulatory_weights=None,
        )
        lm.log = lambda *args, **kwargs: None
        lm.train()

        class _FakeTrainer:
            estimated_stepping_batches = 4
            global_step = 0

        lm._trainer = _FakeTrainer()
        opt_cfg = lm.configure_optimizers()
        optimizer = opt_cfg["optimizer"] if isinstance(opt_cfg, dict) else opt_cfg[0]
        lm.optimizers = lambda: optimizer

        # Manually initialize optimizer state for hubert params to test reset
        hubert_params = optimizer.param_groups[0]["params"]
        for p in hubert_params:
            if p.requires_grad:
                optimizer.state[p] = {'exp_avg': torch.zeros_like(p), 'exp_avg_sq': torch.zeros_like(p), 'step': 1}

        # Verify state was initialized
        assert any(optimizer.state[p] for p in hubert_params if p in optimizer.state)

        # Reset should clear the state
        lm._reset_hubert_lr_warmup()

        assert all(optimizer.state[p] == {} for p in hubert_params if p in optimizer.state)

    def test_ce_loss_aligned_forced_align_fallback(self):
        """Verify _compute_ce_loss_aligned falls back gracefully when forced_align raises."""
        phn_to_id = {"<BLANK>": 0, "<PAD>": 1, "<UNK>": 2, "P": 3, "B": 4}
        id_to_phn = {v: k for k, v in phn_to_id.items()}

        cfg = get_default_config()
        cfg.training.ablation_mode = "neural_only"
        cfg.model.use_learnable_constraint = False
        cfg.training.use_forced_alignment = True
        cfg.training.forced_alignment_fallback_warn = True

        model = _DummyNeuroSymbolicASR(num_classes=len(phn_to_id), time_steps=6)
        lm = DysarthriaASRLightning(
            model=model,
            config=cfg,
            phn_to_id=phn_to_id,
            id_to_phn=id_to_phn,
            class_weights=torch.ones(len(phn_to_id), dtype=torch.float32),
            articulatory_weights=None,
        )
        lm.log = lambda *args, **kwargs: None
        lm.train()

        # Force fallback by mocking TAF.forced_align to raise on every call
        import train as train_mod
        original_forced_align = train_mod.TAF.forced_align
        call_count = 0

        def _raising_forced_align(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Simulated forced_align failure")

        train_mod.TAF.forced_align = _raising_forced_align
        try:
            logits_neural = torch.randn(2, 6, 5, requires_grad=True)
            input_lengths = torch.tensor([6, 6])
            label_lengths = torch.tensor([6, 6])
            labels = torch.tensor([[3, 4, 3, 4, 3, 4], [4, 3, 4, 3, 4, 3]], dtype=torch.long)

            loss = lm._compute_ce_loss_aligned(logits_neural, labels, input_lengths, label_lengths)
            assert torch.isfinite(loss), "Loss should be finite even with fallback"
            assert loss.item() >= 0.0, "Loss should be non-negative"
            assert call_count > 0, "Mock was never called"
        finally:
            train_mod.TAF.forced_align = original_forced_align

    def test_prepare_step_filters_invalid_samples(self):
        """Verify _prepare_step correctly filters samples where label_length > input_length."""
        phn_to_id = {"<BLANK>": 0, "<PAD>": 1, "<UNK>": 2, "P": 3, "B": 4}
        id_to_phn = {v: k for k, v in phn_to_id.items()}

        cfg = get_default_config()
        cfg.training.ablation_mode = "neural_only"
        cfg.model.use_learnable_constraint = False
        cfg.training.use_forced_alignment = True

        model = _DummyNeuroSymbolicASR(num_classes=len(phn_to_id), time_steps=6)
        lm = DysarthriaASRLightning(
            model=model,
            config=cfg,
            phn_to_id=phn_to_id,
            id_to_phn=id_to_phn,
            class_weights=torch.ones(len(phn_to_id), dtype=torch.float32),
            articulatory_weights=None,
        )
        lm.log = lambda *args, **kwargs: None
        lm.train()

        # Batch: one valid (label_len=3 <= time=6) and one invalid (label_len=8 > time=6)
        logits = torch.randn(2, 6, 5, requires_grad=True)
        labels = torch.tensor([[3, 4, 3, -100, -100, -100, -100, -100],
                               [3, 4, 3, 4, 3, 4, 3, 4]], dtype=torch.long)
        outputs = {
            'logits_constrained': F.log_softmax(logits, dim=-1),
            'logits_neural': logits,
            'hidden_states': logits,
            'output_lengths': torch.tensor([6, 6], dtype=torch.long),
            'logits_manner': None,
            'logits_place': None,
            'logits_voice': None,
        }
        batch = {
            'labels': labels,
            'label_lengths': torch.tensor([3, 8], dtype=torch.long),
            'attention_mask': torch.ones(2, 1920, dtype=torch.long),
            'status': torch.tensor([1, 0], dtype=torch.long),
            'articulatory_labels': None,
        }

        prepared = lm._prepare_step(batch, outputs, log_invalid_frac=False)

        assert prepared is not None, "prepare_step should return a dict even with partial invalid"
        assert prepared['labels'].size(0) == 1, "Only 1 sample should survive filtering"
        assert prepared['input_lengths'].size(0) == 1
        assert prepared['label_lengths'].size(0) == 1
        labels_padded = prepared['labels'][0].tolist()
        active_labels = [v for v in labels_padded if v != -100]
        assert active_labels == [3, 4, 3], "Should keep the valid sample's labels (excluding padding)"
        assert prepared['input_lengths'][0].item() == 6, "Valid sample input_length=6"
        assert prepared['label_lengths'][0].item() == 3, "Valid sample label_length=3"
        assert prepared['valid_mask'].tolist() == [True, False], "Second sample should be masked"

    def test_prepare_step_all_invalid_returns_none(self):
        """When ALL samples have label_length > input_length, prepare_step returns None."""
        phn_to_id = {"<BLANK>": 0, "<PAD>": 1, "<UNK>": 2, "P": 3, "B": 4}
        id_to_phn = {v: k for k, v in phn_to_id.items()}

        cfg = get_default_config()
        cfg.training.ablation_mode = "neural_only"
        cfg.model.use_learnable_constraint = False

        model = _DummyNeuroSymbolicASR(num_classes=len(phn_to_id), time_steps=3)
        lm = DysarthriaASRLightning(
            model=model,
            config=cfg,
            phn_to_id=phn_to_id,
            id_to_phn=id_to_phn,
            class_weights=torch.ones(len(phn_to_id), dtype=torch.float32),
            articulatory_weights=None,
        )
        lm.log = lambda *args, **kwargs: None
        lm.train()

        logits = torch.randn(2, 3, 5)
        labels = torch.full((2, 4), 3, dtype=torch.long)
        outputs = {
            'logits_constrained': F.log_softmax(logits, dim=-1),
            'logits_neural': logits,
            'output_lengths': torch.tensor([3, 3], dtype=torch.long),
            'logits_manner': None,
            'logits_place': None,
            'logits_voice': None,
        }
        batch = {
            'labels': labels,
            'label_lengths': torch.tensor([4, 4], dtype=torch.long),
            'attention_mask': torch.ones(2, 1920, dtype=torch.long),
            'status': torch.tensor([1, 0], dtype=torch.long),
        }

        prepared = lm._prepare_step(batch, outputs, log_invalid_frac=False)
        assert prepared is None, "Should return None when all samples are invalid"


class TestEndToEndDataFlow:
    """End-to-end test: synthetic samples → collator → model → loss → backward."""

    def _make_sample(self, audio_len: int, label_len: int, speaker: str = "M01",
                     is_dysarthric: int = 1):
        """Build a minimal collator-compatible sample dict (no file I/O)."""
        return {
            "input_values": torch.zeros(audio_len),
            "labels": torch.zeros(label_len, dtype=torch.long) + 5,
            "articulatory_labels": {
                "manner": torch.zeros(label_len, dtype=torch.long),
                "place": torch.zeros(label_len, dtype=torch.long),
                "voice": torch.zeros(label_len, dtype=torch.long),
            },
            "metadata": {
                "is_dysarthric": torch.tensor(is_dysarthric, dtype=torch.long),
                "speaker": speaker,
                "transcript": "AH B C D E",
            },
        }

    def test_full_training_step_via_collator(self):
        """Synthetic samples → collator → model → loss → backward."""
        phn_to_id = {"<BLANK>": 0, "<PAD>": 1, "<UNK>": 2}
        for i in range(3, 10):
            phn_to_id[f"PHN{i}"] = i
        id_to_phn = {v: k for k, v in phn_to_id.items()}

        cfg = get_default_config()
        cfg.training.ablation_mode = "neural_only"
        cfg.model.use_learnable_constraint = False

        collator = NeuroSymbolicCollator(processor=None, pad_id=1, ctc_stride=320)
        samples = [
            self._make_sample(audio_len=3200, label_len=4, speaker="M01", is_dysarthric=1),
            self._make_sample(audio_len=4800, label_len=6, speaker="FC02", is_dysarthric=0),
        ]
        batch = collator(samples)

        model = _DummyNeuroSymbolicASR(num_classes=len(phn_to_id), time_steps=15, hidden_dim=16)
        lm = DysarthriaASRLightning(
            model=model,
            config=cfg,
            phn_to_id=phn_to_id,
            id_to_phn=id_to_phn,
            class_weights=torch.ones(len(phn_to_id), dtype=torch.float32),
            articulatory_weights=None,
        )
        lm.log = lambda *args, **kwargs: None  # stub logging
        lm.train()

        opt = torch.optim.SGD(lm.parameters(), lr=0.01)
        opt.zero_grad()

        loss = lm.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor), f"Expected Tensor, got {type(loss)}"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0.0, "Loss should be positive"

        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in lm.parameters() if p.grad is not None)
        assert grad_norm > 0.0, "Gradients should flow"

        opt.step()
        # After step, the model params should have changed
        # (not asserting specific values — just confirming no crash)
