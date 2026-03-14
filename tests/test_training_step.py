"""
Integration test for one full training step.

Covers: forward -> compute_loss -> backward -> optimizer step on a synthetic batch.
Uses a lightweight dummy model to avoid HuBERT downloads in CI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import get_default_config
from train import DysarthriaASRLightning


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
        model.hubert = model.proj_in
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

        batch = {
            "input_values": torch.randn(2, 1920),
            "attention_mask": torch.ones(2, 1920, dtype=torch.long),
            "labels": torch.tensor([[3, 4, 3, 4], [4, 3, 4, 3]], dtype=torch.long),
            "label_lengths": torch.tensor([4, 4], dtype=torch.long),
            "status": torch.tensor([1, 0], dtype=torch.long),
        }

        loss = lm.training_step(batch, batch_idx=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        hubert_params = optimizer.param_groups[0]["params"]
        assert any(optimizer.state[p] for p in hubert_params if p in optimizer.state)

        for p in hubert_params:
            p.requires_grad = True
        lm._reset_hubert_lr_warmup()

        assert all(optimizer.state[p] == {} for p in hubert_params if p in optimizer.state)

        loss2 = lm.training_step(batch, batch_idx=1)
        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()

        assert all("exp_avg" in optimizer.state[p] for p in hubert_params if p in optimizer.state)
