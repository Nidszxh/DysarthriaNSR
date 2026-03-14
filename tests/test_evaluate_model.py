"""
Integration test for evaluate_model on a synthetic batch.

Covers: model forward + greedy decoding + metric aggregation + JSON artifact write.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate import evaluate_model


class _DummyEvalModel(nn.Module):
    """Small model stub with evaluate_model-compatible outputs."""

    def __init__(self, num_classes: int, time_steps: int = 6):
        super().__init__()
        self.time_steps = time_steps
        self.num_classes = num_classes

    def forward(
        self,
        input_values,
        attention_mask=None,
        speaker_severity=None,
        output_attentions=False,
        ablation_mode="full",
    ):
        batch_size = input_values.size(0)
        device = input_values.device

        # Deterministic, non-special-token predictions so confusion plots always
        # have content in this integration test.
        logits_neural = torch.full(
            (batch_size, self.time_steps, self.num_classes),
            -6.0,
            dtype=torch.float32,
            device=device,
        )
        for b in range(batch_size):
            for t in range(self.time_steps):
                token_id = 3 if (b + t) % 2 == 0 else 4
                logits_neural[b, t, token_id] = 6.0

        if speaker_severity is not None:
            logits_neural = logits_neural + (speaker_severity.view(batch_size, 1, 1) / 100.0)

        log_probs = F.log_softmax(logits_neural, dim=-1)

        return {
            "logits_constrained": log_probs,
            "logits_neural": logits_neural,
            "output_lengths": torch.full((batch_size,), self.time_steps, dtype=torch.long, device=device),
            "beta": torch.tensor(0.1, device=device),
            "logits_manner": None,
            "logits_place": None,
            "logits_voice": None,
        }


class TestEvaluateModelIntegration:
    def test_evaluate_model_synthetic_batch(self, tmp_path: Path):
        phn_to_id = {"<BLANK>": 0, "<PAD>": 1, "<UNK>": 2, "P": 3, "B": 4}
        id_to_phn = {v: k for k, v in phn_to_id.items()}

        model = _DummyEvalModel(num_classes=len(phn_to_id), time_steps=6)
        model.eval()

        batch = {
            "input_values": torch.randn(2, 1920),
            "attention_mask": torch.ones(2, 1920, dtype=torch.long),
            "labels": torch.tensor([[3, 4, 3, -100], [4, 3, -100, -100]], dtype=torch.long),
            "status": torch.tensor([1, 0], dtype=torch.long),
            "speakers": ["M01", "MC01"],
        }

        dataloader = [batch]
        results = evaluate_model(
            model=model,
            dataloader=dataloader,
            device="cpu",
            phn_to_id=phn_to_id,
            id_to_phn=id_to_phn,
            results_dir=tmp_path,
            use_beam_search=False,
            generate_explanations=False,
            compute_uncertainty=False,
            ablation_mode="neural_only",
        )

        assert "overall" in results
        assert "avg_per" in results
        assert results["overall"]["n_samples"] == 2
        assert (tmp_path / "evaluation_results.json").exists()
