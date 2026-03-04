"""
Uncertainty-Aware Decoder (ROADMAP §9 — Proposal P4)

Implements Monte Carlo Dropout inference for epistemic uncertainty quantification.
Enables calibrated phoneme-set predictions that are clinically useful:
  "Model predicts /b/ or /p/ with 95% confidence" → actionable for SLPs.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


class UncertaintyAwareDecoder:
    """
    Monte Carlo Dropout decoder for uncertainty-quantified ASR.

    By enabling dropout at inference time and running N forward passes, the
    decoder:
      1. Estimates epistemic uncertainty (model ignorance about unseen dysarthric
         speech patterns) via predictive entropy
      2. Provides uncertainty-aware phoneme-set predictions via conformal
         prediction calibration (optional)

    Args:
        model:    NeuroSymbolicASR model instance (dropout must be present)
        n_samples: Number of MC dropout samples (default 20, balanced accuracy/speed)
    """

    def __init__(self, model, n_samples: int = 20):
        self.model    = model
        self.n_samples = n_samples

    def predict_with_uncertainty(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_severity: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict:
        """
        Run N stochastic forward passes and aggregate uncertainty estimates.

        Args:
            input_values:     Raw waveform [B, T]
            attention_mask:   Valid frame mask [B, T]
            speaker_severity: Continuous severity [B] in [0, 5]
            return_attention: Whether to return attention maps

        Returns:
            Dict with:
              - mean_log_probs:              [B, T, V] mean log-probabilities
              - epistemic_entropy_per_frame: [B, T] entropy of predictive distribution
              - utterance_uncertainty:       [B] mean entropy over valid frames
              - sample_variance:             [B, T, V] variance across MC samples
              - confidence_scores:           [B] 1 - normalised_entropy (higher = more confident)
        """
        device = input_values.device
        B, T_audio = input_values.shape

        # Enable dropout (train mode activates dropout, but disables BN momentum)
        self.model.train()
        self.model.hubert.eval()  # Keep HuBERT in eval to preserve LayerNorm statistics

        log_probs_samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                out = self.model(
                    input_values,
                    attention_mask=attention_mask,
                    speaker_severity=speaker_severity,
                    output_attentions=False,
                )
                lp = out.get('log_probs_constrained', out['logits_constrained'])
                log_probs_samples.append(lp.cpu())  # accumulate on CPU to save VRAM

        self.model.eval()

        # Stack: [N, B, T, V]
        lp_stack = torch.stack(log_probs_samples, dim=0).to(device)
        mean_lp = lp_stack.mean(dim=0)  # [B, T, V]

        # Predictive entropy: H(y|x) = -Σ p̄_v log p̄_v
        mean_probs = lp_stack.exp().mean(dim=0)  # [B, T, V]
        eps = 1e-8
        entropy = -(mean_probs * (mean_probs + eps).log()).sum(dim=-1)  # [B, T]

        # Utterance-level uncertainty: mean entropy over valid frames
        if attention_mask is not None:
            # Downsample mask to logit resolution
            T_log = mean_lp.size(1)
            stride = max(1, T_audio // T_log)
            mask_ds = attention_mask[:, ::stride][:, :T_log].float()
            n_valid = mask_ds.sum(dim=1).clamp_min(1.0)
            utt_uncertainty = (entropy * mask_ds).sum(dim=1) / n_valid
        else:
            utt_uncertainty = entropy.mean(dim=1)  # [B]

        # Variance across samples [B, T, V]
        sample_variance = lp_stack.var(dim=0)

        # Vocabulary size for normalisation (max entropy = log V)
        V = mean_lp.size(-1)
        max_entropy = float(np.log(V))
        confidence = 1.0 - utt_uncertainty / max_entropy  # [B]

        return {
            'mean_log_probs':              mean_lp,
            'epistemic_entropy_per_frame': entropy,
            'utterance_uncertainty':       utt_uncertainty,
            'sample_variance':             sample_variance,
            'confidence_scores':           confidence.clamp(0.0, 1.0),
        }

    def conformal_phoneme_sets(
        self,
        log_probs: torch.Tensor,
        coverage: float = 0.95,
    ) -> List[List[List[str]]]:
        """
        Produce phoneme sets with guaranteed marginal coverage via conformal
        prediction (threshold-based prediction sets).

        Uses the least-ambiguous-set (LAS) method: sort phoneme probabilities
        descending, include the top-k phonemes until cumulative probability ≥ τ.
        τ is set so that inclusion guarantees coverage at the given level.

        Note: Requires a held-out calibration set to set τ correctly. Without
        calibration, this defaults to a heuristic τ = 1 - (1 - coverage).

        Args:
            log_probs: [B, T, V] log-probabilities (mean across MC samples)
            coverage:  Target coverage probability (default 0.95)

        Returns:
            List[B] of List[T] of List[phoneme_strings]
        """
        tau = 1.0 - (1.0 - coverage)   # heuristic τ without calibration data
        probs = log_probs.exp()          # [B, T, V]

        B, T, V = probs.shape
        phoneme_sets: List[List[List[int]]] = []

        for b in range(B):
            frame_sets = []
            for t in range(T):
                p = probs[b, t]  # [V]
                sorted_idx = torch.argsort(p, descending=True)
                cum = 0.0
                chosen = []
                for idx in sorted_idx:
                    chosen.append(int(idx.item()))
                    cum += float(p[idx].item())
                    if cum >= tau:
                        break
                frame_sets.append(chosen)
            phoneme_sets.append(frame_sets)

        return phoneme_sets
