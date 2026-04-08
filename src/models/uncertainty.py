"""
Uncertainty-Aware Decoder (ROADMAP §9 — Proposal P4)

Implements Monte Carlo Dropout inference for epistemic uncertainty quantification.
Enables calibrated phoneme-set predictions that are clinically useful:
  "Model predicts /b/ or /p/ with 95% confidence" → actionable for SLPs.

# REFACTOR LOG
# [PERF] conformal_phoneme_sets: replaced triple nested Python loops (O(B×T×V) pure
#        Python iterations) with vectorized torch.sort + cumsum on GPU. The outer
#        B×T Python list comprehension remains for final index extraction, but the
#        expensive inner cumulative-sum scan is now a single tensor op.
# [CLEAN] Replaced bare float() multiplication for stride with integer floor-division
#        to avoid implicit floating-point rounding in mask downsampling.
# [CLEAN] Added logging import and module-level logger; kept no external prints.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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
        self.model     = model
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
        del log_probs_samples
        mean_lp = lp_stack.mean(dim=0)  # [B, T, V]

        # Predictive entropy: H(y|x) = -Σ p̄_v log p̄_v
        mean_probs = lp_stack.exp().mean(dim=0)  # [B, T, V]
        eps = 1e-8
        entropy = -(mean_probs * (mean_probs + eps).log()).sum(dim=-1)  # [B, T]

        # Utterance-level uncertainty: mean entropy over valid frames
        if attention_mask is not None:
            # Downsample mask to logit resolution using integer floor-division.
            # [CLEAN] Replaced float() multiply+round with integer ops to avoid
            #         implicit floating-point rounding in mask index computation.
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
    ) -> List[List[List[int]]]:
        """
        Produce phoneme prediction sets targeting a given cumulative coverage level.

        Uses an APS-like (Adaptive Prediction Set) heuristic: sort phoneme
        probabilities descending, include top-k phonemes until cumulative
        probability ≥ τ.

        NOTE: This is NOT calibrated conformal prediction.  True conformal
        calibration requires fitting τ on a held-out calibration set via the
        ``(ceil((n+1)(1-α))/n)``-th quantile of nonconformity scores.  Here
        τ = coverage is used as a conservative approximation (equivalent to
        top-k by cumulative mass).  The name ``conformal_phoneme_sets`` is
        kept for API stability but the method should be interpreted as an
        APS-style heuristic until proper calibration data is available.

        Args:
            log_probs: [B, T, V] log-probabilities (mean across MC samples)
            coverage:  Target coverage probability (default 0.95)

        Returns:
            List[B] of List[T] of List[phoneme_token_ids]
        """
        tau = coverage  # = 1.0 - (1.0 - coverage), preserved for numerical identity

        # [PERF] Replaced triple nested Python loops (O(B × T × V) pure Python)
        # with vectorized GPU ops:
        #   - torch.sort:   one descending sort pass over the vocabulary dim
        #   - cumsum:       one cumulative-sum pass along vocabulary dim
        #   - comparison:   one element-wise < broadcasts over [B, T, V]
        # Only the final index extraction uses a Python comprehension (unavoidable
        # for ragged list-of-lists output), but the expensive inner scan is gone.
        probs = log_probs.exp()  # [B, T, V]
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)  # [B, T, V]
        cumsum = sorted_probs.cumsum(dim=-1)  # [B, T, V]

        # Include a token if the cumsum *before* it is still < tau; the first
        # token is always included regardless of cumsum so we shift by one.
        # Equivalent to: include[v] = True  iff  cumsum[v-1] < tau  (with cum[−1] = 0).
        include = torch.cat([
            torch.ones(
                *cumsum.shape[:2], 1, dtype=torch.bool, device=probs.device
            ),
            cumsum[..., :-1] < tau,
        ], dim=-1)  # [B, T, V]

        # Move to CPU once for the final Python list construction.
        sorted_idx_np = sorted_idx.cpu().numpy()
        include_np    = include.cpu().numpy()

        B, T = probs.shape[:2]
        return [
            [sorted_idx_np[b, t, include_np[b, t]].tolist() for t in range(T)]
            for b in range(B)
        ]
