"""
Research-Grade Loss Functions for Dysarthric Speech Recognition

Three novel loss functions implementing the audit proposals:

1. OrdinalContrastiveLoss   — Proposal P1: severity-aware representation learning
2. BlankPriorKLLoss         — Fix for CTC insertion pathology (56× I/D ratio)
3. SymbolicKLLoss           — Proposal P2: KL anchor for learnable constraint matrix

All losses are additive and flag-controlled; existing CTC+CE baseline is unaffected
when weights are set to zero.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalContrastiveLoss(nn.Module):
    """
    Ordinal Contrastive Loss for Severity-Graded Representations (Proposal P1).

    Hypothesis: utterance embeddings from speakers of similar severity should be
    closer in representation space; those from very different severity levels should
    be farther apart. The margin is proportional to the severity difference.

    This replaces the binary (dysarthric / control) distinction with a soft, graded
    objective — enabling the severity adapter's β to become genuinely adaptive.

    Formula:
        For each pair (i, j):
            margin_ij  = margin_per_level * |sev_i − sev_j|
            target_ij  = +1  if |sev_i − sev_j| == 0  (same severity → pull)
                         -1  otherwise                   (different → push)
            loss_ij    = ReLU(margin_ij − target_ij * cosine_sim(z_i, z_j))

    Args:
        margin_per_level: Base margin multiplied by severity difference. A value of
            0.3 means pairs 5 severity-units apart require 1.5 cosine-distance.
        temperature: Optional softmax temperature for scaled cosine similarity.
    """

    def __init__(self, margin_per_level: float = 0.3, temperature: float = 1.0):
        super().__init__()
        self.margin_per_level = margin_per_level
        self.temperature = temperature

    def forward(
        self,
        hidden_states: torch.Tensor,
        severity: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: HuBERT frame representations [B, T, 768]
            severity:      Continuous severity scores [B], range [0, 5]
            attention_mask: Valid-frame mask [B, T] (1 = valid, 0 = padding)

        Returns:
            Scalar contrastive loss.
        """
        # ── Mean-pool over valid frames → utterance embedding ────────────────
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            z = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            z = hidden_states.mean(dim=1)  # [B, 768]

        z = F.normalize(z, dim=-1)  # Unit-sphere projection

        # ── Pairwise cosine similarity ────────────────────────────────────────
        sim = torch.matmul(z, z.T) / self.temperature  # [B, B]

        # ── Ordinal margin: proportional to severity distance ─────────────────
        sev_diff = (severity.unsqueeze(0) - severity.unsqueeze(1)).abs()  # [B, B]
        margin = self.margin_per_level * sev_diff  # [B, B]

        # target: +1 if same severity bucket, -1 otherwise
        target = torch.where(sev_diff == 0, torch.ones_like(sim), -torch.ones_like(sim))

        # ── Loss: contrastive hinge ───────────────────────────────────────────
        loss_mat = F.relu(margin - target * sim)

        # Exclude self-pairs on diagonal
        diag_mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        valid_pairs = loss_mat[diag_mask]
        # Guard: batch_size=1 after valid_mask → no pairs → return zero loss (not NaN)
        if valid_pairs.numel() == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)
        loss = valid_pairs.mean()
        return loss


class BlankPriorKLLoss(nn.Module):
    """
    Blank-Prior KL Regularisation — CTC Insertion Fix (Audit Issue R3).

    Problem: The frame-level CE loss applies NLLLoss on padded labels (fill=-100),
    which never penalises non-blank emissions on silence frames. This causes the
    56× insertion/deletion ratio (21,290 insertions vs 376 deletions).

    Solution: Regularise the mean blank probability across all valid frames toward
    a target that matches typical CTC blank-emission rates (default 0.85 — empirically
    observed in HuBERT-based CTC systems; 0.30 would actively push blank probability
    DOWN and worsen insertion bias). Uses KL divergence
    from the posterior mean over the 2-class Bernoulli (blank / non-blank).

    Formula:
        p̄_blank = mean(softmax(log_probs)[:, :, blank_id])   over valid frames
        loss = KL(Bernoulli(p̄_blank) || Bernoulli(target_prob))
             = p̄_blank * log(p̄_blank / target) + (1-p̄_blank) * log((1-p̄_blank)/(1-target))

    Args:
        blank_id:    Index of the CTC blank token in the vocabulary.
        target_prob: Target mean blank probability. Default 0.85
            (empirically matches HuBERT-based CTC blank emission rates;
            lower values would push blank probability down and worsen
            insertion bias).
    """

    def __init__(self, blank_id: int = 0, target_prob: float = 0.85):
        super().__init__()
        self.blank_id = blank_id
        self.register_buffer("target_prob", torch.tensor(target_prob, dtype=torch.float32))

    def forward(
        self,
        log_probs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            log_probs:      Log-probabilities [B, T, V] (output of log_softmax)
            attention_mask: Valid-frame mask [B, T] (1 = valid, 0 = padding)

        Returns:
            Scalar KL loss ≥ 0.
        """
        probs = log_probs.exp()  # [B, T, V]
        blank_probs = probs[:, :, self.blank_id]  # [B, T]

        if attention_mask is not None:
            mask = attention_mask.float()  # [B, T]
            total = mask.sum().clamp_min(1.0)
            p_blank_mean = (blank_probs * mask).sum() / total
        else:
            p_blank_mean = blank_probs.mean()

        # Clamp to numerical stability
        eps = 1e-7
        p = p_blank_mean.clamp(eps, 1.0 - eps)
        q = self.target_prob.clamp(eps, 1.0 - eps)

        # KL(Bernoulli(p) || Bernoulli(q))
        kl = p * (p / q).log() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).log()
        return kl.mean()

    def mean_blank_prob(
        self,
        log_probs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mean blank probability (for monitoring, not gradients)."""
        with torch.no_grad():
            probs = log_probs.exp()
            blank_probs = probs[:, :, self.blank_id]
            if attention_mask is not None:
                mask = attention_mask.float()
                return (blank_probs * mask).sum() / mask.sum().clamp_min(1.0)
            return blank_probs.mean()


class SymbolicKLLoss(nn.Module):
    """
    Symbolic KL Anchor Loss (Proposal P2 — Learnable Constraint Matrix).

    Prevents the learned constraint matrix C from diverging arbitrarily from the
    articulatory/phonological prior C_static. Each row of C is a probability
    distribution over target phonemes; the KL penalty keeps each row near the
    corresponding row of the prior.

    Formula:
        loss = (1/V) Σ_i KL(C_learned[i] || C_static[i])

    Args:
        static_matrix: Static symbolic prior [V, V], row-normalised.
    """

    def __init__(self, static_matrix: torch.Tensor):
        super().__init__()
        # Store as log-probability reference (not a parameter)
        self.register_buffer("log_prior", static_matrix.clamp(1e-8).log())

    def forward(self, logit_C: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logit_C: Raw (pre-softmax) learned constraint matrix [V, V]

        Returns:
            Scalar mean-row KL divergence ≥ 0.
        """
        log_learned = F.log_softmax(logit_C, dim=-1)  # [V, V]
        # KL(learned || prior): F.kl_div(input=log_prior, target=log_learned, log_target=True)
        # F.kl_div(log_q, log_p, log_target=True) = KL(p || q), so to get KL(learned || prior)
        # we pass input=log_prior, target=log_learned.
        # Use reduction="sum" then divide by V (number of rows) to get the
        # mean per-row KL divergence, matching the docstring formula:
        #   loss = (1/V) Σ_i KL(C_learned[i] || C_static[i])
        # "batchmean" is mathematically identical for a 2-D [V,V] tensor
        # (divides by V, the first dimension), but "sum/size(0)" makes the
        # normalisation explicit and avoids confusion if the tensor shape ever
        # changes (e.g. a batched C in future work).
        # B1 fix: explicit sum/V normalisation
        kl = F.kl_div(
            self.log_prior,
            log_learned,
            reduction="sum",
            log_target=True,
        ) / log_learned.size(0)
        return kl
