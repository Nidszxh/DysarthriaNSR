"""
Neuro-Symbolic ASR Model for Dysarthric Speech Recognition

Architecture upgrades (February 2026 research audit):
  - LearnableConstraintMatrix: End-to-end learnable C with KL-anchor to symbolic prior
  - SeverityAdapter: Cross-attention adapter conditioning on continuous severity score
  - output_attentions forwarding: Route HuBERT attention maps for explainability

Components:
    1. ArticulatoryFeatureEncoder  — Encodes phonemes into articulatory feature vectors
    2. LearnableConstraintMatrix   — Parameterized constraint matrix (Proposal P2)
    3. SymbolicConstraintLayer     — Fusion layer: P_final = β·(P_n@C) + (1-β)·P_n
    4. SeverityAdapter             — Cross-attention severity conditioning (Proposal P3)
    5. PhonemeClassifier           — 768→512→|V| phoneme prediction head
    6. NeuroSymbolicASR            — Full model combining all components

# REFACTOR LOG
# [PERF] HuBERT forward runs inside eval()+no_grad() context when all encoder
#        parameters are frozen. Eliminates intermediate activation graph construction
#        for frozen params (~15-25% VRAM reduction at batch_size=8 on RTX 4060)
#        and keeps LayerNorm running statistics from being perturbed by dropout.
# [PERF] SpecAugmentLayer mask tensors now created directly on the target device
#        (device=x.device passed to torch.randint) — avoids a CPU→GPU transfer
#        for each batch during training.
# [CLEAN] Replaced print() calls throughout __init__ and freeze/unfreeze methods
#         with logging.getLogger(__name__).info() for structured log control.
"""

import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel

logger = logging.getLogger(__name__)

try:
    from src.utils.config import ModelConfig, SymbolicConfig, normalize_phoneme
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.config import ModelConfig, SymbolicConfig, normalize_phoneme

try:
    from src.explainability.rule_tracker import SymbolicRuleTracker
except ImportError:
    SymbolicRuleTracker = None  # type: ignore


# Articulatory Feature Encoder  (unchanged — forms the symbolic prior)

class ArticulatoryFeatureEncoder:
    """
    Encodes phonemes into articulatory feature vectors for symbolic reasoning.

    Features based on phonetic theory:
    - Manner of articulation: How airflow is obstructed (stop, fricative, nasal, etc.)
    - Place of articulation:  Where obstruction occurs (bilabial, alveolar, velar, etc.)
    - Voicing:                Vocal fold vibration (voiced vs voiceless)
    """

    PHONEME_FEATURES = {
        # STOPS
        'P':  {'manner': 'stop', 'place': 'bilabial',      'voice': 'voiceless'},
        'B':  {'manner': 'stop', 'place': 'bilabial',      'voice': 'voiced'},
        'T':  {'manner': 'stop', 'place': 'alveolar',      'voice': 'voiceless'},
        'D':  {'manner': 'stop', 'place': 'alveolar',      'voice': 'voiced'},
        'K':  {'manner': 'stop', 'place': 'velar',         'voice': 'voiceless'},
        'G':  {'manner': 'stop', 'place': 'velar',         'voice': 'voiced'},
        # FRICATIVES
        'F':  {'manner': 'fricative', 'place': 'labiodental',  'voice': 'voiceless'},
        'V':  {'manner': 'fricative', 'place': 'labiodental',  'voice': 'voiced'},
        'TH': {'manner': 'fricative', 'place': 'dental',       'voice': 'voiceless'},
        'DH': {'manner': 'fricative', 'place': 'dental',       'voice': 'voiced'},
        'S':  {'manner': 'fricative', 'place': 'alveolar',     'voice': 'voiceless'},
        'Z':  {'manner': 'fricative', 'place': 'alveolar',     'voice': 'voiced'},
        'SH': {'manner': 'fricative', 'place': 'postalveolar', 'voice': 'voiceless'},
        'ZH': {'manner': 'fricative', 'place': 'postalveolar', 'voice': 'voiced'},
        'HH': {'manner': 'fricative', 'place': 'glottal',      'voice': 'voiceless'},
        # AFFRICATES
        'CH': {'manner': 'affricate', 'place': 'postalveolar', 'voice': 'voiceless'},
        'JH': {'manner': 'affricate', 'place': 'postalveolar', 'voice': 'voiced'},
        # NASALS
        'M':  {'manner': 'nasal', 'place': 'bilabial', 'voice': 'voiced'},
        'N':  {'manner': 'nasal', 'place': 'alveolar', 'voice': 'voiced'},
        'NG': {'manner': 'nasal', 'place': 'velar',    'voice': 'voiced'},
        # LIQUIDS
        'L':  {'manner': 'liquid', 'place': 'alveolar', 'voice': 'voiced'},
        'R':  {'manner': 'liquid', 'place': 'alveolar', 'voice': 'voiced'},
        # GLIDES
        'W':  {'manner': 'glide', 'place': 'labio-velar', 'voice': 'voiced'},
        'Y':  {'manner': 'glide', 'place': 'palatal',     'voice': 'voiced'},
        # VOWELS
        'IY': {'manner': 'vowel', 'place': 'front',   'voice': 'voiced', 'height': 'high'},
        'IH': {'manner': 'vowel', 'place': 'front',   'voice': 'voiced', 'height': 'high'},
        'EH': {'manner': 'vowel', 'place': 'front',   'voice': 'voiced', 'height': 'mid'},
        'EY': {'manner': 'vowel', 'place': 'front',   'voice': 'voiced', 'height': 'mid'},
        'AE': {'manner': 'vowel', 'place': 'front',   'voice': 'voiced', 'height': 'low'},
        'AA': {'manner': 'vowel', 'place': 'back',    'voice': 'voiced', 'height': 'low'},
        'AO': {'manner': 'vowel', 'place': 'back',    'voice': 'voiced', 'height': 'mid'},
        'OW': {'manner': 'vowel', 'place': 'back',    'voice': 'voiced', 'height': 'mid'},
        'UH': {'manner': 'vowel', 'place': 'back',    'voice': 'voiced', 'height': 'high'},
        'UW': {'manner': 'vowel', 'place': 'back',    'voice': 'voiced', 'height': 'high'},
        'AH': {'manner': 'vowel', 'place': 'central', 'voice': 'voiced', 'height': 'mid'},
        'ER': {'manner': 'vowel', 'place': 'central', 'voice': 'voiced', 'height': 'mid'},
        'AX': {'manner': 'vowel', 'place': 'central', 'voice': 'voiced', 'height': 'mid'},
        # DIPHTHONGS
        'AY': {'manner': 'diphthong', 'place': 'front', 'voice': 'voiced'},
        'AW': {'manner': 'diphthong', 'place': 'back',  'voice': 'voiced'},
        'OY': {'manner': 'diphthong', 'place': 'back',  'voice': 'voiced'},
    }

    @classmethod
    def compute_distance(
        cls,
        ph1: str,
        ph2: str,
        weights: Dict[str, float],
        decay_factor: float = 3.0,
    ) -> float:
        """
        Compute articulatory similarity between two phonemes.

        distance   = sqrt(Σ w_i * (f1_i ≠ f2_i)^2)
        similarity = exp(-decay_factor * distance)

        Returns similarity in [0, 1].
        """
        if ph1 not in cls.PHONEME_FEATURES or ph2 not in cls.PHONEME_FEATURES:
            return 0.0

        f1, f2 = cls.PHONEME_FEATURES[ph1], cls.PHONEME_FEATURES[ph2]
        manner_dist = 0.0 if f1['manner'] == f2['manner'] else 1.0
        place_dist  = 0.0 if f1['place']  == f2['place']  else 1.0
        voice_dist  = 0.0 if f1['voice']  == f2['voice']  else 1.0

        total_dist = np.sqrt(
            weights['manner'] ** 2 * manner_dist +
            weights['place']  ** 2 * place_dist  +
            weights['voice']  ** 2 * voice_dist
        )
        return float(np.exp(-decay_factor * total_dist))


# Learnable Constraint Matrix (Proposal P2)

class LearnableConstraintMatrix(nn.Module):
    """
    End-to-end Learnable Phoneme Constraint Matrix (Audit Proposal P2).

    Initialised from the articulatory symbolic prior C_static, then updated
    jointly with the rest of the model. A KL-anchor loss (SymbolicKLLoss) keeps
    the learned matrix near the prior, preventing arbitrary drift.

    Row i of C represents the distribution over target phonemes given source
    phoneme i — analogous to a soft confusion matrix that the model can refine.

    The matrix is kept row-stochastic via softmax parameterisation:
        C[i] = softmax(logit_C[i])

    Args:
        num_phonemes: Vocabulary size |V|
        init_matrix:  Static prior C [|V|, |V|], row-normalised probabilities
    """

    def __init__(self, num_phonemes: int, init_matrix: torch.Tensor, init_temperature: float = 0.5):
        super().__init__()
        # Temperature-sharpened log initialisation (§3.2 fix).
        # Plain log(C_static) passed through softmax produces a much flatter
        # distribution than C_static because softmax re-normalises.  Dividing
        # by init_temperature < 1.0 preserves the peakedness of the prior so
        # that softmax(logit_C / T) ≈ C_static at epoch 0.
        log_init = init_matrix.clamp(1e-8).log() / init_temperature
        self.logit_C = nn.Parameter(log_init)

    @property
    def C(self) -> torch.Tensor:
        """Row-stochastic constraint matrix [|V|, |V|]."""
        return F.softmax(self.logit_C, dim=-1)

    def forward(self, P_neural: torch.Tensor) -> torch.Tensor:
        """
        Apply constraint: P_constrained = P_neural @ C

        Args:
            P_neural: [B, T, |V|] probabilities

        Returns:
            [B, T, |V|] constrained probabilities (row-normalised)
        """
        P_c = torch.matmul(P_neural, self.C)
        return P_c / P_c.sum(dim=-1, keepdim=True).clamp_min(1e-8)


# Symbolic Constraint Layer  (refactored to use LearnableConstraintMatrix)

class SymbolicConstraintLayer(nn.Module):
    """
    Neuro-symbolic fusion layer applying dysarthric phoneme confusion rules.

    Fusion formula:
        P_final = β · P_constrained + (1-β) · P_neural
    where β adapts to speaker severity (continuous [0, 5]) and is learnable.

    Supports both:
      - Static constraint matrix (original behaviour)
      - Learnable constraint matrix (Proposal P2)
    """

    def __init__(
        self,
        num_phonemes: int,
        phn_to_id: Dict[str, int],
        id_to_phn: Dict[int, str],
        symbolic_config: SymbolicConfig,
        constraint_weight: float = 0.05,
        learnable: bool = True,
        use_learnable_matrix: bool = True,
    ):
        super().__init__()
        self.num_phonemes = num_phonemes
        self.phn_to_id = phn_to_id
        self.id_to_phn = id_to_phn
        self.config = symbolic_config
        self.use_learnable_matrix = use_learnable_matrix

        # Learnable fusion weight β — keep as a trainable parameter so that
        # the blending factor can be tuned even when the constraint matrix
        # itself is static. Tests expect `beta` to be a Parameter in static
        # mode as well.
        self.beta = nn.Parameter(torch.tensor(constraint_weight))

        # Build static prior matrix (always stored as reference)
        static_C = self._build_static_matrix()
        self.register_buffer('static_constraint_matrix', static_C)

        # Learnable wrapper (Proposal P2) or static buffer
        if use_learnable_matrix:
            self.learnable_matrix = LearnableConstraintMatrix(num_phonemes, static_C)
        else:
            self.learnable_matrix = None

        # Rule activation tracking (for explainability)
        self.rule_activations = [] if symbolic_config.track_rule_activations else None

        # SymbolicRuleTracker: structured logging API used by explainability module
        # (ROADMAP §6.2 — log_rule_activation called from _track_activations)
        self.rule_tracker = None
        if symbolic_config.track_rule_activations and SymbolicRuleTracker is not None:
            self.rule_tracker = SymbolicRuleTracker(
                min_confidence=symbolic_config.min_rule_confidence
            )

    def _build_static_matrix(self) -> torch.Tensor:
        """Build the symbolic prior constraint matrix from articulatory knowledge."""
        C = torch.eye(self.num_phonemes)
        special_tokens = {'<BLANK>', '<PAD>', '<UNK>'}
        weights = {
            'manner': self.config.manner_weight,
            'place':  self.config.place_weight,
            'voice':  self.config.voice_weight,
        }

        for i in range(self.num_phonemes):
            for j in range(self.num_phonemes):
                if i == j:
                    continue
                ph_i = normalize_phoneme(self.id_to_phn.get(i, '<UNK>'))
                ph_j = normalize_phoneme(self.id_to_phn.get(j, '<UNK>'))

                if ph_i in special_tokens or ph_j in special_tokens:
                    C[i, j] = 0.0
                    continue

                if (ph_i, ph_j) in self.config.substitution_rules:
                    C[i, j] = self.config.substitution_rules[(ph_i, ph_j)]
                else:
                    C[i, j] = ArticulatoryFeatureEncoder.compute_distance(
                        ph_i, ph_j, weights,
                        decay_factor=self.config.distance_decay_factor,
                    )

        row_sums = C.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return C / row_sums

    def forward(
        self,
        logits: torch.Tensor,
        speaker_severity: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply symbolic constraints to neural phoneme predictions.

        Args:
            logits:          Neural logits [B, T, |V|]
            speaker_severity: Continuous severity [B] in [0, 5]

        Returns:
            Dict with log_probs, beta, P_neural, P_constrained, rule_shift
        """
        P_neural = F.softmax(logits, dim=-1)

        # Apply constraint matrix (learnable or static)
        if self.use_learnable_matrix and self.learnable_matrix is not None:
            P_constrained = self.learnable_matrix(P_neural)
        else:
            P_constrained = torch.matmul(P_neural, self.static_constraint_matrix)
            P_constrained = P_constrained / P_constrained.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Adaptive β based on severity
        if speaker_severity is not None:
            beta_adaptive = self._compute_adaptive_beta(speaker_severity)
        else:
            beta_adaptive = self.beta

        # Weighted fusion
        P_final = beta_adaptive * P_constrained + (1 - beta_adaptive) * P_neural
        P_final = P_final / P_final.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Phase 3 Fix B: blank-frame constraint masking.
        # ~85% of CTC frames are blank-dominant.  Applying C to those frames only
        # amplifies the blank posterior through the row of C that maps blank→blank,
        # degrading PER.  Only apply the constrained distribution where the
        # neural model is not already blank-dominant.
        blank_threshold = getattr(self.config, 'blank_constraint_threshold', 0.5)
        non_blank_mask = (P_neural[:, :, 0] < blank_threshold).unsqueeze(-1)  # [B, T, 1]
        P_final = torch.where(non_blank_mask, P_final, P_neural)
        P_final = P_final / P_final.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Track rule activations (inference only)
        shift_metric = None
        if self.rule_activations is not None and not self.training:
            beta_scalar = (
                float(beta_adaptive.mean().item())
                if isinstance(beta_adaptive, torch.Tensor)
                else float(beta_adaptive)
            )
            shift_metric = self._track_activations(P_neural, P_final, beta_value=beta_scalar)

        log_probs = torch.log(P_final.clamp_min(1e-6))  # B4a fix: +1e-12 underflows to 0 in BF16; clamp_min(1e-6) is always representable

        return {
            'log_probs': log_probs,
            'beta': beta_adaptive.mean() if isinstance(beta_adaptive, torch.Tensor) and beta_adaptive.numel() > 1 else self.beta,
            'P_neural': P_neural,
            'P_constrained': P_constrained,
            'rule_shift': shift_metric,
        }

    def _compute_adaptive_beta(self, speaker_severity: torch.Tensor) -> torch.Tensor:
        """
        β_adaptive = clamp(β_base + slope * severity/5.0, 0.0, 0.8)
        Higher severity → heavier reliance on symbolic constraints.
        """
        sev_norm = speaker_severity.float() / 5.0
        beta_adaptive = self.beta + self.config.severity_beta_slope * sev_norm
        return torch.clamp(beta_adaptive, 0.0, 0.8).view(-1, 1, 1)

    def _track_activations(
        self,
        P_neural: torch.Tensor,
        P_final: torch.Tensor,
        beta_value: float = 0.5,
    ) -> Optional[torch.Tensor]:
        """Track probability shifts and log to SymbolicRuleTracker for explainability."""
        if self.rule_activations is None:
            return None

        with torch.no_grad():  # B4b fix: no_grad prevents building unused gradient graph during inference
            shift = (P_final - P_neural).abs().sum(dim=-1).mean().detach()

            if len(self.rule_activations) < 10000:
                neural_preds = torch.argmax(P_neural, dim=-1)
                final_preds = torch.argmax(P_final, dim=-1)
                changed = (neural_preds != final_preds)

                if changed.any():
                    indices = torch.nonzero(changed, as_tuple=False)
                    for b, t in indices[:100].tolist():
                        n_id = int(neural_preds[b, t].item())
                        f_id = int(final_preds[b, t].item())
                        n_phn = self.id_to_phn.get(n_id, '<UNK>')
                        f_phn = self.id_to_phn.get(f_id, '<UNK>')
                        rule_id = f"{n_phn}->{f_phn}"

                        # Log to SymbolicRuleTracker (ROADMAP §6.2)
                        if self.rule_tracker is not None:
                            self.rule_tracker.log_rule_activation(
                                rule_id=rule_id,
                                input_phoneme=n_phn,
                                output_phoneme=f_phn,
                                blend_weight=beta_value,
                                prediction_confidence=float(P_final[b, t, f_id].item()),
                            )

                        self.rule_activations.append({
                            'neural_pred': n_phn,
                            'final_pred':  f_phn,
                            'shift': float(shift.item()),
                        })

        return shift

    def get_rule_statistics(self) -> Dict:
        """Return rule activation statistics from the SymbolicRuleTracker.

        Prefers the structured SymbolicRuleTracker output; falls back to the
        legacy list-based statistics when the tracker is unavailable.
        """
        # Prefer structured tracker output (ROADMAP §6.2)
        if self.rule_tracker is not None:
            return self.rule_tracker.generate_explanation()

        # Legacy fallback: use self.rule_activations list
        if not self.rule_activations:
            return {'total_activations': 0, 'unique_rules': set(), 'top_rules': [], 'avg_shift': 0.0}

        from collections import Counter
        subs = Counter(
            (a['neural_pred'], a['final_pred'])
            for a in self.rule_activations
            if 'neural_pred' in a
        )
        shifts = [a.get('shift', 0.0) for a in self.rule_activations if 'shift' in a]
        return {
            'total_activations': len(self.rule_activations),
            'unique_rules': set(subs.keys()),
            'top_rules': subs.most_common(10),
            'avg_shift': float(np.mean(shifts)) if shifts else 0.0,
        }

    def clear_activations(self) -> None:
        if self.rule_activations is not None:
            self.rule_activations = []
        if self.rule_tracker is not None:
            self.rule_tracker.clear()


# Severity Adapter (Proposal P3)

class SeverityAdapter(nn.Module):
    """
    Cross-Attention Severity Adapter (Audit Proposal P3).

    Injects a continuous severity signal into HuBERT hidden states via
    cross-attention, replacing the coarse scalar β adjustment.

    Architecture:
        1. Project severity scalar → context vector:  Linear(1,64) → SiLU → Linear(64,768)
        2. Cross-attention: Q = hidden_states, K = V = severity_context [B, 1, 768]
        3. Residual + LayerNorm

    This allows the model to selectively amplify or suppress dysarthria-relevant
    features in a severity-conditioned, spatially-aware manner.

    Args:
        hidden_dim:    HuBERT hidden dimension (default 768)
        adapter_dim:   Bottleneck dimension for severity projection (default 64)
        n_heads:       Number of attention heads (default 8)
        dropout:       Attention dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        adapter_dim: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Severity → context vector
        self.severity_proj = nn.Sequential(
            nn.Linear(1, adapter_dim),
            nn.SiLU(),
            nn.Linear(adapter_dim, hidden_dim),
        )
        # Cross-attention: hidden states attend to severity context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        severity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, 768] HuBERT encoder output
            severity:      [B] continuous severity in [0, 5]

        Returns:
            Adapted hidden states [B, T, 768]
        """
        # [B, 1, 768] severity context
        sev_ctx = self.severity_proj(severity.float().view(-1, 1, 1))

        # Cross-attention: hidden_states as query, severity_ctx as key/value
        attn_out, _ = self.cross_attn(hidden_states, sev_ctx, sev_ctx)
        attn_out = self.dropout(attn_out)

        return self.layer_norm(hidden_states + attn_out)


# SpecAugment — time and frequency masking on HuBERT hidden states

class SpecAugmentLayer(nn.Module):
    """
    SpecAugment applied to HuBERT hidden states [B, T, D] during training.

    Applies independent random time and frequency masks per batch item.
    Masked positions are zeroed (zero is close to the mean of layer-normalised
    representations and is the standard default for hidden-state SpecAugment).

    Applied only when model.training is True; a no-op at eval/inference time.

    Args:
        time_mask_prob:   Expected fraction of time steps to mask per utterance.
        time_mask_length: Max consecutive frames per mask (uniform in [1, length]).
        freq_mask_prob:   Expected fraction of feature dims to mask.
        freq_mask_length: Max consecutive dims per mask (uniform in [1, length]).
    """

    def __init__(
        self,
        time_mask_prob: float = 0.05,
        time_mask_length: int = 10,
        freq_mask_prob: float = 0.05,
        freq_mask_length: int = 8,
    ):
        super().__init__()
        self.time_mask_prob   = time_mask_prob
        self.time_mask_length = time_mask_length
        self.freq_mask_prob   = freq_mask_prob
        self.freq_mask_length = freq_mask_length
        # Dedicated RNG makes mask sampling reproducible independent of Python random state.
        self._rng = torch.Generator()
        self._rng.manual_seed(torch.initial_seed())

    def set_seed(self, seed: int) -> None:
        """Set deterministic RNG seed for SpecAugment mask sampling."""
        self._rng.manual_seed(int(seed))

    def _randint_inclusive(self, low: int, high: int, device: torch.device = None) -> int:
        """Inclusive integer sampling helper backed by torch.Generator."""
        if high <= low:
            return int(low)
        # [PERF] Use on-device RNG when possible to avoid host-device sync.
        if device is not None and device.type != 'cpu':
            return int(torch.randint(low, high + 1, (1,), device=device).item())
        return int(torch.randint(low, high + 1, (1,), generator=self._rng).item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] hidden states
        Returns:
            [B, T, D] with random time/freq regions zeroed (training only)
        """
        if not self.training:
            return x

        B, T, D = x.shape
        x = x.clone()

        # ─ Time masking — independent masks per sample (§2.3 fix) ─────────
        # [PERF] Mask tensor is built as a zeros_like so it lives on the same device
        #        as x without any explicit .to() call.
        n_time_masks = max(1, int(T * self.time_mask_prob))
        for b in range(B):
            for _ in range(n_time_masks):
                t_len = self._randint_inclusive(1, self.time_mask_length, device=x.device)
                t0 = self._randint_inclusive(0, max(0, T - t_len), device=x.device)
                x[b, t0:t0 + t_len, :] = 0.0

        # ─ Frequency masking — independent masks per sample ────────────────
        n_freq_masks = max(1, int(D * self.freq_mask_prob))
        for b in range(B):
            for _ in range(n_freq_masks):
                f_len = self._randint_inclusive(1, self.freq_mask_length, device=x.device)
                f0 = self._randint_inclusive(0, max(0, D - f_len), device=x.device)
                x[b, :, f0:f0 + f_len] = 0.0

        return x


# Phoneme Classifier Head (unchanged)

class TemporalDownsampler(nn.Module):
    """
    Stride-2 Conv1d bottleneck inserted between HuBERT hidden states and the
    phoneme classifier.

    Motivation (dysarthric CTC)
    ---------------------------
    With HuBERT mostly frozen the model must map ~50 Hz frame embeddings to
    phoneme posteriors in one projection step.  Dysarthric speech has elongated,
    slurred phonemes whose acoustic realisation spans many frames.  A stride-2
    conv forces the model to aggregate context from 3 neighbouring frames before
    predicting each phoneme, halving the effective frame rate to ~25 Hz and making
    CTC alignment substantially more stable.

    Length formula (Conv1d, kernel=3, stride=2, padding=1)
    ------------------------------------------------------
        T_out = floor((T_in + 2*padding - kernel) / stride) + 1
               = floor((T_in - 1) / 2) + 1
               = ceil(T_in / 2)  =  (T_in + 1) // 2

    Callers must apply the same formula to ``output_lengths``::

        output_lengths = (output_lengths + 1) // 2
    """

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.norm    = nn.LayerNorm(hidden_dim)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, hidden_dim]  (HuBERT convention: batch-first)
        Returns:
            [B, ceil(T/2), hidden_dim]
        """
        x = x.transpose(1, 2)   # [B, hidden_dim, T]
        x = self.conv(x)         # [B, hidden_dim, ceil(T/2)]
        x = x.transpose(1, 2)   # [B, ceil(T/2), hidden_dim]
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class PhonemeClassifier(nn.Module):
    """
    Phoneme prediction head: 768 → LayerNorm → GELU → Dropout → 512 → |V|
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.projection  = nn.Linear(768, config.hidden_dim)
        self.layer_norm  = nn.LayerNorm(config.hidden_dim)
        self.activation  = nn.GELU()
        self.dropout     = nn.Dropout(config.classifier_dropout)
        self.classifier  = nn.Linear(config.hidden_dim, config.num_phonemes)

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_features: bool = False,
    ):
        x = self.projection(hidden_states)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        if return_features:
            return logits, x
        return logits


# Full NeuroSymbolicASR Model

class NeuroSymbolicASR(nn.Module):
    """
    Complete Neuro-Symbolic ASR model (February 2026 research audit revision).

    Forward flow:
        Waveform
          → HuBERT encoder  [B, T, 768]
          → SeverityAdapter (optional, Proposal P3)  [B, T, 768]
          → PhonemeClassifier  [B, T, |V|]   (logits_neural)
          → SymbolicConstraintLayer  [B, T, |V|]  (log_probs_constrained)
          → Loss computation (CTC + CE + Art + Ordinal + BlankKL + SymbolicKL)

    Args:
        model_config:   Architecture configuration
        symbolic_config: Symbolic reasoning configuration
        phn_to_id:      Phoneme → ID mapping
        id_to_phn:      ID → Phoneme reverse mapping
        manner_to_id:   Manner class vocabulary (for auxiliary head)
        place_to_id:    Place class vocabulary
        voice_to_id:    Voice class vocabulary
    """

    def __init__(
        self,
        model_config: ModelConfig,
        symbolic_config: SymbolicConfig,
        phn_to_id: Dict[str, int],
        id_to_phn: Dict[int, str],
        manner_to_id: Optional[Dict[str, int]] = None,
        place_to_id: Optional[Dict[str, int]] = None,
        voice_to_id: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.symbolic_config = symbolic_config
        model_config.num_phonemes = len(phn_to_id)

        # ── HuBERT Encoder ────────────────────────────────────────────────────
        logger.info("Loading HuBERT: %s", model_config.hubert_model_id)
        _hf_kwargs = dict(attn_implementation="eager")  # SDPA does not support output_attentions=True
        if getattr(model_config, 'hubert_model_revision', None):
            _hf_kwargs['revision'] = model_config.hubert_model_revision
            logger.info("   HuBERT revision pinned: %s", model_config.hubert_model_revision)
        self.hubert = HubertModel.from_pretrained(
            model_config.hubert_model_id,
            **_hf_kwargs,
        )

        if getattr(model_config, "use_gradient_checkpointing", True):
            try:
                self.hubert.gradient_checkpointing_enable()
                logger.info("   Gradient checkpointing enabled")
            except Exception:
                if hasattr(self.hubert, "config"):
                    setattr(self.hubert.config, "gradient_checkpointing", True)
                    logger.info("   Gradient checkpointing enabled (legacy mode)")
        else:
            logger.info("   Gradient checkpointing disabled (speed mode)")

        self._configure_frozen_layers()

        # ── Severity Adapter (Proposal P3) ───────────────────────────────────
        self.severity_adapter: Optional[SeverityAdapter] = None
        if model_config.use_severity_adapter:
            self.severity_adapter = SeverityAdapter(
                hidden_dim=768,
                adapter_dim=model_config.severity_adapter_dim,
            )
            logger.info("   SeverityAdapter enabled (cross-attention, continuous severity)")

        # ── SpecAugment (time/freq masking on hidden states, training only) ───
        self.spec_augment: Optional[SpecAugmentLayer] = None
        if model_config.use_spec_augment:
            self.spec_augment = SpecAugmentLayer(
                time_mask_prob=model_config.spec_time_mask_prob,
                time_mask_length=model_config.spec_time_mask_length,
                freq_mask_prob=model_config.spec_freq_mask_prob,
                freq_mask_length=model_config.spec_freq_mask_length,
            )
            logger.info("   SpecAugment enabled (time/freq masking on HuBERT hidden states)")

        # ── Temporal Downsampler (halves frame rate; stabilises CTC alignment) ─
        self.temporal_downsampler: Optional[TemporalDownsampler] = None
        if model_config.use_temporal_downsample:
            self.temporal_downsampler = TemporalDownsampler(
                hidden_dim=768,
                dropout=model_config.classifier_dropout,
            )
            logger.info("   TemporalDownsampler enabled (~50 Hz → ~25 Hz, stride-2 Conv1d)")

        # ── Phoneme Classifier ────────────────────────────────────────────────
        self.phoneme_classifier = PhonemeClassifier(model_config)

        # ── Articulatory Auxiliary Heads ──────────────────────────────────────
        self.manner_head = self.place_head = self.voice_head = None
        if manner_to_id and place_to_id and voice_to_id:
            self.manner_head = nn.Linear(model_config.hidden_dim, len(manner_to_id))
            self.place_head  = nn.Linear(model_config.hidden_dim, len(place_to_id))
            self.voice_head  = nn.Linear(model_config.hidden_dim, len(voice_to_id))
            logger.info("   Articulatory auxiliary heads enabled (manner/place/voice)")

        # ── Symbolic Constraint Layer ─────────────────────────────────────────
        self.symbolic_layer = SymbolicConstraintLayer(
            num_phonemes=model_config.num_phonemes,
            phn_to_id=phn_to_id,
            id_to_phn=id_to_phn,
            symbolic_config=symbolic_config,
            constraint_weight=model_config.constraint_weight_init,
            learnable=model_config.constraint_learnable,
            use_learnable_matrix=model_config.use_learnable_constraint,
        )
        if model_config.use_learnable_constraint:
            logger.info("   LearnableConstraintMatrix enabled (Proposal P2)")

        # Summary
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        logger.info(
            "NeuroSymbolicASR ready: %s total / %s trainable params (%.1f%%)",
            f"{total:,}", f"{trainable:,}", trainable / total * 100
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Layer freezing
    # ──────────────────────────────────────────────────────────────────────────

    def _configure_frozen_layers(self) -> None:
        """Freeze feature extractor and configured encoder layers."""
        if self.model_config.freeze_feature_extractor:
            for param in self.hubert.feature_extractor.parameters():
                param.requires_grad = False
            logger.info("   Froze feature extractor (CNN)")

        for idx in self.model_config.freeze_encoder_layers:
            for param in self.hubert.encoder.layers[idx].parameters():
                param.requires_grad = False
        if self.model_config.freeze_encoder_layers:
            logger.info("   Froze encoder layers %s", self.model_config.freeze_encoder_layers)

    @property
    def _hubert_is_frozen(self) -> bool:
        """True when every HuBERT parameter has requires_grad=False."""
        return not any(p.requires_grad for p in self.hubert.parameters())

    def _unfreeze_all_hubert(self) -> None:
        """Enable gradients for all HuBERT parameters."""
        for param in self.hubert.parameters():
            param.requires_grad = True

    def freeze_encoder(self) -> None:
        """Freeze the entire HuBERT encoder (CNN + all transformer layers)."""
        for param in self.hubert.parameters():
            param.requires_grad = False
        logger.info("Froze entire HuBERT encoder")

    def unfreeze_encoder(self, layers: Optional[List[int]] = None) -> None:
        """Unfreeze the specified HuBERT transformer layers (all if None)."""
        if layers is None:
            for param in self.hubert.encoder.parameters():
                param.requires_grad = True
            logger.info("Unfroze all encoder layers")
        else:
            for idx in layers:
                for param in self.hubert.encoder.layers[idx].parameters():
                    param.requires_grad = True
            logger.info("Unfroze encoder layers %s", layers)

    def unfreeze_after_warmup(self) -> None:
        """Progressive unfreezing: re-enables upper layers after warmup epochs."""
        self._unfreeze_all_hubert()
        self._configure_frozen_layers()
        logger.info("Unfroze HuBERT encoder (keeping configured frozen layers)")

    # ──────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_severity: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        ablation_mode: str = "full",
    ) -> Dict[str, torch.Tensor]:
        """
        Full neuro-symbolic forward pass.

        Args:
            input_values:     Raw waveform [B, T_audio]
            attention_mask:   Valid-frame mask [B, T_audio]
            speaker_severity: Continuous severity [B] in [0, 5]
            output_attentions: If True, return HuBERT attention weights for
                               explainability (ROADMAP §6.1 Method 2)
            ablation_mode:    One of "full" | "neural_only" | "symbolic_only" |
                              "no_art_heads".  When "neural_only", the
                              SeverityAdapter and SymbolicConstraintLayer are
                              bypassed and log-softmax of the neural logits is
                              returned directly (Q7 — true neural baseline).

        Returns:
            Dict with logits_neural, logits_constrained, log_probs_constrained,
            hidden_states, beta, P_neural, P_constrained, rule_shift,
            logits_manner/place/voice, attention_weights (if requested)
        """
        # ── HuBERT encoding ───────────────────────────────────────────────────
        # output_hidden_states only when attn weights are needed (explainability).
        # Setting it True unconditionally stores all 12 intermediate [B,T,768]
        # tensors (~120 MB at batch=16) even though only last_hidden_state is used.
        #
        # [PERF] When all HuBERT parameters are frozen, run the encoder under
        # eval()+no_grad() to:
        #   1. Disable HuBERT dropout — LayerNorm stats remain stable and batch-
        #      independent, preventing training-mode noise from propagating.
        #   2. Avoid building an intermediate activation graph for frozen params,
        #      saving ~15-25% VRAM and ~8-10% wall-clock per step on RTX 4060.
        # After the frozen forward pass, the original training mode is restored so
        # downstream learnable modules (SeverityAdapter, classifier, etc.) receive
        # backprop normally through hidden_states.
        if self._hubert_is_frozen:
            _prev_training = self.hubert.training
            self.hubert.eval()
            with torch.no_grad():
                hubert_outputs = self.hubert(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    output_attentions=output_attentions,
                    return_dict=True,
                )
            self.hubert.train(_prev_training)  # restore; downstream modules still in training mode
        else:
            hubert_outputs = self.hubert(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=output_attentions,
                return_dict=True,
            )
        hidden_states = hubert_outputs.last_hidden_state  # [B, T, 768]

        # ── SpecAugment (no-op at eval; masks time/freq regions during training) ──
        # NOTE: SpecAugment before SeverityAdapter — do not reorder.
        # Augmentation must run on clean HuBERT encoder features; severity
        # conditioning then operates on the augmented (but content-preserving)
        # representation.  Applying masks AFTER the adapter would zero-out the
        # adapter's severity signal for masked frames, corrupting its training.
        # B3 fix: moved from after SeverityAdapter to before it.
        # Skipped when ablation_mode == 'no_spec_augment' to isolate its contribution.
        if self.spec_augment is not None and ablation_mode != "no_spec_augment":
            hidden_states = self.spec_augment(hidden_states)

        # ── Severity Adapter (Proposal P3) ────────────────────────────────────
        # Skipped in neural_only ablation for a true neural baseline (Q7).
        if ablation_mode != "neural_only":
            if self.severity_adapter is not None and speaker_severity is not None:
                hidden_states = self.severity_adapter(hidden_states, speaker_severity)

        # ── Temporal Downsampler (stride-2, ~50 Hz → ~25 Hz) ─────────────────
        # Track whether downsampling actually ran — output_lengths depends on it.
        _downsample_applied = False
        if self.temporal_downsampler is not None and ablation_mode != "no_temporal_ds":
            hidden_states = self.temporal_downsampler(hidden_states)
            _downsample_applied = True

        # ── Phoneme Classifier ────────────────────────────────────────────────
        logits_neural, shared_features = self.phoneme_classifier(
            hidden_states, return_features=True
        )

        # ── Articulatory Auxiliary Heads (utterance-level via GAP) ──────────────
        # CTC models produce no forced alignment, so frame-level articulatory
        # supervision assigns a single phoneme label to every output frame, which
        # is semantically incorrect (the model receives contradictory gradients for
        # most frame positions).  Global average pooling over the time axis first
        # gives a proper utterance-level representation that captures the dominant
        # articulatory profile of the whole utterance.  The CE loss in train.py
        # then uses the mode of the phoneme label sequence as the utterance target.
        if self.manner_head is not None and ablation_mode not in ("neural_only", "no_art_heads"):
            pooled_for_art = shared_features.mean(dim=1)  # [B, hidden_dim]
            logits_manner = self.manner_head(pooled_for_art)   # [B, num_manner]
            logits_place  = self.place_head(pooled_for_art)    # [B, num_place]
            logits_voice  = self.voice_head(pooled_for_art)    # [B, num_voice]
        else:
            logits_manner = logits_place = logits_voice = None

        # ── Compute output_lengths (shared by neural_only and full paths) ─────
        if attention_mask is not None:
            audio_lengths = attention_mask.sum(dim=-1)
            output_lengths = self.hubert._get_feat_extract_output_lengths(
                audio_lengths
            ).long()
        else:
            pre_downsample_T = hidden_states.size(1)
            output_lengths = torch.full(
                (hidden_states.size(0),), pre_downsample_T,
                dtype=torch.long, device=hidden_states.device,
            )
        if _downsample_applied:
            output_lengths = (output_lengths + 1) // 2
        output_lengths = output_lengths.clamp(max=hidden_states.size(1))

        # ── Q7: Neural-only short-circuit ─────────────────────────────────────
        # When ablation_mode == "neural_only" skip the SymbolicConstraintLayer
        # entirely so gradients never flow through it.  The constrained logits
        # alias the neural log-softmax — providing a true neural-only baseline.
        if ablation_mode == "neural_only":
            log_probs_neural = F.log_softmax(logits_neural, dim=-1)
            result_neural_only = {
                'logits_neural':         logits_neural,
                'logits_constrained':    log_probs_neural,
                'log_probs_constrained': log_probs_neural,
                'hidden_states':         hidden_states,
                'beta':                  torch.zeros(1, device=logits_neural.device),
                'P_neural':              F.softmax(logits_neural, dim=-1),
                'P_constrained':         None,
                'rule_shift':            None,
                'logits_manner':         logits_manner,
                'logits_place':          logits_place,
                'logits_voice':          logits_voice,
                'output_lengths':        output_lengths,
            }
            return result_neural_only

        # ── no_constraint_matrix: keep SeverityAdapter + art heads, skip symbolic layer ──
        # Unlike neural_only (which also skips SeverityAdapter and art heads), this
        # gives a fairer comparison by keeping all components except the constraint blending.
        if ablation_mode == "no_constraint_matrix":
            log_probs_neural = F.log_softmax(logits_neural, dim=-1)
            return {
                'logits_neural':         logits_neural,
                'logits_constrained':    log_probs_neural,
                'log_probs_constrained': log_probs_neural,
                'hidden_states':         hidden_states,
                'beta':                  torch.zeros(1, device=logits_neural.device),
                'P_neural':              F.softmax(logits_neural, dim=-1),
                'P_constrained':         None,
                'rule_shift':            None,
                'logits_manner':         logits_manner,
                'logits_place':          logits_place,
                'logits_voice':          logits_voice,
                'output_lengths':        output_lengths,
            }

        # ── Symbolic Constraint Layer ─────────────────────────────────────────
        symbolic_out = self.symbolic_layer(logits_neural, speaker_severity=speaker_severity)

        result = {
            'logits_neural':         logits_neural,
            'logits_constrained':    symbolic_out['log_probs'],
            'log_probs_constrained': symbolic_out['log_probs'],
            'hidden_states':         hidden_states,
            'beta':                  symbolic_out['beta'],
            'P_neural':              symbolic_out.get('P_neural'),
            'P_constrained':         symbolic_out.get('P_constrained'),
            'rule_shift':            symbolic_out.get('rule_shift'),
            'logits_manner':         logits_manner,
            'logits_place':          logits_place,
            'logits_voice':          logits_voice,
            'output_lengths':        output_lengths,
        }

        if output_attentions and hubert_outputs.attentions is not None:
            result['attention_weights'] = hubert_outputs.attentions

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_constraint_matrix(self) -> torch.Tensor:
        """Return the current constraint matrix (learned or static)."""
        if self.symbolic_layer.use_learnable_matrix and self.symbolic_layer.learnable_matrix is not None:
            return self.symbolic_layer.learnable_matrix.C.detach()
        return self.symbolic_layer.static_constraint_matrix

    def get_rule_statistics(self) -> Dict:
        return self.symbolic_layer.get_rule_statistics()

    def clear_rule_activations(self) -> None:
        self.symbolic_layer.clear_activations()


# Module self-test

def main() -> None:
    """Quick forward-pass smoke test."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.config import get_default_config

    config = get_default_config()
    phn_to_id = {'<BLANK>': 0, '<PAD>': 1, '<UNK>': 2, 'P': 3, 'B': 4, 'T': 5}
    id_to_phn = {v: k for k, v in phn_to_id.items()}

    model = NeuroSymbolicASR(
        model_config=config.model,
        symbolic_config=config.symbolic,
        phn_to_id=phn_to_id,
        id_to_phn=id_to_phn,
    )

    batch_size = 2
    dummy_input    = torch.randn(batch_size, 16000)
    dummy_mask     = torch.ones(batch_size, 16000)
    dummy_severity = torch.tensor([0.0, 4.9])  # Control vs severe

    print("\n🔎 Running forward pass …")
    with torch.no_grad():
        out = model(dummy_input, attention_mask=dummy_mask,
                    speaker_severity=dummy_severity, output_attentions=True)

    print(f"   ✅ logits_neural:     {out['logits_neural'].shape}")
    print(f"   ✅ logits_constrained:{out['logits_constrained'].shape}")
    print(f"   ✅ hidden_states:     {out['hidden_states'].shape}")
    att = out.get('attention_weights')
    if att and att[0] is not None:
        print(f"   ✅ attention_weights: {len(att)} layers × {att[0].shape}")
    beta_val = out['beta']
    if isinstance(beta_val, torch.Tensor) and beta_val.numel() > 1:
        print(f"   ✅ beta (per sample): {beta_val.squeeze().tolist()}")
    else:
        print(f"   ✅ beta: {float(beta_val):.3f}")


if __name__ == "__main__":
    main()
