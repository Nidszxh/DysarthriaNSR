"""
Training Pipeline for Neuro-Symbolic Dysarthric Speech Recognition

Orchestrates model training using PyTorch Lightning with multi-task learning,
symbolic constraints, and comprehensive evaluation metrics.

# REFACTOR LOG
# [CLEAN] Added module-level logger = logging.getLogger(__name__); replaced
#         logger.info() calls in train(), _save_learning_curve(), and run_loso() with
#         logger.info() / logger.warning() for structured log control.
# [REPRO] Added _seed_worker() DataLoader worker_init_fn so each worker's
#         NumPy / Python random states are seeded deterministically from the
#         Lightning global seed, making augmentation reproducible across restarts.
# [CONFIG] Extracted LOSO bootstrap iterations into LOSO_BOOTSTRAP_SAMPLES constant.
# [CLEAN]  Moved matplotlib import to module level (was inside _save_learning_curve
#          on every call, which is fragile if the backend changes mid-session).
# [FIX-T05] Replaced proportional label interpolation with torchaudio forced_align
#         for frame-CE loss; added use_forced_alignment, forced_alignment_fallback_warn,
#         frame_ce_start_epoch=0, lambda_ce=0.15 config fields; added
#         _compute_ce_loss_aligned method with fallback to proportional interpolation.
# [FIX-T04] Replaced OneCycleLR with CosineAnnealingWarmRestarts; removed manual
#         LR warmup workaround (_hubert_warmup_steps_remaining, _hubert_warmup_total_steps,
#         _hubert_warmup_peak_lr); simplified _reset_hubert_lr_warmup to only clear
#         Adam state; changed scheduler interval from 'step' to 'epoch'; removed
#         OneCycleLR step-count drift detection in train().
"""

import logging
import os
import sys
from collections import defaultdict
import warnings
import json
import time
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')   # [CLEAN] Backend set at module level before any pyplot import
import matplotlib.pyplot as plt

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as TAF
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# Add src directory to path for imports
project_root = Path(__file__).resolve().parent
sys.path.insert(2, str(project_root / "src"))

from src.utils.config import Config, get_default_config, get_project_root, get_speaker_severity
from src.utils.sequence_utils import align_labels_to_logits
from src.utils import resolve_best_fold_checkpoint
from src.data.dataloader import NeuroSymbolicCollator, TorgoNeuroSymbolicDataset
from src.models.model import NeuroSymbolicASR
from src.models.losses import OrdinalContrastiveLoss, BlankPriorKLLoss, SymbolicKLLoss

# Import evaluate functions from root level
from evaluate import compute_per, evaluate_model, decode_predictions, decode_references

warnings.filterwarnings('once')

logger = logging.getLogger(__name__)

# [CONFIG] Named constant for LOSO bootstrap iterations — previously bare literal 2000.
# Controls width of the 95% CI reported after LOSO-CV: higher = tighter CI, slower.
LOSO_BOOTSTRAP_SAMPLES: int = 2000


def _seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker deterministically from the Lightning global seed.

    Called automatically by DataLoader on worker process start when set as
    worker_init_fn.  Ensures augmentation and sampling are reproducible across
    restarts and folds — important for multi-run LOSO-CV comparisons.
    """
    # [REPRO] torch.initial_seed() returns the initial seed set by pl.seed_everything();
    # adding worker_id ensures distinct (but deterministic) sequences across workers.
    import numpy as np  # noqa: PLC0415  (local import avoids top-level NumPy fork hazard)
    import random       # noqa: PLC0415
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def _get_batch_severity(batch: Dict, device: torch.device,
                        config: Optional[Config] = None) -> torch.Tensor:
    """Return per-sample severity scores on the requested device.

    Uses speaker IDs when available (continuous severity map), otherwise
    falls back to binary status mapping 0->0.0, 1->severity_normalization_constant.
    """
    speakers = batch.get('speakers', [])
    status = batch['status']
    if speakers and isinstance(speakers[0], str):
        return torch.tensor(
            [get_speaker_severity(s) for s in speakers],
            dtype=torch.float32,
            device=device,
        )
    sev_norm = getattr(config.symbolic, 'severity_normalization_constant', 5.0) if config else 5.0
    return status.float().to(device) * sev_norm


def flatten_config_for_mlflow(config: Config) -> Dict:

    safe_params = {}

    def add_params(prefix: str, obj: object) -> None:
        """Recursively add parameters with safe names."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                # Convert tuple keys to safe strings
                safe_key = str(key).replace("'", "").replace("(", "").replace(")", "").replace(", ", "_")
                new_prefix = f"{prefix}/{safe_key}" if prefix else safe_key
                add_params(new_prefix, value)
        elif isinstance(obj, (list, tuple)):
            # Skip complex nested structures
            if prefix and not any(isinstance(x, (dict, list, tuple)) for x in obj):
                safe_params[prefix] = str(obj)
        elif isinstance(obj, Path):
            safe_params[prefix] = str(obj)
        elif isinstance(obj, (bool, int, float, str)):
            safe_params[prefix] = obj

    # Flatten each sub-config
    add_params("model", config.model.__dict__)
    add_params("training", config.training.__dict__)
    add_params("data", config.data.__dict__)
    add_params("experiment", config.experiment.__dict__)
    add_params("symbolic", config.symbolic.__dict__)
    safe_params["device"] = config.device

    return safe_params


class DysarthriaASRLightning(pl.LightningModule):
    """PyTorch Lightning module for neuro-symbolic dysarthria ASR."""

    def __init__(
        self,
        model: NeuroSymbolicASR,
        config: Config,
        phn_to_id: Dict[str, int],
        id_to_phn: Dict[int, str],
        class_weights: Optional[torch.Tensor] = None,
        articulatory_weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:

        super().__init__()
        self.model = model
        self.config = config
        self.phn_to_id = phn_to_id
        self.id_to_phn = id_to_phn
        self.class_weights = class_weights
        self.articulatory_weights = articulatory_weights or {}

        # Save hyperparameters (exclude complex objects)
        self.save_hyperparameters(ignore=['model', 'config', 'phn_to_id', 'id_to_phn'])

        # Initialize loss functions
        self._init_loss_functions()

        # Freeze HuBERT encoder for warmup epochs to stabilize the head
        self.model.freeze_encoder()
        self.encoder_unfrozen = False

        # Metrics tracking
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # When LOSO must resume with weights-only (fresh optimizer/scheduler),
        # this offset preserves epoch-aware behavior (staged unfreezing, KL ramp)
        # as if training had continued from the checkpoint epoch.
        self.resume_epoch_offset = 0
        self._encoder_unfrozen = False
        self._encoder_deep_unfrozen = False
        self._encoder_deeper_unfrozen = False
        self._sync_unfreeze_flags_from_epoch(self.resume_epoch_offset)

    def _sync_unfreeze_flags_from_epoch(self, effective_epoch: int, apply_to_model: bool = False) -> None:
        """Sync progressive-unfreeze stage flags from an effective epoch.

        This prevents duplicate stage transitions during weights-only resume.
        Optionally applies the implied unfreeze stage to the underlying model
        immediately (useful when resume offset is injected after __init__).
        """
        warmup_ep = self.config.training.encoder_warmup_epochs
        second_unfreeze = getattr(self.config.training, 'encoder_second_unfreeze_epoch', 6)
        third_unfreeze = getattr(self.config.training, 'encoder_third_unfreeze_epoch', 12)

        epoch = int(effective_epoch)
        if epoch >= third_unfreeze:
            self.encoder_unfrozen = True
            self._encoder_unfrozen = True
            self._encoder_deep_unfrozen = True
            self._encoder_deeper_unfrozen = True
            if apply_to_model:
                self.model.unfreeze_encoder(layers=[4, 5, 6, 7, 8, 9, 10, 11])
        elif epoch >= second_unfreeze:
            self.encoder_unfrozen = True
            self._encoder_unfrozen = True
            self._encoder_deep_unfrozen = True
            self._encoder_deeper_unfrozen = False
            if apply_to_model:
                self.model.unfreeze_encoder(layers=[6, 7, 8, 9, 10, 11])
        elif epoch >= warmup_ep:
            self.encoder_unfrozen = True
            self._encoder_unfrozen = True
            self._encoder_deep_unfrozen = False
            self._encoder_deeper_unfrozen = False
            if apply_to_model:
                self.model.unfreeze_encoder(layers=[8, 9, 10, 11])
        else:
            self.encoder_unfrozen = False
            self._encoder_unfrozen = False
            self._encoder_deep_unfrozen = False
            self._encoder_deeper_unfrozen = False

    def _init_loss_functions(self) -> None:
        """Initialize loss functions (simplified to 2 losses)."""
        self.ctc_loss = nn.CTCLoss(
            blank=self.phn_to_id['<BLANK>'],
            reduction='mean',
            zero_infinity=True
        )

        num_classes = len(self.phn_to_id)
        # Use dataset-provided inverse-frequency weights if available
        if self.class_weights is not None:
            ce_weights = self.class_weights.clone().detach().float()
            # Ensure correct length
            if ce_weights.numel() != num_classes:
                ce_weights = torch.ones(num_classes, dtype=torch.float32)
        else:
            ce_weights = torch.ones(num_classes, dtype=torch.float32)

        # NOTE: dataloader BLANK/PAD multipliers set to 1.0 (removed).

        # C1: Use CrossEntropyLoss (accepts raw logits, always non-negative)
        self.ce_loss = nn.CrossEntropyLoss(
            weight=ce_weights,
            ignore_index=-100,
            label_smoothing=self.config.training.label_smoothing
        )

        # C6: Use nn.ModuleDict so Lightning registers and device-manages these modules
        _art_losses = {}
        for key in ["manner", "place", "voice"]:
            weights = self.articulatory_weights.get(key)
            if weights is not None:
                weights = weights.clone().detach().float()
            _art_losses[key] = nn.CrossEntropyLoss(
                weight=weights,
                ignore_index=-100,
                label_smoothing=self.config.training.label_smoothing
            )
        self.art_ce_losses = nn.ModuleDict(_art_losses)

        # --- Phase 2: New research-grade loss functions (audit proposals P1, P2, R3) ---
        blank_id = self.phn_to_id.get('<BLANK>', 0)
        self.ordinal_loss = OrdinalContrastiveLoss(margin_per_level=0.3)
        self.blank_kl_loss = BlankPriorKLLoss(
            blank_id=blank_id,
            target_prob=self.config.training.blank_target_prob,
        )
        # I6: Eagerly initialise SymbolicKLLoss so Lightning registers the module
        # for device management.  The static constraint matrix is already built on
        # model init, so there is no reason to delay construction.
        self.symbolic_kl_loss: Optional[SymbolicKLLoss] = None
        if self.config.model.use_learnable_constraint:
            sl = self.model.symbolic_layer
            if sl.learnable_matrix is not None:
                self.symbolic_kl_loss = SymbolicKLLoss(
                    sl.static_constraint_matrix,
                    blank_penalty_weight=self.config.symbolic.blank_penalty_weight,
                    entropy_penalty_weight=self.config.symbolic.constraint_entropy_penalty_weight,
                )

        # Register core loss weights as buffers
        self.register_buffer('lambda_ctc', torch.tensor(self.config.training.lambda_ctc))
        self.register_buffer('lambda_ce', torch.tensor(self.config.training.lambda_ce))
        self.register_buffer('lambda_articulatory', torch.tensor(self.config.training.lambda_articulatory))

    def on_save_checkpoint(self, checkpoint: Dict) -> None:

        checkpoint['phn_to_id'] = self.phn_to_id
        checkpoint['id_to_phn'] = self.id_to_phn

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        """Warn on vocabulary mismatch between checkpoint and current dataset (§2.10)."""
        import logging as _logging
        _log = _logging.getLogger(__name__)
        saved_vocab = checkpoint.get('phn_to_id')
        if saved_vocab is not None and saved_vocab != self.phn_to_id:
            missing = set(saved_vocab) - set(self.phn_to_id)
            extra = set(self.phn_to_id) - set(saved_vocab)
            _log.warning(
                "Vocabulary mismatch between checkpoint (%d tokens) and current "
                "dataset (%d tokens). Missing from current: %s. Extra in current: %s. "
                "Verify the manifest has not been regenerated with a different phoneme set.",
                len(saved_vocab), len(self.phn_to_id), missing, extra,
            )

    def on_fit_start(self) -> None:

        for loss_fn in [self.ce_loss, *self.art_ce_losses.values()]:
            if hasattr(loss_fn, 'weight') and loss_fn.weight is not None:
                try:
                    loss_fn.weight = loss_fn.weight.to(self.device)
                except Exception:
                    logger.warning("Failed to move loss weight tensor to device %s", self.device)

    def forward(self, batch: Dict) -> Dict:
        severity = _get_batch_severity(batch, batch['status'].device, self.config)

        return self.model(
            input_values=batch['input_values'],
            attention_mask=batch['attention_mask'],
            speaker_severity=severity,
            ablation_mode=self.config.training.ablation_mode,  # Q7: true neural-only ablation
        )

    def compute_loss(
        self,
        outputs: Dict,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
        articulatory_labels: Optional[Dict[str, torch.Tensor]] = None,
        severity: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-task loss: CTC + frame-CE + articulatory + ordinal contrastive +
        blank-prior KL + symbolic KL anchor.
        """
        ablation = self.config.training.ablation_mode

        log_probs_constrained = outputs['logits_constrained']

        # ── Shape assertions ──────────────────────────────────────────────────
        B, T, V = log_probs_constrained.shape
        assert labels.shape[0] == B, f"Batch dim mismatch: logits={B}, labels={labels.shape[0]}"
        assert input_lengths.shape == (B,), f"input_lengths shape {input_lengths.shape} != ({B},)"
        assert label_lengths.shape == (B,), f"label_lengths shape {label_lengths.shape} != ({B},)"
        assert V == len(self.phn_to_id), f"Vocab dim mismatch: {V} vs phn_to_id {len(self.phn_to_id)}"
        # ──────────────────────────────────────────────────────────────────────

        # 1. CTC Loss (primary alignment loss — always)
        loss_ctc = self._compute_ctc_loss(
            log_probs_constrained, labels, input_lengths, label_lengths
        )

        # 2. Frame-level CE applied to NEURAL logits (not constrained log-probs).
        # [FIX-T05] Use CTC forced alignment (torchaudio.functional.forced_align) when enabled.
        # This provides a phonetically grounded frame alignment, removing the need for the
        # epoch gate that was a workaround for the bad proportional interpolation.
        effective_epoch = int(self.current_epoch + int(getattr(self, 'resume_epoch_offset', 0)))
        frame_ce_start_epoch = int(self.config.training.frame_ce_start_epoch)
        use_forced_alignment = getattr(self.config.training, 'use_forced_alignment', True)
        if use_forced_alignment:
            # Forced alignment is accurate, so activate from epoch 0 without gating
            loss_ce = self._compute_ce_loss_aligned(
                outputs.get('logits_neural', log_probs_constrained),
                labels,
                input_lengths,
                label_lengths,
            )
        else:
            # Fallback to proportional interpolation with epoch gate
            frame_ce_enabled = effective_epoch >= frame_ce_start_epoch
            if frame_ce_enabled:
                loss_ce = self._compute_ce_loss(outputs.get('logits_neural', log_probs_constrained), labels)
            else:
                loss_ce = torch.zeros_like(loss_ctc)

        # 3. Articulatory auxiliary heads
        loss_art = None
        if ablation not in ("neural_only", "no_art_heads") and articulatory_labels and outputs.get('logits_manner') is not None:
            art_losses = []
            for key in ["manner", "place", "voice"]:
                logits = outputs.get(f"logits_{key}")
                labels_art = articulatory_labels.get(key)
                if logits is None or labels_art is None:
                    continue
                assert logits.shape[0] == B, f"Articulatory {key} batch dim: {logits.shape[0]} vs {B}"
                art_losses.append(self._compute_articulatory_ce_loss(logits, labels_art, key))
            if art_losses:
                loss_art = torch.stack(art_losses).mean()

        # 4. Blank-prior KL (insertion fix)
        loss_blank_kl = self.blank_kl_loss(log_probs_constrained, attention_mask, severity=severity)

        # 5. Ordinal contrastive severity loss (Proposal P1)
        loss_ordinal = None
        hidden = outputs.get('hidden_states')
        if hidden is not None and severity is not None and ablation != 'symbolic_only':
            if attention_mask is not None:
                assert hidden.size(1) == attention_mask.size(1), (
                    f"Hidden ({hidden.size(1)}) and mask ({attention_mask.size(1)}) "
                    "time dims must match"
                )
            loss_ordinal = self.ordinal_loss(hidden, severity, attention_mask)

        # 6. Symbolic KL anchor (Proposal P2 — learnable constraint matrix)
        loss_symbolic_kl = None
        if self.config.model.use_learnable_constraint and ablation not in ('neural_only', 'symbolic_only', 'no_constraint_matrix'):
            sl = self.model.symbolic_layer
            if sl.learnable_matrix is not None and self.symbolic_kl_loss is not None:
                loss_symbolic_kl = self.symbolic_kl_loss(sl.learnable_matrix.logit_C)

        # ── Ablation-mode weighting ──────────────────────────────────────────
        lambda_ctc = 0.0 if ablation == 'symbolic_only' else float(self.lambda_ctc)
        if use_forced_alignment:
            # Forced alignment is accurate, so no epoch gating needed
            lambda_ce = 0.0 if ablation == 'symbolic_only' else float(self.lambda_ce)
        else:
            # Fallback: use epoch gate (effective_epoch already computed above)
            frame_ce_enabled = effective_epoch >= frame_ce_start_epoch
            lambda_ce = 0.0 if (ablation == 'symbolic_only' or not frame_ce_enabled) else float(self.lambda_ce)
        lambda_art = float(self.lambda_articulatory)
        lambda_ord = getattr(self, '_current_lambda_ordinal', self.config.training.lambda_ordinal)
        # I2: Use staged lambda_blank_kl set by on_train_epoch_start (falls back to
        # config value during evaluation/test when the attribute is not set).
        lambda_bkl = getattr(self, '_current_lambda_blank_kl', self.config.training.lambda_blank_kl)
        lambda_skl = getattr(self, '_current_lambda_symbolic_kl', self.config.training.lambda_symbolic_kl)

        total_loss = lambda_ctc * loss_ctc + lambda_ce * loss_ce
        if loss_art is not None:
            total_loss = total_loss + lambda_art * loss_art
        total_loss = total_loss + lambda_bkl * loss_blank_kl
        if loss_ordinal is not None:
            total_loss = total_loss + lambda_ord * loss_ordinal
        if loss_symbolic_kl is not None:
            total_loss = total_loss + lambda_skl * loss_symbolic_kl

        # Monitor blank probability for insertion tracking (no gradient)
        with torch.no_grad():
            blank_prob_mean = self.blank_kl_loss.mean_blank_prob(log_probs_constrained, attention_mask)

        return {
            'loss':            total_loss,
            'loss_ctc':        loss_ctc,
            'loss_ce':         loss_ce,
            'loss_art':        loss_art,
            'loss_blank_kl':   loss_blank_kl,
            'loss_ordinal':    loss_ordinal,
            'loss_symbolic_kl': loss_symbolic_kl,
            'blank_prob_mean': blank_prob_mean,
        }

    def _compute_ctc_loss(
        self,
        log_probs: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        label_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute CTC loss."""
        log_probs_transposed = log_probs.transpose(0, 1)  # [time, batch, num_classes]

        # Clamp to actual time dimension to avoid off-by-one length mismatches
        input_lengths = torch.clamp(
            input_lengths,
            max=log_probs_transposed.size(0)
        )

        return self.ctc_loss(
            log_probs_transposed,
            labels,
            input_lengths,
            label_lengths
        )

    def _compute_ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss on raw neural logits.

        C1 fix: self.ce_loss is now nn.CrossEntropyLoss which expects raw
        logits (no prior softmax/log_softmax).  Passing raw logits ensures
        the loss is always ≥ 0.
        """
        batch_size, time_steps_logits, num_classes = logits.shape

        # Align labels to logits time dimension
        labels_aligned = align_labels_to_logits(labels, time_steps_logits)

        # Compute loss — CrossEntropyLoss handles softmax internally
        logits_flat = logits.reshape(-1, num_classes)
        labels_flat = labels_aligned.reshape(-1)

        return self.ce_loss(logits_flat, labels_flat)

    def _compute_ce_loss_aligned(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Frame-CE with CTC forced alignment instead of proportional interpolation.

        Uses torchaudio.functional.forced_align to get frame indices for each
        phoneme label, providing a phonetically grounded alignment for frame-level
        cross-entropy supervision.

        Performance notes:
          - Calls forced_align once for the whole batch (not per-sample).
          - Inner per-label search uses precomputed position lookup + searchsorted,
            reducing inner loop from O(L×T) to O(L×log T).

        Args:
            logits: Neural logits [B, T, V]
            labels: Phoneme labels [B, L] (padded with -100)
            input_lengths: Valid frame counts per sample [B]
            label_lengths: Valid label counts per sample [B]

        Returns:
            Scalar CE loss.
        """
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        blank_id = self.phn_to_id['<BLANK>']
        B = logits.size(0)
        device = logits.device

        # Pre-compute per-sample (T, L, seq) tuples, skipping invalid samples
        sample_info = []
        for b in range(B):
            T = int(input_lengths[b].item())
            L = int(label_lengths[b].item())
            if L > 0 and T >= L:
                sample_info.append((b, T, L))

        if not sample_info:
            return torch.zeros((), device=device)

        # ── Batch forced_align ────────────────────────────────────────────────
        # Build batched inputs from valid samples so TAF runs a single kernel
        valid_indices = []
        batched_log_probs = []
        batched_seqs = []
        batched_in_lens = []
        batched_lab_lens = []
        fallback_mask = []

        for b, T, L in sample_info:
            valid_indices.append(b)
            batched_log_probs.append(log_probs[b:b+1, :T])
            batched_seqs.append(labels[b, :L].unsqueeze(0))
            batched_in_lens.append(input_lengths[b:b+1])
            batched_lab_lens.append(label_lengths[b:b+1])
            fallback_mask.append(False)

        fallback_samples = 0
        try:
            # Single batched call instead of B per-sample calls
            aligned = TAF.forced_align(
                torch.cat(batched_log_probs, dim=0),
                torch.cat(batched_seqs, dim=0),
                torch.cat(batched_in_lens, dim=0),
                torch.cat(batched_lab_lens, dim=0),
                blank=blank_id,
            )
            # aligned is (alignment, log_probs) — alignment: [B_valid, max_T]
            alignments = aligned[0]
        except Exception:
            logger.warning("Forced alignment failed — falling back to proportional interpolation")
            # Fallback path: process each sample individually with align_labels_to_logits
            fallback_samples = len(sample_info) if self.config.training.forced_alignment_fallback_warn else 0
            all_frame_losses = []
            for b, T, L in sample_info:
                seq = labels[b, :L]
                seq_aligned = align_labels_to_logits(seq.unsqueeze(0), T).squeeze(0)
                all_frame_losses.append(
                    F.cross_entropy(logits[b, :T], seq_aligned, ignore_index=-100)
                )
            if fallback_samples > 0 and self.config.training.forced_alignment_fallback_warn:
                self.log(
                    'train/forced_align_fallback_rate',
                    fallback_samples / max(len(sample_info), 1),
                    on_step=False, on_epoch=True,
                )
            if not all_frame_losses:
                return torch.zeros((), device=device)
            return torch.stack(all_frame_losses).mean()

        # ── Frame-index extraction via searchsorted ───────────────────────────
        # For each sample, precompute per-label position lists from alignment,
        # then find the first unused position ≥ previous label's position.
        all_frame_losses = []
        num_phonemes = self.config.model.num_phonemes

        for idx, (b, T, L) in enumerate(sample_info):
            align_1d = alignments[idx, :T]  # [T]
            seq = labels[valid_indices[idx], :L]  # [L]

            # Build per-label position lookup using list indexed by label ID
            # (avoids dict overhead and allows tensor-based indexing)
            unique_labels = align_1d.unique()
            pos_lookup = [torch.empty(0, dtype=torch.long, device=device) for _ in range(num_phonemes)]
            for lu in unique_labels:
                lab = int(lu)
                pos_lookup[lab] = torch.where(align_1d == lab)[0]

            frame_indices = torch.zeros(L, dtype=torch.long, device=device)
            seq_cpu = seq.cpu()
            prev_i = torch.zeros((), dtype=torch.long, device=device) - 1
            zero_t = torch.zeros((), dtype=torch.long, device=device)
            T_t = torch.tensor(T, dtype=torch.long, device=device)

            for s in range(L):
                target = int(seq_cpu[s])
                positions = pos_lookup[target]
                n_pos = positions.numel()
                j_idx = torch.searchsorted(positions, prev_i + 1)
                cond_in_bounds = (prev_i + 1) < T_t
                if n_pos > 0:
                    cond_j_valid = j_idx < n_pos
                    use_result = cond_in_bounds & cond_j_valid
                    j_safe = torch.where(
                        cond_j_valid, j_idx,
                        torch.tensor(n_pos - 1, dtype=torch.long, device=device)
                    )
                    pi = torch.where(
                        use_result,
                        positions[j_safe],
                        torch.where(prev_i >= 0, prev_i, zero_t)
                    )
                else:
                    pi = torch.where(prev_i >= 0, prev_i, zero_t)
                frame_indices[s] = pi
                prev_i = pi

            frame_logits = logits[valid_indices[idx], frame_indices]
            all_frame_losses.append(
                F.cross_entropy(frame_logits, seq, ignore_index=-100)
            )

        if fallback_samples > 0 and self.config.training.forced_alignment_fallback_warn:
            self.log(
                'train/forced_align_fallback_rate',
                fallback_samples / max(len(sample_info), 1),
                on_step=False, on_epoch=True,
            )

        if not all_frame_losses:
            return torch.zeros((), device=device)
        return torch.stack(all_frame_losses).mean()

    def _compute_articulatory_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        key: str
    ) -> torch.Tensor:
        """Compute articulatory cross-entropy loss.

        Supports both the new utterance-level path (I5) where ``logits`` is
        ``[B, num_classes]`` and the legacy frame-level path where ``logits``
        is ``[B, T, num_classes]``.

        For the utterance-level path the per-utterance target is the *mode*
        (most frequent class) of the valid (non-padding) phoneme labels in
        that utterance.  This gives a single, semantically consistent label
        that reflects the dominant articulatory feature for the whole utterance.
        """
        if logits.dim() == 2:
            # I5: Utterance-level path (logits [B, C])
            batch_size = labels.size(0)
            utt_labels = torch.full(
                (batch_size,), -100, dtype=torch.long, device=labels.device
            )
            valid_mask = labels != -100  # [B, L]
            for i in range(batch_size):
                valid_seq = labels[i][valid_mask[i]]
                if valid_seq.numel() > 0:
                    # L2 fix: use argmax of counts instead of torch.mode().
                    # torch.mode() breaks ties by returning the smallest ID
                    # (alphabetically-first articulatory class — arbitrary bias).
                    # argmax is also deterministic for non-tied majorities.
                    vals, counts = torch.unique(valid_seq, return_counts=True)
                    utt_labels[i] = vals[counts.argmax()]
            return self.art_ce_losses[key](logits, utt_labels)
        else:
            # Legacy frame-level path [B, T, C]
            batch_size, time_steps_logits, num_classes = logits.shape
            labels_aligned = align_labels_to_logits(labels, time_steps_logits)
            logits_flat = logits.reshape(-1, num_classes)
            labels_flat = labels_aligned.reshape(-1)
            return self.art_ce_losses[key](logits_flat, labels_flat)

    def _prepare_step(self, batch: Dict, outputs: Dict, log_invalid_frac: bool = False) -> Optional[Dict]:
        """Extract and filter step inputs, removing invalid CTC samples.

        Shared by training_step, validation_step, and test_step to eliminate
        the ~50-line copy-paste block that previously existed in each method.

        Returns None when all samples are invalid (caller should return zero loss).
        """
        log_probs_constrained = outputs.get('log_probs_constrained', outputs['logits_constrained'])
        labels = batch['labels']
        device = log_probs_constrained.device

        # Prefer exact output_lengths from model over approximate batch.input_lengths.
        if outputs.get('output_lengths') is not None:
            input_lengths = outputs['output_lengths'].to(device)
        else:
            input_lengths = batch.get('input_lengths', None)
            if input_lengths is None:
                input_lengths = torch.full(
                    (log_probs_constrained.size(0),),
                    log_probs_constrained.size(1),
                    dtype=torch.long, device=device,
                )
            else:
                input_lengths = input_lengths.to(device)

        input_lengths = torch.clamp(input_lengths, max=log_probs_constrained.size(1))

        label_lengths = batch.get('label_lengths', None)
        if label_lengths is None:
            label_lengths = (labels != -100).sum(dim=1)
        else:
            label_lengths = label_lengths.to(device)

        valid_mask = label_lengths <= input_lengths
        if log_invalid_frac and not valid_mask.all():
            self.log('train/ctc_invalid_frac', 1.0 - valid_mask.float().mean(), on_step=True, prog_bar=False)

        if valid_mask.sum() == 0:
            return None

        labels_f = labels[valid_mask]
        input_lengths_f = input_lengths[valid_mask]
        label_lengths_f = label_lengths[valid_mask]

        logits_manner = outputs.get('logits_manner')
        logits_place = outputs.get('logits_place')
        logits_voice = outputs.get('logits_voice')

        art_labels = batch.get('articulatory_labels')
        if art_labels is not None:
            art_labels = {k: v[valid_mask] for k, v in art_labels.items()}

        logits_neural_v = outputs.get('logits_neural')
        hidden_states_v = outputs.get('hidden_states')

        outputs_f = {
            'logits_constrained': log_probs_constrained[valid_mask],
            'logits_neural': (logits_neural_v[valid_mask] if logits_neural_v is not None else log_probs_constrained),
            'hidden_states': (hidden_states_v[valid_mask] if hidden_states_v is not None else None),
            'logits_manner': logits_manner[valid_mask] if logits_manner is not None else None,
            'logits_place': logits_place[valid_mask] if logits_place is not None else None,
            'logits_voice': logits_voice[valid_mask] if logits_voice is not None else None,
        }

        return {
            'outputs_filtered': outputs_f,
            'labels': labels_f,
            'input_lengths': input_lengths_f,
            'label_lengths': label_lengths_f,
            'art_labels': art_labels,
            'valid_mask': valid_mask,
        }

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:

        outputs = self(batch)
        prepared = self._prepare_step(batch, outputs, log_invalid_frac=True)

        if prepared is None:
            zero_loss = torch.zeros((), device=next(self.parameters()).device, requires_grad=True)
            self.log('train/loss', zero_loss, on_step=True, on_epoch=True, prog_bar=True)
            return zero_loss

        outputs_f = prepared['outputs_filtered']
        labels_f = prepared['labels']
        input_lengths_f = prepared['input_lengths']
        label_lengths_f = prepared['label_lengths']
        art_labels = prepared['art_labels']

        # C7: Use per-speaker severity from TORGO_SEVERITY_MAP (continuous [0,5])
        # instead of the coarse binary status * 5.0 used previously.
        severity = _get_batch_severity(batch, outputs_f['logits_constrained'].device, self.config)

        losses = self.compute_loss(
            outputs_f,
            labels_f,
            input_lengths_f,
            label_lengths_f,
            articulatory_labels=art_labels,
            severity=severity[prepared['valid_mask']],
            attention_mask=self._downsample_attn_mask(batch, outputs_f['logits_constrained'], prepared['valid_mask']),
        )

        # Log metrics
        self.log('train/loss',         losses['loss'],       on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/loss_ctc',     losses['loss_ctc'],   on_step=False, on_epoch=True)
        self.log('train/loss_ce',      losses['loss_ce'],    on_step=False, on_epoch=True)
        self.log('train/loss_blank_kl', losses['loss_blank_kl'], on_step=False, on_epoch=True)
        self.log('train/blank_prob_mean', losses['blank_prob_mean'], on_step=True, on_epoch=True, prog_bar=True)
        if losses.get('loss_art') is not None:
            self.log('train/loss_art',      losses['loss_art'],       on_step=False, on_epoch=True)
        if losses.get('loss_ordinal') is not None:
            self.log('train/loss_ordinal',  losses['loss_ordinal'],   on_step=False, on_epoch=True)
        if losses.get('loss_symbolic_kl') is not None:
            self.log('train/loss_sym_kl',   losses['loss_symbolic_kl'], on_step=False, on_epoch=True)
        # Monitor β
        beta_val = outputs['beta']
        if isinstance(beta_val, torch.Tensor):
            beta_val = beta_val.mean()
        self.log('train/avg_beta', beta_val, on_step=False, on_epoch=True)

        # Q4: Gradient norm monitoring — incremental L2 norm to avoid
        # allocating a giant concatenated gradient tensor (~760 MB for 95M params).
        if self.global_step % 50 == 0:
            grad_norm_sq = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    grad_norm_sq += p.grad.detach().norm(2).item() ** 2
            grad_norm = grad_norm_sq ** 0.5
            self.log('train/grad_norm', grad_norm, on_step=True, prog_bar=False)

        return losses['loss']

    def on_train_epoch_start(self) -> None:
        """Three-stage progressive HuBERT unfreezing + staged blank-KL warmup (I2).

        Stage 1 (encoder_warmup_epochs, default 1):
            Unfreeze top 4 layers (8-11) after 1 warm-up epoch so the head
            stabilises quickly without wasting VRAM budget.
        Stage 2 (encoder_second_unfreeze_epoch, default 6):
            Unfreeze layers 6-11; keeps bottom 4 frozen for stability.
        Stage 3 (encoder_third_unfreeze_epoch, default 12):
            Unfreeze layers 4-11 for deepest dysarthric adaptation.
            With freeze_encoder_layers=[0-3] this unlocks the maximum
            number of HuBERT layers while preserving generic acoustic features.

        Blank-KL staging (I2): ramp lambda_blank_kl from a gentle initial value to
        the full target over training to avoid CTC collapse in early epochs when the
        phoneme head has not yet learned basic boundaries.
        """
        epoch = self.current_epoch + int(getattr(self, 'resume_epoch_offset', 0))
        effective_final = max(0, int(self.config.training.max_epochs) - 1)
        local_final = max(0, int(getattr(self.trainer, 'max_epochs', 1)) - 1)
        # Make resumed epoch numbering explicit for logs: when continuing from
        # epoch 21, logs should show effective epochs 22/23/24 even if the
        # local Trainer loop is 0/1/2.
        logger.info(
            "Effective epoch %d/%d (local %d/%d)",
            epoch, effective_final, self.current_epoch, local_final,
        )
        warmup_ep = self.config.training.encoder_warmup_epochs
        second_unfreeze = getattr(self.config.training, 'encoder_second_unfreeze_epoch', 6)
        third_unfreeze = getattr(self.config.training, 'encoder_third_unfreeze_epoch', 12)

        # Resume-safe stage selection: choose the highest stage implied by the
        # current epoch immediately. This avoids one-epoch lag when resuming at
        # late epochs (e.g., epoch 21 must directly enter Stage 3).
        if epoch >= third_unfreeze and not self._encoder_deeper_unfrozen:
            self.model.unfreeze_encoder(layers=[4, 5, 6, 7, 8, 9, 10, 11])
            self.encoder_unfrozen = True
            self._encoder_unfrozen = True
            self._encoder_deep_unfrozen = True
            self._encoder_deeper_unfrozen = True
            self._reset_hubert_lr_warmup()  # T-04: reset Adam state for newly-active params
            logger.info("Stage 3: Unfroze HuBERT layers 4-11 at epoch %d", epoch)

        elif epoch >= second_unfreeze and not self._encoder_deep_unfrozen:
            self.model.unfreeze_encoder(layers=[6, 7, 8, 9, 10, 11])
            self.encoder_unfrozen = True
            self._encoder_unfrozen = True
            self._encoder_deep_unfrozen = True
            self._reset_hubert_lr_warmup()
            logger.info("Stage 2: Unfroze HuBERT layers 6-11 at epoch %d", epoch)

        elif epoch >= warmup_ep and not self.encoder_unfrozen:
            self.model.unfreeze_encoder(layers=[8, 9, 10, 11])
            self.encoder_unfrozen = True
            self._encoder_unfrozen = True
            self._reset_hubert_lr_warmup()
            logger.info("Stage 1: Unfroze HuBERT layers 8-11 at epoch %d", epoch)

        # I2: Staged lambda_blank_kl ramp
        tr = self.config.training
        stage1_end = getattr(tr, 'blank_kl_stage1_end',  10)
        stage2_end = getattr(tr, 'blank_kl_stage2_end',  20)
        val_s1 = getattr(tr, 'blank_kl_stage1_value', 0.10)
        val_s2 = getattr(tr, 'blank_kl_stage2_value', 0.15)
        target = tr.lambda_blank_kl  # Final value (default 0.20)
        if epoch < stage1_end:
            current_bkl = val_s1
        elif epoch < stage2_end:
            current_bkl = val_s2
        else:
            current_bkl = target
        # Store as an instance attribute so compute_loss can read it without
        # mutating the config (which would violate per-run isolation in LOSO).
        self._current_lambda_blank_kl = current_bkl
        self.log('train/lambda_blank_kl', current_bkl, on_epoch=True, prog_bar=False)

        # Staged lambda_ordinal ramp
        stage1_end_ord = getattr(tr, 'ordinal_stage1_end', 10)
        stage2_end_ord = getattr(tr, 'ordinal_stage2_end', 20)
        val_s1_ord = getattr(tr, 'ordinal_stage1_value', 0.01)
        val_s2_ord = getattr(tr, 'ordinal_stage2_value', 0.03)
        target_ord = tr.lambda_ordinal
        if epoch < stage1_end_ord:
            current_ord = val_s1_ord
        elif epoch < stage2_end_ord:
            current_ord = val_s2_ord
        else:
            current_ord = target_ord
        self._current_lambda_ordinal = current_ord
        self.log('train/lambda_ordinal', current_ord, on_epoch=True, prog_bar=False)

        # Staged lambda_symbolic_kl ramp
        stage1_end_skl = getattr(tr, 'symbolic_kl_stage1_end', 5)
        stage2_end_skl = getattr(tr, 'symbolic_kl_stage2_end', 15)
        val_s1_skl = getattr(tr, 'symbolic_kl_stage1_value', 0.1)
        val_s2_skl = getattr(tr, 'symbolic_kl_stage2_value', 0.3)
        target_skl = tr.lambda_symbolic_kl
        if epoch < stage1_end_skl:
            current_skl = val_s1_skl
        elif epoch < stage2_end_skl:
            current_skl = val_s2_skl
        else:
            current_skl = target_skl
        self._current_lambda_symbolic_kl = current_skl
        self.log('train/lambda_symbolic_kl', current_skl, on_epoch=True, prog_bar=False)

    def _reset_hubert_lr_warmup(self) -> None:
        """Clear AdamW optimizer state for newly unfrozen HuBERT params.

        With CosineAnnealingWarmRestarts (interval='epoch'), the scheduler
        automatically restarts at the correct LR on the next epoch boundary.
        We only need to clear Adam state so the optimizer reinitializes
        exp_avg / exp_avg_sq / step for the newly active parameters.
        """
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        for pg in optimizer.param_groups:
            if pg.get('name') == 'hubert_encoder':
                for p in pg['params']:
                    if p.requires_grad and p in optimizer.state:
                        optimizer.state[p].clear()
                break

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        outputs = self(batch)
        prepared = self._prepare_step(batch, outputs)
        if prepared is None:
            return {}

        outputs_f = prepared['outputs_filtered']
        labels_f = prepared['labels']
        input_lengths_f = prepared['input_lengths']
        label_lengths_f = prepared['label_lengths']
        art_labels = prepared['art_labels']
        valid_mask = prepared['valid_mask']

        losses = self.compute_loss(
            outputs_f, labels_f, input_lengths_f, label_lengths_f,
            articulatory_labels=art_labels,
            severity=_get_batch_severity(batch, outputs_f['logits_constrained'].device, self.config)[valid_mask],
            attention_mask=self._downsample_attn_mask(batch, outputs_f['logits_constrained'], valid_mask),
        )

        log_probs_constrained = outputs_f['logits_constrained']

        output_lengths_full = outputs.get('output_lengths')
        output_lengths_pred = output_lengths_full[valid_mask] if output_lengths_full is not None else None
        predictions = decode_predictions(
            log_probs_constrained, self.phn_to_id, self.id_to_phn,
            output_lengths=output_lengths_pred,
        )
        references = decode_references(labels_f, self.id_to_phn)
        per_scores = [compute_per(pred, ref) for pred, ref in zip(predictions, references)]
        avg_per = float(np.mean(per_scores)) if per_scores else 0.0

        all_speakers = batch.get('speakers', [])
        valid_speakers = [
            all_speakers[i] for i in range(len(all_speakers))
            if i < len(valid_mask) and valid_mask[i].item()
        ] if all_speakers else []

        self.validation_step_outputs.append({
            'loss':      losses['loss'],
            'per':       avg_per,
            'per_scores': per_scores,
            'speakers':  valid_speakers,
            'predictions': predictions,
            'references':  references,
            'status':    batch['status'][valid_mask]
        })

        return losses

    def on_validation_epoch_end(self) -> None:
        """Aggregate and log validation metrics."""
        if not self.validation_step_outputs:
            return

        # Aggregate losses
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()

        # Macro-speaker PER: average per-speaker mean, then across speakers (§2.6 fix).
        # The old approach averaged batch-mean PER values, which over-weighted speakers
        # with more utterances.  The publication metric averages per-speaker first.
        speaker_per_map: Dict[str, List[float]] = defaultdict(list)
        for output in self.validation_step_outputs:
            for spk, per in zip(output.get('speakers', []), output.get('per_scores', [output['per']])):
                speaker_per_map[spk].append(per)
        if speaker_per_map:
            avg_per = float(np.mean([np.mean(v) for v in speaker_per_map.values()]))
        else:
            # Fallback when speaker info is unavailable
            avg_per = float(np.mean([x['per'] for x in self.validation_step_outputs]))

        # Stratified PER (dysarthric vs control)
        dysarthric_per, control_per = self._compute_stratified_per()

        # Log metrics
        self.log('val/loss', avg_loss, prog_bar=True)
        self.log('val/per', avg_per, prog_bar=True)
        # val_per: slash-free alias used in ModelCheckpoint filename.
        # PL 2.x expands {key:fmt} → "key=value" in filenames, so a template
        # like 'epoch={epoch:02d}-val_per={val/per:.3f}' would double-prefix:
        # epoch=epoch=04-val_per=val/per=0.885. Using a separately-logged
        # alias avoids the slash and the doubled key.
        self.log('val_per', avg_per, prog_bar=False, sync_dist=True)
        self.log('val/per_dysarthric', dysarthric_per)
        self.log('val/per_control', control_per)

        # Log learned constraint matrix diagnostics (implementation plan Phase 2).
        # Row entropy measures how diffuse/peaked each row of C has become;
        # low entropy → confident (peaked) constraint; high entropy → near-uniform
        # (constraint not learning phonologically meaningful patterns).
        # KL from prior measures drift away from the static symbolic initialisation.
        try:
            sl = self.model.symbolic_layer
            lm = getattr(sl, 'learnable_matrix', None)
            if lm is not None:
                C = torch.softmax(lm.logit_C.detach().float(), dim=-1)  # [V, V]
                eps = 1e-6
                row_entropy = -(C * torch.log(C + eps)).sum(dim=-1).mean()
                self.log('val/constraint_row_entropy', float(row_entropy), prog_bar=False)
                C_prior = sl.static_constraint_matrix.detach().float()
                kl_from_prior = (C * torch.log((C + eps) / (C_prior + eps))).sum(dim=-1).mean()
                self.log('val/constraint_kl_from_prior', float(kl_from_prior), prog_bar=False)
        except RuntimeError:
            logger.warning("Constraint diagnostics failed (non-fatal)", exc_info=True)

        # Clear for next epoch
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        """Aggregate and log test metrics using macro-speaker PER."""
        if not self.test_step_outputs:
            return

        avg_loss = torch.stack([x['loss'] for x in self.test_step_outputs]).mean()

        speaker_per_map: Dict[str, List[float]] = defaultdict(list)
        for output in self.test_step_outputs:
            for spk, per in zip(output.get('speakers', []), output.get('per_scores', [])):
                speaker_per_map[spk].append(per)
        if speaker_per_map:
            avg_per = float(np.mean([np.mean(v) for v in speaker_per_map.values()]))
        else:
            all_scores = [s for o in self.test_step_outputs for s in o.get('per_scores', [])]
            avg_per = float(np.mean(all_scores)) if all_scores else 0.0

        self.log('test/per', avg_per)

        self.test_step_outputs.clear()

    def _downsample_attn_mask(
        self,
        batch: Dict,
        log_probs: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """[FIX-6] Downsample attention mask to logit resolution with explicit stride.

        Rewritten to use explicit two-step stride calculation (HuBERT output
        length → optional stride-2 downsampling) for robustness on edge cases.
        Previously implicit calculation could fail on unusual sequence lengths.
        """
        attn_mask = batch.get('attention_mask')
        if attn_mask is None:
            return None
        T_log = log_probs.size(1)
        if T_log <= 0:
            return None

        # P2.2: mirror the model's actual temporal path.
        # Step 1) HuBERT feature-extractor output lengths from audio-frame mask.
        audio_lengths = attn_mask.sum(dim=1).long()
        hubert_lengths = self.model.hubert._get_feat_extract_output_lengths(audio_lengths).long()

        # Step 2) Optional stride-2 temporal downsampling (when active in forward).
        ablation_mode = getattr(self.config.training, 'ablation_mode', 'full')
        if self.model.temporal_downsampler is not None and ablation_mode != 'no_temporal_ds':
            logit_lengths = (hubert_lengths + 1) // 2
        else:
            logit_lengths = hubert_lengths

        positions = torch.arange(T_log, device=log_probs.device).unsqueeze(0)  # [1, T]
        lengths = logit_lengths.clamp(max=T_log).unsqueeze(1)  # [B, 1]
        ds = (positions < lengths).to(dtype=attn_mask.dtype)
        if valid_mask is not None:
            return ds[valid_mask]
        return ds

    def _compute_stratified_per(self) -> Tuple[float, float]:
        """
        Compute PER stratified by dysarthric vs control speakers.
        
        Returns:
            Tuple of (dysarthric_per, control_per)
        """
        dysarthric_per = []
        control_per = []

        for output in self.validation_step_outputs:
            # H-6: Reuse pre-computed per_scores (already in validation_step_outputs)
            # instead of calling compute_per() again for each utterance.
            stored_per = output.get('per_scores', [])
            for per, status in zip(stored_per, output['status']):
                if int(status) == 1:  # Dysarthric
                    dysarthric_per.append(per)
                else:  # Control
                    control_per.append(per)

        avg_dysarthric = np.mean(dysarthric_per) if dysarthric_per else 0.0
        avg_control = np.mean(control_per) if control_per else 0.0

        return avg_dysarthric, avg_control

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:

        outputs = self(batch)
        prepared = self._prepare_step(batch, outputs)
        if prepared is None:
            return {}

        outputs_f = prepared['outputs_filtered']
        labels_f = prepared['labels']
        input_lengths_f = prepared['input_lengths']
        label_lengths_f = prepared['label_lengths']
        art_labels = prepared['art_labels']
        valid_mask = prepared['valid_mask']

        losses = self.compute_loss(
            outputs_f, labels_f, input_lengths_f, label_lengths_f,
            articulatory_labels=art_labels,
            severity=_get_batch_severity(batch, outputs_f['logits_constrained'].device, self.config)[valid_mask],
            attention_mask=self._downsample_attn_mask(batch, outputs_f['logits_constrained'], valid_mask),
        )

        log_probs_constrained = outputs_f['logits_constrained']

        output_lengths_full = outputs.get('output_lengths')
        output_lengths_pred = output_lengths_full[valid_mask] if output_lengths_full is not None else None
        predictions = decode_predictions(
            log_probs_constrained, self.phn_to_id, self.id_to_phn,
            output_lengths=output_lengths_pred,
        )
        references = decode_references(labels_f, self.id_to_phn)
        per_scores = [compute_per(pred, ref) for pred, ref in zip(predictions, references)]

        all_speakers = batch.get('speakers', [])
        valid_speakers = [
            all_speakers[i] for i in range(len(all_speakers))
            if i < len(valid_mask) and valid_mask[i].item()
        ] if all_speakers else []

        self.test_step_outputs.append({
            'loss': losses['loss'],
            'per_scores': per_scores,
            'speakers': valid_speakers,
            'status': batch['status'][valid_mask],
        })

        self.log('test/loss', losses['loss'])

        return losses

    def configure_optimizers(self):
        """
        Configure optimizer with differential learning rates (audit S4):
          - HuBERT encoder:               lr × 0.1  (slow fine-tuning)
          - PhonemeClassifier + Adapter:  lr × 1.0
          - SymbolicConstraintLayer:      lr × 0.5  (keeps C near prior)
        """
        lr = self.config.training.learning_rate

        def params_of(*modules):
            for m in modules:
                if m is not None:
                    yield from m.parameters()

        param_groups = [
            # HuBERT encoder — slower LR to avoid catastrophic forgetting
            {
                'params': list(self.model.hubert.parameters()),
                'lr': lr * self.config.training.encoder_lr_multiplier,
                'name': 'hubert_encoder',
            },
            # Classification head + severity adapter
            {
                'params': list(params_of(
                    self.model.phoneme_classifier,
                    self.model.severity_adapter,
                    self.model.manner_head,
                    self.model.place_head,
                    self.model.voice_head,
                )),
                'lr': lr * 1.0,
                'name': 'classifier_heads',
            },
            # Symbolic layer (learnable C + β)
            {
                'params': list(self.model.symbolic_layer.parameters()),
                'lr': lr * self.config.training.symbolic_lr_multiplier,
                'name': 'symbolic_layer',
            },
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.training.weight_decay,
        )

        # [FIX-T04] Use CosineAnnealingWarmRestarts instead of OneCycleLR
        # T_0 = epochs until first restart (matches encoder_warmup_epochs = 1)
        # T_mult = 2 means each restart cycle is twice as long
        # eta_min = lr * cosine_eta_min_ratio
        # interval='epoch' steps per epoch, not per batch
        warmup_epochs = self.config.training.encoder_warmup_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.training.cosine_T0,
            T_mult=self.config.training.cosine_T_mult,
            eta_min=lr * self.config.training.cosine_eta_min_ratio,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1},
        }


class StratifiedMicroBatchSampler:
    """Ensures each micro-batch contains both dysarthric and control samples.

    Interleaves dysarthric and control indices so gradient accumulation sees
    balanced batches, preventing OrdinalContrastiveLoss from degenerating when
    all items in a micro-batch share the same severity.

    Args:
        labels: List of 0/1 labels for each sample (control=0, dysarthric=1).
        batch_size: Number of samples per micro-batch.
        dysarthric_ratio: Target fraction of dysarthric samples per batch.
    """

    def __init__(
        self,
        labels,
        batch_size: int,
        dysarthric_ratio: float = 0.75,
        seed: Optional[int] = None,
    ) -> None:
        self.dysarthric_idx = [i for i, l in enumerate(labels) if l == 1]
        self.control_idx = [i for i, l in enumerate(labels) if l == 0]
        self.batch_size = batch_size
        self.n_dys = max(1, int(dysarthric_ratio * batch_size))
        self.n_ctrl = batch_size - self.n_dys
        self._seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self._seed)
        n_dys_total = len(self.dysarthric_idx)
        n_ctrl_total = len(self.control_idx)
        n_total = n_dys_total + n_ctrl_total
        n_batches = n_total // self.batch_size
        n_dys_needed = n_batches * self.n_dys
        n_ctrl_needed = n_batches * self.n_ctrl

        # Tile minority class so all data is used every epoch
        if n_dys_needed > n_dys_total:
            repeats = n_dys_needed // n_dys_total + 1
            dys = (rng.permutation(self.dysarthric_idx).tolist() * repeats)[:n_dys_needed]
        else:
            dys = rng.permutation(self.dysarthric_idx).tolist()[:n_dys_needed]
        if n_ctrl_needed > n_ctrl_total:
            repeats = n_ctrl_needed // n_ctrl_total + 1
            ctrl = (rng.permutation(self.control_idx).tolist() * repeats)[:n_ctrl_needed]
        else:
            ctrl = rng.permutation(self.control_idx).tolist()[:n_ctrl_needed]

        batches = []
        for i in range(n_batches):
            batch = dys[i * self.n_dys:(i + 1) * self.n_dys] + ctrl[i * self.n_ctrl:(i + 1) * self.n_ctrl]
            rng.shuffle(batch)
            batches.append(batch)

        yield from batches

    def __len__(self):
        n_total = len(self.dysarthric_idx) + len(self.control_idx)
        return n_total // self.batch_size


def create_dataloaders(
    config: Config,
    dataset: TorgoNeuroSymbolicDataset
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    logger.info("Creating data splits...")

    # Speaker-stratified split
    df = dataset.df
    speakers = df['speaker'].unique()

    # Shuffle speakers
    np.random.seed(config.experiment.seed)
    np.random.shuffle(speakers)

    # Calculate split indices
    n_train = int(len(speakers) * config.data.train_split)
    n_val = int(len(speakers) * config.data.val_split)

    train_speakers = speakers[:n_train]
    val_speakers = speakers[n_train:n_train + n_val]
    test_speakers = speakers[n_train + n_val:]

    # Create sample indices for each split
    train_idx = df[df['speaker'].isin(train_speakers)].index.tolist()
    val_idx = df[df['speaker'].isin(val_speakers)].index.tolist()
    test_idx = df[df['speaker'].isin(test_speakers)].index.tolist()

    # C3: Guard against data leakage from sample-level fallback splits.
    # When the requested ratio produces empty partitions (e.g. only 3 speakers
    # with a 70/15/15 ratio), enforce a round-robin 1-speaker-per-split
    # assignment rather than silently mixing speakers across train/test.
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        n_spk = len(speakers)
        if n_spk < 3:
            raise ValueError(
                f"Speaker-stratified splits require at least 3 speakers; "
                f"found {n_spk}: {list(speakers)}. "
                "Add more speakers or switch to a different split strategy."
            )
        # Round-robin: one speaker per critical split, rest to train
        test_speakers = speakers[-1:]
        val_speakers = speakers[-2:-1]
        train_speakers = speakers[:-2]

        train_idx = df[df['speaker'].isin(train_speakers)].index.tolist()
        val_idx = df[df['speaker'].isin(val_speakers)].index.tolist()
        test_idx = df[df['speaker'].isin(test_speakers)].index.tolist()

        logger.warning(
            "Ratio-based split produced empty partition(s). Using round-robin speaker assignment: "
            "train=%s, val=%s, test=%s",
            list(train_speakers), list(val_speakers), list(test_speakers),
        )

    # Create subset datasets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    logger.info("Train: %d samples (%d speakers)", len(train_dataset), len(train_speakers))
    logger.info("Val: %d samples (%d speakers)", len(val_dataset), len(val_speakers))
    logger.info("Test: %d samples (%d speakers)", len(test_dataset), len(test_speakers))

    # Create collator
    collator = NeuroSymbolicCollator(dataset.processor)

    # Stratified micro-batch sampler ensures each gradient accumulation step
    # contains both dysarthric and control samples. This prevents
    # OrdinalContrastiveLoss from degenerating when all items in a micro-batch
    # share the same severity.
    train_df = dataset.df.iloc[train_idx]
    use_stratified = getattr(config.training, 'use_stratified_micro_batch', True)
    if use_stratified:
        train_labels = train_df['label'].tolist()
        train_sampler = StratifiedMicroBatchSampler(
            train_labels,
            batch_size=config.training.batch_size,
            dysarthric_ratio=config.training.stratified_dysarthric_ratio,
            seed=config.experiment.seed,
        )
        train_batch_sampler = train_sampler
        loader_batch_size = None
    else:
        speaker_counts = train_df['speaker'].value_counts().to_dict()
        train_weights = [1.0 / speaker_counts[dataset.df.iloc[i]['speaker']] for i in train_idx]
        train_weighted_sampler = WeightedRandomSampler(
            train_weights, num_samples=len(train_weights), replacement=True
        )
        train_batch_sampler = None
        loader_batch_size = config.training.batch_size

    # Create dataloaders with split-specific worker/prefetch settings.
    base_loader_kwargs = dict(
        batch_size=loader_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        worker_init_fn=_seed_worker,
    )

    train_loader_kwargs = dict(base_loader_kwargs)
    val_loader_kwargs = dict(base_loader_kwargs)
    test_loader_kwargs = dict(base_loader_kwargs)
    # Val/test loaders always use explicit batch_size
    val_loader_kwargs["batch_size"] = config.training.batch_size
    test_loader_kwargs["batch_size"] = config.training.batch_size
    if config.training.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = config.training.prefetch_factor
        val_loader_kwargs["persistent_workers"] = False
        val_loader_kwargs["prefetch_factor"] = 2
        test_loader_kwargs["persistent_workers"] = False
        test_loader_kwargs["prefetch_factor"] = 2

    if use_stratified:
        # batch_sampler is mutually exclusive with batch_size, shuffle, sampler, drop_last
        stratified_kwargs = {k: v for k, v in train_loader_kwargs.items()
                            if k not in ('batch_size', 'shuffle', 'sampler', 'drop_last')}
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            **stratified_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            sampler=train_weighted_sampler,
            **train_loader_kwargs,
        )

    val_loader = DataLoader(
        val_dataset,
        **val_loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        **test_loader_kwargs,
    )

    return train_loader, val_loader, test_loader


# Training-curve helpers (E4)

class _MetricLoggerCallback(pl.Callback):
    """
    Lightweight PyTorch Lightning callback that records epoch-level training
    and validation metrics for offline learning-curve plotting.

    Collected lists:
      - train_loss : from ``trainer.callback_metrics['train/loss']`` at epoch end
      - val_per    : from ``trainer.callback_metrics['val/per']`` at validation end
    """

    def __init__(self) -> None:
        super().__init__()
        self.train_loss: List[float] = []
        self.val_per:    List[float] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        loss = trainer.callback_metrics.get('train/loss')
        if loss is not None:
            self.train_loss.append(float(loss))

        # Log raw loss magnitudes and weighted contributions for hyperparameter analysis
        cm = trainer.callback_metrics
        cfg = pl_module.config.training
        components = [
            ('loss_ctc',      'CTC',          getattr(cfg, 'lambda_ctc', 0.8)),
            ('loss_ce',       'CE',           getattr(cfg, 'lambda_ce', 0.15)),
            ('loss_art',      'Art',          getattr(cfg, 'lambda_articulatory', 0.08)),
            ('loss_blank_kl', 'BlankKL',      getattr(cfg, 'lambda_blank_kl', 0.20)),
            ('loss_ordinal',  'Ordinal',      getattr(cfg, 'lambda_ordinal', 0.05)),
            ('loss_sym_kl',   'SymKL',        getattr(cfg, 'lambda_symbolic_kl', 0.5)),
        ]
        lines = []
        total_weighted = 0.0
        for metric_key, label, lam in components:
            raw = cm.get(f'train/{metric_key}')
            if raw is not None:
                raw_val = float(raw)
                weighted = raw_val * lam
                total_weighted += weighted
                pct = 100.0 * lam * raw_val / max(float(loss or 1), 1e-6)
                lines.append(f"  {label:>8}  {raw_val:8.4f} × {lam:.3f} = {weighted:8.4f} ({pct:5.1f}%)")
            else:
                lines.append(f"  {label:>8}  (not logged)")
        logger.info(
            "Epoch %d weighted loss breakdown:\n%s\n  Total = %8.4f",
            int(pl_module.current_epoch),
            "\n".join(lines),
            total_weighted,
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        per = trainer.callback_metrics.get('val/per')
        if per is not None:
            self.val_per.append(float(per))


class _CompactFoldProgressCallback(pl.Callback):
    """Compact, one-line epoch progress for LOSO folds (no tqdm/model table)."""

    @staticmethod
    def _fmt_metric(value: Optional[torch.Tensor]) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.3f}"
        except Exception:
            logger.warning("Failed to format metric %s", value)
            return "n/a"

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        effective_epoch = int(pl_module.current_epoch + int(getattr(pl_module, 'resume_epoch_offset', 0)))
        effective_final = max(0, int(getattr(pl_module.config.training, 'max_epochs', trainer.max_epochs)) - 1)

        train_loss = self._fmt_metric(metrics.get('train/loss_epoch') or metrics.get('train/loss'))
        val_loss = self._fmt_metric(metrics.get('val/loss'))
        val_per = self._fmt_metric(metrics.get('val/per'))
        blank_mean = self._fmt_metric(metrics.get('train/blank_prob_mean_epoch') or metrics.get('train/blank_prob_mean'))

        msg = (
            f"Epoch {effective_epoch}/{effective_final} | "
            f"train/loss={train_loss} | val/loss={val_loss} | "
            f"val/per={val_per} | blank={blank_mean}"
        )
        logger.info(msg)


def _save_learning_curve(
    train_losses: List[float],
    val_pers: List[float],
    results_dir: Path,
) -> None:

    if not train_losses and not val_pers:
        logger.warning("No metric history available; skipping learning curve.")
        return

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    epochs_loss = list(range(1, len(train_losses) + 1))
    epochs_per = list(range(1, len(val_pers) + 1))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_loss = '#e74c3c'
    color_per = '#3498db'

    if train_losses:
        ax1.plot(epochs_loss, train_losses, color=color_loss, linewidth=2.0,
                 marker='o', markersize=4, label='train/loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss', color=color_loss, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color_loss)

    ax2 = ax1.twinx()
    if val_pers:
        ax2.plot(epochs_per, val_pers, color=color_per, linewidth=2.0,
                 marker='s', markersize=4, label='val/per')
        ax2.set_ylabel('Validation PER', color=color_per, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color_per)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.title('Learning Curve: Training Loss & Validation PER', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    plt.tight_layout()

    save_path = results_dir / 'learning_curve.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Learning curve saved to %s", save_path)


def train(
    config: Optional[Config] = None,
    limit_train_batches: Optional[int] = None,
    ckpt_path: Optional[str] = None,
) -> Tuple[DysarthriaASRLightning, pl.Trainer]:

    # Mitigate memory fragmentation — critical for variable-length audio batches.
    # PyTorch ≥2.0 uses PYTORCH_ALLOC_CONF (PYTORCH_CUDA_ALLOC_CONF is deprecated).
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    torch.set_float32_matmul_precision('high')

    if config is None:
        config = get_default_config()

    # Set seeds for reproducibility
    pl.seed_everything(config.experiment.seed, workers=True)

    # Configure deterministic algorithms (CTC requires warn_only)
    if config.experiment.deterministic:
        # L6: For production / final submission runs, non-CTC ops should be
        # deterministic.  CTC itself requires warn_only=True (CUDA WARP_CTC
        # uses non-deterministic atomics); wrap the strict attempt in a try.
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except RuntimeError:
            torch.use_deterministic_algorithms(True, warn_only=True)
            warnings.warn(
                "CTC (or another op) requires non-deterministic CUDA ops; "
                "exact reproducibility cannot be guaranteed."
            )

    # Setup MLflow logging
    mlflow.set_tracking_uri(config.experiment.tracking_uri)
    mlflow.set_experiment(config.experiment.experiment_name)

    mlflow_logger = MLFlowLogger(
        experiment_name=config.experiment.experiment_name,
        run_name=config.experiment.run_name,
        tracking_uri=config.experiment.tracking_uri
    )

    # Log configuration
    config_dict = flatten_config_for_mlflow(config)
    mlflow_logger.log_hyperparams(config_dict)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = TorgoNeuroSymbolicDataset(
        manifest_path=str(config.data.manifest_path),
        processor_id=config.model.hubert_model_id,
        sampling_rate=config.data.sampling_rate,
        max_audio_length=config.data.max_audio_length
    )

    # Create dataloaders (test split unused here; evaluation is the caller's responsibility)
    train_loader, val_loader, _ = create_dataloaders(config, dataset)

    # Initialize model
    logger.info("Initializing neuro-symbolic model...")
    model = NeuroSymbolicASR(
        model_config=config.model,
        symbolic_config=config.symbolic,
        phn_to_id=dataset.phn_to_id,
        id_to_phn=dataset.id_to_phn,
        manner_to_id=dataset.manner_to_id,
        place_to_id=dataset.place_to_id,
        voice_to_id=dataset.voice_to_id
    )

    # Wrap in Lightning module
    lightning_model = DysarthriaASRLightning(
        model=model,
        config=config,
        phn_to_id=dataset.phn_to_id,
        id_to_phn=dataset.id_to_phn,
        class_weights=dataset.get_loss_weights(),
        articulatory_weights=dataset.get_articulatory_loss_weights()
    )

    # Setup callbacks
    checkpoint_dir = get_project_root() / "checkpoints" / config.experiment.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # D4: Persist config alongside checkpoint for reproducibility
    config.save(checkpoint_dir / "config.yaml")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='{epoch:02d}-{val_per:.3f}',
        monitor=config.training.monitor_metric,
        mode=config.training.monitor_mode,
        save_top_k=config.training.save_top_k,
        save_last=True
    )

    early_stop_patience = int(config.training.early_stopping_patience)
    if config.training.ablation_mode == 'full':
        min_full_patience = int(getattr(config.training, 'encoder_third_unfreeze_epoch', 12)) + 10
        early_stop_patience = max(early_stop_patience, min_full_patience)

    early_stop_callback = EarlyStopping(
        monitor=config.training.monitor_metric,
        patience=early_stop_patience,
        mode=config.training.monitor_mode,
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    metric_logger_cb = _MetricLoggerCallback()   # E4: collect epoch metrics for learning curve

    # Create trainer
    trainer_kwargs: Dict = dict(
        max_epochs=config.training.max_epochs,
        accelerator='gpu' if config.device == 'cuda' else 'cpu',
        devices=1,
        precision=config.training.precision,
        accumulate_grad_batches=config.training.gradient_accumulation_steps,
        gradient_clip_val=config.training.gradient_clip_val,
        val_check_interval=config.training.val_check_interval,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
        num_sanity_val_steps=config.training.num_sanity_val_steps,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, metric_logger_cb],
        deterministic=False,  # Handled manually above
        log_every_n_steps=config.experiment.log_every_n_steps,
    )
    if limit_train_batches is not None:
        trainer_kwargs['limit_train_batches'] = limit_train_batches
    trainer = pl.Trainer(**trainer_kwargs)

    # Train
    if ckpt_path:
        logger.info("Resuming from checkpoint: %s", ckpt_path)
    else:
        logger.info("Starting training from scratch...")
    trainer.fit(lightning_model, train_loader, val_loader, ckpt_path=ckpt_path)

    # Save learning curve immediately after fit (E4)
    _save_learning_curve(
        train_losses=metric_logger_cb.train_loss,
        val_pers=metric_logger_cb.val_per,
        results_dir=get_project_root() / "results" / config.experiment.run_name,
    )

    # Resolve checkpoint: prefer best scored ckpt, fall back to last.ckpt
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            best_model_path = str(last_ckpt)
            logger.warning("No best checkpoint saved; using last.ckpt: %s", best_model_path)
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir} after training."
            )
    else:
        logger.info("Best checkpoint: %s", best_model_path)

    # Load best checkpoint and return
    logger.info("Loading best model from checkpoint...")
    best_model = DysarthriaASRLightning.load_from_checkpoint(
        best_model_path,
        model=model,
        config=config,
        phn_to_id=dataset.phn_to_id,
        id_to_phn=dataset.id_to_phn,
        strict=False
    )

    logger.info("Training complete.")
    logger.info("Checkpoints: %s", checkpoint_dir)

    return best_model, trainer


def create_loso_splits(
    config: Config,
    dataset: TorgoNeuroSymbolicDataset,
    held_out_speaker: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    df = dataset.df
    # B5 fix: sort lexicographically before seeded shuffle so fold assignments
    # are independent of manifest row order (df['speaker'].unique() returns
    # first-occurrence order, which changes if the manifest is regenerated).
    # Must match pl.seed_everything seed used in train() / run_pipeline.py.
    all_speakers = sorted([s for s in df['speaker'].unique() if s != held_out_speaker])

    np.random.seed(config.experiment.seed)
    np.random.shuffle(all_speakers)

    n_val = max(1, int(len(all_speakers) * 0.15))
    val_speakers = all_speakers[:n_val]
    train_speakers = all_speakers[n_val:]

    train_idx = df[df['speaker'].isin(train_speakers)].index.tolist()
    val_idx = df[df['speaker'].isin(val_speakers)].index.tolist()
    test_idx = df[df['speaker'] == held_out_speaker].index.tolist()

    collator = NeuroSymbolicCollator(dataset.processor)

    # Speaker-balanced sampler for train split (§3.5 fix: was using class-level
    # dysarthric/control weights, causing high-utterance speakers to dominate).
    train_df = df.iloc[train_idx]
    speaker_counts_loso = train_df['speaker'].value_counts().to_dict()
    train_weights = [1.0 / max(speaker_counts_loso.get(df.iloc[i]['speaker'], 1), 1) for i in train_idx]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    def _make_loader(idx: List[int], shuffle_sampler: Optional[WeightedRandomSampler] = None, is_train: bool = False) -> DataLoader:
        loader_kwargs = dict(
            batch_size=config.training.batch_size,
            shuffle=False,
            sampler=shuffle_sampler,
            collate_fn=collator,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            worker_init_fn=_seed_worker,
        )
        if config.training.num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(is_train)
            loader_kwargs["prefetch_factor"] = (
                config.training.prefetch_factor if is_train else 2
            )
        return DataLoader(
            Subset(dataset, idx),
            **loader_kwargs,
        )

    return (
        _make_loader(train_idx, shuffle_sampler=sampler, is_train=True),
        _make_loader(val_idx, is_train=False),
        _make_loader(test_idx, is_train=False),
    )


def run_loso(
    config: Config,
    dataset: TorgoNeuroSymbolicDataset,
    resume: bool = False,
    force_speakers: Optional[List[str]] = None,
    use_beam_search: bool = False,
    beam_width: int = 10,
    generate_explanations: bool = False,
    compute_uncertainty: bool = False,
    uncertainty_n_samples: int = 20,
) -> Dict:

    # Deterministic speaker order for LOSO folds:
    # 1) Preserve group order by first appearance in the manifest (e.g. FC, MC, M, F)
    # 2) Sort numerically within each group so MC01 comes before MC02.
    raw_speakers = list(dataset.df['speaker'].unique())

    def _split_speaker_id(spk: str) -> Tuple[str, int]:
        m = re.match(r"([A-Za-z]+)(\d+)$", str(spk))
        if not m:
            return str(spk), 10**9
        return m.group(1), int(m.group(2))

    prefix_order: Dict[str, int] = {}
    for spk in raw_speakers:
        prefix, _ = _split_speaker_id(spk)
        if prefix not in prefix_order:
            prefix_order[prefix] = len(prefix_order)

    speakers = sorted(
        raw_speakers,
        key=lambda s: (prefix_order.get(_split_speaker_id(s)[0], 10**9), _split_speaker_id(s)[1], str(s)),
    )
    force_set = {s.strip() for s in (force_speakers or []) if str(s).strip()}
    logger.info(f"\n🔁 LOSO: {len(speakers)} folds ({speakers})")
    if force_set:
        unknown_force = sorted([s for s in force_set if s not in speakers])
        if unknown_force:
            logger.info(f"⚠️  Unknown --loso-force-speakers ignored: {unknown_force}")
        valid_force = sorted([s for s in force_set if s in speakers])
        if valid_force:
            logger.info(f"🎯 Forcing clean re-run for folds: {valid_force}")

    # Snapshot the base run name before the loop so that fold names are derived
    # from the original name rather than accumulating across folds (Bug B6).
    base_run_name = config.experiment.run_name

    per_per_fold: List[float] = []
    wer_per_fold: List[float] = []
    n_samples_per_fold: List[int] = []

    # Crash-safe progress file: allows resuming LOSO from last completed fold
    # (or continuing an interrupted fold from checkpoints/<fold>/last.ckpt).
    summary_dir = get_project_root() / "results"
    summary_dir.mkdir(parents=True, exist_ok=True)
    progress_path = summary_dir / f"{base_run_name}_loso_progress.json"

    completed_folds: Dict[str, Dict] = {}
    new_fold_elapsed_sec: List[float] = []
    if resume and progress_path.exists():
        try:
            with open(progress_path, "r", encoding="utf-8") as _pf:
                progress = json.load(_pf)
            for fold in progress.get("folds", []):
                if fold.get("status") == "completed" and "speaker" in fold:
                    completed_folds[fold["speaker"]] = fold
            if completed_folds:
                logger.info(
                    f"♻️  Resuming LOSO from progress file: {progress_path} "
                    f"({len(completed_folds)} completed fold(s) found)"
                )
        except Exception as exc:
            logger.info(f"⚠️  Failed to parse LOSO progress file ({progress_path}): {exc}")

    def _write_progress() -> None:
        """Persist fold-level progress atomically for crash-safe LOSO resume."""
        progress_payload = {
            "run_name": base_run_name,
            "fold_speakers": speakers,
            "max_epochs": config.training.max_epochs,
            "folds": [
                {
                    "speaker": sp,
                    "status": "completed",
                    "per": float(completed_folds[sp].get("per", float("nan"))),
                    "wer": float(completed_folds[sp].get("wer", float("nan"))),
                    "n_samples": int(completed_folds[sp].get("n_samples", 0)),
                    "trained_epochs": int(completed_folds[sp].get("trained_epochs", 0)),
                    "elapsed_sec": float(completed_folds[sp].get("elapsed_sec", 0.0)),
                    "checkpoint": completed_folds[sp].get("checkpoint"),
                    "results_dir": completed_folds[sp].get("results_dir"),
                }
                for sp in speakers if sp in completed_folds
            ],
        }
        tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as _tf:
            json.dump(progress_payload, _tf, indent=2)
        tmp_path.replace(progress_path)

    for i, spk in enumerate(speakers):
        logger.info(f"\n── Fold {i+1}/{len(speakers)}: held-out speaker = {spk} ──")
        fold_run_name = f"{base_run_name}_loso_{spk}"
        config.experiment.run_name = fold_run_name

        if spk in force_set:
            # Explicitly invalidate this fold so it is retrained with the latest
            # code path (used for recovering from prior resume/stage bugs).
            completed_folds.pop(spk, None)
            force_ckpt_dir = get_project_root() / "checkpoints" / fold_run_name
            force_results_dir = get_project_root() / "results" / fold_run_name
            if force_ckpt_dir.exists():
                shutil.rmtree(force_ckpt_dir)
            if force_results_dir.exists():
                shutil.rmtree(force_results_dir)
            logger.info(f"   🔄 Forced clean rerun: cleared {force_ckpt_dir.name} and {force_results_dir.name}")

        if spk in completed_folds:
            # Trust crash-safe progress file as source-of-truth for completed folds.
            # Re-opening based on checkpoint epoch causes completed folds to be
            # retrained repeatedly after weights-only continuation flows.
            cached = completed_folds[spk]
            fold_per = float(cached.get('per', float('nan')))
            fold_wer = float(cached.get('wer', float('nan')))
            fold_n = int(cached.get('n_samples', 0))
            per_per_fold.append(fold_per)
            wer_per_fold.append(fold_wer)
            n_samples_per_fold.append(fold_n)
            logger.info(
                f"   ⏭️  Skipping completed fold {spk}: "
                f"PER={fold_per:.4f} | WER={fold_wer:.4f} | n={fold_n}"
            )
            continue

        fold_start_time = time.time()

        train_loader, val_loader, test_loader = create_loso_splits(config, dataset, spk)

        # Initialise a fresh model for each fold
        model = NeuroSymbolicASR(
            model_config=config.model,
            symbolic_config=config.symbolic,
            phn_to_id=dataset.phn_to_id,
            id_to_phn=dataset.id_to_phn,
            manner_to_id=dataset.manner_to_id,
            place_to_id=dataset.place_to_id,
            voice_to_id=dataset.voice_to_id,
        )
        lm = DysarthriaASRLightning(
            model=model, config=config,
            phn_to_id=dataset.phn_to_id, id_to_phn=dataset.id_to_phn,
            class_weights=dataset.get_loss_weights(),
            articulatory_weights=dataset.get_articulatory_loss_weights(),
        )

        total_params = sum(p.numel() for p in lm.model.parameters())
        trainable_params = sum(p.numel() for p in lm.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        logger.info(
            f"   🧠 Params: total={total_params/1e6:.1f}M | "
            f"trainable={trainable_params/1e6:.1f}M | frozen={frozen_params/1e6:.1f}M"
        )

        ckpt_dir = get_project_root() / "checkpoints" / fold_run_name
        ckpt_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir), monitor='val_per', mode='min',
            filename='{epoch:02d}-{val_per:.3f}',
            save_top_k=1, save_last=True,
        )
        # C-02: Create a per-fold MLFlowLogger so each fold's metrics are tracked
        # individually in MLflow under the same experiment.  The trainer is rebuilt
        # per fold anyway (fresh lm + ckpt_cb), so there is no extra cost.
        fold_logger = MLFlowLogger(
            experiment_name=config.experiment.experiment_name,
            run_name=fold_run_name,
            tracking_uri=config.experiment.tracking_uri,
        )
        trainer_max_epochs = int(config.training.max_epochs)
        weights_only_resume = False
        resumed_epoch = -1
        resume_ckpt = ckpt_dir / 'last.ckpt'
        # Persisted marker for interrupted weights-only resumes. This avoids
        # repeatedly re-entering scheduler-exhausted detection on every rerun
        # when last.ckpt has not advanced yet.
        weights_only_marker = ckpt_dir / 'weights_only_resume.pt'
        ckpt_path = str(resume_ckpt) if resume and resume_ckpt.exists() else None
        skip_fit = False

        # If a prior run already switched this fold to weights-only continuation,
        # continue directly from the persisted model state + epoch offset.
        if resume and weights_only_marker.exists() and not skip_fit:
            try:
                marker = torch.load(weights_only_marker, map_location='cpu')
                marker_start_epoch = int(marker.get('start_epoch', 0))
                marker_state = marker.get('state_dict')
                if marker_state is not None and marker_start_epoch < int(config.training.max_epochs):
                    lm.load_state_dict(marker_state, strict=False)
                    lm.resume_epoch_offset = marker_start_epoch
                    lm._sync_unfreeze_flags_from_epoch(lm.resume_epoch_offset, apply_to_model=True)
                    trainer_max_epochs = max(1, int(config.training.max_epochs) - marker_start_epoch)
                    resumed_epoch = marker_start_epoch - 1
                    weights_only_resume = True
                    ckpt_path = None
                    logger.info(
                        f"   ♻️  Continuing existing weights-only resume "
                        f"(start_epoch={marker_start_epoch}, remaining_epochs={trainer_max_epochs})"
                    )
                elif marker_start_epoch >= int(config.training.max_epochs):
                    # Stale marker from an older run target; drop it.
                    weights_only_marker.unlink(missing_ok=True)
            except Exception as exc:
                logger.info(f"   ⚠️  Could not load weights-only marker {weights_only_marker.name}: {exc}")

        if ckpt_path:
            # Lightning cannot restore a checkpoint whose current_epoch is
            # >= Trainer(max_epochs). In deadline-driven runs we may reduce
            # max_epochs after some folds have progressed further; when that
            # happens, skip training for this fold and evaluate the checkpoint.
            ckpt_meta = torch.load(resume_ckpt, map_location='cpu')
            resumed_epoch = int(ckpt_meta.get('epoch', -1))
            # Zero-based epoch indexing: if resumed_epoch >= max_epochs-1, the
            # target training budget is already fully consumed.
            if resumed_epoch >= (int(config.training.max_epochs) - 1):
                logger.info(
                    f"   ⏭️  Resume checkpoint epoch={resumed_epoch} is already >= "
                    f"final_epoch={int(config.training.max_epochs) - 1}; skipping fit and "
                    f"evaluating checkpoint directly."
                )
                lm = DysarthriaASRLightning.load_from_checkpoint(
                    str(resume_ckpt),
                    model=model,
                    config=config,
                    phn_to_id=dataset.phn_to_id,
                    id_to_phn=dataset.id_to_phn,
                    class_weights=dataset.get_loss_weights(),
                    articulatory_weights=dataset.get_articulatory_loss_weights(),
                    strict=False,
                )
                skip_fit = True
                ckpt_path = None
            else:
                # Guard against CosineAnnealingWarmRestarts exhaustion when
                # extending runs or changing batch/accum settings between
                # resumed launches. If scheduler state is already at/over
                # total_steps, Lightning resume will fail with: "Tried to step
                # X times... total steps Y".  In that case, resume model
                # weights only (fresh optimizer/scheduler).
                sched_exhausted = False
                sched_total_steps = None
                sched_last_epoch = None
                lr_states = ckpt_meta.get('lr_schedulers', [])
                if isinstance(lr_states, list) and lr_states:
                    s0 = lr_states[0] if isinstance(lr_states[0], dict) else {}
                    sched_total_steps = s0.get('total_steps')
                    sched_last_epoch = s0.get('last_epoch', s0.get('_step_count'))
                    if sched_total_steps is not None and sched_last_epoch is not None:
                        sched_exhausted = int(sched_last_epoch) >= int(sched_total_steps)

                # Also treat an undersized total_steps as exhausted for the new run.
                if sched_total_steps is not None:
                    steps_per_epoch = int(np.ceil(len(train_loader) / max(config.training.gradient_accumulation_steps, 1)))
                    required_total_steps = int(config.training.max_epochs) * steps_per_epoch
                    if int(sched_total_steps) < required_total_steps:
                        sched_exhausted = True

                if sched_exhausted:
                    logger.info(
                        f"   ⚠️  Scheduler state in {resume_ckpt.name} is exhausted/undersized "
                        f"(last_epoch={sched_last_epoch}, total_steps={sched_total_steps}); "
                        f"resuming weights-only with fresh optimizer/scheduler."
                    )
                    lm = DysarthriaASRLightning.load_from_checkpoint(
                        str(resume_ckpt),
                        model=model,
                        config=config,
                        phn_to_id=dataset.phn_to_id,
                        id_to_phn=dataset.id_to_phn,
                        class_weights=dataset.get_loss_weights(),
                        articulatory_weights=dataset.get_articulatory_loss_weights(),
                        strict=False,
                    )
                    # Continue only the remaining epochs, not a full fresh run.
                    # Example: resumed_epoch=21, max_epochs=25 -> run 3 epochs (22,23,24).
                    start_epoch = resumed_epoch + 1
                    remaining_epochs = max(1, int(config.training.max_epochs) - start_epoch)
                    lm.resume_epoch_offset = start_epoch
                    lm._sync_unfreeze_flags_from_epoch(lm.resume_epoch_offset, apply_to_model=True)
                    trainer_max_epochs = remaining_epochs
                    weights_only_resume = True
                    ckpt_path = None
                    # Persist lightweight resume intent + model weights so if
                    # this run is interrupted before a new last.ckpt is written,
                    # subsequent reruns continue directly without repeating this
                    # scheduler-exhausted path.
                    try:
                        torch.save(
                            {
                                'start_epoch': start_epoch,
                                'state_dict': lm.state_dict(),
                            },
                            weights_only_marker,
                        )
                    except Exception as exc:
                        logger.info(f"   ⚠️  Could not write weights-only marker {weights_only_marker.name}: {exc}")
                    logger.info(
                        f"   ↪️  Weights-only continuation: start_epoch={start_epoch}, "
                        f"remaining_epochs={remaining_epochs}"
                    )
                else:
                    logger.info(f"   ♻️  Resuming fold training from {resume_ckpt}")

        trainer = pl.Trainer(
            max_epochs=trainer_max_epochs,
            accelerator='gpu' if config.device == 'cuda' else 'cpu',
            devices=1,
            precision=config.training.precision,
            accumulate_grad_batches=config.training.gradient_accumulation_steps,
            gradient_clip_val=config.training.gradient_clip_val,
            callbacks=[
                ckpt_cb,
                EarlyStopping(
                    monitor='val_per',
                    patience=(
                        max(
                            int(getattr(config.training, 'loso_early_stopping_patience', config.training.early_stopping_patience)),
                            int(getattr(config.training, 'encoder_third_unfreeze_epoch', 12)) + 10,
                        )
                        if config.training.ablation_mode == 'full'
                        else int(config.training.early_stopping_patience)
                    ),
                    mode='min'
                ),
                _CompactFoldProgressCallback(),
            ],
            val_check_interval=config.training.val_check_interval,
            check_val_every_n_epoch=config.training.check_val_every_n_epoch,
            num_sanity_val_steps=config.training.num_sanity_val_steps,
            logger=fold_logger,
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=config.experiment.log_every_n_steps,
        )

        if not skip_fit:
            trainer.fit(lm, train_loader, val_loader, ckpt_path=ckpt_path)

        # Evaluate on held-out speaker using the best checkpoint available for
        # this fold (across prior + newly resumed epochs), not just the last
        # in-memory epoch state.
        results_dir = get_project_root() / "results" / fold_run_name
        results_dir.mkdir(parents=True, exist_ok=True)

        best_eval_ckpt = resolve_best_fold_checkpoint(ckpt_dir)
        eval_model_obj = lm.model
        if best_eval_ckpt is not None:
            try:
                eval_lm = DysarthriaASRLightning.load_from_checkpoint(
                    str(best_eval_ckpt),
                    model=model,
                    config=config,
                    phn_to_id=dataset.phn_to_id,
                    id_to_phn=dataset.id_to_phn,
                    class_weights=dataset.get_loss_weights(),
                    articulatory_weights=dataset.get_articulatory_loss_weights(),
                    strict=False,
                )
                eval_model_obj = eval_lm.model
                logger.info(f"   🎯 Evaluating best fold checkpoint: {best_eval_ckpt.name}")
            except Exception as exc:
                logger.info(f"   ⚠️  Failed to load best checkpoint for eval ({best_eval_ckpt}): {exc}")

        eval_results = evaluate_model(
            model=eval_model_obj, dataloader=test_loader,
            device=config.device,
            phn_to_id=dataset.phn_to_id, id_to_phn=dataset.id_to_phn,
            results_dir=results_dir,
            use_beam_search=use_beam_search,
            beam_width=beam_width,
            generate_explanations=generate_explanations,
            compute_uncertainty=compute_uncertainty,
            uncertainty_n_samples=uncertainty_n_samples,
            val_loader=val_loader,
            config=config,
            ablation_mode=config.training.ablation_mode,
        )
        fold_per = eval_results.get('avg_per', float('nan'))
        fold_wer = eval_results.get('wer', float('nan'))
        fold_n = eval_results.get('overall', {}).get('n_samples', 0)
        fold_elapsed_sec = max(0.0, time.time() - fold_start_time)
        if skip_fit:
            trained_epochs = resumed_epoch
        elif weights_only_resume:
            trained_epochs = int(getattr(lm, 'resume_epoch_offset', 0) + getattr(trainer, 'current_epoch', 0))
        else:
            trained_epochs = int(getattr(trainer, 'current_epoch', 0))
        per_per_fold.append(fold_per)
        wer_per_fold.append(fold_wer)
        n_samples_per_fold.append(fold_n)
        completed_folds[spk] = {
            'per': fold_per,
            'wer': fold_wer,
            'n_samples': fold_n,
            'trained_epochs': trained_epochs,
            'elapsed_sec': fold_elapsed_sec,
            'checkpoint': str(best_eval_ckpt) if best_eval_ckpt is not None else str(ckpt_dir / 'last.ckpt'),
            'results_dir': str(results_dir),
        }
        new_fold_elapsed_sec.append(float(fold_elapsed_sec))
        _write_progress()
        elapsed_min = fold_elapsed_sec / 60.0
        done_folds = len(completed_folds)
        remaining_folds = len(speakers) - done_folds
        avg_elapsed_sec = (
            float(np.mean(new_fold_elapsed_sec))
            if new_fold_elapsed_sec
            else float(fold_elapsed_sec)
        )
        eta_total_sec = max(0.0, avg_elapsed_sec * remaining_folds)
        eta_h = int(eta_total_sec // 3600)
        eta_m = int((eta_total_sec % 3600) // 60)
        logger.info(
            f"   Fold PER ({spk}): {fold_per:.4f}  |  WER: {fold_wer:.4f}  |  n={fold_n}"
            f"  |  epochs={trained_epochs}  |  time={elapsed_min:.1f} min"
        )
        if done_folds > 0:
            logger.info(
                f"   Progress: {done_folds}/{len(speakers)} folds complete"
                f"  |  avg/fold={avg_elapsed_sec/60.0:.1f} min"
                f"  |  ETA remaining≈{eta_h}h {eta_m}m"
            )

        # Fold finished successfully; cleanup any persisted weights-only marker.
        if weights_only_marker.exists():
            try:
                weights_only_marker.unlink()
            except Exception:
                logger.warning("Failed to clean up weights-only marker %s", weights_only_marker)

        # H-1: Explicit VRAM cleanup between folds — prevents OOM on 8 GB card.
        del trainer, lm, model
        torch.cuda.empty_cache()

    # Bootstrap 95% CI over fold PERs
    per_arr = np.array(per_per_fold)
    macro_per = float(np.nanmean(per_arr))
    rng = np.random.default_rng(42)
    n_bootstrap = int(getattr(config.training, 'loso_bootstrap_samples', LOSO_BOOTSTRAP_SAMPLES))
    boot = np.array([rng.choice(per_arr, size=len(per_arr), replace=True).mean()
                     for _ in range(n_bootstrap)])
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # Weighted-average PER (weighted by number of test samples per fold)
    n_arr = np.array(n_samples_per_fold, dtype=float)
    valid_mask = ~np.isnan(per_arr) & (n_arr > 0)
    weighted_per = (
        float(np.sum(per_arr[valid_mask] * n_arr[valid_mask]) / np.sum(n_arr[valid_mask]))
        if valid_mask.any() else float('nan')
    )

    severity_weights = np.array([get_speaker_severity(spk) for spk in speakers], dtype=float)
    dysarthric_mask = severity_weights > 0.0
    dysarthric_avg_per = float(np.nanmean(per_arr[dysarthric_mask])) if np.any(dysarthric_mask) else float('nan')
    control_avg_per = float(np.nanmean(per_arr[~dysarthric_mask])) if np.any(~dysarthric_mask) else float('nan')
    sev_weight = 1.0 + (severity_weights / 5.0)
    sev_valid = ~np.isnan(per_arr)
    severity_weighted_per = (
        float(np.nansum(per_arr[sev_valid] * sev_weight[sev_valid]) / np.nansum(sev_weight[sev_valid]))
        if np.any(sev_valid) else float('nan')
    )

    # WER aggregation (both simple mean and weighted)
    wer_arr = np.array([w for w in wer_per_fold if not np.isnan(w)])
    macro_wer = float(np.nanmean(wer_arr)) if len(wer_arr) > 0 else float('nan')
    wer_arr_full = np.array(wer_per_fold, dtype=float)
    wer_valid_mask = ~np.isnan(wer_arr_full) & (n_arr > 0)
    weighted_wer = (
        float(np.sum(wer_arr_full[wer_valid_mask] * n_arr[wer_valid_mask]) / np.sum(n_arr[wer_valid_mask]))
        if wer_valid_mask.any() else float('nan')
    )

    logger.info(f"\n✅ LOSO Complete:")
    logger.info(f"   Mean PER     = {macro_per:.4f}  [95% CI: {ci_lo:.4f} – {ci_hi:.4f}]")
    logger.info(f"   Weighted PER = {weighted_per:.4f}  (weighted by fold sample count)")
    logger.info(f"   Dys PER      = {dysarthric_avg_per:.4f}  |  Ctrl PER = {control_avg_per:.4f}")
    logger.info(f"   Sev-weighted = {severity_weighted_per:.4f}")
    logger.info(f"   Mean WER     = {macro_wer:.4f}  |  Weighted WER = {weighted_wer:.4f}")

    loso_summary = {
        'per_per_fold': per_per_fold,
        'wer_per_fold': wer_per_fold,
        'n_samples_per_fold': n_samples_per_fold,
        'fold_speakers': speakers,
        # Simple (macro) averages
        'macro_avg_per': macro_per,
        'per_95ci': [ci_lo, ci_hi],
        'macro_avg_wer': macro_wer,
        # Sample-weighted averages (I3 requirement)
        'weighted_avg_per': weighted_per,
        'weighted_avg_wer': weighted_wer,
        # Severity-stratified LOSO summaries
        'dysarthric_avg_per': dysarthric_avg_per,
        'control_avg_per': control_avg_per,
        'severity_weighted_per': severity_weighted_per,
    }

    # Save aggregated LOSO summary to results/{base_run_name}_loso_summary.json (I3)
    summary_path = summary_dir / f"{base_run_name}_loso_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as _sf:
        json.dump(loso_summary, _sf, indent=2)
    logger.info(f"💾 LOSO summary saved to {summary_path}")

    return loso_summary


def main() -> None:
    """Main entry point for training script."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Neuro-Symbolic Dysarthria ASR")
    parser.add_argument('--config',              type=str, help='Path to config YAML file')
    parser.add_argument('--run-name',            type=str, help='MLflow run name')
    parser.add_argument('--ablation',            type=str, default='full',
                        choices=['full', 'neural_only', 'symbolic_only', 'no_art_heads',
                                 'no_constraint_matrix', 'no_spec_augment', 'no_temporal_ds',
                                 'no_severity_adapter'],
                        help='Ablation mode (default: full)')
    parser.add_argument('--loso',                action='store_true',
                        help='Run Leave-One-Speaker-Out cross-validation')
    parser.add_argument('--resume-loso', '--resume_loso', dest='resume_loso', action='store_true',
                        help='Resume LOSO from progress/checkpoints when available')
    parser.add_argument('--max_epochs', '--max-epochs', type=int,
                        help='Override max_epochs')
    parser.add_argument('--early_stopping_patience', type=int,
                        help='Override early_stopping_patience')
    parser.add_argument('--limit_train_batches', type=int,
                        help='Limit batches per epoch (smoke test)')
    args = parser.parse_args()

    # Load or create config
    config = Config.load(Path(args.config)) if args.config else get_default_config()

    if args.run_name:
        config.experiment.run_name = args.run_name
    config.training.ablation_mode = args.ablation
    config.training.use_loso = args.loso
    if args.max_epochs:
        config.training.max_epochs = args.max_epochs
    if args.early_stopping_patience is not None:
        config.training.early_stopping_patience = args.early_stopping_patience

    if args.loso:
        # LOSO mode: load full dataset then run CV
        pl.seed_everything(config.experiment.seed, workers=True)
        dataset = TorgoNeuroSymbolicDataset(
            manifest_path=str(config.data.manifest_path),
            processor_id=config.model.hubert_model_id,
            sampling_rate=config.data.sampling_rate,
            max_audio_length=config.data.max_audio_length,
        )
        run_loso(config, dataset, resume=args.resume_loso)
    else:
        train(config, limit_train_batches=args.limit_train_batches)


if __name__ == "__main__":
    main()
