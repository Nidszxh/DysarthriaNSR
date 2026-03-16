"""
Training Pipeline for Neuro-Symbolic Dysarthric Speech Recognition

Orchestrates model training using PyTorch Lightning with multi-task learning,
symbolic constraints, and comprehensive evaluation metrics.
"""

import sys
from collections import defaultdict
import warnings
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# Add src directory to path for imports
project_root = Path(__file__).resolve().parent
sys.path.insert(2, str(project_root / "src"))

from src.utils.config import Config, get_default_config, get_project_root, get_speaker_severity
from src.utils.sequence_utils import align_labels_to_logits
from src.data.dataloader import NeuroSymbolicCollator, TorgoNeuroSymbolicDataset
import os
from src.models.model import NeuroSymbolicASR
from src.models.losses import OrdinalContrastiveLoss, BlankPriorKLLoss, SymbolicKLLoss

# Import evaluate functions from root level
from evaluate import compute_per, evaluate_model, decode_predictions, decode_references

warnings.filterwarnings('ignore')


def flatten_config_for_mlflow(config: Config) -> Dict:

    safe_params = {}
    
    def add_params(prefix: str, obj) -> None:
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
    ):
        """
        Initialize Lightning module.
        
        Args:
            model: Neuro-symbolic ASR model
            config: Training configuration
            phn_to_id: Phoneme to ID mapping
            id_to_phn: ID to phoneme mapping
        """
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

        # B2 fix: removed hard ×1.5 multiplier for <BLANK> and <PAD>.
        # Those were insertion-era heuristics (baseline_v1/v2, I/D=4.6×).
        # I/D is now 0.87× (resolved in v4); adding extra blank weight compounds
        # with blank_target_prob KL and natural CTC blank-dominance, producing
        # triple blank pressure that contributes to deletion/substitution dominance.
        # blank_priority_weight config field remains (set to 1.0) for future use.

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
                self.symbolic_kl_loss = SymbolicKLLoss(sl.static_constraint_matrix)

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
            extra   = set(self.phn_to_id) - set(saved_vocab)
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
                    pass
    
    def forward(self, batch: Dict) -> Dict:

        speakers = batch.get('speakers', [])
        status   = batch['status']

        # Use per-speaker severity scores when speaker IDs are available
        if speakers and isinstance(speakers[0], str):
            severity = torch.tensor(
                [get_speaker_severity(s) for s in speakers],
                dtype=torch.float32,
                device=status.device,
            )
        else:
            # Fallback: binary mapping 0→0.0, 1→5.0
            severity = status.float() * 5.0

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

        # 1. CTC Loss (primary alignment loss — always)
        loss_ctc = self._compute_ctc_loss(
            log_probs_constrained, labels, input_lengths, label_lengths
        )

        # 2. Frame-level CE applied to NEURAL logits (not constrained log-probs)
        #    Fixes the original misaligned-label gradient pathology (audit S2)
        loss_ce = self._compute_ce_loss(outputs.get('logits_neural', log_probs_constrained), labels)

        # 3. Articulatory auxiliary heads
        loss_art = None
        if ablation not in ("neural_only", "no_art_heads") and articulatory_labels and outputs.get('logits_manner') is not None:
            art_losses = []
            for key in ["manner", "place", "voice"]:
                logits = outputs.get(f"logits_{key}")
                labels_art = articulatory_labels.get(key)
                if logits is None or labels_art is None:
                    continue
                art_losses.append(self._compute_articulatory_ce_loss(logits, labels_art, key))
            if art_losses:
                loss_art = torch.stack(art_losses).mean()

        # 4. Blank-prior KL (insertion fix)
        loss_blank_kl = self.blank_kl_loss(log_probs_constrained, attention_mask)

        # 5. Ordinal contrastive severity loss (Proposal P1)
        loss_ordinal = None
        hidden = outputs.get('hidden_states')
        if hidden is not None and severity is not None and ablation != 'symbolic_only':
            loss_ordinal = self.ordinal_loss(hidden, severity, attention_mask)

        # 6. Symbolic KL anchor (Proposal P2 — learnable constraint matrix)
        loss_symbolic_kl = None
        if self.config.model.use_learnable_constraint and ablation not in ('neural_only', 'symbolic_only', 'no_constraint_matrix'):
            sl = self.model.symbolic_layer
            if sl.learnable_matrix is not None and self.symbolic_kl_loss is not None:
                loss_symbolic_kl = self.symbolic_kl_loss(sl.learnable_matrix.logit_C)

        # ── Ablation-mode weighting ──────────────────────────────────────────
        lambda_ctc  = 0.0 if ablation == 'symbolic_only' else float(self.lambda_ctc)
        lambda_ce   = 0.0 if ablation == 'symbolic_only' else float(self.lambda_ce)
        lambda_art  = float(self.lambda_articulatory)
        lambda_ord  = self.config.training.lambda_ordinal
        # I2: Use staged lambda_blank_kl set by on_train_epoch_start (falls back to
        # config value during evaluation/test when the attribute is not set).
        lambda_bkl  = getattr(self, '_current_lambda_blank_kl', self.config.training.lambda_blank_kl)
        lambda_skl  = self.config.training.lambda_symbolic_kl

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

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Training batch
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        outputs = self(batch)
        
        # Compute lengths
        log_probs_constrained = outputs.get('log_probs_constrained', outputs['logits_constrained'])
        labels = batch['labels']

        # Prefer exact output_lengths from model (computed via HuBERT CNN formula)
        # over approximate batch.input_lengths (collator uses // 320 which is off by
        # 0-2 frames and passes wrong values to CTCLoss).
        if outputs.get('output_lengths') is not None:
            input_lengths = outputs['output_lengths'].to(log_probs_constrained.device)
        else:
            input_lengths = batch.get('input_lengths')
            if input_lengths is None:
                input_lengths = torch.full(
                    (log_probs_constrained.size(0),),
                    log_probs_constrained.size(1),
                    dtype=torch.long,
                    device=log_probs_constrained.device
                )
            else:
                input_lengths = input_lengths.to(log_probs_constrained.device)

        # Guard against off-by-one or padding-induced length mismatch
        input_lengths = torch.clamp(
            input_lengths,
            max=log_probs_constrained.size(1)
        )

        label_lengths = batch.get('label_lengths')
        if label_lengths is None:
            label_lengths = (labels != -100).sum(dim=1)
        else:
            label_lengths = label_lengths.to(log_probs_constrained.device)

        # Drop samples where target length exceeds input length (invalid for CTC)
        valid_mask = label_lengths <= input_lengths
        if not valid_mask.all():
            invalid_frac = 1.0 - valid_mask.float().mean()
            self.log('train/ctc_invalid_frac', invalid_frac, on_step=True, prog_bar=False)

        if valid_mask.sum() == 0:
            zero_loss = torch.zeros((), device=log_probs_constrained.device, requires_grad=True)
            self.log('train/loss', zero_loss, on_step=True, on_epoch=True, prog_bar=True)
            return zero_loss

        logits_manner = outputs.get('logits_manner')
        logits_place = outputs.get('logits_place')
        logits_voice = outputs.get('logits_voice')

        # Build filtered outputs dict once (Bug B4: previous code built it twice;
        # the first construction lacked logits_neural and was silently discarded).
        labels_filtered = labels[valid_mask]
        input_lengths_filtered = input_lengths[valid_mask]
        label_lengths_filtered = label_lengths[valid_mask]

        # Compute losses
        art_labels = batch.get('articulatory_labels')
        if art_labels is not None:
            art_labels = {key: value[valid_mask] for key, value in art_labels.items()}

        outputs_filtered = {
            'logits_constrained': log_probs_constrained[valid_mask],
            'logits_neural':      outputs.get('logits_neural',  log_probs_constrained)[valid_mask],
            'hidden_states':      outputs.get('hidden_states',  None),
            'logits_manner': (outputs.get('logits_manner')[valid_mask]
                              if outputs.get('logits_manner') is not None else None),
            'logits_place':  (outputs.get('logits_place')[valid_mask]
                              if outputs.get('logits_place') is not None else None),
            'logits_voice':  (outputs.get('logits_voice')[valid_mask]
                              if outputs.get('logits_voice') is not None else None),
        }
        if outputs_filtered['hidden_states'] is not None:
            outputs_filtered['hidden_states'] = outputs_filtered['hidden_states'][valid_mask]

        # C7: Use per-speaker severity from TORGO_SEVERITY_MAP (continuous [0,5])
        # instead of the coarse binary status * 5.0 used previously.
        speakers_batch = batch.get('speakers', [])
        if speakers_batch and isinstance(speakers_batch[0], str):
            severity = torch.tensor(
                [get_speaker_severity(s) for s in speakers_batch],
                dtype=torch.float32,
                device=log_probs_constrained.device,
            )
        else:
            severity = batch['status'].float().to(log_probs_constrained.device) * 5.0

        attn_mask = batch.get('attention_mask')
        if attn_mask is not None:
            # Downsample attention mask from audio-frame resolution to logit resolution.
            # The effective stride is 320 (HuBERT CNN) * 2 (TemporalDownsampler) when
            # the downsampler is active (§2.7 fix: was always using 320, causing ~2×
            # too many valid-frame entries after truncation).
            T_log = log_probs_constrained.size(1)
            ctc_stride = 320 * (2 if self.model.temporal_downsampler is not None else 1)
            attn_mask_ds = (
                attn_mask[:, ::ctc_stride].to(log_probs_constrained.device)
                if attn_mask.size(1) > T_log
                else attn_mask.to(log_probs_constrained.device)
            )
            attn_mask_ds = attn_mask_ds[:, :T_log]
        else:
            attn_mask_ds = None

        losses = self.compute_loss(
            outputs_filtered,
            labels_filtered,
            input_lengths_filtered,
            label_lengths_filtered,
            articulatory_labels=art_labels,
            severity=severity[valid_mask],
            attention_mask=attn_mask_ds[valid_mask] if attn_mask_ds is not None else None,
        )

        # Log metrics
        self.log('train/loss',         losses['loss'],       on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/loss_ctc',     losses['loss_ctc'],   on_step=False, on_epoch=True)
        self.log('train/loss_ce',      losses['loss_ce'],    on_step=False, on_epoch=True)
        self.log('train/loss_blank_kl',losses['loss_blank_kl'], on_step=False, on_epoch=True)
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

        # Q4: Gradient norm monitoring — log L2 norm every 50 steps for stability diagnostics.
        # Logged after the backward pass has populated .grad attributes via Lightning's
        # on_before_optimizer_step hook ordering; accessing here is safe because
        # training_step is called before the optimizer step in Lightning 2.x.
        if self.global_step % 50 == 0:
            total_norm_sq = sum(
                p.grad.detach().norm(2).item() ** 2
                for p in self.parameters()
                if p.grad is not None
            )
            grad_norm = total_norm_sq ** 0.5
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
        epoch           = self.current_epoch
        warmup_ep       = self.config.training.encoder_warmup_epochs
        second_unfreeze = getattr(self.config.training, 'encoder_second_unfreeze_epoch', 6)
        third_unfreeze  = getattr(self.config.training, 'encoder_third_unfreeze_epoch', 12)

        if not self.encoder_unfrozen and epoch >= warmup_ep:
            # Stage 1: unfreeze top 4 layers only
            self.model.unfreeze_encoder(layers=[8, 9, 10, 11])
            self.encoder_unfrozen = True
            self._reset_hubert_lr_warmup()  # T-04: reset Adam state for newly-active params
            print(f"🔥 Stage 1: Unfroze HuBERT layers 8-11 at epoch {epoch}")

        elif (self.encoder_unfrozen
              and not getattr(self, '_encoder_deep_unfrozen', False)
              and epoch >= second_unfreeze):
            # Stage 2: unfreeze layers 6-11 only (§2.4 fix: was calling
            # unfreeze_after_warmup() which unfroze ALL layers 4-11, making
            # Stage 3 a no-op)
            self.model.unfreeze_encoder(layers=[6, 7, 8, 9, 10, 11])
            self._encoder_deep_unfrozen = True
            self._reset_hubert_lr_warmup()  # T-04: reset Adam state for newly-active params
            print(f"🔥 Stage 2: Unfroze HuBERT layers 6-11 at epoch {epoch}")

        elif (getattr(self, '_encoder_deep_unfrozen', False)
              and not getattr(self, '_encoder_deeper_unfrozen', False)
              and epoch >= third_unfreeze):
            # Stage 3: unfreeze layers 4-11 (deepest adaptation to dysarthric speech)
            self.model.unfreeze_encoder(layers=[4, 5, 6, 7, 8, 9, 10, 11])
            self._encoder_deeper_unfrozen = True
            self._reset_hubert_lr_warmup()  # T-04: reset Adam state for newly-active params
            print(f"🔥 Stage 3: Unfroze HuBERT layers 4-11 at epoch {epoch}")

        # I2: Staged lambda_blank_kl ramp ─────────────────────────────────────
        tr = self.config.training
        stage1_end = getattr(tr, 'blank_kl_stage1_end',   5)
        stage2_end = getattr(tr, 'blank_kl_stage2_end',  15)
        val_s1     = getattr(tr, 'blank_kl_stage1_value', 0.10)
        val_s2     = getattr(tr, 'blank_kl_stage2_value', 0.20)
        target     = tr.lambda_blank_kl  # Final value (default 0.35)
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

    def _reset_hubert_lr_warmup(self) -> None:
        """T-04: Reset AdamW optimizer state for newly-unfrozen HuBERT params.

        OneCycleLR is built once at configure_optimizers() and its step counter
        cannot be rewound.  Newly unfrozen layers therefore start at the current
        (potentially decayed) LR rather than at the original peak LR.  While a
        full fix requires switching to CosineAnnealingWarmRestarts (T_mult=2)
        or per-group schedulers, resetting the Adam first/second moment estimates
        gives newly-active parameters the closest equivalent to a "fresh start":
        gradients are not polluted by stale momentum from pre-unfreeze noise.
        Called after each unfreeze event in on_train_epoch_start.

        Important: Adam expects either a fully initialized state dict or an empty
        one for each parameter. Removing only exp_avg / exp_avg_sq leaves a
        partially initialized state that crashes on the next optimizer.step()
        with KeyError('exp_avg').
        """
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        for pg in optimizer.param_groups:
            if pg.get('name') == 'hubert_encoder':
                for p in pg['params']:
                    if p.requires_grad and p in optimizer.state:
                        # Clear the whole per-parameter state so Adam fully
                        # reinitializes exp_avg / exp_avg_sq / step on next use.
                        optimizer.state[p].clear()
                break

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """
        Validation step with metrics computation.
        
        Args:
            batch: Validation batch
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """
        outputs = self(batch)
        
        # Compute lengths
        log_probs_constrained = outputs.get('log_probs_constrained', outputs['logits_constrained'])
        labels = batch['labels']

        # Prefer exact output_lengths from model over approximate batch.input_lengths.
        if outputs.get('output_lengths') is not None:
            input_lengths = outputs['output_lengths'].to(log_probs_constrained.device)
        else:
            input_lengths = batch.get('input_lengths')
            if input_lengths is None:
                input_lengths = torch.full(
                    (log_probs_constrained.size(0),),
                    log_probs_constrained.size(1),
                    dtype=torch.long,
                    device=log_probs_constrained.device
                )
            else:
                input_lengths = input_lengths.to(log_probs_constrained.device)

        # Guard against off-by-one or padding-induced length mismatch
        input_lengths = torch.clamp(
            input_lengths,
            max=log_probs_constrained.size(1)
        )

        label_lengths = batch.get('label_lengths')
        if label_lengths is None:
            label_lengths = (labels != -100).sum(dim=1)
        else:
            label_lengths = label_lengths.to(log_probs_constrained.device)

        # Drop invalid CTC samples for validation metrics
        valid_mask = label_lengths <= input_lengths
        if valid_mask.sum() == 0:
            return {}

        log_probs_constrained = log_probs_constrained[valid_mask]
        labels = labels[valid_mask]
        input_lengths = input_lengths[valid_mask]
        label_lengths = label_lengths[valid_mask]

        logits_manner = outputs.get('logits_manner')
        logits_place = outputs.get('logits_place')
        logits_voice = outputs.get('logits_voice')

        art_labels = batch.get('articulatory_labels')
        if art_labels is not None:
            art_labels = {key: value[valid_mask] for key, value in art_labels.items()}

        # Bug B5: logits_neural and hidden_states were missing from this dict;
        # CE loss fell back silently to constrained log-probs in val/test.
        logits_neural_v = outputs.get('logits_neural')
        hidden_states_v = outputs.get('hidden_states')
        losses = self.compute_loss(
            {
                'logits_constrained': log_probs_constrained,
                'logits_neural': (
                    logits_neural_v[valid_mask]
                    if logits_neural_v is not None
                    else log_probs_constrained
                ),
                'hidden_states': (
                    hidden_states_v[valid_mask]
                    if hidden_states_v is not None
                    else None
                ),
                'logits_manner': logits_manner[valid_mask] if logits_manner is not None else None,
                'logits_place': logits_place[valid_mask] if logits_place is not None else None,
                'logits_voice': logits_voice[valid_mask] if logits_voice is not None else None
            },
            labels,
            input_lengths,
            label_lengths,
            articulatory_labels=art_labels,
            attention_mask=self._downsample_attn_mask(batch, log_probs_constrained, valid_mask),
        )

        # Decode predictions for PER computation
        # Pass output_lengths so padding frames are excluded (prevents insertion inflation
        # on shorter utterances padded to batch-max length).
        output_lengths_full = outputs.get('output_lengths')
        output_lengths_pred = output_lengths_full[valid_mask] if output_lengths_full is not None else None
        predictions = decode_predictions(
            log_probs_constrained, self.phn_to_id, self.id_to_phn,
            output_lengths=output_lengths_pred,
        )
        references = decode_references(labels, self.id_to_phn)
        per_scores = [compute_per(pred, ref) for pred, ref in zip(predictions, references)]
        avg_per = float(np.mean(per_scores)) if per_scores else 0.0

        # Collect speakers aligned to valid_mask for macro-speaker PER (§2.6)
        all_speakers   = batch.get('speakers', [])
        valid_speakers = [
            all_speakers[i] for i in range(len(all_speakers))
            if i < len(valid_mask) and valid_mask[i].item()
        ] if all_speakers else []

        # Store outputs for epoch-level metrics
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
                eps = 1e-8
                # Row entropy H(C[i]) — mean over rows
                row_entropy = -(C * torch.log(C + eps)).sum(dim=-1).mean()
                self.log('val/constraint_row_entropy', float(row_entropy), prog_bar=False)
                # KL from static prior: KL(C_learned || C_prior)
                C_prior = sl.static_constraint_matrix.detach().float()  # static buffer (correct attr name)
                kl_from_prior = (C * torch.log((C + eps) / (C_prior + eps))).sum(dim=-1).mean()
                self.log('val/constraint_kl_from_prior', float(kl_from_prior), prog_bar=False)
        except Exception:
            pass  # Non-fatal: logging diagnostics should never crash training

        # Clear for next epoch
        self.validation_step_outputs.clear()
    
    def _downsample_attn_mask(
        self,
        batch: Dict,
        log_probs: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Downsample the batch attention mask to logit resolution and apply valid_mask.

        Shared by validation_step and test_step (§3.4 fix: both were passing
        attention_mask=None to compute_loss, causing BlankPriorKLLoss to include
        padding frames and biasing the loss upward).
        """
        attn_mask = batch.get('attention_mask')
        if attn_mask is None:
            return None
        T_log = log_probs.size(1)
        stride = 320 * (2 if self.model.temporal_downsampler is not None else 1)
        ds = attn_mask[:, ::stride].to(log_probs.device)
        ds = ds[:, :T_log]
        return ds[valid_mask]

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
        """
        Test step.
        
        Args:
            batch: Test batch
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """
        outputs = self(batch)
        
        # Compute lengths
        log_probs_constrained = outputs.get('log_probs_constrained', outputs['logits_constrained'])
        labels = batch['labels']

        # Prefer exact output_lengths from model over approximate batch.input_lengths.
        if outputs.get('output_lengths') is not None:
            input_lengths = outputs['output_lengths'].to(log_probs_constrained.device)
        else:
            input_lengths = batch.get('input_lengths')
            if input_lengths is None:
                input_lengths = torch.full(
                    (log_probs_constrained.size(0),),
                    log_probs_constrained.size(1),
                    dtype=torch.long,
                    device=log_probs_constrained.device
                )
            else:
                input_lengths = input_lengths.to(log_probs_constrained.device)

        input_lengths = torch.clamp(input_lengths, max=log_probs_constrained.size(1))

        label_lengths = batch.get('label_lengths')
        if label_lengths is None:
            label_lengths = (labels != -100).sum(dim=1)
        else:
            label_lengths = label_lengths.to(log_probs_constrained.device)

        # Drop invalid CTC samples for test metrics
        valid_mask = label_lengths <= input_lengths
        if valid_mask.sum() == 0:
            return {}

        log_probs_constrained = log_probs_constrained[valid_mask]
        labels = labels[valid_mask]
        input_lengths = input_lengths[valid_mask]
        label_lengths = label_lengths[valid_mask]
        
        logits_manner = outputs.get('logits_manner')
        logits_place = outputs.get('logits_place')
        logits_voice = outputs.get('logits_voice')

        art_labels = batch.get('articulatory_labels')
        if art_labels is not None:
            art_labels = {key: value[valid_mask] for key, value in art_labels.items()}

        # Bug B5 (test_step mirror): add logits_neural and hidden_states so CE loss
        # is computed on neural logits, not constrained log-probs.
        logits_neural_t = outputs.get('logits_neural')
        hidden_states_t = outputs.get('hidden_states')
        losses = self.compute_loss(
            {
                'logits_constrained': log_probs_constrained,
                'logits_neural': (
                    logits_neural_t[valid_mask]
                    if logits_neural_t is not None
                    else log_probs_constrained
                ),
                'hidden_states': (
                    hidden_states_t[valid_mask]
                    if hidden_states_t is not None
                    else None
                ),
                'logits_manner': logits_manner[valid_mask] if logits_manner is not None else None,
                'logits_place': logits_place[valid_mask] if logits_place is not None else None,
                'logits_voice': logits_voice[valid_mask] if logits_voice is not None else None
            },
            labels,
            input_lengths,
            label_lengths,
            articulatory_labels=art_labels,
            attention_mask=self._downsample_attn_mask(batch, log_probs_constrained, valid_mask),
        )

        # Decode and compute PER
        # Pass output_lengths so padding frames are excluded.
        output_lengths_full_t = outputs.get('output_lengths')
        output_lengths_pred_t = output_lengths_full_t[valid_mask] if output_lengths_full_t is not None else None
        predictions = decode_predictions(
            log_probs_constrained, self.phn_to_id, self.id_to_phn,
            output_lengths=output_lengths_pred_t,
        )
        references = decode_references(labels, self.id_to_phn)
        per_scores = [compute_per(pred, ref) for pred, ref in zip(predictions, references)]
        avg_per = np.mean(per_scores) if per_scores else 0.0
        
        # Log test metrics
        self.log('test/loss', losses['loss'])
        self.log('test/per', avg_per)
        
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
                'lr': lr * 0.1,
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
                'lr': lr * 0.5,
                'name': 'symbolic_layer',
            },
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.training.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[lr * 0.1, lr * 1.0, lr * 0.5],
            total_steps=total_steps,
            pct_start=self.config.training.warmup_ratio,
            anneal_strategy='cos',
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1},
        }


def create_dataloaders(
    config: Config,
    dataset: TorgoNeuroSymbolicDataset
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with stratified speaker splits.
    
    Args:
        config: Configuration object
        dataset: Full dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("📊 Creating data splits...")
    
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
        test_speakers  = speakers[-1:]
        val_speakers   = speakers[-2:-1]
        train_speakers = speakers[:-2]

        train_idx = df[df['speaker'].isin(train_speakers)].index.tolist()
        val_idx   = df[df['speaker'].isin(val_speakers)].index.tolist()
        test_idx  = df[df['speaker'].isin(test_speakers)].index.tolist()

        print(
            f"⚠️  Ratio-based split produced empty partition(s). "
            f"Using round-robin speaker assignment: "
            f"train={list(train_speakers)}, val={list(val_speakers)}, "
            f"test={list(test_speakers)}"
        )
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    print(f"Train: {len(train_dataset)} samples ({len(train_speakers)} speakers)")
    print(f"Val: {len(val_dataset)} samples ({len(val_speakers)} speakers)")
    print(f"Test: {len(test_dataset)} samples ({len(test_speakers)} speakers)")
    
    # Create collator
    collator = NeuroSymbolicCollator(dataset.processor)

    # Speaker-balanced sampling so high-utterance speakers don't dominate gradient
    # signal; each speaker contributes equally per epoch (§3.5 fix: was using
    # class-level dysarthric/control inverse-frequency weights).
    train_df = dataset.df.iloc[train_idx]
    speaker_counts = train_df['speaker'].value_counts().to_dict()
    train_weights = [1.0 / speaker_counts[dataset.df.iloc[i]['speaker']] for i in train_idx]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    
    # Create dataloaders with shared worker/prefetch settings for throughput.
    common_loader_kwargs = dict(
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    if config.training.num_workers > 0:
        common_loader_kwargs["persistent_workers"] = True
        common_loader_kwargs["prefetch_factor"] = config.training.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        **common_loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        **common_loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        **common_loader_kwargs,
    )
    
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training-curve helpers (E4)
# ---------------------------------------------------------------------------

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

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        per = trainer.callback_metrics.get('val/per')
        if per is not None:
            self.val_per.append(float(per))


def _save_learning_curve(
    train_losses: List[float],
    val_pers: List[float],
    results_dir: Path,
) -> None:
    """
    Produce and save a two-axis learning-curve plot.

    Left axis  : ``train/loss`` vs epoch.
    Right axis : ``val/per``   vs epoch.

    The plot is saved to ``results_dir/learning_curve.png``.

    Args:
        train_losses : Per-epoch training loss values.
        val_pers     : Per-epoch validation PER values.
        results_dir  : Directory in which to write the PNG.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not train_losses and not val_pers:
        print("\u26a0\ufe0f  No metric history available \u2014 skipping learning curve.")
        return

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    epochs_loss = list(range(1, len(train_losses) + 1))
    epochs_per  = list(range(1, len(val_pers)    + 1))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_loss = '#e74c3c'
    color_per  = '#3498db'

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
    print(f"\u2705 Learning curve saved to {save_path}")


def train(
    config: Optional[Config] = None,
    limit_train_batches: Optional[int] = None,
) -> Tuple[DysarthriaASRLightning, pl.Trainer]:
    """
    Main training function.

    Args:
        config: Configuration object, uses default if None
        limit_train_batches: If set, cap batches per epoch (smoke/dev mode).
            Passed directly to ``pl.Trainer(limit_train_batches=...)``.

    Returns:
        Tuple of (trained_model, trainer)
    """
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
    print("📦 Loading dataset...")
    dataset = TorgoNeuroSymbolicDataset(
        manifest_path=str(config.data.manifest_path),
        processor_id=config.model.hubert_model_id,
        sampling_rate=config.data.sampling_rate,
        max_audio_length=config.data.max_audio_length
    )
    
    # Create dataloaders (test split unused here; evaluation is the caller's responsibility)
    train_loader, val_loader, _ = create_dataloaders(config, dataset)
    
    # Initialize model
    print("🧠 Initializing neuro-symbolic model...")
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
    
    early_stop_callback = EarlyStopping(
        monitor=config.training.monitor_metric,
        patience=config.training.early_stopping_patience,
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
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, metric_logger_cb],
        deterministic=False,  # Handled manually above
        log_every_n_steps=config.experiment.log_every_n_steps,
    )
    if limit_train_batches is not None:
        trainer_kwargs['limit_train_batches'] = limit_train_batches
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train
    print("🚀 Starting training...")
    trainer.fit(lightning_model, train_loader, val_loader)

    # L5: Detect OneCycleLR step-count drift (>10% deviation raises a warning;
    # drift causes the scheduler to silently plateau before training ends).
    _estimated_steps = trainer.estimated_stepping_batches
    _actual_steps    = trainer.global_step
    if _estimated_steps and _estimated_steps > 0:
        _drift = abs(_actual_steps - _estimated_steps) / _estimated_steps
        if _drift > 0.10:
            warnings.warn(
                f"OneCycleLR step drift detected: estimated={_estimated_steps}, "
                f"actual={_actual_steps} ({_drift:.1%}).  "
                f"Adjust max_epochs or gradient_accumulation_steps to match.",
                stacklevel=2,
            )

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
            print(f"⚠️  No best checkpoint saved; using last.ckpt: {best_model_path}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir} after training."
            )
    else:
        print(f"✅ Best checkpoint: {best_model_path}")

    # Load best checkpoint and return
    print("🔄 Loading best model from checkpoint...")
    best_model = DysarthriaASRLightning.load_from_checkpoint(
        best_model_path,
        model=model,
        config=config,
        phn_to_id=dataset.phn_to_id,
        id_to_phn=dataset.id_to_phn,
        strict=False
    )

    print("✅ Training complete!")
    print(f"  Checkpoints: {checkpoint_dir}")

    return best_model, trainer



def create_loso_splits(
    config: Config,
    dataset: TorgoNeuroSymbolicDataset,
    held_out_speaker: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for a single LOSO fold.

    The held-out speaker is used exclusively for testing. The remaining speakers
    are split 85/15 into train/val by speaker.

    Args:
        config:             Training configuration
        dataset:            Full TorgoNeuroSymbolicDataset
        held_out_speaker:   Speaker ID to hold out for this fold

    Returns:
        (train_loader, val_loader, test_loader)
    """
    df = dataset.df
    # B5 fix: sort lexicographically before seeded shuffle so fold assignments
    # are independent of manifest row order (df['speaker'].unique() returns
    # first-occurrence order, which changes if the manifest is regenerated).
    # Must match pl.seed_everything seed used in train() / run_pipeline.py.
    all_speakers = sorted([s for s in df['speaker'].unique() if s != held_out_speaker])

    np.random.seed(config.experiment.seed)
    np.random.shuffle(all_speakers)

    n_val = max(1, int(len(all_speakers) * 0.15))
    val_speakers  = all_speakers[:n_val]
    train_speakers = all_speakers[n_val:]

    train_idx = df[df['speaker'].isin(train_speakers)].index.tolist()
    val_idx   = df[df['speaker'].isin(val_speakers)].index.tolist()
    test_idx  = df[df['speaker'] == held_out_speaker].index.tolist()

    collator = NeuroSymbolicCollator(dataset.processor)

    # Speaker-balanced sampler for train split (§3.5 fix: was using class-level
    # dysarthric/control weights, causing high-utterance speakers to dominate).
    train_df = df.iloc[train_idx]
    speaker_counts_loso = train_df['speaker'].value_counts().to_dict()
    train_weights = [1.0 / max(speaker_counts_loso.get(df.iloc[i]['speaker'], 1), 1) for i in train_idx]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    def _make_loader(idx, shuffle_sampler=None):
        loader_kwargs = dict(
            batch_size=config.training.batch_size,
            shuffle=False,
            sampler=shuffle_sampler,
            collate_fn=collator,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
        )
        if config.training.num_workers > 0:
            # Keep worker processes alive across epochs/folds to reduce startup
            # overhead and improve I/O throughput for long LOSO runs.
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = config.training.prefetch_factor
        return DataLoader(
            Subset(dataset, idx),
            **loader_kwargs,
        )

    return (
        _make_loader(train_idx, shuffle_sampler=sampler),
        _make_loader(val_idx),
        _make_loader(test_idx),
    )


def run_loso(
    config: Config,
    dataset: TorgoNeuroSymbolicDataset,
    resume: bool = False,
) -> Dict:
    """
    Run Leave-One-Speaker-Out cross-validation.

    Trains one model per speaker fold, aggregates macro-average PER and
    95% bootstrap CIs across folds.

    Returns:
        Dict with per_per_fold, macro_avg_per, per_95ci, fold_speakers
    """
    speakers = list(dataset.df['speaker'].unique())
    print(f"\n🔁 LOSO: {len(speakers)} folds ({speakers})")

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
    if resume and progress_path.exists():
        try:
            with open(progress_path, "r", encoding="utf-8") as _pf:
                progress = json.load(_pf)
            for fold in progress.get("folds", []):
                if fold.get("status") == "completed" and "speaker" in fold:
                    completed_folds[fold["speaker"]] = fold
            if completed_folds:
                print(
                    f"♻️  Resuming LOSO from progress file: {progress_path} "
                    f"({len(completed_folds)} completed fold(s) found)"
                )
        except Exception as exc:
            print(f"⚠️  Failed to parse LOSO progress file ({progress_path}): {exc}")

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
        print(f"\n── Fold {i+1}/{len(speakers)}: held-out speaker = {spk} ──")
        fold_run_name = f"{base_run_name}_loso_{spk}"
        config.experiment.run_name = fold_run_name

        if spk in completed_folds:
            cached = completed_folds[spk]
            fold_per = float(cached.get('per', float('nan')))
            fold_wer = float(cached.get('wer', float('nan')))
            fold_n = int(cached.get('n_samples', 0))
            per_per_fold.append(fold_per)
            wer_per_fold.append(fold_wer)
            n_samples_per_fold.append(fold_n)
            print(
                f"   ⏭️  Skipping completed fold {spk}: "
                f"PER={fold_per:.4f} | WER={fold_wer:.4f} | n={fold_n}"
            )
            continue

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
        trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            accelerator='gpu' if config.device == 'cuda' else 'cpu',
            devices=1,
            precision=config.training.precision,
            accumulate_grad_batches=config.training.gradient_accumulation_steps,
            gradient_clip_val=config.training.gradient_clip_val,
            callbacks=[
                ckpt_cb,
                EarlyStopping(
                    monitor='val_per',
                    patience=config.training.early_stopping_patience,
                    mode='min'
                ),
            ],
            val_check_interval=config.training.val_check_interval,
            logger=fold_logger,
            enable_progress_bar=True,
            log_every_n_steps=50,
        )
        resume_ckpt = ckpt_dir / 'last.ckpt'
        ckpt_path = str(resume_ckpt) if resume and resume_ckpt.exists() else None
        if ckpt_path:
            print(f"   ♻️  Resuming fold training from {resume_ckpt}")
        trainer.fit(lm, train_loader, val_loader, ckpt_path=ckpt_path)

        # Evaluate on held-out speaker
        results_dir = get_project_root() / "results" / fold_run_name
        results_dir.mkdir(parents=True, exist_ok=True)
        eval_results = evaluate_model(
            model=lm.model, dataloader=test_loader,
            device=config.device,
            phn_to_id=dataset.phn_to_id, id_to_phn=dataset.id_to_phn,
            results_dir=results_dir,
        )
        fold_per = eval_results.get('avg_per', float('nan'))
        fold_wer = eval_results.get('wer', float('nan'))
        fold_n   = eval_results.get('overall', {}).get('n_samples', 0)
        per_per_fold.append(fold_per)
        wer_per_fold.append(fold_wer)
        n_samples_per_fold.append(fold_n)
        completed_folds[spk] = {
            'per': fold_per,
            'wer': fold_wer,
            'n_samples': fold_n,
            'checkpoint': str(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else str(ckpt_dir / 'last.ckpt'),
            'results_dir': str(results_dir),
        }
        _write_progress()
        print(f"   Fold PER ({spk}): {fold_per:.4f}  |  WER: {fold_wer:.4f}  |  n={fold_n}")

        # H-1: Explicit VRAM cleanup between folds — prevents OOM on 8 GB card.
        del trainer, lm, model
        torch.cuda.empty_cache()

    # Bootstrap 95% CI over fold PERs
    per_arr = np.array(per_per_fold)
    macro_per = float(np.nanmean(per_arr))
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(per_arr, size=len(per_arr), replace=True).mean()
                     for _ in range(2000)])
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # Weighted-average PER (weighted by number of test samples per fold)
    n_arr = np.array(n_samples_per_fold, dtype=float)
    valid_mask = ~np.isnan(per_arr) & (n_arr > 0)
    weighted_per = (
        float(np.sum(per_arr[valid_mask] * n_arr[valid_mask]) / np.sum(n_arr[valid_mask]))
        if valid_mask.any() else float('nan')
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

    print(f"\n✅ LOSO Complete:")
    print(f"   Mean PER     = {macro_per:.4f}  [95% CI: {ci_lo:.4f} – {ci_hi:.4f}]")
    print(f"   Weighted PER = {weighted_per:.4f}  (weighted by fold sample count)")
    print(f"   Mean WER     = {macro_wer:.4f}  |  Weighted WER = {weighted_wer:.4f}")

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
    }

    # Save aggregated LOSO summary to results/{base_run_name}_loso_summary.json (I3)
    summary_path = summary_dir / f"{base_run_name}_loso_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as _sf:
        json.dump(loso_summary, _sf, indent=2)
    print(f"💾 LOSO summary saved to {summary_path}")

    return loso_summary


def main() -> None:
    """Main entry point for training script."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Neuro-Symbolic Dysarthria ASR")
    parser.add_argument('--config',              type=str, help='Path to config YAML file')
    parser.add_argument('--run-name',            type=str, help='MLflow run name')
    parser.add_argument('--ablation',            type=str, default='full',
                        choices=['full', 'neural_only', 'symbolic_only', 'no_art_heads',
                                 'no_constraint_matrix', 'no_spec_augment', 'no_temporal_ds'],
                        help='Ablation mode (default: full)')
    parser.add_argument('--loso',                action='store_true',
                        help='Run Leave-One-Speaker-Out cross-validation')
    parser.add_argument('--resume-loso', '--resume_loso', dest='resume_loso', action='store_true',
                        help='Resume LOSO from progress/checkpoints when available')
    parser.add_argument('--max_epochs',          type=int, help='Override max_epochs')
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