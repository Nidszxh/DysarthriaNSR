"""
Training Pipeline for Neuro-Symbolic Dysarthric Speech Recognition

Orchestrates model training using PyTorch Lightning with multi-task learning,
symbolic constraints, and comprehensive evaluation metrics.
"""

import sys
import warnings
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
from src.data.dataloader import NeuroSymbolicCollator, TorgoNeuroSymbolicDataset
import os
from src.models.model import NeuroSymbolicASR
from src.models.losses import OrdinalContrastiveLoss, BlankPriorKLLoss, SymbolicKLLoss

# Import evaluate functions from root level
from evaluate import compute_per, evaluate_model, decode_predictions, decode_references

warnings.filterwarnings('ignore')


def flatten_config_for_mlflow(config: Config) -> Dict:
    """
    Flatten config to dict with MLflow-safe parameter names.
    
    MLflow requires alphanumerics, underscores, dashes, periods, spaces, colons, slashes.
    Converts nested dicts and tuple keys to safe formats.
    
    Args:
        config: Configuration object
        
    Returns:
        Flattened dictionary with safe parameter names
    """
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

        # Optional emphasis for special tokens if present
        if '<BLANK>' in self.phn_to_id:
            ce_weights[self.phn_to_id['<BLANK>']] = float(ce_weights[self.phn_to_id['<BLANK>']]) * 1.5
        if '<PAD>' in self.phn_to_id:
            ce_weights[self.phn_to_id['<PAD>']] = float(ce_weights[self.phn_to_id['<PAD>']]) * 1.5

        self.ce_loss = nn.NLLLoss(
            weight=ce_weights,
            ignore_index=-100
        )

        self.art_ce_losses = {}
        for key in ["manner", "place", "voice"]:
            weights = self.articulatory_weights.get(key)
            if weights is not None:
                weights = weights.clone().detach().float()
            self.art_ce_losses[key] = nn.CrossEntropyLoss(
                weight=weights,
                ignore_index=-100,
                label_smoothing=self.config.training.label_smoothing
            )

        # --- Phase 2: New research-grade loss functions (audit proposals P1, P2, R3) ---
        blank_id = self.phn_to_id.get('<BLANK>', 0)
        self.ordinal_loss = OrdinalContrastiveLoss(margin_per_level=0.3)
        self.blank_kl_loss = BlankPriorKLLoss(
            blank_id=blank_id,
            target_prob=self.config.training.blank_target_prob,
        )
        # SymbolicKLLoss is initialised lazily on first use (needs static matrix from model)
        self._symbolic_kl_loss: Optional[SymbolicKLLoss] = None

        # Register core loss weights as buffers
        self.register_buffer('lambda_ctc', torch.tensor(self.config.training.lambda_ctc))
        self.register_buffer('lambda_ce', torch.tensor(self.config.training.lambda_ce))
        self.register_buffer('lambda_articulatory', torch.tensor(self.config.training.lambda_articulatory))

    def on_fit_start(self) -> None:
        """Ensure loss weights reside on the correct device once available."""
        if hasattr(self, 'ce_loss') and hasattr(self.ce_loss, 'weight') and self.ce_loss.weight is not None:
            try:
                self.ce_loss.weight = self.ce_loss.weight.to(self.device)
            except Exception:
                pass
        for loss_fn in self.art_ce_losses.values():
            if hasattr(loss_fn, 'weight') and loss_fn.weight is not None:
                try:
                    loss_fn.weight = loss_fn.weight.to(self.device)
                except Exception:
                    pass
    
    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass through model.

        Converts binary speaker status (0/1) to continuous severity [0, 5] using the
        TORGO_SEVERITY_MAP for known speakers, falling back to status*5.0 for others.
        """
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
        if self.config.model.use_learnable_constraint and ablation not in ('neural_only', 'symbolic_only'):
            sl = self.model.symbolic_layer
            if sl.learnable_matrix is not None:
                if self._symbolic_kl_loss is None:
                    static_C = sl.static_constraint_matrix
                    self._symbolic_kl_loss = SymbolicKLLoss(static_C).to(self.device)
                loss_symbolic_kl = self._symbolic_kl_loss(sl.learnable_matrix.logit_C)

        # ── Ablation-mode weighting ──────────────────────────────────────────
        lambda_ctc  = 0.0 if ablation == 'symbolic_only' else float(self.lambda_ctc)
        lambda_ce   = 0.0 if ablation == 'symbolic_only' else float(self.lambda_ce)
        lambda_art  = float(self.lambda_articulatory)
        lambda_ord  = self.config.training.lambda_ordinal
        lambda_bkl  = self.config.training.lambda_blank_kl
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
    
    def _compute_ce_loss(self, log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute NLL loss using constrained log-probabilities."""
        batch_size, time_steps_logits, num_classes = log_probs.shape
        
        # Align labels to logits time dimension
        labels_aligned = self._align_labels_to_logits(labels, time_steps_logits)
        
        # Compute loss
        logits_flat = log_probs.reshape(-1, num_classes)
        labels_flat = labels_aligned.reshape(-1)
        
        return self.ce_loss(logits_flat, labels_flat)

    def _compute_articulatory_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        key: str
    ) -> torch.Tensor:
        batch_size, time_steps_logits, num_classes = logits.shape
        labels_aligned = self._align_labels_to_logits(labels, time_steps_logits)
        logits_flat = logits.reshape(-1, num_classes)
        labels_flat = labels_aligned.reshape(-1)
        return self.art_ce_losses[key](logits_flat, labels_flat)
    
    def _align_labels_to_logits(self, labels: torch.Tensor, time_steps_logits: int) -> torch.Tensor:
        """
        Align label sequence length to match logits time dimension.
        
        Args:
            labels: Label tensor [batch, seq_len]
            time_steps_logits: Target time dimension
            
        Returns:
            Aligned labels [batch, time_steps_logits]
        """
        batch_size = labels.size(0)
        time_steps_labels = labels.size(1)
        
        if time_steps_labels < time_steps_logits:
            # Pad with -100 (ignored by loss)
            padding = torch.full(
                (batch_size, time_steps_logits - time_steps_labels),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            return torch.cat([labels, padding], dim=1)
        else:
            # Truncate to match logits
            return labels[:, :time_steps_logits]
    
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

        severity  = batch['status'].float() * 5.0
        attn_mask = batch.get('attention_mask')
        if attn_mask is not None:
            # Downsample attention mask from audio-frame resolution to logit resolution.
            # Stride must match the HuBERT CNN feature extractor stride (320 samples),
            # NOT batch_size.  Using batch_size=4 was Bug B3 — it caused the mask to
            # cover only the first ~1.3 s of each utterance.
            T_log = log_probs_constrained.size(1)
            ctc_stride = 320  # matches NeuroSymbolicCollator.ctc_stride
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

        return losses['loss']

    def on_train_epoch_start(self) -> None:
        """Gradually unfreeze HuBERT after warmup epochs."""
        if (not self.encoder_unfrozen and
                self.current_epoch >= self.config.training.encoder_warmup_epochs):
            self.model.unfreeze_after_warmup()
            self.encoder_unfrozen = True
            print(f"🧊 Unfroze HuBERT encoder at epoch {self.current_epoch}")
    
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
            articulatory_labels=art_labels
        )

        # Decode predictions for PER computation
        predictions = decode_predictions(log_probs_constrained, self.phn_to_id, self.id_to_phn)
        references = decode_references(labels, self.id_to_phn)
        
        # Compute PER
        per_scores = [compute_per(pred, ref) for pred, ref in zip(predictions, references)]
        avg_per = np.mean(per_scores) if per_scores else 0.0
        
        # Store outputs for epoch-level metrics
        self.validation_step_outputs.append({
            'loss': losses['loss'],
            'per': avg_per,
            'predictions': predictions,
            'references': references,
            'status': batch['status'][valid_mask]
        })
        
        return losses
    
    def on_validation_epoch_end(self) -> None:
        """Aggregate and log validation metrics."""
        if not self.validation_step_outputs:
            return
        
        # Aggregate losses
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        
        # Aggregate PER
        all_per = [x['per'] for x in self.validation_step_outputs]
        avg_per = np.mean(all_per)
        
        # Stratified PER (dysarthric vs control)
        dysarthric_per, control_per = self._compute_stratified_per()
        
        # Log metrics
        self.log('val/loss', avg_loss, prog_bar=True)
        self.log('val/per', avg_per, prog_bar=True)
        self.log('val/per_dysarthric', dysarthric_per)
        self.log('val/per_control', control_per)
        
        # Clear for next epoch
        self.validation_step_outputs.clear()
    
    def _compute_stratified_per(self) -> Tuple[float, float]:
        """
        Compute PER stratified by dysarthric vs control speakers.
        
        Returns:
            Tuple of (dysarthric_per, control_per)
        """
        dysarthric_per = []
        control_per = []
        
        for output in self.validation_step_outputs:
            for pred, ref, status in zip(
                output['predictions'],
                output['references'],
                output['status']
            ):
                per = compute_per(pred, ref)
                if status == 1:  # Dysarthric
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
            articulatory_labels=art_labels
        )

        # Decode and compute PER
        predictions = decode_predictions(log_probs_constrained, self.phn_to_id, self.id_to_phn)
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

    # Fallback for tiny speaker counts: ensure non-empty splits
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        print(
            "⚠️  Speaker-level split produced empty split(s). "
            "Falling back to sample-level split for this run."
        )
        total_samples = len(df)
        rng = np.random.default_rng(config.experiment.seed)
        shuffled_idx = rng.permutation(total_samples)

        if total_samples <= 2:
            n_train = max(1, total_samples)
            n_val = max(0, total_samples - n_train)
            n_test = 0
        else:
            n_train = max(1, int(total_samples * config.data.train_split))
            n_val = max(1, int(total_samples * config.data.val_split))
            n_test = total_samples - n_train - n_val

            # Ensure test split is non-empty when possible
            if n_test == 0:
                n_test = 1
                if n_train > n_val:
                    n_train -= 1
                else:
                    n_val -= 1

        train_idx = shuffled_idx[:n_train].tolist()
        val_idx = shuffled_idx[n_train:n_train + n_val].tolist()
        test_idx = shuffled_idx[n_train + n_val:n_train + n_val + n_test].tolist()
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    print(f"Train: {len(train_dataset)} samples ({len(train_speakers)} speakers)")
    print(f"Val: {len(val_dataset)} samples ({len(val_speakers)} speakers)")
    print(f"Test: {len(test_dataset)} samples ({len(test_speakers)} speakers)")
    
    # Create collator
    collator = NeuroSymbolicCollator(dataset.processor)

    # Weighted sampling to balance dysarthric vs control speakers
    train_df = dataset.df.iloc[train_idx]
    status_counts = train_df['status'].value_counts().to_dict()
    train_weights = [1.0 / status_counts[dataset.df.iloc[i]['status']] for i in train_idx]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=collator,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        prefetch_factor=config.training.prefetch_factor
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    return train_loader, val_loader, test_loader


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
    # Mitigate memory fragmentation for long audio sequences
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    torch.set_float32_matmul_precision('high')

    if config is None:
        config = get_default_config()
    
    # Set seeds for reproducibility
    pl.seed_everything(config.experiment.seed, workers=True)
    
    # Configure deterministic algorithms (CTC requires warn_only)
    if config.experiment.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    
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
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='epoch={epoch:02d}-val_per={val/per:.3f}',
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
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        deterministic=False,  # Handled manually above
        log_every_n_steps=config.experiment.log_every_n_steps,
    )
    if limit_train_batches is not None:
        trainer_kwargs['limit_train_batches'] = limit_train_batches
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train
    print("🚀 Starting training...")
    trainer.fit(lightning_model, train_loader, val_loader)

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
    all_speakers = [s for s in df['speaker'].unique() if s != held_out_speaker]

    np.random.seed(config.experiment.seed)
    np.random.shuffle(all_speakers)

    n_val = max(1, int(len(all_speakers) * 0.15))
    val_speakers  = all_speakers[:n_val]
    train_speakers = all_speakers[n_val:]

    train_idx = df[df['speaker'].isin(train_speakers)].index.tolist()
    val_idx   = df[df['speaker'].isin(val_speakers)].index.tolist()
    test_idx  = df[df['speaker'] == held_out_speaker].index.tolist()

    collator = NeuroSymbolicCollator(dataset.processor)

    # Weighted sampler for train split
    train_df = df.iloc[train_idx]
    status_counts = train_df['status'].value_counts().to_dict()
    train_weights = [1.0 / max(status_counts.get(df.iloc[i]['status'], 1), 1) for i in train_idx]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    def _make_loader(idx, shuffle_sampler=None):
        return DataLoader(
            Subset(dataset, idx),
            batch_size=config.training.batch_size,
            shuffle=False,
            sampler=shuffle_sampler,
            collate_fn=collator,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
        )

    return (
        _make_loader(train_idx, shuffle_sampler=sampler),
        _make_loader(val_idx),
        _make_loader(test_idx),
    )


def run_loso(
    config: Config,
    dataset: TorgoNeuroSymbolicDataset,
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

    per_per_fold = []
    for i, spk in enumerate(speakers):
        print(f"\n── Fold {i+1}/{len(speakers)}: held-out speaker = {spk} ──")
        fold_run_name = f"{base_run_name}_loso_{spk}"
        config.experiment.run_name = fold_run_name

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
            dirpath=str(ckpt_dir), monitor='val/per', mode='min',
            save_top_k=1, save_last=False,
        )
        trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            accelerator='gpu' if config.device == 'cuda' else 'cpu',
            devices=1,
            precision=config.training.precision,
            accumulate_grad_batches=config.training.gradient_accumulation_steps,
            gradient_clip_val=config.training.gradient_clip_val,
            callbacks=[ckpt_cb, EarlyStopping(monitor='val/per', patience=5, mode='min')],
            logger=False,
            enable_progress_bar=True,
            log_every_n_steps=50,
        )
        trainer.fit(lm, train_loader, val_loader)

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
        per_per_fold.append(fold_per)
        print(f"   Fold PER ({spk}): {fold_per:.4f}")

    # Bootstrap 95% CI over fold PERs
    per_arr = np.array(per_per_fold)
    macro_per = float(np.nanmean(per_arr))
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(per_arr, size=len(per_arr), replace=True).mean()
                     for _ in range(2000)])
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    print(f"\n✅ LOSO Complete: macro-avg PER = {macro_per:.4f} "
          f"[95% CI: {ci_lo:.4f} – {ci_hi:.4f}]")

    return {
        'per_per_fold': per_per_fold,
        'fold_speakers': speakers,
        'macro_avg_per': macro_per,
        'per_95ci': (ci_lo, ci_hi),
    }


def main() -> None:
    """Main entry point for training script."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Neuro-Symbolic Dysarthria ASR")
    parser.add_argument('--config',              type=str, help='Path to config YAML file')
    parser.add_argument('--run-name',            type=str, help='MLflow run name')
    parser.add_argument('--ablation',            type=str, default='full',
                        choices=['full', 'neural_only', 'symbolic_only', 'no_art_heads'],
                        help='Ablation mode (default: full)')
    parser.add_argument('--loso',                action='store_true',
                        help='Run Leave-One-Speaker-Out cross-validation')
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
        run_loso(config, dataset)
    else:
        train(config, limit_train_batches=args.limit_train_batches)


if __name__ == "__main__":
    main()