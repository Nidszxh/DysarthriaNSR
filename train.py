"""
Training Pipeline for Neuro-Symbolic Dysarthric Speech Recognition

Orchestrates model training using PyTorch Lightning with multi-task learning,
symbolic constraints, and comprehensive evaluation metrics.
"""

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
from torch.utils.data import DataLoader, Subset

from config import Config, get_default_config
from dataloader import NeuroSymbolicCollator, TorgoNeuroSymbolicDataset
from evaluate import compute_per
from model import NeuroSymbolicASR

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


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in phoneme distribution.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = -100):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for balanced classes
            gamma: Focusing parameter for hard examples
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Predictions [batch, time, num_classes]
            targets: Ground truth [batch, time]
            
        Returns:
            Scalar loss value
        """
        batch_size, time_steps, num_classes = logits.shape
        
        # Reshape for cross-entropy
        logits_flat = logits.reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)
        
        # Compute cross-entropy
        ce_loss = F.cross_entropy(
            logits_flat, 
            targets_flat, 
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        # Compute focal weight
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()


class DysarthriaASRLightning(pl.LightningModule):
    """PyTorch Lightning module for neuro-symbolic dysarthria ASR."""
    
    def __init__(
        self,
        model: NeuroSymbolicASR,
        config: Config,
        phn_to_id: Dict[str, int],
        id_to_phn: Dict[int, str]
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
        
        # Save hyperparameters (exclude complex objects)
        self.save_hyperparameters(ignore=['model', 'config', 'phn_to_id', 'id_to_phn'])
        
        # Initialize loss functions
        self._init_loss_functions()
        
        # Metrics tracking
        self.validation_step_outputs = []
    
    def _init_loss_functions(self) -> None:
        """Initialize all loss functions."""
        self.ctc_loss = nn.CTCLoss(
            blank=self.phn_to_id['<BLANK>'],
            reduction='mean',
            zero_infinity=True
        )
        
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=self.config.training.label_smoothing
        )
        
        self.focal_loss = FocalLoss(
            alpha=self.config.training.focal_alpha,
            gamma=self.config.training.focal_gamma,
            ignore_index=-100
        )
        
        # Register loss weights as buffers
        self.register_buffer('lambda_ctc', torch.tensor(self.config.training.lambda_ctc))
        self.register_buffer('lambda_ce', torch.tensor(self.config.training.lambda_ce))
        self.register_buffer('lambda_focal', torch.tensor(self.config.training.lambda_focal))
        self.register_buffer('lambda_symbolic', torch.tensor(self.config.training.lambda_symbolic))
    
    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass through model.
        
        Args:
            batch: Batch dictionary
            
        Returns:
            Model outputs
        """
        return self.model(
            input_values=batch['input_values'],
            attention_mask=batch['attention_mask'],
            speaker_severity=batch.get('severity'),
            return_activations=False
        )
    
    def compute_loss(
        self,
        outputs: Dict,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        label_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            outputs: Model outputs
            labels: Ground truth phoneme labels
            input_lengths: Lengths of input sequences
            label_lengths: Lengths of label sequences
            
        Returns:
            Dictionary with individual and total losses
        """
        logits_neural = outputs['logits_neural']
        logits_constrained = outputs['logits_constrained']
        
        # CTC Loss (alignment-free)
        loss_ctc = self._compute_ctc_loss(logits_constrained, labels, input_lengths, label_lengths)
        
        # Cross-Entropy Loss (frame-level supervision)
        loss_ce = self._compute_ce_loss(logits_neural, labels)
        
        # Focal Loss (class imbalance)
        loss_focal = self._compute_focal_loss(logits_neural, labels)
        
        # Symbolic Constraint Loss
        loss_symbolic = self._compute_symbolic_loss(outputs)
        
        # Total weighted loss
        total_loss = (
            self.lambda_ctc * loss_ctc +
            self.lambda_ce * loss_ce +
            self.lambda_focal * loss_focal +
            self.lambda_symbolic * loss_symbolic
        )
        
        return {
            'loss': total_loss,
            'loss_ctc': loss_ctc,
            'loss_ce': loss_ce,
            'loss_focal': loss_focal,
            'loss_symbolic': loss_symbolic
        }
    
    def _compute_ctc_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        label_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute CTC loss."""
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_transposed = log_probs.transpose(0, 1)  # [time, batch, num_classes]
        
        return self.ctc_loss(
            log_probs_transposed,
            labels,
            input_lengths,
            label_lengths
        )
    
    def _compute_ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss with label alignment."""
        batch_size, time_steps_logits, num_classes = logits.shape
        
        # Align labels to logits time dimension
        labels_aligned = self._align_labels_to_logits(labels, time_steps_logits)
        
        # Compute loss
        logits_flat = logits.reshape(-1, num_classes)
        labels_flat = labels_aligned.reshape(-1)
        
        return self.ce_loss(logits_flat, labels_flat)
    
    def _compute_focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal loss with label alignment."""
        time_steps_logits = logits.size(1)
        labels_aligned = self._align_labels_to_logits(labels, time_steps_logits)
        return self.focal_loss(logits, labels_aligned)
    
    def _compute_symbolic_loss(self, outputs: Dict) -> torch.Tensor:
        """Compute symbolic constraint loss using KL divergence."""
        P_neural = outputs.get('P_neural')
        P_constrained = outputs.get('P_constrained')
        
        if P_neural is None or P_constrained is None:
            return torch.tensor(0.0, device=self.device)
        
        # KL divergence: encourage constrained to be close to neural when confident
        return F.kl_div(
            P_constrained.log(),
            P_neural,
            reduction='batchmean',
            log_target=False
        )
    
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
        logits_constrained = outputs['logits_constrained']
        input_lengths = torch.full(
            (logits_constrained.size(0),),
            logits_constrained.size(1),
            dtype=torch.long,
            device=logits_constrained.device
        )
        
        labels = batch['labels']
        label_lengths = (labels != -100).sum(dim=1)
        
        # Compute losses
        losses = self.compute_loss(outputs, labels, input_lengths, label_lengths)
        
        # Log metrics
        self.log('train/loss', losses['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_ctc', losses['loss_ctc'], on_step=False, on_epoch=True)
        self.log('train/loss_ce', losses['loss_ce'], on_step=False, on_epoch=True)
        self.log('train/loss_focal', losses['loss_focal'], on_step=False, on_epoch=True)
        self.log('train/loss_symbolic', losses['loss_symbolic'], on_step=False, on_epoch=True)
        self.log('train/beta', outputs['beta'], on_step=False, on_epoch=True)
        
        return losses['loss']
    
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
        
        # Compute losses
        logits_constrained = outputs['logits_constrained']
        input_lengths = torch.full(
            (logits_constrained.size(0),),
            logits_constrained.size(1),
            dtype=torch.long,
            device=logits_constrained.device
        )
        
        labels = batch['labels']
        label_lengths = (labels != -100).sum(dim=1)
        
        losses = self.compute_loss(outputs, labels, input_lengths, label_lengths)
        
        # Decode predictions for PER computation
        predictions = self._decode_predictions(logits_constrained)
        references = self._decode_references(labels)
        
        # Compute PER
        per_scores = [compute_per(pred, ref) for pred, ref in zip(predictions, references)]
        avg_per = np.mean(per_scores) if per_scores else 0.0
        
        # Store outputs for epoch-level metrics
        self.validation_step_outputs.append({
            'loss': losses['loss'],
            'per': avg_per,
            'predictions': predictions,
            'references': references,
            'status': batch['status']
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
        
        # Compute losses
        logits_constrained = outputs['logits_constrained']
        input_lengths = torch.full(
            (logits_constrained.size(0),),
            logits_constrained.size(1),
            dtype=torch.long,
            device=logits_constrained.device
        )
        
        labels = batch['labels']
        label_lengths = (labels != -100).sum(dim=1)
        
        losses = self.compute_loss(outputs, labels, input_lengths, label_lengths)
        
        # Decode and compute PER
        predictions = self._decode_predictions(logits_constrained)
        references = self._decode_references(labels)
        
        per_scores = [compute_per(pred, ref) for pred, ref in zip(predictions, references)]
        avg_per = np.mean(per_scores) if per_scores else 0.0
        
        # Log test metrics
        self.log('test/loss', losses['loss'])
        self.log('test/per', avg_per)
        
        return losses
    
    def _decode_predictions(self, logits: torch.Tensor) -> List[List[str]]:
        """
        Decode logits to phoneme sequences using greedy decoding.
        
        Args:
            logits: Model logits [batch, time, num_classes]
            
        Returns:
            List of phoneme sequences
        """
        predictions = []
        pred_ids = torch.argmax(logits, dim=-1)  # [batch, time]
        
        for seq in pred_ids:
            phonemes = []
            prev_id = None
            
            for phone_id in seq.cpu().numpy():
                # Skip blanks and padding
                if phone_id == self.phn_to_id['<BLANK>']:
                    prev_id = None
                    continue
                if phone_id == self.phn_to_id['<PAD>']:
                    continue
                # Skip repetitions (CTC-style)
                if phone_id == prev_id:
                    continue
                
                phoneme = self.id_to_phn.get(phone_id, '<UNK>')
                phonemes.append(phoneme)
                prev_id = phone_id
            
            predictions.append(phonemes)
        
        return predictions
    
    def _decode_references(self, labels: torch.Tensor) -> List[List[str]]:
        """
        Decode reference labels to phoneme sequences.
        
        Args:
            labels: Label tensor [batch, seq_len]
            
        Returns:
            List of phoneme sequences
        """
        references = []
        
        for seq in labels:
            phonemes = []
            for phone_id in seq.cpu().numpy():
                phone_id = int(phone_id)
                if phone_id == -100:  # Padding
                    break
                phoneme = self.id_to_phn.get(phone_id, '<UNK>')
                phonemes.append(phoneme)
            references.append(phonemes)
        
        return references
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Learning rate scheduler with warmup
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.training.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.training.warmup_ratio,
            anneal_strategy='cos'
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
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
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    print(f"Train: {len(train_dataset)} samples ({len(train_speakers)} speakers)")
    print(f"Val: {len(val_dataset)} samples ({len(val_speakers)} speakers)")
    print(f"Test: {len(test_dataset)} samples ({len(test_speakers)} speakers)")
    
    # Create collator
    collator = NeuroSymbolicCollator(dataset.processor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
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


def train(config: Optional[Config] = None) -> Tuple[DysarthriaASRLightning, pl.Trainer]:
    """
    Main training function.
    
    Args:
        config: Configuration object, uses default if None
        
    Returns:
        Tuple of (trained_model, trainer)
    """
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
        sampling_rate=config.data.sampling_rate
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config, dataset)
    
    # Initialize model
    print("🧠 Initializing neuro-symbolic model...")
    model = NeuroSymbolicASR(
        model_config=config.model,
        symbolic_config=config.symbolic,
        phn_to_id=dataset.phn_to_id,
        id_to_phn=dataset.id_to_phn
    )
    
    # Wrap in Lightning module
    lightning_model = DysarthriaASRLightning(
        model=model,
        config=config,
        phn_to_id=dataset.phn_to_id,
        id_to_phn=dataset.id_to_phn
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path("checkpoints") / config.experiment.run_name,
        filename='epoch={epoch:02d}-val_per={val/per:.3f}',
        monitor=config.training.checkpoint_metric,
        mode=config.training.checkpoint_mode,
        save_top_k=config.training.save_top_k,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor=config.training.early_stopping_metric,
        patience=config.training.early_stopping_patience,
        mode=config.training.early_stopping_mode,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Create trainer
    trainer = pl.Trainer(
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
        log_every_n_steps=config.experiment.log_every_n_steps
    )
    
    # Train
    print("🚀 Starting training...")
    trainer.fit(lightning_model, train_loader, val_loader)
    
    # Test on best checkpoint
    print("📈 Evaluating on test set...")
    trainer.test(lightning_model, test_loader, ckpt_path='best')
    
    print("✅ Training complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    
    return lightning_model, trainer


def main() -> None:
    """Main entry point for training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Neuro-Symbolic Dysarthria ASR")
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--run-name', type=str, help='MLflow run name')
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = Config.load(Path(args.config))
    else:
        config = get_default_config()
    
    if args.run_name:
        config.experiment.run_name = args.run_name
    
    # Train
    train(config)


if __name__ == "__main__":
    main()