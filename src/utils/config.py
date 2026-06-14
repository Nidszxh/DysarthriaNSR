import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import yaml

logger = logging.getLogger(__name__)

# REFACTOR LOG
# [FIX-T05] Added use_forced_alignment, forced_alignment_fallback_warn,
#         frame_ce_start_epoch=0, lambda_ce=0.15 config fields for CTC
#         forced alignment frame-CE loss (replaces proportional interpolation).


def get_project_root() -> Path:
    """Detects project root even if running in Notebooks or as a script."""
    return Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()

def normalize_phoneme(phoneme: str) -> str:
    """Normalize ARPABET by stripping stress markers (e.g., AH0 -> AH)."""
    return str(phoneme).rstrip('012')

# --- TORGO Continuous Severity Map ---
# Source: Rudzicz et al. (2012), "The TORGO Database of Acoustic and Articulatory
# Speech from Dysarthric Speakers", CSL.
# Intelligibility scores are publish mean percent-word-correct by trained listeners.
# Severity = (1 - intelligibility/100) * 5.0  → range [0.0 (control), 5.0 (most severe)]
#
# Raw intelligibility (approx.):  F01≈2%, F03≈7%, F04≈39%, F05≈73%
#                                 M01≈2%, M02≈4.5%, M03≈8.2%, M04≈43%, M05≈58%
#                                 FC01-FC03, MC01-MC04: controls ≈ 100%

TORGO_SEVERITY_MAP: Dict[str, float] = {
    # Dysarthric speakers — converted to severity score in [0, 5]
    "F01": 4.90,   # ~2% intelligibility   (severe)
    "F03": 4.65,   # ~7% intelligibility   (severe)
    "F04": 3.05,   # ~39% intelligibility  (moderate)
    "M01": 4.90,   # ~2% intelligibility   (severe)
    "M02": 4.78,   # ~4.5% intelligibility (severe)
    "M03": 4.59,   # ~8.2% intelligibility (severe)
    "M04": 2.85,   # ~43% intelligibility  (moderate)
    "M05": 2.10,   # ~58% intelligibility  (mild)
    # Control speakers — severity 0.0
    "FC01": 0.0,
    "FC02": 0.0,
    "FC03": 0.0,
    "MC01": 0.0,
    "MC02": 0.0,
    "MC03": 0.0,
    "MC04": 0.0,
}

def get_speaker_severity(speaker_id: str) -> float:
    """
    Return continuous severity score [0.0, 5.0] for a given TORGO speaker.
    Unknown speakers default to 2.5 (mid-range, conservative fallback).
    """
    # Strip session suffixes if present (e.g., "M01_session1" → "M01")
    base_id = speaker_id.split("_")[0].upper()
    return TORGO_SEVERITY_MAP.get(base_id, 2.5)


# --- Configuration Components ---

@dataclass(frozen=True)
class ProjectPaths:
    """Immutable path structure with auto-directory creation."""
    root: Path = field(default_factory=get_project_root)
    
    @property
    def data_dir(self) -> Path: return self.root / "data"
    @property
    def raw_dir(self) -> Path: return self.data_dir / "raw"
    @property
    def processed_dir(self) -> Path: return self.data_dir / "processed"
    @property
    def external_dir(self) -> Path: return self.data_dir / "external"
    @property
    def checkpoints_dir(self) -> Path: return self.root / "checkpoints"
    @property
    def results_dir(self) -> Path: return self.root / "results"
    @property
    def mlruns_dir(self) -> Path: return self.root / "mlruns"

    def ensure_directories(self) -> None:
        """Creates all sub-folders needed for the pipeline."""
        for path in [self.raw_dir, self.processed_dir, self.external_dir, 
                     self.checkpoints_dir, self.results_dir, self.mlruns_dir]:
            path.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """Architecture settings optimized for 8GB VRAM."""
    hubert_model_id: str = "facebook/hubert-base-ls960"
    # Pin the exact HuBERT Hub commit hash for reproducibility (H7/R-01).
    # Pinned to dba3bb02 — verified from local cache 2026-03-06.
    # Override to None only for development / CI where offline access is required.
    hubert_model_revision: Optional[str] = "dba3bb02fda4248b6e082697eee756de8fe8aa8a"
    freeze_feature_extractor: bool = True
    # Memory-vs-speed tradeoff: checkpointing reduces VRAM but adds compute.
    # Disable for faster training/eval when VRAM headroom allows.
    use_gradient_checkpointing: bool = True
    
    # Permanently freeze only the bottom 4 layers (robust generic acoustic features).
    # Layers 4-11 are fine-tuned progressively via the two-stage warmup schedule in
    # on_train_epoch_start (epoch 1: unfreeze 8-11; epoch 6: unfreeze 4-11).
    # Keeping 8 layers trainable at full training maximizes VRAM utilisation on the RTX 4060.
    freeze_encoder_layers: List[int] = field(default_factory=lambda: list(range(0, 4)))

    hidden_dim: int = 512
    # Runtime vocab is rebuilt from the manifest and includes 3 special tokens
    # (<BLANK>, <PAD>, <UNK>) plus the observed ARPABET phonemes.
    num_phonemes: Optional[int] = None
    classifier_dropout: float = 0.1
    
    # Symbolic Neural-Fusion
    constraint_weight_init: float = 0.05
    # UNUSED (reserved; articulatory distance always computed in _build_static_matrix)
    use_articulatory_distance: bool = True

    # --- Phase 3: Architectural additions (audit Proposals P2, P3) ---
    # Learnable constraint matrix (Proposal P2)
    use_learnable_constraint: bool = True
    # Cross-attention severity adapter (Proposal P3)
    use_severity_adapter: bool = True
    severity_adapter_dim: int = 128  # P2.4: increase severity-conditioning capacity

    # Temporal downsampling bottleneck (stride-2 Conv1d before phoneme head).
    # Halves frame rate (~50 Hz → ~25 Hz), forcing the model to aggregate
    # local acoustic context before predicting phonemes.  Especially useful
    # with mostly-frozen HuBERT where elongated dysarthric phonemes otherwise
    # receive direct per-frame predictions with no context window.
    use_temporal_downsample: bool = True

    # SpecAugment: time/frequency masking on HuBERT hidden states during training.
    # Prevents overfitting on the small TORGO dataset; especially effective with
    # only 16k samples and a partially-frozen encoder.
    use_spec_augment: bool = True
    spec_time_mask_prob: float = 0.05   # Fraction of frames to mask per utterance
    spec_time_mask_length: int = 10     # Max consecutive frames to mask
    spec_freq_mask_prob: float = 0.05   # Fraction of feature dims to mask
    spec_freq_mask_length: int = 8      # Max consecutive feature dims to mask

@dataclass
class TrainingConfig:
    """Training hyperparams tuned for RTX 4060 stability and speed."""
    
    # RTX 4060 (Ada) supports BF16 which is much more stable than FP16
    precision: str = "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "16-mixed"
    
    learning_rate: float = 3e-5
    # Per-group LR multipliers relative to base learning_rate.
    # HuBERT encoder: slower to avoid catastrophic forgetting.
    # Symbolic layer: moderate to stabilise constraint matrix learning.
    encoder_lr_multiplier: float = 0.1
    symbolic_lr_multiplier: float = 0.5
    weight_decay: float = 0.01
    optimizer: str = "AdamW" 
    # [FIX-T04] Replaced OneCycleLR with CosineAnnealingWarmRestarts
    # OneCycleLR's step counter cannot be rewound; warm restarts provide proper
    # LR cycling per unfreeze stage.
    lr_scheduler: str = "cosine_warm_restarts"
    cosine_T0: int = 1          # epochs until first restart (matches encoder_warmup_epochs)
    cosine_T_mult: int = 2      # each cycle is 2× the previous
    cosine_eta_min_ratio: float = 0.001  # eta_min = lr * this ratio
    warmup_steps: int = 250
    warmup_ratio: float = 0.05       # UNUSED (reserved; scheduler uses cosine_warm_restarts)
    
    batch_size: int = 12   # RTX 4060: safe upper bound after OOM at batch=16; effective batch stays 36 with gradient accumulation
    gradient_accumulation_steps: int = 3   # Effective batch=36 (12×3)
    use_stratified_micro_batch: bool = True
    stratified_dysarthric_ratio: float = 0.75
    max_epochs: int = 40
    encoder_warmup_epochs: int = 1          # Unfreeze top layers after 1 epoch (was 3) — spend less time VRAM-idle
    encoder_second_unfreeze_epoch: int = 6  # Stage 2: unfreeze layers 6-11 at epoch 6
    encoder_third_unfreeze_epoch: int = 12  # Stage 3: unfreeze layers 4-11 at epoch 12 (deepest adaptation)
    val_check_interval: float = 1.0        # Validate once per epoch (was 0.5) — halves eval overhead
    check_val_every_n_epoch: int = 1        # Evaluate every N epochs (set 2 to halve val overhead)
    num_sanity_val_steps: int = 0           # Skip startup sanity-validation for faster fold startup
    
    # Regularization & Loss
    dropout: float = 0.1
    layer_dropout: float = 0.05      # UNUSED (reserved for future stochastic depth)
    label_smoothing: float = 0.1
    gradient_clip_val: float = 1.0

    # Multi-task loss weights (primary)
    lambda_ctc: float = 0.8
    # [FIX-2] (Historical) Frame-CE was originally gated behind epoch 15 because CTC
    # lacked forced alignment. [FIX-T05] replaced proportional interpolation with
    # torchaudio.functional.forced_align, so the gate is disabled (frame_ce_start_epoch=0).
    frame_ce_start_epoch: int = 0
    use_forced_alignment: bool = True
    forced_alignment_fallback_warn: bool = True
    lambda_ce: float = 0.15
    lambda_articulatory: float = 0.08  # T-01: reduced from 0.15 — articulatory accuracy already ~78–92%; marginal gain is low

    # --- Phase 2: New loss weights (audit Proposals P1, P2, R3) ---
    # Ordinal contrastive severity loss (Proposal P1)
    lambda_ordinal: float = 0.05
    # Staged lambda_ordinal warmup — ordinal loss is most meaningful after basic
    # phoneme discrimination is established (epoch 10+).
    ordinal_stage1_end: int = 10
    ordinal_stage1_value: float = 0.01
    ordinal_stage2_end: int = 20
    ordinal_stage2_value: float = 0.03
    # Blank-prior KL regularisation (fix CTC insertion pathology)
    lambda_blank_kl: float = 0.20   # Full-stage target (epochs 20+); gentler than old 0.35
    blank_target_prob: float = 0.75   # B2/T-03 fix: 0.82 overshoots deletion rate; 0.75 allows slightly more phoneme emission (audit §3.3 / §4-T03)
    # I2: Staged lambda_blank_kl warmup — prevents early CTC collapse from aggressively
    # suppressing blanks before the model has learned basic phoneme boundaries.
    # Schedule:  epochs < stage1_end  → stage1_value (gentle push)
    #            epochs < stage2_end  → stage2_value (moderate push)
    #            epochs >= stage2_end → lambda_blank_kl (full target)
    blank_kl_stage1_end: int = 10
    blank_kl_stage1_value: float = 0.10
    blank_kl_stage2_end: int = 20
    blank_kl_stage2_value: float = 0.15
    # Symbolic KL anchor (keeps learnable C near symbolic prior)
    # §3.8 fix: Old 0.05 with batchmean/V=47 had effective per-row weight ≈0.001,
    # too weak to prevent degenerate constraint matrix.  0.5 gives ~0.01 per row.
    lambda_symbolic_kl: float = 0.5
    # Staged lambda_symbolic_kl warmup — gentle constraint at first, full after
    # the neural head stabilises (epoch 15+).
    symbolic_kl_stage1_end: int = 5
    symbolic_kl_stage1_value: float = 0.1
    symbolic_kl_stage2_end: int = 15
    symbolic_kl_stage2_value: float = 0.3

    # Logging & Checkpointing
    monitor_metric: str = "val/per"
    monitor_mode: str = "min"
    early_stopping_patience: int = 8
    # [FIX-4] LOSO runs use higher patience (22 vs 8) due to fold variance
    # Single-split has ~2 test speakers; LOSO has 15 folds with different speaker
    # combinations. Higher patience prevents early stopping on unfavorable folds.
    # Audit §1.4: early stopping was biasing aggregate metrics.
    loso_early_stopping_patience: int = 22
    beam_length_norm_alpha: float = 0.6  # Exponent for beam-search length normalisation: score / len^alpha
    beam_lm_weight: float = 0.0           # UNUSED (reserved for LM shallow-fusion; evaluate.py passes it via CLI flag)
    save_top_k: int = 2
    
    # HW Acceleration — optimised for RTX 4060 + modern NVMe
    num_workers: int = 8          # saturate PCIe; was 4
    pin_memory: bool = True
    prefetch_factor: int = 4       # deeper prefetch queue reduces GPU stalls; was 2
    blank_priority_weight: float = 1.0  # UNUSED (reserved) — dataloader BLANK/PAD multipliers set to 1.0 directly

    # --- Phase 4: Training infrastructure (audit G2, ablations) ---
    # Leave-One-Speaker-Out cross-validation (opt-in, keeps fast single-split default)
    use_loso: bool = False
    loso_bootstrap_samples: int = 2000  # LOSO PER CI bootstrap draws
    # Ablation mode: "full" | "neural_only" | "symbolic_only" | "no_art_heads"
    ablation_mode: str = "full"

@dataclass
class DataConfig:
    """Data processing limits."""
    data_dir: Path = field(default_factory=lambda: ProjectPaths().data_dir)
    manifest_path: Path = field(default_factory=lambda: ProjectPaths().processed_dir / "torgo_neuro_symbolic_manifest.csv")
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    split_strategy: str = "speaker_stratified"
    sampling_rate: int = 16000
    max_audio_length: float = 6.0  # 6s — reverted from 8s after OOM at batch=16+8s; safe on 8 GB VRAM

@dataclass
class ExperimentConfig:
    """Experiment settings (MLflow + runtime config).

    Only fields that are read via direct attribute access must be kept.
    Fields accessed via getattr(..., default) are safe to remove —
    we keep temperature_calibration_range for YAML round-trip consistency.
    """
    experiment_name: str = "DysarthriaNSR"
    run_name: str = "rtx4060_optimized_v1"
    tracking_uri: str = field(default_factory=lambda: (
        os.environ.get("MLFLOW_TRACKING_URI")
        or f"file://{ProjectPaths().mlruns_dir.resolve()}"
    ))
    log_every_n_steps: int = 20
    generate_explanations: bool = True
    compute_uncertainty: bool = False
    uncertainty_n_samples: int = 20
    use_temperature_calibration: bool = False
    temperature_calibration_range: tuple = (0.5, 3.0)
    seed: int = 42
    deterministic: bool = True

@dataclass
class SymbolicConfig:
    """Evidence-based phoneme substitution rules for Dysarthria."""
    substitution_rules: Dict[Tuple[str, str], float] = field(default_factory=lambda: {
        # Devoicing
        ('B', 'P'): 0.85, ('D', 'T'): 0.82, ('G', 'K'): 0.80,
        ('V', 'F'): 0.75, ('Z', 'S'): 0.78, ('ZH', 'SH'): 0.70, ('JH', 'CH'): 0.72,
        # Fronting
        ('K', 'T'): 0.65, ('G', 'D'): 0.65, ('NG', 'N'): 0.60, ('SH', 'S'): 0.60,
        # Liquid Gliding
        ('R', 'W'): 0.70, ('L', 'W'): 0.60, ('L', 'Y'): 0.55,
        # Stopping
        ('S', 'T'): 0.55, ('Z', 'D'): 0.55, ('F', 'P'): 0.50, ('V', 'B'): 0.50,
        ('TH', 'T'): 0.58, ('DH', 'D'): 0.58,
        # Vowel Centralization
        ('IY', 'IH'): 0.45, ('UW', 'UH'): 0.45, ('EY', 'EH'): 0.42,
    })
    manner_weight: float = 0.4
    place_weight: float = 0.35
    voice_weight: float = 0.25
    distance_decay_factor: float = 3.0
    min_rule_confidence: float = 0.05  # C5: lowered to 0.05; β ≈ 0.05–0.20 at typical operating range
    severity_threshold_mild: float = 2.0   # UNUSED (reserved for future severity binning)
    severity_threshold_severe: float = 4.0 # UNUSED (reserved for future severity binning)
    constraint_entropy_penalty_weight: float = 0.05
    track_rule_activations: bool = True
    generate_confusion_matrix: bool = True  # UNUSED (reserved; confusion matrix generated unconditionally in evaluate.py)
    severity_beta_slope: float = 0.2
    # TORGO severity range spans [0.0, 4.9]; 5.0 is the ceiling for the
    # binary status → severity fallback.  Used to normalise severity values
    # into [0, 1] for adaptive beta and severity adapter conditioning.
    severity_normalization_constant: float = 5.0
    # Upper bound for adaptive beta clamping. 0.8 means symbolic constraint
    # can contribute at most 80% weight even for the most severe dysarthria.
    max_beta: float = 0.8
    # [FIX-1] Blank-frame constraint threshold lowered from 0.5 → 0.25
    # Previously ~85% of CTC frames bypassed the constraint, producing near-zero
    # gradient signal to logit_C and beta. 0.25 allows constraint to act on
    # ambiguous transition frames while still protecting blank-dominant frames.
    # Audit §1.1: blank-frame bypass threshold too aggressive.
    blank_constraint_threshold: float = 0.25
    # Blank-column penalty weight in SymbolicKLLoss (default 0.1).
    # Penalises blank-dominated rows in the learned constraint matrix.
    blank_penalty_weight: float = 0.1


# --- Master Config Handler ---

class Config:
    def __init__(self) -> None:
        self.paths = ProjectPaths()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.experiment = ExperimentConfig()
        self.symbolic = SymbolicConfig()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.effective_batch_size = self.training.batch_size * self.training.gradient_accumulation_steps
        
        self.paths.ensure_directories()
        # NOTE: _print_vram_status() is intentionally NOT called here.
        # run_pipeline.py sets config.training.ablation_mode *after* Config() is
        # constructed, so printing the banner here would always show 'Ablation: full'.
        # Call config.print_vram_status() explicitly after all overrides are applied.

    def print_vram_status(self) -> None:
        """Print the RTX 4060 VRAM safety report.  Call AFTER all config overrides."""
        self._print_vram_status()

    def _print_vram_status(self) -> None:
        """Prints a rough VRAM safety report for the 4060's 8GB limit.

        This is an estimate, not runtime telemetry. It intentionally reports a
        conservative peak budget for later-stage fine-tuning.
        """
        if self.device == "cuda":
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            # Estimate breakdown:
            # - Parameters: stored in fp32 regardless of mixed precision (4 bytes each)
            # - Gradients: use compute dtype (bf16 → 2 bytes, fp32 → 4 bytes)
            # - AdamW optimizer states: always fp32 m+v (8 bytes per trainable param)
            # - Activations: use compute dtype, scaled by layers × batch × frames
            # - Runtime reserve: CUDA workspaces, caching allocator overhead, NCCL
            is_mixed = str(self.training.precision).startswith("bf16") or str(self.training.precision).startswith("16")
            param_bytes = 4       # params always fp32 in Lightning mixed precision
            compute_bytes = 2 if is_mixed else 4
            # Temporal downsampler halves HuBERT 50Hz output to ~25Hz.
            T_frames = self.data.max_audio_length * 25  # post-downsampler frames
            hidden = 768

            # Architecture constants for HuBERT-base + task heads.
            total_params = 99_000_000
            trainable_warmup = 4_600_000
            trainable_peak = 66_400_000

            # Stage-dependent activation proxies.
            active_layers_warmup = 0   # encoder frozen warmup
            active_layers_peak = max(1, 12 - len(self.model.freeze_encoder_layers))
            # Gradient checkpointing stores far fewer intermediates; without it,
            # saved activations scale ~2-3× per layer.  16× is for no-checkpointing.
            ckpt = self.model.use_gradient_checkpointing
            activation_factor = 3.0 if ckpt else 16.0

            def _estimate(trainable_params: int, active_layers: int) -> float:
                param_gb = (total_params * param_bytes) / 1e9
                grad_gb = (trainable_params * compute_bytes) / 1e9
                optim_gb = (trainable_params * 8) / 1e9  # AdamW m+v always fp32
                act_gb = (
                    self.training.batch_size * T_frames * hidden * max(1, active_layers) *
                    compute_bytes * activation_factor
                ) / 1e9
                runtime_reserve_gb = 1.2
                return param_gb + grad_gb + optim_gb + act_gb + runtime_reserve_gb

            est_warmup = _estimate(trainable_warmup, active_layers_warmup)
            est_peak = _estimate(trainable_peak, active_layers_peak)
            margin = total_vram - est_peak
            ckpt_label = "on" if ckpt else "off"
            logger.info("--- ⚡ VRAM ESTIMATE ---")
            logger.info(
                "Warmup→Peak: %.2fGB → %.2fGB / %.2fGB (margin: %.2fGB) "
                "[batch=%d, T=%.0f, peak_active_layers=%d, ckpt=%s]",
                est_warmup, est_peak, total_vram, margin,
                self.training.batch_size, T_frames,
                active_layers_peak, ckpt_label,
            )
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            logger.info("CUDA now (allocated/reserved): %.2fGB / %.2fGB", allocated, reserved)
            logger.info("Precision: %s | Batch Size: %d", self.training.precision, self.training.batch_size)
            logger.info("SeverityAdapter: %s | LearnableC: %s | Ablation: %s",
                        self.model.use_severity_adapter,
                        self.model.use_learnable_constraint,
                        self.training.ablation_mode)
            logger.info("Note: estimate is conservative and not a direct profiler reading.")
            logger.info("----------------------------------")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the config for saving/logging."""
        data = {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": {k: str(v) if isinstance(v, Path) else v for k, v in self.data.__dict__.items()},
            "experiment": {k: list(v) if isinstance(v, tuple) else v for k, v in self.experiment.__dict__.items()},
            "symbolic": {k: v for k, v in self.symbolic.__dict__.items() if k != 'substitution_rules'},
        }
        # Flatten rules for YAML (Tuples are not YAML standard)
        data["symbolic"]["substitution_rules"] = {f"{k[0]}_{k[1]}": v for k, v in self.symbolic.substitution_rules.items()}
        # H8: Persist TORGO_SEVERITY_MAP so any change is tracked in saved configs
        data["severity_map"] = dict(TORGO_SEVERITY_MAP)
        return data

    def save(self, path: Path) -> None:
        """Saves config to YAML."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        logger.info("Config saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """
        Load config from a YAML file previously written by Config.save().

        Only keys present in the YAML override defaults; all other fields
        retain their dataclass defaults. This makes the method forward-
        compatible when new fields are added to a dataclass.

        Args:
            path: Path to the YAML config file.

        Returns:
            Config instance with fields overridden from the YAML.
        """
        config = cls()
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        def _apply(target_obj: object, section: dict) -> None:
            """Recursively apply YAML section onto a dataclass instance."""
            for key, value in section.items():
                if not hasattr(target_obj, key):
                    logger.warning("Unknown config key '%s' ignored in %s", key, path)
                    continue
                current = getattr(target_obj, key)
                if isinstance(current, Path):
                    setattr(target_obj, key, Path(value))
                elif isinstance(current, list) and isinstance(value, list):
                    setattr(target_obj, key, value)
                else:
                    setattr(target_obj, key, value)

        _apply(config.model,      data.get('model',      {}))
        _apply(config.training,   data.get('training',   {}))
        _apply(config.data,       data.get('data',       {}))
        _apply(config.experiment, data.get('experiment', {}))

        # SymbolicConfig — handle substitution_rules separately
        symbolic_data = dict(data.get('symbolic', {}))
        rules_raw = symbolic_data.pop('substitution_rules', None)
        _apply(config.symbolic, symbolic_data)
        if rules_raw and isinstance(rules_raw, dict):
            # Reconstruct tuple keys from "X_Y" → ('X', 'Y')
            reconstructed = {}
            for key_str, prob in rules_raw.items():
                parts = str(key_str).split('_', 1)
                if len(parts) == 2:
                    reconstructed[(parts[0], parts[1])] = float(prob)
            config.symbolic.substitution_rules = reconstructed

        return config

_default_config: Optional["Config"] = None


def get_default_config() -> "Config":

    global _default_config
    if _default_config is None:
        _default_config = Config()
    return copy.deepcopy(_default_config)
