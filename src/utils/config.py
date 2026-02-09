from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import yaml

# --- Utilities ---

def get_project_root() -> Path:
    """Detects project root even if running in Notebooks or as a script."""
    return Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()

def normalize_phoneme(phoneme: str) -> str:
    """Normalize ARPABET by stripping stress markers (e.g., AH0 -> AH)."""
    return str(phoneme).rstrip('012')

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
    freeze_feature_extractor: bool = True
    
    # 4060 TWEAK: Freezing 10/12 layers provides the best VRAM/Accuracy trade-off
    freeze_encoder_layers: List[int] = field(default_factory=lambda: list(range(0, 10)))
    
    hidden_dim: int = 512
    num_phonemes: int = 44
    classifier_dropout: float = 0.1
    
    # Symbolic Neural-Fusion
    constraint_weight_init: float = 0.05
    constraint_learnable: bool = True
    use_articulatory_distance: bool = True

@dataclass
class TrainingConfig:
    """Training hyperparams tuned for RTX 4060 stability and speed."""
    
    # RTX 4060 (Ada) supports BF16 which is much more stable than FP16
    precision: str = "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "16-mixed"
    
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    optimizer: str = "AdamW" 
    lr_scheduler: str = "onecycle"
    warmup_steps: int = 250
    warmup_ratio: float = 0.05
    
    batch_size: int = 4  # Increased from 1 or 2 to maximize 4060 core utilization
    gradient_accumulation_steps: int = 8 # Effective batch = 32
    max_epochs: int = 30
    encoder_warmup_epochs: int = 3
    val_check_interval: float = 0.5
    
    # Regularization & Loss
    dropout: float = 0.1
    layer_dropout: float = 0.05
    label_smoothing: float = 0.1
    gradient_clip_val: float = 1.0
    lambda_ctc: float = 0.8
    lambda_ce: float = 0.2
    lambda_articulatory: float = 0.1
    
    # Logging & Checkpointing
    monitor_metric: str = "val/per"
    monitor_mode: str = "min"
    early_stopping_patience: int = 8
    save_top_k: int = 2
    
    # HW Acceleration
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    blank_priority_weight: float = 1.5 

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
    max_audio_length: float = 6.0 # 6s covers ~99% of TORGO clips

@dataclass
class ExperimentConfig:
    """MLflow and Experiment Tracking."""
    experiment_name: str = "DysarthriaNSR"
    run_name: str = "rtx4060_optimized_v1"
    tracking_uri: str = field(default_factory=lambda: f"file:{ProjectPaths().mlruns_dir}")
    log_every_n_steps: int = 20
    log_gradients: bool = False
    log_model_architecture: bool = True
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    save_attention_maps: bool = False
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
    min_rule_confidence: float = 0.5
    severity_threshold_mild: float = 2.0
    severity_threshold_severe: float = 4.0
    track_rule_activations: bool = True
    generate_confusion_matrix: bool = True
    severity_beta_slope: float = 0.2

# --- Master Config Handler ---

class Config:
    def __init__(self):
        self.paths = ProjectPaths()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.experiment = ExperimentConfig()
        self.symbolic = SymbolicConfig()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.effective_batch_size = self.training.batch_size * self.training.gradient_accumulation_steps
        
        self.paths.ensure_directories()
        self._print_vram_status()

    def _print_vram_status(self):
        """Prints a safety report for the 4060's 8GB limit."""
        if self.device == "cuda":
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            # Estimation based on BF16 + 10-layer freeze
            est_vram = (self.training.batch_size * self.data.max_audio_length * 0.24) + 1.8
            margin = total_vram - est_vram
            print(f"--- ⚡ RTX 4060 OPTIMIZATION ---")
            print(f"Est. Peak VRAM: {est_vram:.2f}GB / {total_vram:.2f}GB (Margin: {margin:.2f}GB)")
            print(f"Precision: {self.training.precision} | Batch Size: {self.training.batch_size}")
            print(f"----------------------------------")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the config for saving/logging."""
        data = {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": {k: str(v) if isinstance(v, Path) else v for k, v in self.data.__dict__.items()},
            "experiment": {k: str(v) if isinstance(v, Path) else v for k, v in self.experiment.__dict__.items()},
            "symbolic": {k: v for k, v in self.symbolic.__dict__.items() if k != 'substitution_rules'},
        }
        # Flatten rules for YAML (Tuples are not YAML standard)
        data["symbolic"]["substitution_rules"] = {f"{k[0]}_{k[1]}": v for k, v in self.symbolic.substitution_rules.items()}
        return data

    def save(self, path: Path):
        """Saves config to YAML."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        print(f"✅ Config saved to {path}")

def get_default_config() -> "Config":
    return cfg

cfg = Config()