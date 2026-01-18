from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import torch


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate from src/utils/config.py to project root
    return Path(__file__).resolve().parent.parent.parent


def get_processed_dir() -> Path:
    """Get the processed data directory."""
    return get_project_root() / "data" / "processed"


@dataclass
class ModelConfig:
    
    # HuBERT Encoder
    hubert_model_id: str = "facebook/hubert-base-ls960"
    freeze_feature_extractor: bool = True
    freeze_encoder_layers: List[int] = field(default_factory=lambda: list(range(0, 6)))
    
    # Phoneme Classifier
    hidden_dim: int = 512
    num_phonemes: int = 42  # Updated from dataset vocabulary
    classifier_dropout: float = 0.1
    
    # Symbolic Constraint Layer
    constraint_weight_init: float = 0.3
    constraint_learnable: bool = True
    use_articulatory_distance: bool = True


@dataclass
class TrainingConfig:
    # Training hyperparameters optimized for RTX 4060 8GB.
    
    # Optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    optimizer: str = "AdamW"
    
    # Learning Rate Scheduling
    lr_scheduler: str = "cosine_with_warmup"
    warmup_steps: int = 500
    warmup_ratio: float = 0.1
    
    # Training Dynamics (VRAM-optimized)
    batch_size: int = 2  # Small batch for 8GB VRAM
    gradient_accumulation_steps: int = 12  # Effective batch size = 24
    max_epochs: int = 2  # For experimentation
    val_check_interval: float = 0.25  # Validate 4 times per epoch
    
    # Regularization
    dropout: float = 0.1
    layer_dropout: float = 0.05
    label_smoothing: float = 0.1
    gradient_clip_val: float = 1.0
    
    # Loss Weights (simplified to 2 losses)
    lambda_ctc: float = 0.7  # Primary alignment loss
    lambda_ce: float = 0.3   # Auxiliary frame-level loss
    
    # Monitoring (unified metric for both checkpointing and early stopping)
    monitor_metric: str = "val/per"
    monitor_mode: str = "min"
    early_stopping_patience: int = 10
    save_top_k: int = 3
    
    # Memory Optimization
    precision: str = "16-mixed"  # Mixed precision for RTX 4060
    use_cpu_offload: bool = False
    
    # Data Loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


@dataclass
class DataConfig:
    # Data pipeline configuration.
    
    # Paths
    data_dir: Path = field(default_factory=lambda: get_project_root() / "data")
    manifest_path: Path = field(
        default_factory=lambda: get_processed_dir() / "torgo_neuro_symbolic_manifest.csv")
    
    # Dataset Split
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    split_strategy: str = "speaker_stratified"
    
    # Audio Processing
    sampling_rate: int = 16000
    max_audio_length: float = 10.0  # seconds


@dataclass
class ExperimentConfig:
    # MLflow and logging configuration.
    
    # Experiment Tracking
    experiment_name: str = "DysarthriaNSR"
    run_name: str = "baseline_v1"
    tracking_uri: str = field(default_factory=lambda: f"file:{get_project_root() / 'mlruns'}")
    
    # Logging
    log_every_n_steps: int = 10
    log_gradients: bool = False
    log_model_architecture: bool = True
    
    # Artifacts
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    save_attention_maps: bool = False  # Memory intensive
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


@dataclass
class SymbolicConfig:
    # Symbolic reasoning and explainability configuration.
    
    # Dysarthric Substitution Rules (from clinical phonetics literature)
    substitution_rules: Dict[Tuple[str, str], float] = field(default_factory=lambda: {
        # Devoicing (high probability in dysarthria)
        ('B', 'P'): 0.85,
        ('D', 'T'): 0.82,
        ('G', 'K'): 0.80,
        ('V', 'F'): 0.75,
        ('Z', 'S'): 0.78,
        ('ZH', 'SH'): 0.70,
        
        # Fronting
        ('K', 'T'): 0.65,
        ('G', 'D'): 0.65,
        ('SH', 'S'): 0.60,
        
        # Liquid Gliding
        ('R', 'W'): 0.70,
        ('L', 'W'): 0.60,
        ('L', 'Y'): 0.55,
        
        # Stopping (fricatives → stops)
        ('S', 'T'): 0.55,
        ('Z', 'D'): 0.55,
        ('F', 'P'): 0.50,
        ('V', 'B'): 0.50,
    })
    
    # Articulatory Feature Weights for Similarity Computation
    manner_weight: float = 0.4
    place_weight: float = 0.35
    voice_weight: float = 0.25
    
    # Rule Application Thresholds
    min_rule_confidence: float = 0.5
    severity_threshold_mild: float = 2.0
    severity_threshold_severe: float = 4.0
    
    # Explainability (disabled until implemented)
    track_rule_activations: bool = False
    generate_confusion_matrix: bool = True


class Config:
    # Master configuration combining all sub-configs.
    
    def __init__(
        self,
        model: ModelConfig = None,
        training: TrainingConfig = None,
        data: DataConfig = None,
        experiment: ExperimentConfig = None,
        symbolic: SymbolicConfig = None,
    ):
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.data = data or DataConfig()
        self.experiment = experiment or ExperimentConfig()
        self.symbolic = symbolic or SymbolicConfig()
        
        self._post_init()
    
    def _post_init(self) -> None:
        # Compute effective batch size
        self.training.effective_batch_size = (
            self.training.batch_size * self.training.gradient_accumulation_steps
        )
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate VRAM constraints for CUDA
        if self.device == "cuda":
            self._validate_vram_constraints()
    
    def _validate_vram_constraints(self) -> None:
        # Validate and adjust batch size based on available VRAM
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if total_vram < 10 and self.training.batch_size > 2:
            print(f"⚠️  Detected {total_vram:.1f}GB VRAM. Reducing batch size to 2.")
            self.training.batch_size = 2
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for logging.
        
        Returns:
            Dictionary representation of all configuration parameters
        """
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "experiment": self.experiment.__dict__,
            "symbolic": self.symbolic.__dict__,
            "device": self.device,
        }
    
    def save(self, path: Path) -> None:
        # Save configuration to YAML file.

        import yaml
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: Path) -> 'Config':
        # Load configuration from YAML file.
        import yaml
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            data=DataConfig(**config_dict['data']),
            experiment=ExperimentConfig(**config_dict['experiment']),
            symbolic=SymbolicConfig(**config_dict['symbolic']),
        )


def get_default_config() -> Config:
    return Config()

def main() -> None:
    config = get_default_config()
    
    print(f"Configuration Summary:")
    print(f"Device: {config.device}")
    print(f"Effective Batch Size: {config.training.effective_batch_size}")
    print(f"Max Epochs: {config.training.max_epochs}")
    print(f"Precision: {config.training.precision}")
    print(f"Loss Weights: CTC={config.training.lambda_ctc}, CE={config.training.lambda_ce}")
    print(f"Monitor Metric: {config.training.monitor_metric} ({config.training.monitor_mode})")
    print(f"Top-K Checkpoints: {config.training.save_top_k}")


if __name__ == "__main__":
    main()