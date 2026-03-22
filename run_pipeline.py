"""
run_pipeline.py — DysarthriaNSR End-to-End Training & Evaluation Orchestrator
==============================================================================

Unified entry point that coordinates training (train.py) and evaluation
(evaluate.py) under a single, consistent --run-name namespace.

All output paths derive from --run-name:
    checkpoints/{run_name}/   ← model checkpoints (written by train.py)
    results/{run_name}/       ← evaluation artifacts (written by evaluate_model)

Usage examples
--------------
Full pipeline (train + eval):
    python run_pipeline.py --run-name experiment_v1

Smoke run (1 epoch, 5 batches):
    python run_pipeline.py --run-name smoke --smoke

Train only:
    python run_pipeline.py --run-name experiment_v1 --skip-eval

Eval only (checkpoint must already exist):
    python run_pipeline.py --run-name experiment_v1 --skip-train

With explainability + uncertainty:
    python run_pipeline.py --run-name experiment_v1 --skip-train --explain --uncertainty

LOSO cross-validation:
    python run_pipeline.py --run-name loso_v1 --loso --skip-eval

Warm feature cache only:
    python run_pipeline.py --run-name cache_warmup --warm-cache --warm-cache-only
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

# Project-root on sys.path (supports running from any CWD)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.utils.config import Config, get_default_config, get_project_root
from src.data.dataloader import NeuroSymbolicCollator, TorgoNeuroSymbolicDataset
from src.data.warm_feature_cache import warm_feature_cache
from src.models.model import NeuroSymbolicASR
from train import (
    DysarthriaASRLightning,
    create_dataloaders,
    run_loso,
    train,
)
from evaluate import evaluate_model, BigramLMScorer

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# Internal helpers

def _load_dataset(config: Config) -> TorgoNeuroSymbolicDataset:
    """Instantiate TorgoNeuroSymbolicDataset from config paths."""
    log.info("Loading dataset from %s", config.data.manifest_path)
    return TorgoNeuroSymbolicDataset(
        manifest_path=str(config.data.manifest_path),
        processor_id=config.model.hubert_model_id,
        sampling_rate=config.data.sampling_rate,
        max_audio_length=config.data.max_audio_length,
    )


def _build_test_loader(
    config: Config,
    dataset: TorgoNeuroSymbolicDataset,
) -> torch.utils.data.DataLoader:
    """Return the test DataLoader by running the same deterministic split as train()."""
    _, _, test_loader = create_dataloaders(config, dataset)
    return test_loader


def _resolve_checkpoint(run_name: str) -> Path:
    """
    Find the best available checkpoint under checkpoints/{run_name}/.

    Resolution order:
    1. Non-last ckpt files with ``val_per=X.XXX`` in the filename — pick lowest.
    2. ``last.ckpt`` fallback.

    Raises
    ------
    FileNotFoundError
        If no .ckpt file exists in the checkpoint directory.
    """
    ckpt_dir = get_project_root() / "checkpoints" / run_name
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            "Run without --skip-train first, or verify --run-name is correct."
        )

    all_ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not all_ckpts:
        raise FileNotFoundError(
            f"No .ckpt files found in {ckpt_dir}."
        )

    # Prefer best-metric checkpoint (val_per=X.XXX in filename, lowest wins)
    _per_pattern = re.compile(r"val_per=([0-9]+\.[0-9]+)")
    scored: list[Tuple[float, Path]] = []
    for p in all_ckpts:
        if p.name == "last.ckpt":
            continue
        m = _per_pattern.search(p.name)
        if m:
            scored.append((float(m.group(1)), p))

    if scored:
        scored.sort(key=lambda t: t[0])
        best = scored[0][1]
        log.info("Resolved best checkpoint: %s (val_per=%.4f)", best.name, scored[0][0])
        return best

    # Fallback: last.ckpt
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        log.warning(
            "No scored checkpoint found in %s; using last.ckpt.", ckpt_dir
        )
        return last_ckpt

    # Last resort: alphabetically last .ckpt
    fallback = sorted(all_ckpts)[-1]
    log.warning("Falling back to checkpoint: %s", fallback.name)
    return fallback


def _load_lightning_model(
    ckpt_path: Path,
    config: Config,
    dataset: TorgoNeuroSymbolicDataset,
) -> DysarthriaASRLightning:
    """
    Reconstruct a DysarthriaASRLightning instance from a checkpoint file.

    Mirrors the pattern used inside train.py after fitting.
    """
    log.info("Loading model from checkpoint: %s", ckpt_path)
    model_arch = NeuroSymbolicASR(
        model_config=config.model,
        symbolic_config=config.symbolic,
        phn_to_id=dataset.phn_to_id,
        id_to_phn=dataset.id_to_phn,
        manner_to_id=dataset.manner_to_id,
        place_to_id=dataset.place_to_id,
        voice_to_id=dataset.voice_to_id,
    )
    lightning_model = DysarthriaASRLightning.load_from_checkpoint(
        str(ckpt_path),
        model=model_arch,
        config=config,
        phn_to_id=dataset.phn_to_id,
        id_to_phn=dataset.id_to_phn,
        strict=False,
    )
    return lightning_model


# Pipeline stages

def run_training(
    config: Config,
    loso: bool,
    resume_loso: bool,
    loso_force_speakers: Optional[list[str]],
    loso_use_beam_search: bool,
    loso_beam_width: int,
    loso_generate_explanations: bool,
    loso_compute_uncertainty: bool,
    loso_uncertainty_n_samples: int,
    limit_train_batches: Optional[int],
) -> Tuple[Optional[DysarthriaASRLightning], Optional[TorgoNeuroSymbolicDataset]]:
    """
    Execute the training stage.

    Parameters
    ----------
    config:
        Fully populated Config instance with run_name, ablation, epochs etc.
    loso:
        If True, run Leave-One-Speaker-Out CV via ``run_loso()``.
        Each fold is trained and evaluated independently inside run_loso.
        No single model is returned in this mode.
    limit_train_batches:
        If set, passed to ``train()`` to cap batches per epoch (smoke mode).

    Returns
    -------
    (lightning_model, dataset)
        lightning_model is None when loso=True (each fold has its own model).
        dataset is returned so run_evaluation can reuse it without reloading.
    """
    import pytorch_lightning as pl  # local to avoid polluting the module namespace

    pl.seed_everything(config.experiment.seed, workers=True)

    # Runtime throughput configuration: LOSO bypasses train(), so set these
    # here to ensure both single-run and LOSO paths use fast CUDA defaults.
    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available() and str(config.device).startswith("cuda"):
        # TF32 accelerates matmul/conv on Ampere+ with negligible quality impact
        # for this ASR workload.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # For variable-length audio, cuDNN benchmarking usually helps; keep
        # deterministic behavior if explicitly requested.
        torch.backends.cudnn.benchmark = not bool(config.experiment.deterministic)

    if loso:
        log.info("=== LOSO mode — loading dataset ===")
        dataset = _load_dataset(config)
        log.info("=== Starting LOSO cross-validation ===")
        loso_results = run_loso(
            config,
            dataset,
            resume=resume_loso,
            force_speakers=loso_force_speakers,
            use_beam_search=loso_use_beam_search,
            beam_width=loso_beam_width,
            generate_explanations=loso_generate_explanations,
            compute_uncertainty=loso_compute_uncertainty,
            uncertainty_n_samples=loso_uncertainty_n_samples,
        )
        log.info(
            "LOSO complete: mean PER = %.4f [95%% CI: %.4f – %.4f] | weighted PER = %.4f | mean WER = %.4f",
            loso_results["macro_avg_per"],
            loso_results["per_95ci"][0],
            loso_results["per_95ci"][1],
            loso_results.get("weighted_avg_per", float("nan")),
            loso_results.get("macro_avg_wer", float("nan")),
        )
        # Each fold evaluated internally; no single model or test_loader to return.
        return None, None

    log.info("=== Starting single-run training (run_name=%s) ===", config.experiment.run_name)
    best_model, _trainer = train(config, limit_train_batches=limit_train_batches)
    # Reload the dataset so the test_loader for comprehensive eval is built fresh
    # (train() discards the dataset after returning; we need vocab + splits again)
    dataset = _load_dataset(config)
    return best_model, dataset


def run_evaluation(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    dataset: TorgoNeuroSymbolicDataset,
    config: Config,
    device: str,
    results_dir: Path,
    use_beam_search: bool,
    beam_width: int,
    generate_explanations: bool,
    compute_uncertainty: bool,
    uncertainty_n_samples: int,
    lm_scorer: Optional["BigramLMScorer"] = None,
    lm_weight: float = 0.0,
    ablation_mode: str = "full",
) -> Dict:
    """
    Execute the evaluation stage via ``evaluate_model()``.

    Parameters
    ----------
    model:
        The raw ``NeuroSymbolicASR`` instance (``lightning_model.model``).
    test_loader:
        DataLoader for the test split.
    dataset:
        Full dataset — used to extract phn_to_id / id_to_phn vocabularies.
    config:
        Config instance (provides symbolic_rules for rule-impact chart).
    device:
        Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
    results_dir:
        All evaluation artifacts are written here.
    use_beam_search, beam_width:
        Decoder configuration.
    generate_explanations:
        Enable explainability pipeline (phoneme attributor, formatter, etc.).
    compute_uncertainty, uncertainty_n_samples:
        Enable MC-Dropout uncertainty estimation.
    lm_scorer:
        Optional BigramLMScorer built from training phoneme sequences.
    lm_weight:
        Bigram LM shallow-fusion weight λ (0.0 = disabled).

    Returns
    -------
    dict
        Full evaluation results as returned by ``evaluate_model()``.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "=== Starting evaluation (beam_search=%s, explain=%s, uncertainty=%s) ===",
        use_beam_search, generate_explanations, compute_uncertainty,
    )

    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        phn_to_id=dataset.phn_to_id,
        id_to_phn=dataset.id_to_phn,
        results_dir=results_dir,
        symbolic_rules=config.symbolic.substitution_rules,
        use_beam_search=use_beam_search,
        beam_width=beam_width,
        generate_explanations=generate_explanations,
        compute_uncertainty=compute_uncertainty,
        uncertainty_n_samples=uncertainty_n_samples,
        lm_scorer=lm_scorer,
        lm_weight=lm_weight,
        ablation_mode=ablation_mode,
    )

    log.info(
        "Evaluation complete: macro-speaker PER=%.4f  [95%% CI: %.4f–%.4f]",
        results["overall"]["per_macro_speaker"],
        results["overall"]["ci"][0],
        results["overall"]["ci"][1],
    )
    log.info("Artifacts written to: %s", results_dir)
    return results


# Top-level orchestrator

def run_auto(args: argparse.Namespace) -> None:
    """
    Full pipeline: load config → inject flags → train → evaluate.

    This function embodies the ownership rules:
    - run_name is the single source of truth for all paths.
    - checkpoint path is derived from run_name, not hardcoded.
    - device is owned here and threaded to evaluate_model.
    """
    # ── 1. Load or create config ────────────────────────────────────────────
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            log.error("Config file not found: %s", config_path)
            sys.exit(1)
        config = Config.load(config_path)
        log.info("Loaded config from %s", config_path)
    else:
        config = get_default_config()
        log.info("Using default config")

    # ── 2. Inject orchestration-level overrides ──────────────────────────────
    if args.run_name:
        config.experiment.run_name = args.run_name

    run_name: str = config.experiment.run_name

    if not run_name:
        log.error(
            "--run-name is required (or set experiment.run_name in your config YAML)."
        )
        sys.exit(1)

    # Forwarded training overrides
    config.training.ablation_mode = args.ablation
    config.training.use_loso = args.loso
    if args.no_gradient_checkpointing:
        config.model.use_gradient_checkpointing = False
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.grad_accum is not None:
        config.training.gradient_accumulation_steps = args.grad_accum
    if args.check_val_every_n_epoch is not None:
        config.training.check_val_every_n_epoch = args.check_val_every_n_epoch

    # Print VRAM banner AFTER all overrides are applied so ablation mode is correct.
    config.print_vram_status()
    if args.max_epochs is not None:
        config.training.max_epochs = args.max_epochs
    if args.early_stopping_patience is not None:
        config.training.early_stopping_patience = args.early_stopping_patience

    # Smoke mode: cap batches and epochs; never surfaced as a public flag
    limit_train_batches: Optional[int] = None
    if args.smoke:
        config.training.max_epochs = 1
        limit_train_batches = 5
        log.info("Smoke mode: max_epochs=1, limit_train_batches=5")

    # Q4: anomaly detection for NaN/Inf gradient debugging
    if getattr(args, "detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)
        log.warning("Anomaly detection ENABLED — training will be significantly slower.")

    # Resolve device (override config.device if explicitly supplied)
    device: str = args.device if args.device else config.device
    config.device = device

    # Derive canonical output paths from run_name
    project_root = get_project_root()
    ckpt_dir = project_root / "checkpoints" / run_name
    results_dir = project_root / "results" / run_name

    log.info("run_name       : %s", run_name)
    log.info("device         : %s", device)
    log.info("checkpoints/   : %s", ckpt_dir)
    log.info("results/       : %s", results_dir)

    # D4: Save config snapshot to results dir for reproducibility
    results_dir.mkdir(parents=True, exist_ok=True)
    config.save(results_dir / "config.yaml")

    # Optional feature-cache warm-up stage (safe speed optimization).
    if args.warm_cache:
        log.info("=== Warming feature cache ===")
        warm_feature_cache(
            config=config,
            workers=args.warm_cache_workers,
            batch_size=args.warm_cache_batch_size,
            enable_disk_cache=not args.warm_cache_disable_disk,
            enable_memory_cache=not args.warm_cache_disable_memory,
        )
        if args.warm_cache_only:
            log.info("--warm-cache-only set; exiting after cache warm-up.")
            return

    # ── 3. Training stage 
    trained_model: Optional[DysarthriaASRLightning] = None
    dataset: Optional[TorgoNeuroSymbolicDataset] = None

    if not args.skip_train:
        loso_force_speakers = None
        if getattr(args, "loso_force_speakers", None):
            loso_force_speakers = [
                s.strip() for s in str(args.loso_force_speakers).split(",") if s.strip()
            ]
        trained_model, dataset = run_training(
            config=config,
            loso=args.loso,
            resume_loso=args.resume_loso,
            loso_force_speakers=loso_force_speakers,
            loso_use_beam_search=args.beam_search,
            loso_beam_width=args.beam_width,
            loso_generate_explanations=args.explain,
            loso_compute_uncertainty=(args.uncertainty or config.experiment.compute_uncertainty),
            loso_uncertainty_n_samples=(
                args.uncertainty_samples if args.uncertainty
                else config.experiment.uncertainty_n_samples
            ),
            limit_train_batches=limit_train_batches,
        )
        # LOSO mode does its own per-fold evaluation; pipeline ends here.
        if args.loso:
            log.info("LOSO pipeline complete.")
            return
    else:
        log.info("--skip-train set; skipping training stage.")

    # ── 4. Evaluation stage 
    if args.skip_eval:
        log.info("--skip-eval set; skipping evaluation stage.")
        return

    # Need dataset for vocab + test_loader regardless of training path
    if dataset is None:
        dataset = _load_dataset(config)

    # Resolve which model to evaluate
    if trained_model is not None:
        # train() already returned the best model — use it directly
        model_to_eval = trained_model.model
    else:
        # --skip-train: load from checkpoint on disk
        ckpt_path = _resolve_checkpoint(run_name)
        lightning_model = _load_lightning_model(ckpt_path, config, dataset)
        model_to_eval = lightning_model.model

    # Build test DataLoader (deterministic split matches training)
    test_loader = _build_test_loader(config, dataset)

    # Build bigram LM from training phoneme sequences (if LM weight requested)
    _lm_scorer: Optional[BigramLMScorer] = None
    if args.beam_search and args.lm_weight > 0.0:
        log.info("Building bigram LM from training phoneme sequences (λ=%.2f)...", args.lm_weight)
        try:
            from src.utils.config import normalize_phoneme
            # Use manifest train split to gather phoneme ID sequences
            import pandas as pd
            train_df = dataset.df[dataset.df['split'] == 'train'] if 'split' in dataset.df.columns else dataset.df
            phn_seqs: list = []
            for phn_str in train_df['phonemes'].dropna():
                ids = [
                    dataset.phn_to_id.get(normalize_phoneme(p), dataset.phn_to_id.get('<UNK>', 2))
                    for p in str(phn_str).split()
                ]
                if ids:
                    phn_seqs.append(ids)
            _lm_scorer = BigramLMScorer(k=0.5)
            _lm_scorer.fit(phn_seqs, vocab_size=len(dataset.phn_to_id))
            log.info("Bigram LM built from %d training sequences.", len(phn_seqs))
        except Exception as _lm_exc:
            log.warning("Bigram LM build failed (non-fatal, falling back to acoustic-only): %s", _lm_exc)
            _lm_scorer = None

    run_evaluation(
        model=model_to_eval,
        test_loader=test_loader,
        dataset=dataset,
        config=config,
        device=device,
        results_dir=results_dir,
        use_beam_search=args.beam_search,
        beam_width=args.beam_width,
        generate_explanations=args.explain,
        ablation_mode=config.training.ablation_mode,
        # I4: CLI flag OR config.experiment flag enables uncertainty; CLI takes priority for n_samples
        compute_uncertainty=args.uncertainty or config.experiment.compute_uncertainty,
        uncertainty_n_samples=(
            args.uncertainty_samples if args.uncertainty
            else config.experiment.uncertainty_n_samples
        ),
        lm_scorer=_lm_scorer,
        lm_weight=args.lm_weight,
    )


# CLI

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="DysarthriaNSR end-to-end training and evaluation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Orchestration-level flags (owned by run_pipeline.py) 
    orch = parser.add_argument_group("Orchestration")
    orch.add_argument(
        "--run-name",
        type=str,
        default=None,
        metavar="STR",
        help=(
            "Run identifier. Drives all output paths: "
            "checkpoints/{run_name}/ and results/{run_name}/."
        ),
    )
    orch.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a YAML config file produced by Config.save(). "
             "All fields not present in the file retain dataclass defaults.",
    )
    orch.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="STR",
        help="Torch device override, e.g. 'cuda' or 'cpu'. "
             "Defaults to config.device (auto-detected).",
    )
    orch.add_argument(
        "--skip-train",
        action="store_true",
        default=False,
        help="Skip the training stage. A checkpoint must already exist under "
             "checkpoints/{run_name}/.",
    )
    orch.add_argument(
        "--skip-eval",
        action="store_true",
        default=False,
        help="Skip the evaluation stage. Train only.",
    )
    orch.add_argument(
        "--detect-anomaly",
        action="store_true",
        default=False,
        dest="detect_anomaly",
        help="Enable torch.autograd.set_detect_anomaly(True) for NaN/Inf gradient "
             "debugging. Significantly slower; use only during debugging runs.",
    )
    orch.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Fast smoke-test mode: forces max_epochs=1 and caps batches to 5 "
             "per epoch. Intended for CI / pre-commit sanity checks.",
    )
    orch.add_argument(
        "--warm-cache",
        action="store_true",
        default=False,
        help="Warm data/processed/feature_cache before training/evaluation.",
    )
    orch.add_argument(
        "--warm-cache-only",
        action="store_true",
        default=False,
        help="Run cache warm-up stage and exit without train/eval.",
    )
    orch.add_argument(
        "--warm-cache-workers",
        type=int,
        default=None,
        metavar="INT",
        help="Override DataLoader worker count for cache warm-up.",
    )
    orch.add_argument(
        "--warm-cache-batch-size",
        type=int,
        default=1,
        metavar="INT",
        help="Warm-up batch size (recommended: 1).",
    )
    orch.add_argument(
        "--warm-cache-disable-disk",
        action="store_true",
        default=False,
        help="Disable writing cache files to disk during warm-up.",
    )
    orch.add_argument(
        "--warm-cache-disable-memory",
        action="store_true",
        default=False,
        help="Disable in-process memory cache during warm-up.",
    )

    # ── Forwarded verbatim to train() / run_loso() ───────────────────────────
    train_grp = parser.add_argument_group(
        "Training (forwarded to train.py)"
    )
    train_grp.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "neural_only", "symbolic_only", "no_art_heads",
                 "no_constraint_matrix", "no_spec_augment", "no_temporal_ds"],
        metavar="STR",
        help="Ablation mode.",
    )
    train_grp.add_argument(
        "--loso",
        action="store_true",
        default=False,
        help="Run Leave-One-Speaker-Out cross-validation. "
             "Evaluation is performed per-fold inside run_loso(); "
             "the post-training evaluation stage is skipped.",
    )
    train_grp.add_argument(
        "--resume-loso",
        action="store_true",
        default=False,
        help="Resume LOSO from results/{run_name}_loso_progress.json and "
             "checkpoints/{run_name}_loso_<speaker>/last.ckpt when present.",
    )
    train_grp.add_argument(
        "--loso-force-speakers",
        type=str,
        default=None,
        metavar="CSV",
        help="Force re-run specific LOSO held-out speakers (comma-separated). "
             "Their old fold checkpoints/results are cleared first. "
             "Example: MC02,MC03",
    )
    train_grp.add_argument(
        "--max_epochs",
        "--max-epochs",
        type=int,
        default=None,
        metavar="INT",
        help="Override config.training.max_epochs.",
    )
    train_grp.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        metavar="INT",
        help="Override config.training.early_stopping_patience.",
    )
    train_grp.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        default=False,
        help="Disable HuBERT gradient checkpointing (faster, uses more VRAM).",
    )
    train_grp.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="INT",
        help="Override config.training.batch_size.",
    )
    train_grp.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        metavar="INT",
        help="Override config.training.gradient_accumulation_steps.",
    )
    train_grp.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=None,
        metavar="INT",
        help="Override config.training.check_val_every_n_epoch (e.g., 2 for faster LOSO).",
    )

    # ── Forwarded as kwargs to evaluate_model() ──────────────────────────────
    eval_grp = parser.add_argument_group(
        "Evaluation (forwarded to evaluate_model)"
    )
    eval_grp.add_argument(
        "--beam-search",
        action="store_true",
        default=False,
        help="Use BeamSearchDecoder instead of greedy decoding.",
    )
    eval_grp.add_argument(
        "--beam-width",
        type=int,
        default=10,
        metavar="INT",
        help="Beam width for beam search (ignored unless --beam-search is set).",
    )
    eval_grp.add_argument(
        "--explain",
        action="store_true",
        default=False,
        help="Run the explainability pipeline: phoneme attributor, articulatory "
             "confusion analysis, and ExplainableOutputFormatter. "
             "Writes explanations.json and articulatory_confusion.png to results_dir.",
    )
    eval_grp.add_argument(
        "--lm-weight",
        type=float,
        default=0.0,
        metavar="FLOAT",
        dest="lm_weight",
        help="Bigram LM shallow-fusion weight λ for beam search. "
             "0.0 = disabled (acoustic-only). Typical range: 0.1–0.5. "
             "LM is built from training phoneme sequences automatically.",
    )
    eval_grp.add_argument(
        "--uncertainty",
        action="store_true",
        default=False,
        help="Enable MC-Dropout uncertainty estimation via UncertaintyAwareDecoder.",
    )
    eval_grp.add_argument(
        "--uncertainty-samples",
        type=int,
        default=20,
        metavar="INT",
        dest="uncertainty_samples",
        help="Number of MC-Dropout forward passes for uncertainty estimation.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        run_auto(args)
    except FileNotFoundError as exc:
        log.error("File not found: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("Pipeline interrupted by user.")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        log.exception("Pipeline failed with unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
