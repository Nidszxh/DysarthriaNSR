from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import evaluate_model
from src.data.dataloader import TorgoNeuroSymbolicDataset
from src.models.model import NeuroSymbolicASR
from src.utils.config import Config, get_project_root
from train import DysarthriaASRLightning, create_loso_splits


@dataclass
class FoldSpec:
    speaker: str
    run_name: str
    checkpoint_path: Path
    results_dir: Path


def _split_speaker_id(spk: str) -> Tuple[str, int]:
    match = re.match(r"([A-Za-z]+)(\d+)$", str(spk))
    if not match:
        return str(spk), 10**9
    return match.group(1), int(match.group(2))


def _speaker_sort_key(speaker: str) -> Tuple[str, int, str]:
    prefix, number = _split_speaker_id(speaker)
    return prefix, number, str(speaker)


def _resolve_best_fold_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None

    scored = list(ckpt_dir.glob("epoch=*-val_per=*.ckpt"))
    if scored:
        pattern = re.compile(r"val_per=([0-9]+(?:\.[0-9]+)?)")
        best_path: Optional[Path] = None
        best_score = float("inf")

        for checkpoint_path in scored:
            match = pattern.search(checkpoint_path.name)
            if not match:
                continue
            score = float(match.group(1))
            if score < best_score:
                best_score = score
                best_path = checkpoint_path

        if best_path is not None:
            return best_path

    last_checkpoint = ckpt_dir / "last.ckpt"
    return last_checkpoint if last_checkpoint.exists() else None


def _load_config(base_run_name: str, config_path: Optional[Path]) -> Config:
    if config_path is not None and config_path.exists():
        config = Config.load(config_path)
        print(f"✅ Loaded config: {config_path}")
    else:
        config = Config()
        if config_path is not None:
            print(f"⚠️ Config not found ({config_path}); using defaults from Config().")
        else:
            print("ℹ️ No --config provided; using defaults from Config().")

    config.experiment.run_name = base_run_name
    return config


def _build_dataset(config: Config) -> TorgoNeuroSymbolicDataset:
    return TorgoNeuroSymbolicDataset(
        manifest_path=str(config.data.manifest_path),
        sampling_rate=config.data.sampling_rate,
        max_audio_length=config.data.max_audio_length,
    )


def _discover_folds(project_root: Path, base_run_name: str) -> Dict[str, FoldSpec]:
    fold_specs: Dict[str, FoldSpec] = {}

    progress_path = project_root / "results" / f"{base_run_name}_loso_progress.json"
    if progress_path.exists():
        try:
            progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
            for fold in progress_payload.get("folds", []):
                speaker = str(fold.get("speaker", "")).strip()
                checkpoint_str = fold.get("checkpoint")
                if not speaker or not checkpoint_str:
                    continue

                fold_run_name = f"{base_run_name}_loso_{speaker}"
                results_dir_str = fold.get("results_dir")
                results_dir = (
                    Path(results_dir_str)
                    if results_dir_str
                    else project_root / "results" / fold_run_name
                )

                checkpoint_path = Path(checkpoint_str)
                if checkpoint_path.exists():
                    fold_specs[speaker] = FoldSpec(
                        speaker=speaker,
                        run_name=fold_run_name,
                        checkpoint_path=checkpoint_path,
                        results_dir=results_dir,
                    )
        except Exception as exc:
            print(f"⚠️ Failed to parse progress file {progress_path}: {exc}")

    if fold_specs:
        return fold_specs

    print("ℹ️ Progress file missing/incomplete; falling back to checkpoint discovery.")
    checkpoints_root = project_root / "checkpoints"
    for checkpoint_dir in checkpoints_root.glob(f"{base_run_name}_loso_*"):
        if not checkpoint_dir.is_dir():
            continue

        fold_run_name = checkpoint_dir.name
        speaker = fold_run_name.replace(f"{base_run_name}_loso_", "", 1)
        if not speaker:
            continue

        checkpoint_path = _resolve_best_fold_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            continue

        results_dir = project_root / "results" / fold_run_name
        fold_specs[speaker] = FoldSpec(
            speaker=speaker,
            run_name=fold_run_name,
            checkpoint_path=checkpoint_path,
            results_dir=results_dir,
        )

    return fold_specs


def _archive_existing_outputs(results_dir: Path, overwrite_archive: bool) -> None:
    archive_dir = results_dir / "eval_greedy"
    archive_dir.mkdir(parents=True, exist_ok=True)

    archive_candidates = [
        "evaluation_results.json",
        "explanations.json",
    ]

    for filename in archive_candidates:
        source_path = results_dir / filename
        target_path = archive_dir / filename

        if not source_path.exists():
            continue

        if target_path.exists() and not overwrite_archive:
            continue

        shutil.copy2(source_path, target_path)


def _evaluate_fold(
    config: Config,
    dataset: TorgoNeuroSymbolicDataset,
    fold_spec: FoldSpec,
    device: str,
    beam_width: int,
    uncertainty_samples: int,
) -> Dict:
    _train_loader, _val_loader, test_loader = create_loso_splits(
        config=config,
        dataset=dataset,
        held_out_speaker=fold_spec.speaker,
    )

    model = NeuroSymbolicASR(
        model_config=config.model,
        symbolic_config=config.symbolic,
        phn_to_id=dataset.phn_to_id,
        id_to_phn=dataset.id_to_phn,
        manner_to_id=dataset.manner_to_id,
        place_to_id=dataset.place_to_id,
        voice_to_id=dataset.voice_to_id,
    )

    lightning_model = DysarthriaASRLightning.load_from_checkpoint(
        str(fold_spec.checkpoint_path),
        model=model,
        config=config,
        phn_to_id=dataset.phn_to_id,
        id_to_phn=dataset.id_to_phn,
        class_weights=dataset.get_loss_weights(),
        articulatory_weights=dataset.get_articulatory_loss_weights(),
        strict=False,
    )

    fold_spec.results_dir.mkdir(parents=True, exist_ok=True)

    return evaluate_model(
        model=lightning_model.model,
        dataloader=test_loader,
        device=device,
        phn_to_id=dataset.phn_to_id,
        id_to_phn=dataset.id_to_phn,
        results_dir=fold_spec.results_dir,
        use_beam_search=True,
        beam_width=beam_width,
        generate_explanations=True,
        compute_uncertainty=True,
        uncertainty_n_samples=uncertainty_samples,
    )


def _parse_requested_speakers(raw_speakers: Optional[str]) -> Optional[set[str]]:
    if raw_speakers is None:
        return None

    parsed = {token.strip() for token in raw_speakers.split(",") if token.strip()}
    return parsed if parsed else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive greedy LOSO outputs and re-evaluate folds with beam+uncertainty+explain."
    )
    parser.add_argument("--base-run-name", default="loso_v1", help="Base LOSO run name (default: loso_v1)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config path. If omitted, tries results/<base-run-name>/config.yaml then defaults.",
    )
    parser.add_argument(
        "--fold-speakers",
        type=str,
        default=None,
        help="Comma-separated speaker IDs for fold-wise evaluation (e.g., F01,M02). Omit to run all folds.",
    )
    parser.add_argument("--beam-width", type=int, default=25, help="Beam width for beam search (default: 25)")
    parser.add_argument(
        "--uncertainty-samples",
        type=int,
        default=20,
        help="MC-dropout sample count for uncertainty (default: 20)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g., cuda or cpu). Defaults to config.device.",
    )
    parser.add_argument(
        "--no-archive-greedy",
        action="store_true",
        help="Skip archiving existing fold outputs into eval_greedy/ before re-evaluation.",
    )
    parser.add_argument(
        "--overwrite-archive",
        action="store_true",
        help="Overwrite existing files inside eval_greedy/ when archiving.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned folds/checkpoints without running evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = get_project_root()

    default_config_path = project_root / "results" / args.base_run_name / "config.yaml"
    config_path = Path(args.config) if args.config is not None else default_config_path

    config = _load_config(args.base_run_name, config_path)
    pl.seed_everything(config.experiment.seed, workers=True)

    requested_speakers = _parse_requested_speakers(args.fold_speakers)
    fold_map = _discover_folds(project_root=project_root, base_run_name=args.base_run_name)

    if not fold_map:
        raise RuntimeError(
            f"No LOSO folds discovered for base run '{args.base_run_name}'. "
            f"Expected progress file or checkpoints under {project_root / 'checkpoints'}."
        )

    speakers = sorted(fold_map.keys(), key=_speaker_sort_key)
    if requested_speakers is not None:
        unknown = sorted([speaker for speaker in requested_speakers if speaker not in fold_map], key=_speaker_sort_key)
        if unknown:
            print(f"⚠️ Unknown speakers ignored: {unknown}")
        speakers = [speaker for speaker in speakers if speaker in requested_speakers]

    if not speakers:
        raise RuntimeError("No valid folds selected after filtering --fold-speakers.")

    print(f"\n🔁 Selected folds ({len(speakers)}): {speakers}")
    for speaker in speakers:
        spec = fold_map[speaker]
        print(f"   - {speaker}: ckpt={spec.checkpoint_path} | results={spec.results_dir}")

    if args.dry_run:
        print("\n✅ Dry-run complete. No evaluations were executed.")
        return

    dataset = _build_dataset(config)
    device = args.device if args.device is not None else config.device

    for index, speaker in enumerate(speakers, start=1):
        fold_spec = fold_map[speaker]
        print(f"\n── Re-eval fold {index}/{len(speakers)}: {speaker} ──")

        if not args.no_archive_greedy:
            _archive_existing_outputs(
                results_dir=fold_spec.results_dir,
                overwrite_archive=args.overwrite_archive,
            )
            print(f"   📦 Archived existing outputs to: {fold_spec.results_dir / 'eval_greedy'}")

        metrics = _evaluate_fold(
            config=config,
            dataset=dataset,
            fold_spec=fold_spec,
            device=device,
            beam_width=args.beam_width,
            uncertainty_samples=args.uncertainty_samples,
        )

        avg_per = float(metrics.get("avg_per", float("nan")))
        wer = float(metrics.get("wer", float("nan")))
        print(f"   ✅ Done {speaker}: avg_per={avg_per:.4f}, wer={wer:.4f}")

    print("\n🎉 LOSO re-evaluation complete (beam + uncertainty + explain).")


if __name__ == "__main__":
    main()
