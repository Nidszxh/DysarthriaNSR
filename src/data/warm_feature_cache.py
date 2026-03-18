"""
warm_feature_cache.py — Pre-materialize HuBERT input feature cache from manifest.

Run once before training to avoid repeated preprocessing overhead on each epoch.

# REFACTOR LOG
# [CLEAN] Replaced print() with logging.getLogger(__name__) calls so output
#         respects the caller's logging configuration and is silent under
#         logging.disable(logging.INFO) (e.g., during pytest runs).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from src.data.dataloader import TorgoNeuroSymbolicDataset
from src.utils.config import Config, get_default_config


def warm_feature_cache(
    config: Config,
    manifest_path: str | None = None,
    workers: int | None = None,
    batch_size: int = 1,
    enable_disk_cache: bool = True,
    enable_memory_cache: bool = True,
) -> Path | None:
    """
    Materialize cached input feature tensors for all manifest samples.

    Args:
        config: Active project configuration.
        manifest_path: Optional manifest override.
        workers: Optional DataLoader worker override.
        batch_size: Warm-up DataLoader batch size (default 1).
        enable_disk_cache: Persist cache files under data/processed/feature_cache.
        enable_memory_cache: Keep per-process LRU cache enabled while warming.

    Returns:
        Path to the cache namespace directory when disk cache is enabled,
        otherwise None.
    """
    manifest = Path(manifest_path) if manifest_path else config.data.manifest_path
    loader_workers = workers if workers is not None else config.training.num_workers

    dataset = TorgoNeuroSymbolicDataset(
        manifest_path=str(manifest),
        processor_id=config.model.hubert_model_id,
        sampling_rate=config.data.sampling_rate,
        max_audio_length=config.data.max_audio_length,
        enable_feature_cache=enable_disk_cache,
        enable_memory_cache=enable_memory_cache,
    )

    loader_kwargs = dict(
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(loader_workers)),
        collate_fn=lambda x: x,
    )
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = config.training.prefetch_factor

    loader = DataLoader(dataset, **loader_kwargs)

    start = time.time()
    n_samples = len(dataset)

    for batch_idx, batch in enumerate(loader):
        _ = batch
        if (batch_idx + 1) % 500 == 0:
            elapsed = time.time() - start
            logger.info("Warmed %d/%d samples in %.1fs", batch_idx + 1, n_samples, elapsed)

    elapsed = time.time() - start
    logger.info("Done. Warmed %d samples in %.1fs", n_samples, elapsed)

    if dataset.enable_feature_cache:
        cache_dir = dataset.feature_cache_dir / dataset._cache_namespace
        logger.info("Feature cache dir: %s", cache_dir.resolve())
        return cache_dir

    return None


def main() -> None:
    """CLI entry point for feature cache warm-up."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Warm HuBERT input feature cache from manifest")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers for warm-up")
    parser.add_argument("--batch-size", type=int, default=1, help="Warm-up batch size")
    parser.add_argument("--disable-disk-cache", action="store_true", help="Disable on-disk cache writes")
    parser.add_argument("--disable-memory-cache", action="store_true", help="Disable in-process memory cache")
    args = parser.parse_args()

    cfg = get_default_config()
    warm_feature_cache(
        config=cfg,
        manifest_path=args.manifest,
        workers=args.workers,
        batch_size=args.batch_size,
        enable_disk_cache=not args.disable_disk_cache,
        enable_memory_cache=not args.disable_memory_cache,
    )


if __name__ == "__main__":
    main()