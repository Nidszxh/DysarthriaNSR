import logging
import os
import csv
import shutil
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional

import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DatasetConfig:
    """Read-only configuration for dataset paths."""
    root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    external_dir: Path = field(init=False)

    def __post_init__(self):
        # Set derived paths
        object.__setattr__(self, "data_dir", self.root / "data")
        object.__setattr__(self, "raw_dir", self.data_dir / "raw")
        object.__setattr__(self, "processed_dir", self.data_dir / "processed")
        object.__setattr__(self, "external_dir", self.data_dir / "external")

def setup_environment(config: DatasetConfig):
    """Initializes directories and environment variables."""
    os.environ["HF_HOME"] = str(config.data_dir)
    
    for path in [config.raw_dir, config.processed_dir, config.external_dir]:
        path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Environment initialized at {config.root}")

class TorgoManager:
    def __init__(self, config: DatasetConfig, target_sr: int = 16000):
        self.config = config
        self.target_sr = target_sr  # Standardize to 16kHz for consistency

    def download_and_load(self, repo_id: str = "abnerh/TORGO-database"):
        logger.info(f"Loading dataset from {repo_id}...")
        # Load and immediately save to raw_dir to avoid cache management headaches
        dataset = load_dataset(repo_id, cache_dir=self.config.data_dir)
        
        arrow_path = self.config.raw_dir / "torgo_hf"
        if not arrow_path.exists():
            dataset.save_to_disk(arrow_path)
            logger.info(f"Dataset saved to disk at {arrow_path}")
        
        return dataset

    def organize_arrows(self) -> None:
        """Collect cached arrow files into data/raw/arrow_data for easier inspection."""
        arrow_root = self.config.raw_dir / "arrow_data"
        arrow_root.mkdir(parents=True, exist_ok=True)

        cache_root = self.config.data_dir / "abnerh___torgo-database" / "default"
        if not cache_root.exists():
            logger.warning("No HuggingFace cache found at %s", cache_root)
            return

        arrow_files = list(cache_root.rglob("*.arrow"))
        if not arrow_files:
            logger.warning("No .arrow files found under %s", cache_root)
            return

        moved = 0
        for src in arrow_files:
            dest = arrow_root / src.name
            if dest.exists():
                continue
            try:
                shutil.copy2(src, dest)
                moved += 1
            except Exception as exc:
                logger.warning("Failed to copy %s: %s", src, exc)

        logger.info("Arrow files organized: %d copied to %s", moved, arrow_root)

    def _save_single_sample(self, args):
        """Memory-safe helper for parallel execution."""
        sample_dec, sample_raw, split_dir = args
        
        audio_struct = sample_dec["audio"]
        audio_data = audio_struct["array"]
        original_sr = audio_struct["sampling_rate"]
        
        # 1. Collision-resistant naming using path hashes
        original_path_str = sample_raw["audio"].get("path", "unknown")
        path_hash = hashlib.md5(original_path_str.encode()).hexdigest()[:8]
        speaker = sample_dec.get('speaker_id', 'unknown')
        
        filename = f"{speaker}_{path_hash}_{Path(original_path_str).name}"
        dest_path = split_dir / filename
        
        # 2. Resampling & Writing
        if not dest_path.exists():
            # For now, we standardize the output rate:
            sf.write(dest_path, audio_data, self.target_sr)
        
        return {
            "speaker": speaker,
            "filename": filename,
            "path": str(dest_path),
            "original_sr": original_sr,
            "target_sr": self.target_sr,
            "transcript": sample_dec.get("transcription", "")
        }

    def _get_task_generator(self, ds_dec, ds_no_dec, split_dir):
        """Prevents OOM by yielding items one by one instead of loading all to RAM."""
        for i in range(len(ds_dec)):
            yield (ds_dec[i], ds_no_dec[i], split_dir)

    def extract_audio(self, dataset) -> Path:
        audio_root = self.config.raw_dir / "audio"
        metadata_records = []
        
        # Ensure audio is cast to the target sampling rate during decoding
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.target_sr))
        ds_no_dec = dataset.cast_column("audio", Audio(decode=False))
        
        for split in dataset.keys():
            split_dir = audio_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Extracting {split} audio at {self.target_sr}Hz...")
            
            # map() accepts a generator, keeping memory usage constant
            tasks = self._get_task_generator(dataset[split], ds_no_dec[split], split_dir)
            
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(tqdm(
                    executor.map(self._save_single_sample, tasks), 
                    total=len(dataset[split]), 
                    desc=f"Processing {split}"
                ))
                metadata_records.extend(results)
        
        self._write_raw_manifest(metadata_records)
        return audio_root

    def _write_raw_manifest(self, records):
        manifest_path = self.config.processed_dir / "raw_extraction_map.csv"
        if records:
            keys = records[0].keys()
            with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(records)
            logger.info(f"Raw extraction map saved to {manifest_path}")

def main():
    config = DatasetConfig()
    setup_environment(config)
    
    manager = TorgoManager(config, target_sr=16000)
    
    try:
        # 1. Load dataset
        dataset = manager.download_and_load()
        logger.info(f"Splits found: {list(dataset.keys())}")
        
        # 2. Inspection
        sample = dataset["train"][0]
        text_snippet = sample['transcription'][:50]
        logger.info(f"Sample Transcription: {text_snippet}...")
        
        # 3. File Management
        manager.organize_arrows()
        audio_path = manager.extract_audio(dataset)
        
        print(f"SETUP COMPLETE")
        print(f"  Raw Audio: {audio_path}")
        print(f"  Project Root: {config.root}")

    except Exception as e:
        logger.error(f"Failed to setup dataset: {e}")
        raise

# Main Guard
if __name__ == "__main__":
    main()