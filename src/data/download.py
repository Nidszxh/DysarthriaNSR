import shutil
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

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
    def __init__(self, config: DatasetConfig):
        self.config = config

    def download_and_load(self, repo_id: str = "abnerh/TORGO-database"):
        """Fetch the TORGO dataset from HuggingFace."""
        logger.info(f"Loading dataset metadata from {repo_id}...")
        dataset = load_dataset(repo_id, cache_dir=self.config.data_dir)
        
        # Cast audio to access paths without loading waveforms into memory
        return dataset.cast_column("audio", Audio(decode=False))

    def extract_audio(self, dataset) -> Path:
        """Extracts audio files by writing binary data to raw directory."""
        audio_root = self.config.raw_dir / "audio"
        
        # 1. First, get the paths WITHOUT decoding to avoid the decoder object issue
        # We create a map of indices to original paths
        dataset_no_decode = dataset.cast_column("audio", Audio(decode=False))
        
        # 2. Now cast to decode for the actual signal data
        dataset_decode = dataset.cast_column("audio", Audio(decode=True))
        
        for split in dataset.keys():
            split_dir = audio_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Access both decoded and non-decoded data
            data_dec = dataset_decode[split]
            data_raw = dataset_no_decode[split]
            
            logger.info(f"Saving {split} audio to disk...")
            
            for i in tqdm(range(len(data_dec)), desc=f"Writing {split}", leave=False):
                sample_dec = data_dec[i]
                sample_raw = data_raw[i]
                
                # Extract audio signal and rate
                audio_struct = sample_dec["audio"]
                audio_data = audio_struct["array"]
                sr = audio_struct["sampling_rate"]
                
                # Get original path from the non-decoded sample
                # In decode=False mode, "audio" is a dict containing "path"
                original_path = sample_raw["audio"].get("path")
                
                if original_path:
                    filename = Path(original_path).name
                else:
                    speaker = sample_dec.get('speaker_id', 'unknown')
                    word = sample_dec.get('word_id', 'sample')
                    filename = f"{speaker}_{word}_{i}.wav"
                
                dest_path = split_dir / filename
                
                if not dest_path.exists():
                    sf.write(dest_path, audio_data, sr)
        
        logger.info(f"Audio files saved to: {audio_root}")
        return audio_root

    def organize_arrows(self):
        """Moves arrow files to raw directory for local dataset loading."""
        # Note: Globbing for the version folder to avoid hardcoding "0.0.0"
        hf_cache = self.config.data_dir / "abnerh___torgo-database"
        arrow_dest = self.config.raw_dir / "arrow_data"
        
        if not hf_cache.exists():
            logger.warning("HF cache not found. Please run download_and_load first.")
            return

        arrow_dest.mkdir(parents=True, exist_ok=True)
        arrow_files = list(hf_cache.rglob("*.arrow"))
        
        logger.info(f"Organizing {len(arrow_files)} arrow files...")
        for arrow_file in arrow_files:
            dest = arrow_dest / arrow_file.name
            if not dest.exists():
                shutil.copy2(arrow_file, dest)

def main():
    config = DatasetConfig()
    setup_environment(config)
    
    manager = TorgoManager(config)
    
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