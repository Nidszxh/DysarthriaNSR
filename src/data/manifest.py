import argparse
import os
import warnings
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from functools import lru_cache

import librosa
import numpy as np
import pandas as pd
from datasets import Audio, load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---

@dataclass(frozen=True)
class DataPaths:
    """Path configuration matching your finalized download.py structure."""
    root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = None
    raw_dir: Path = None
    processed_dir: Path = None
    
    def __post_init__(self):
        # Set derived paths to match the finalized project structure
        object.__setattr__(self, "data_dir", self.root / "data")
        object.__setattr__(self, "raw_dir", self.data_dir / "raw")
        object.__setattr__(self, "processed_dir", self.data_dir / "processed")

# Phoneme-to-Articulatory-Class Mapping (American English ARPABET)
PHONEME_ARTICULATION = {
    # Stops (Plosives)
    'P': 'stop', 'B': 'stop', 'T': 'stop', 'D': 'stop', 'K': 'stop', 'G': 'stop',
    # Fricatives
    'F': 'fricative', 'V': 'fricative', 'TH': 'fricative', 'DH': 'fricative',
    'S': 'fricative', 'Z': 'fricative', 'SH': 'fricative', 'ZH': 'fricative',
    'HH': 'fricative',
    # Affricates
    'CH': 'affricate', 'JH': 'affricate',
    # Nasals
    'M': 'nasal', 'N': 'nasal', 'NG': 'nasal',
    # Liquids
    'L': 'liquid', 'R': 'liquid',
    # Glides
    'W': 'glide', 'Y': 'glide',
    # Vowels (Monophthongs)
    'IY': 'vowel', 'IH': 'vowel', 'EH': 'vowel', 'EY': 'vowel', 'AE': 'vowel',
    'AA': 'vowel', 'AO': 'vowel', 'OW': 'vowel', 'UH': 'vowel', 'UW': 'vowel',
    'AH': 'vowel', 'ER': 'vowel', 'AX': 'vowel', 'IX': 'vowel', 'AXR': 'vowel',
    # Diphthongs
    'AY': 'diphthong', 'AW': 'diphthong', 'OY': 'diphthong',
}

# --- HELPERS ---

class SymbolicProcessor:
    """Handles G2P and phoneme-to-articulatory mapping."""
    def __init__(self):
        try:
            import g2p_en, nltk
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            self.g2p = g2p_en.G2p()
        except Exception:
            self.g2p = None
            print("⚠️ g2p_en not found. Phoneme extraction will be skipped.")

    @lru_cache(maxsize=2048)
    def get_features(self, text: str) -> Tuple[str, str, int]:
        """Convert transcript to phonemes and classes with caching for speed."""
        if not self.g2p or not text:
            return "", "", 0
        
        # Clean: keep alphanumeric phonemes, strip stress markers (0,1,2)
        raw_phonemes = self.g2p(text)
        phonemes = [p.rstrip('012') for p in raw_phonemes if p.strip() and p.isalnum()]
        
        # Map to articulatory classes
        classes = [PHONEME_ARTICULATION.get(p, 'other') for p in phonemes]
        
        return " ".join(phonemes), " ".join(classes), len(phonemes)

def calculate_audio_metrics(audio_path: Path) -> Dict:
    """Extracts RMS energy and duration. Designed for parallel execution."""
    try:
        # Load with native sampling rate
        y, sr = librosa.load(str(audio_path), sr=None)
        if len(y) == 0:
            return {"rms": 0.0, "duration": 0.0}
        
        rms = np.mean(librosa.feature.rms(y=y))
        duration = len(y) / sr
        return {
            "rms": round(float(rms), 6), 
            "duration": round(float(duration), 3)
        }
    except Exception:
        return {"rms": 0.0, "duration": 0.0}

# --- MAIN PIPELINE ---

def main():
    paths = DataPaths()
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    processor = SymbolicProcessor()
    
    # 1. Load Metadata with DECODING DISABLED
    print(f"Loading Metadata (Cache: {paths.data_dir})")
    # By using decode=False here, 'audio' stays a dict with a 'path' key
    ds = load_dataset("abnerh/TORGO-database", cache_dir=str(paths.data_dir))
    ds = ds.cast_column("audio", Audio(decode=False))
    
    # 2. Index Local Files
    print(f"Indexing {paths.raw_dir}/audio...")
    local_files = {f.name: f for f in paths.raw_dir.rglob("*.wav")}
    print(f"Found {len(local_files)} local files.")
    
    # 3. Match Metadata to Local Files
    matched_samples = []
    for split in ds.keys():
        for sample in tqdm(ds[split], desc=f"Matching {split}"):
            # Because decode=False, 'audio' is a dict containing 'path'
            audio_info = sample.get("audio", {})
            hf_raw_path = audio_info.get("path", "")
            
            if hf_raw_path:
                fname = Path(hf_raw_path).name
                if fname in local_files:
                    matched_samples.append({
                        "metadata": sample,
                        "local_path": local_files[fname]
                    })

    if not matched_samples:
        print("\nZero matches. Debugging path strings...")
        test_sample = ds['train'][0]['audio']
        print(f"Metadata Audio Dict: {test_sample}")
        print(f"Sample Filename: {Path(test_sample.get('path', '')).name}")
        return

    print(f"Matched {len(matched_samples)} local files.")

    # 4. Parallel Audio Metrics Extraction
    print(f"Analyzing audio metrics...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        local_paths = [s["local_path"] for s in matched_samples]
        metrics_results = list(tqdm(
            executor.map(calculate_audio_metrics, local_paths), 
            total=len(local_paths),
            desc="Audio Analysis"
        ))

    # 5. Assemble Final Manifest
    print("Generating symbolic phoneme sequences...")
    rows = []
    for bundle, metrics in zip(matched_samples, metrics_results):
        if metrics["duration"] < 0.1:
            continue

        meta = bundle["metadata"]
        path = bundle["local_path"]
        transcript = meta.get("transcription", "").strip().upper()
        
        phn, art, count = processor.get_features(transcript)
        
        rows.append({
            "sample_id": path.name,
            "path": str(path),
            "speaker": path.name.split('_')[0],
            "status": meta["speech_status"],
            "label": 1 if meta["speech_status"] == "dysarthria" else 0,
            "transcript": transcript,
            "phonemes": phn,
            "articulatory_classes": art,
            "phn_count": count,
            "duration": metrics["duration"],
            "rms_energy": metrics["rms"],
            "gender": meta.get("gender", "unknown")
        })

    # 6. Save and Report
    df = pd.DataFrame(rows)
    out_path = paths.processed_dir / "torgo_neuro_symbolic_manifest.csv"
    df.to_csv(out_path, index=False)
    
    print(f"\nMANIFEST GENERATION COMPLETE")
    print(f"Output: {out_path}")
    print(f"Total Samples: {len(df)}")
    print(f"Total Hours:   {df['duration'].sum() / 3600:.2f} hrs")



if __name__ == "__main__":
    main()