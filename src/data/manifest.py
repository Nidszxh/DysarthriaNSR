import hashlib
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
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
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    
    def __post_init__(self):
        # Set derived paths to match the finalized project structure
        object.__setattr__(self, "data_dir", self.root / "data")
        object.__setattr__(self, "raw_dir", self.data_dir / "raw")
        object.__setattr__(self, "processed_dir", self.data_dir / "processed")

# Phoneme-to-Articulatory details (American English ARPABET)
# Format: (Manner, Place, Voicing)
PHONEME_DETAILS = {
    # Stops (Plosives)
    'P': ('stop', 'bilabial', 'voiceless'),
    'B': ('stop', 'bilabial', 'voiced'),
    'T': ('stop', 'alveolar', 'voiceless'),
    'D': ('stop', 'alveolar', 'voiced'),
    'K': ('stop', 'velar', 'voiceless'),
    'G': ('stop', 'velar', 'voiced'),
    # Fricatives
    'F': ('fricative', 'labiodental', 'voiceless'),
    'V': ('fricative', 'labiodental', 'voiced'),
    'TH': ('fricative', 'dental', 'voiceless'),
    'DH': ('fricative', 'dental', 'voiced'),
    'S': ('fricative', 'alveolar', 'voiceless'),
    'Z': ('fricative', 'alveolar', 'voiced'),
    'SH': ('fricative', 'palatal', 'voiceless'),
    'ZH': ('fricative', 'palatal', 'voiced'),
    'HH': ('fricative', 'glottal', 'voiceless'),
    # Affricates
    'CH': ('affricate', 'palatal', 'voiceless'),
    'JH': ('affricate', 'palatal', 'voiced'),
    # Nasals
    'M': ('nasal', 'bilabial', 'voiced'),
    'N': ('nasal', 'alveolar', 'voiced'),
    'NG': ('nasal', 'velar', 'voiced'),
    # Liquids
    'L': ('liquid', 'alveolar', 'voiced'),
    'R': ('liquid', 'palatal', 'voiced'),
    # Glides
    'W': ('glide', 'bilabial', 'voiced'),
    'Y': ('glide', 'palatal', 'voiced'),
    # Vowels (Monophthongs)
    'IY': ('vowel', 'front', 'voiced'),
    'IH': ('vowel', 'front', 'voiced'),
    'EH': ('vowel', 'front', 'voiced'),
    'EY': ('vowel', 'front', 'voiced'),
    'AE': ('vowel', 'front', 'voiced'),
    'AA': ('vowel', 'back', 'voiced'),
    'AO': ('vowel', 'back', 'voiced'),
    'OW': ('vowel', 'back', 'voiced'),
    'UH': ('vowel', 'back', 'voiced'),
    'UW': ('vowel', 'back', 'voiced'),
    'AH': ('vowel', 'central', 'voiced'),
    'ER': ('vowel', 'central', 'voiced'),
    'AX': ('vowel', 'central', 'voiced'),
    'IX': ('vowel', 'central', 'voiced'),
    'AXR': ('vowel', 'central', 'voiced'),
    # Diphthongs
    'AY': ('diphthong', 'front', 'voiced'),
    'AW': ('diphthong', 'back', 'voiced'),
    'OY': ('diphthong', 'back', 'voiced'),
}

MIN_PHONEME_COUNT = 2
MAX_PHONEMES_PER_SEC = 20


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
    def get_features(self, text: str) -> Dict[str, object]:
        """Convert transcript to phonemes and multi-dimensional classes."""
        if not self.g2p or not text:
            return {"phn": "", "manner": "", "place": "", "voice": "", "count": 0}

        # Remove bracketed noise tokens like [SILENCE], [NOISE]
        clean_text = " ".join(
            word for word in text.split()
            if not (word.startswith("[") and word.endswith("]"))
        )
        if not clean_text:
            return {"phn": "", "manner": "", "place": "", "voice": "", "count": 0}

        # Keep alphanumeric phonemes, strip stress markers (0,1,2)
        raw_phonemes = self.g2p(clean_text)
        phonemes = [p.rstrip('012') for p in raw_phonemes if p.strip() and p.isalnum()]
        if not phonemes:
            return {"phn": "", "manner": "", "place": "", "voice": "", "count": 0}

        manners, places, voices = [], [], []
        for phn in phonemes:
            manner, place, voice = PHONEME_DETAILS.get(phn, ('other', 'other', 'other'))
            manners.append(manner)
            places.append(place)
            voices.append(voice)

        return {
            "phn": " ".join(phonemes),
            "manner": " ".join(manners),
            "place": " ".join(places),
            "voice": " ".join(voices),
            "count": len(phonemes),
        }

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
            audio_info = sample.get("audio", {})
            hf_raw_path = audio_info.get("path", "")
        
            if hf_raw_path:
                # Replicate the hashing logic from your download script
                path_hash = hashlib.md5(hf_raw_path.encode()).hexdigest()[:8]
                speaker = sample.get('speaker_id', 'unknown')
                original_name = Path(hf_raw_path).name
            
                # This matches the filename format: {speaker}_{hash}_{name}
                expected_filename = f"{speaker}_{path_hash}_{original_name}"
            
                if expected_filename in local_files:
                    matched_samples.append({
                        "metadata": sample,
                        "local_path": local_files[expected_filename],
                        "hf_path": hf_raw_path,
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
        
        features = processor.get_features(transcript)
        if features["count"] < MIN_PHONEME_COUNT:
            continue

        phn_rate = features["count"] / metrics["duration"]
        if phn_rate > MAX_PHONEMES_PER_SEC:
            continue

        hf_path = bundle.get("hf_path", "")
        session_id = Path(hf_path).parent.name if hf_path else "unknown"
        speaker_id = meta.get("speaker_id", path.name.split('_')[0])

        rows.append({
            "sample_id": f"{speaker_id}_{session_id}_{Path(hf_path).name}" if hf_path else path.name,
            "path": str(path),
            "speaker": speaker_id,
            "status": meta["speech_status"],
            "label": 1 if meta["speech_status"] == "dysarthria" else 0,
            "transcript": transcript,
            "phonemes": features["phn"],
            "articulatory_classes": features["manner"],
            "manner_classes": features["manner"],
            "place_classes": features["place"],
            "voice_classes": features["voice"],
            "phn_count": features["count"],
            "phonemes_per_sec": round(float(phn_rate), 3),
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