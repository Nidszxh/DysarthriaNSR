import hashlib
import concurrent.futures
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from functools import lru_cache

import librosa
import numpy as np
import pandas as pd
from datasets import Audio, load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.constants import PHONEME_DETAILS

logger = logging.getLogger(__name__)

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


MIN_PHONEME_COUNT = 2
MAX_PHONEMES_PER_SEC = 20


class SymbolicProcessor:
    """Handles G2P and phoneme-to-articulatory mapping."""

    def __init__(self):
        try:
            import g2p_en
            import nltk
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            self.g2p = g2p_en.G2p()
        except Exception:
            self.g2p = None
            logger.warning("g2p_en not found. Phoneme extraction will be skipped.")

    @lru_cache(maxsize=2048)
    def get_features(self, text: str) -> Dict[str, object]:
        """Convert transcript to phonemes and multi-dimensional classes.
        LRU cache avoids re-running G2P for repeated transcripts across speakers.
        G2P text cache (this) is separate from the per-sample feature cache in dataloader.py."""
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
        logger.warning("Failed to compute audio features for %s", audio_path)
        return {"rms": 0.0, "duration": 0.0}

# --- MAIN PIPELINE ---


def main():
    paths = DataPaths()
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    processor = SymbolicProcessor()

    # 1. Load Metadata with DECODING DISABLED
    logger.info("Loading Metadata (Cache: %s)", paths.data_dir)
    # By using decode=False here, 'audio' stays a dict with a 'path' key
    ds = load_dataset("abnerh/TORGO-database", cache_dir=str(paths.data_dir))
    ds = ds.cast_column("audio", Audio(decode=False))

    # 2. Index Local Files
    logger.info("Indexing %s/audio...", paths.raw_dir)
    local_files = {f.name: f for f in paths.raw_dir.rglob("*.wav")}
    logger.info("Found %d local files.", len(local_files))

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
        logger.warning("Zero matches. Debugging path strings...")
        test_sample = ds['train'][0]['audio']
        logger.warning("Metadata Audio Dict: %s", test_sample)
        logger.warning("Sample Filename: %s", Path(test_sample.get('path', '')).name)
        return

    logger.info("Matched %d local files.", len(matched_samples))

    # 4. Parallel Audio Metrics Extraction
    logger.info("Analyzing audio metrics...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        local_paths = [s["local_path"] for s in matched_samples]
        metrics_results = list(tqdm(
            executor.map(calculate_audio_metrics, local_paths),
            total=len(local_paths),
            desc="Audio Analysis"
        ))

    # 5. Assemble Final Manifest
    logger.info("Generating symbolic phoneme sequences...")
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
        # Local filenames are written as unknown_{hash}_{SPEAKER}_{session}_{mic}_{n}.wav
        # because the HuggingFace dataset has no speaker_id field.  The actual TORGO
        # speaker ID (e.g. "FC02") is therefore at split position [2], not [0].
        speaker_id = meta.get("speaker_id", path.name.split('_')[2])

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

    logger.info("MANIFEST GENERATION COMPLETE")
    logger.info("Output: %s", out_path)
    logger.info("Total Samples: %d", len(df))
    logger.info("Total Hours: %.2f hrs", df['duration'].sum() / 3600)


if __name__ == "__main__":
    main()
