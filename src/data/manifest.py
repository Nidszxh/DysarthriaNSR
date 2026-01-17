import argparse
import os
from pathlib import Path
from typing import List, Optional

import librosa
import pandas as pd
from datasets import Audio, load_dataset
from tqdm import tqdm


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


def setup_g2p():
    try:
        import g2p_en, nltk
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        return g2p_en.G2p()
    except Exception: return None

def get_articulatory_classes(phoneme_list: List[str]) -> List[str]:
    """
    Map phonemes to their articulatory classes.
    
    Args:
        phoneme_list: List of phoneme strings (ARPABET format)
        
    Returns:
        List of articulatory class labels
    """
    classes = []
    for phoneme in phoneme_list:
        # Remove stress markers (0, 1, 2) from ARPABET phonemes
        phoneme_base = phoneme.rstrip('012')
        articulatory_class = PHONEME_ARTICULATION.get(phoneme_base, 'other')
        classes.append(articulatory_class)
    return classes


def calculate_rms_energy(audio_path: str) -> Optional[float]:
    """
    Calculate RMS energy for signal quality assessment.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Mean RMS energy value or None if calculation fails
    """
    try:
        y, _ = librosa.load(audio_path, sr=None)
        rms = librosa.feature.rms(y=y)[0]
        return round(float(rms.mean()), 6)
    except Exception:
        return None


def extract_phonemes(text: str, g2p_converter) -> tuple[List[str], List[str]]:
    """
    Convert text to phoneme sequence and articulatory classes.
    
    Args:
        text: Input text transcript
        g2p_converter: G2p instance or None
        
    Returns:
        Tuple of (phoneme_list, articulatory_class_list)
    """
    if not g2p_converter:
        return [], []
    
    # Convert text to phonemes, excluding punctuation
    phoneme_list = g2p_converter(text)
    phonemes = [p for p in phoneme_list if p.strip() and p not in ['.', ',', '?', '!']]
    
    # Map to articulatory classes
    articulatory_classes = get_articulatory_classes(phonemes)
    
    return phonemes, articulatory_classes


def extract_speaker_id(audio_path: str) -> str:
    """
    Extract speaker ID from audio filename.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Speaker ID (e.g., 'F01', 'MC01')
    """
    return os.path.basename(audio_path).split('_')[0]


def create_manifest_row(sample: dict, idx: int, g2p_converter) -> Optional[dict]:
    """
    Create a single manifest row from a dataset sample.
    
    Args:
        sample: Dataset sample dictionary
        idx: Sample index
        g2p_converter: G2p instance or None
        
    Returns:
        Dictionary representing manifest row or None if invalid
    """
    
    transcript = sample.get("transcription", "").strip().upper()
    if not transcript:
        return None
    
    audio_path = sample["audio"]["path"]
    sample_id = os.path.basename(audio_path)
    speaker_id = extract_speaker_id(audio_path)
    
    # Extract phoneme information
    phonemes, articulatory_classes = extract_phonemes(transcript, g2p_converter)
    
    # Determine dysarthria status
    is_dysarthric = 1 if sample["speech_status"] == "dysarthria" else 0
    
    # Calculate audio metrics
    duration = sample.get("duration", 0.0)
    rms_energy = calculate_rms_energy(audio_path)
    
    if duration < 0.1 or (rms_energy is not None and rms_energy < 0.0001):
        return None # Skip silent or corrupted files
    
    return {
        "sample_id": sample_id,
        "hf_index": idx,
        "path": audio_path,
        "speaker": speaker_id,
        "status": sample["speech_status"],
        "label": is_dysarthric,
        "transcript": transcript,
        "phonemes": " ".join(phonemes),
        "articulatory_classes": " ".join(articulatory_classes),
        "phn_count": len(phonemes),
        "duration": round(duration, 3),
        "rms_energy": rms_energy,
        "gender": sample.get("gender", "unknown")
    }


def create_neuro_symbolic_manifest(cache_dir: Path) -> pd.DataFrame:
    """
    Generate complete manifest DataFrame from TORGO dataset.
    
    Args:
        cache_dir: Directory to cache dataset files
        
    Returns:
        DataFrame containing manifest data
    """
    print(f"📦 Loading TORGO from HF (Cache: {cache_dir})")
    
    # Load dataset with audio decoding disabled
    dataset = load_dataset("abnerh/TORGO-database", cache_dir=str(cache_dir))
    dataset = dataset.cast_column("audio", Audio(decode=False))
    
    # Initialize phoneme converter
    g2p_converter = setup_g2p()
    
    # Collect all samples across splits
    all_samples = []
    for split in dataset.keys():
        all_samples.extend(dataset[split])
    
    print(f"🧠 Generating symbolic phoneme sequences for {len(all_samples)} samples...")
    
    # Process each sample
    rows = []
    for idx, sample in enumerate(tqdm(all_samples)):
        row = create_manifest_row(sample, idx, g2p_converter)
        if row:
            rows.append(row)
    
    return pd.DataFrame(rows)


def print_manifest_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics for generated manifest."""
    print("\n✅ Manifest Created Successfully!")
    print(f"Total Audio Hours: {df['duration'].sum() / 3600:.2f} hrs")
    print(f"Dysarthric Speakers: {df[df['label'] == 1]['speaker'].nunique()}")
    print(f"Control Speakers: {df[df['label'] == 0]['speaker'].nunique()}")
    print(f"Average Phonemes per sample: {df['phn_count'].mean():.1f}")
    
    # RMS energy statistics (excluding missing values)
    valid_rms = df[df['rms_energy'].notna()]['rms_energy']
    if len(valid_rms) > 0:
        print(f"Average RMS Energy: {valid_rms.mean():.6f} ({len(valid_rms)}/{len(df)} samples)")
    else:
        print("⚠️  RMS Energy: Could not compute (audio files may not be accessible)")


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate neuro-symbolic manifest for TORGO dataset"
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory for dataset cache"
    )
    parser.add_argument(
        "--out",
        default="./data/torgo_neuro_symbolic_manifest.csv",
        help="Output CSV file path"
    )
    args = parser.parse_args()
    
    # Generate manifest
    data_dir = Path(args.data_dir)
    df = create_neuro_symbolic_manifest(data_dir)
    
    # Save to CSV
    df.to_csv(args.out, index=False)
    
    # Display statistics
    print_manifest_statistics(df)
    print(f"Manifest saved to: {args.out}")


if __name__ == "__main__":
    main()