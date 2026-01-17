import argparse
import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset, Audio
from tqdm import tqdm
import librosa

try:
    from g2p_en import G2p
    import nltk
    
    # Download required NLTK data for g2p_en
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("📥 Downloading required NLTK data...")
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    
    g2p = G2p()
except Exception as e:
    print(f"Warning: g2p_en setup failed: {e}")
    print("Install with: pip install g2p_en")
    g2p = None

# Phoneme-to-Articulatory-Class Mapping (American English ARPABET)
# This enables explainability at the articulatory level for dysarthric patterns
PHONEME_ARTICULATION = {
    # Stops (Plosives) - often substituted in dysarthria
    'P': 'stop', 'B': 'stop', 'T': 'stop', 'D': 'stop', 'K': 'stop', 'G': 'stop',
    
    # Fricatives - require precise airflow control (commonly impaired)
    'F': 'fricative', 'V': 'fricative', 'TH': 'fricative', 'DH': 'fricative',
    'S': 'fricative', 'Z': 'fricative', 'SH': 'fricative', 'ZH': 'fricative',
    'HH': 'fricative',
    
    # Affricates - combination movements
    'CH': 'affricate', 'JH': 'affricate',
    
    # Nasals - velopharyngeal control
    'M': 'nasal', 'N': 'nasal', 'NG': 'nasal',
    
    # Liquids - tongue control
    'L': 'liquid', 'R': 'liquid',
    
    # Glides (Semivowels)
    'W': 'glide', 'Y': 'glide',
    
    # Vowels (Monophthongs)
    'IY': 'vowel', 'IH': 'vowel', 'EH': 'vowel', 'EY': 'vowel', 'AE': 'vowel',
    'AA': 'vowel', 'AO': 'vowel', 'OW': 'vowel', 'UH': 'vowel', 'UW': 'vowel',
    'AH': 'vowel', 'ER': 'vowel', 'AX': 'vowel', 'IX': 'vowel', 'AXR': 'vowel',
    
    # Diphthongs
    'AY': 'diphthong', 'AW': 'diphthong', 'OY': 'diphthong',
}

def get_articulatory_classes(phoneme_list):
    """Map phonemes to their articulatory classes for robustness analysis."""
    classes = []
    for phn in phoneme_list:
        # Remove stress markers (0, 1, 2) from ARPABET phonemes
        phn_base = phn.rstrip('012')
        art_class = PHONEME_ARTICULATION.get(phn_base, 'other')
        classes.append(art_class)
    return classes

def calculate_rms_energy(audio_path):
    """Calculate RMS energy for signal quality assessment."""
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=None)
        rms = librosa.feature.rms(y=y)[0]
        mean_rms = float(rms.mean())
        return round(mean_rms, 6)
    except Exception as e:
        # Return None if calculation fails - will be handled gracefully in aggregation
        return None

def create_neuro_symbolic_manifest(cache_dir: Path):
    print(f"📦 Loading TORGO from HF (Cache: {cache_dir})")
    
    # Load the lightweight dataset
    dataset = load_dataset("abnerh/TORGO-database", cache_dir=str(cache_dir))
    
    # Disable audio decoding to access paths without torchcodec errors
    dataset = dataset.cast_column('audio', Audio(decode=False))
    
    rows = []
    # Process all splits (usually 'train' in this HF version)
    all_data = []
    for split in dataset.keys():
        for item in dataset[split]:
            all_data.append(item)

    print(f"🧠 Generating Symbolic Phoneme sequences for {len(all_data)} samples...")

    for idx, sample in enumerate(tqdm(all_data)):
        transcript = sample.get("transcription", "").strip().upper()
        if not transcript:
            continue

        # 1. Neural Representation Prep: Audio Path
        wav_path = sample["audio"]["path"]
        
        # Extract unique sample identifier from filename
        sample_id = os.path.basename(wav_path)
        
        # 2. Symbolic Prep: Phonemes (The 'Symbolic' labels for your reasoning layer)
        phonemes = []
        articulatory_classes = []
        if g2p:
            # g2p returns a list of phonemes and punctuation
            phoneme_list = g2p(transcript)
            phonemes = [p for p in phoneme_list if p.strip() and p not in ['.', ',', '?', '!']]
            
            # Map phonemes to articulatory classes for robustness analysis
            articulatory_classes = get_articulatory_classes(phonemes)
        
        # 3. Robustness Prep: Speaker Metadata
        # TORGO Speaker IDs: Dysarthric (F01, F03, F04, M01-M05) vs Controls (FC01, MC01, etc.)
        # Extract speaker ID from filename (e.g., 'F01_Session1_0001.wav' -> 'F01')
        speaker_id = os.path.basename(wav_path).split('_')[0]
        is_dysarthric = 1 if sample["speech_status"] == "dysarthria" else 0
        
        # 4. Explainability Prep: Metadata for error analysis
        # Duration is provided by the dataset (calculated during HF dataset creation)
        duration = sample.get("duration", 0.0)
        
        # 5. Signal Quality: RMS Energy for distinguishing articulation vs quality issues
        rms_energy = calculate_rms_energy(wav_path)

        rows.append({
            "sample_id": sample_id,
            "hf_index": idx,
            "path": wav_path,
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
        })

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out", default="./data/torgo_neuro_symbolic_manifest.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df = create_neuro_symbolic_manifest(data_dir)

    # Save manifest
    df.to_csv(args.out, index=False)
    
    print("\n✅ Manifest Created Successfully!")
    print(f"Total Audio Hours: {df['duration'].sum()/3600:.2f} hrs")
    print(f"Dysarthric Speakers: {df[df['label']==1]['speaker'].nunique()}")
    print(f"Control Speakers: {df[df['label']==0]['speaker'].nunique()}")
    print(f"Average Phonemes per sample: {df['phn_count'].mean():.1f}")
    
    # Calculate RMS energy stats (excluding None values)
    valid_rms = df[df['rms_energy'].notna()]['rms_energy']
    if len(valid_rms) > 0:
        print(f"Average RMS Energy: {valid_rms.mean():.6f} ({len(valid_rms)}/{len(df)} samples loaded)")
    else:
        print(f"⚠️  RMS Energy: Could not compute (audio files may not be directly accessible)")
    
    print(f"Manifest saved to: {args.out}")


if __name__ == "__main__":
    main()