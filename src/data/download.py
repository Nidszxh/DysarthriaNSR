import os
from pathlib import Path

from datasets import Audio, load_dataset

def setup_cache_directory() -> Path:
    
    # Configure and return the data cache directory path.
    data_dir = Path(__file__).parent / "data"
    os.environ["HF_HOME"] = str(data_dir)
    return data_dir


def load_torgo_dataset(cache_dir: Path):
    
    # Load TORGO dataset with audio decoding disabled.
    print("📦 Loading dataset metadata...")
    dataset = load_dataset("abnerh/TORGO-database", cache_dir=cache_dir)
    
    # Disable audio decoding to access file paths directly
    dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset


def extract_speaker_id(sample: dict, audio_path: str) -> str:
    
    # Extract speaker ID from sample metadata or filename.
    if "speaker_id" in sample and sample["speaker_id"]:
        return sample["speaker_id"]
    
    # Fallback: extract from filename (e.g., 'F01_Session1_0001.wav' -> 'F01')
    return os.path.basename(audio_path).split("_")[0]


def display_sample_info(sample: dict, speaker_id: str) -> None:

    print("\nCompleted loading dataset. Sample information:")
    print(f"Transcription: {sample['transcription']}")
    print(f"Path: {sample['audio']['path']}")
    print(f"Speaker ID: {speaker_id}")
    print(f"Speech Status: {sample['speech_status']}")


def main() -> None:
    cache_dir = setup_cache_directory()
    dataset = load_torgo_dataset(cache_dir)
    
    # Display available columns
    print(f"Available columns: {dataset['train'].column_names}")
    
    # Verify dataset accessibility with first sample
    sample = dataset["train"][0]
    audio_path = sample["audio"]["path"]
    speaker_id = extract_speaker_id(sample, audio_path)
    
    display_sample_info(sample, speaker_id)


if __name__ == "__main__":
    main()