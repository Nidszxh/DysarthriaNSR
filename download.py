import os
from pathlib import Path
from datasets import load_dataset, Audio

data_dir = Path(__file__).parent / "data"
os.environ["HF_HOME"] = str(data_dir)

print("📦 Loading dataset metadata...")
# Load dataset and disable audio decoding to access paths directly
dataset = load_dataset("abnerh/TORGO-database", cache_dir=data_dir)

# Disable audio decoding to access the 'path' field
dataset = dataset.cast_column('audio', Audio(decode=False))

sample_audio_metadata = dataset['train'].column_names
print(f"Available columns: {sample_audio_metadata}")

# Now we can safely access the path without triggering torchcodec decoder
sample = dataset['train'][0]
actual_path = sample['audio']['path']

# Extract speaker ID - try dataset field first, fall back to filename parsing
speaker_id = sample.get('speaker_id') or os.path.basename(actual_path).split('_')[0]

print("\n✅ Success! ---")
print(f"Transcription: {sample['transcription']}")
print(f"Path: {actual_path}")
print(f"Speaker ID: {speaker_id}")
print(f"Speech Status: {sample['speech_status']}")