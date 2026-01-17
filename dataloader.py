import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor
from datasets import load_dataset, Audio
from typing import Dict, List, Any

# --- 1. Symbolic Articulatory Knowledge Base ---
# Mapping ARPABET phonemes to Manners of Articulation for the Reasoning Layer
PHONEME_MAP = {
    'P': 'stop', 'B': 'stop', 'T': 'stop', 'D': 'stop', 'K': 'stop', 'G': 'stop',
    'F': 'fricative', 'V': 'fricative', 'TH': 'fricative', 'DH': 'fricative',
    'S': 'fricative', 'Z': 'fricative', 'SH': 'fricative', 'ZH': 'fricative',
    'HH': 'fricative', 'CH': 'affricate', 'JH': 'affricate',
    'M': 'nasal', 'N': 'nasal', 'NG': 'nasal',
    'L': 'liquid', 'R': 'liquid', 'W': 'glide', 'Y': 'glide',
    'IY': 'vowel', 'IH': 'vowel', 'EH': 'vowel', 'EY': 'vowel', 'AE': 'vowel',
    'AA': 'vowel', 'AO': 'vowel', 'OW': 'vowel', 'UH': 'vowel', 'UW': 'vowel',
    'AH': 'vowel', 'ER': 'vowel', 'AX': 'vowel', 'AY': 'diphthong', 
    'AW': 'diphthong', 'OY': 'diphthong'
}

class TorgoNeuroSymbolicDataset(Dataset):
    def __init__(
        self, 
        manifest_path: str, 
        processor_id: str = "facebook/wav2vec2-base-960h", 
        sampling_rate: int = 16000,
        use_hf_dataset: bool = True,
        hf_cache_dir: str = "./data"
    ):
        """
        Custom Dataset for Dysarthric Speech Recognition.
        Combines Neural (Audio) and Symbolic (Phoneme/Articulatory) features.
        
        Args:
            manifest_path: Path to the neuro-symbolic manifest CSV
            processor_id: HuggingFace processor model ID
            sampling_rate: Target sampling rate for audio
            use_hf_dataset: If True, load audio from HF dataset (recommended)
            hf_cache_dir: Cache directory for HF dataset
        """
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
            
        self.df = pd.read_csv(manifest_path)
        self.processor = Wav2Vec2Processor.from_pretrained(processor_id)
        self.sampling_rate = sampling_rate
        self.use_hf_dataset = use_hf_dataset
        
        # Load HuggingFace dataset for audio streaming (avoids path issues)
        if self.use_hf_dataset:
            print("📥 Loading HF TORGO dataset for audio streaming...")
            raw_dataset = load_dataset("abnerh/TORGO-database", cache_dir=hf_cache_dir)
            
            # Flatten all splits - keep decoding ENABLED for audio loading
            all_data = []
            for split in raw_dataset.keys():
                all_data.extend(raw_dataset[split])
            self.hf_dataset = all_data
            
            # Build sample_id to HF index mapping for reliable matching
            # Use speaker prefix to avoid filename collisions across speakers
            self.sample_id_to_hf_idx = {}
            for idx, sample in enumerate(self.hf_dataset):
                # Extract speaker and filename from path
                # Access path safely - try both decoded and non-decoded formats
                try:
                    audio_info = sample['audio']
                    if isinstance(audio_info, dict) and 'path' in audio_info:
                        audio_path = audio_info['path']
                    else:
                        # Fallback: reconstruct from other fields
                        continue
                except:
                    continue
                    
                filename = os.path.basename(audio_path)
                # Create unique key: speaker_filename
                speaker = sample.get('speaker_id', filename.split('_')[0])
                sample_id = f"{speaker}_{filename}"
                self.sample_id_to_hf_idx[sample_id] = idx
            
            print("⚙️  Audio loading configured")
        
        # Build Vocab from Manifest
        self.phonemes_list = sorted(list(set(
            [p for sublist in self.df['phonemes'].fillna("").str.split() for p in sublist]
        )))
        
        # CTC-specific vocabulary: BLANK for alignment, PAD for batching, UNK for OOV
        self.phn_to_id = {p: i + 3 for i, p in enumerate(self.phonemes_list)}
        self.phn_to_id['<BLANK>'] = 0  # CTC blank token for alignment
        self.phn_to_id['<PAD>'] = 1    # Padding token for batching
        self.phn_to_id['<UNK>'] = 2    # Unknown token for OOV phonemes
        
        self.id_to_phn = {i: p for p, i in self.phn_to_id.items()}
        self.blank_id = 0
        self.pad_id = 1
        self.unk_id = 2
        
        print(f"📊 Dataset initialized: {len(self.df)} samples.")
        print(f"🔤 Vocabulary Size: {len(self.phn_to_id)} phonemes.")

    def __len__(self):
        return len(self.df)

    def _load_audio(self, manifest_row):
        """Loads audio from HuggingFace dataset (preferred) or fallback to file path."""
        
        if self.use_hf_dataset:
            # Match using speaker-prefixed sample_id for reliable alignment
            speaker = manifest_row['speaker']
            filename = manifest_row['sample_id']
            sample_id = f"{speaker}_{filename}"
            hf_idx = self.sample_id_to_hf_idx.get(sample_id)
            
            if hf_idx is not None and hf_idx < len(self.hf_dataset):
                try:
                    sample = self.hf_dataset[hf_idx]
                    waveform_np = np.array(sample['audio']['array'], dtype=np.float32)
                    sr = sample['audio']['sampling_rate']
                    
                    # Convert to tensor
                    waveform = torch.from_numpy(waveform_np).unsqueeze(0)  # [1, T]
                    
                except Exception as e:
                    print(f"⚠️  HF dataset load failed for {sample_id}: {e}")
                    waveform = torch.zeros(1, self.sampling_rate)
                    sr = self.sampling_rate
            else:
                print(f"⚠️  Sample {sample_id} not found in HF dataset")
                waveform = torch.zeros(1, self.sampling_rate)
                sr = self.sampling_rate
        else:
            # Fallback: try to load from file path
            path = manifest_row['path']
            
            if os.path.exists(path):
                try:
                    waveform, sr = torchaudio.load(path)
                except Exception as e:
                    print(f"⚠️  File load failed for {path}: {e}")
                    waveform = torch.zeros(1, self.sampling_rate)
                    sr = self.sampling_rate
            else:
                # Last resort: return silence
                waveform = torch.zeros(1, self.sampling_rate)
                sr = self.sampling_rate
        
        # Convert Stereo to Mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if necessary
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            waveform = resampler(waveform)
        
        # Peak normalization (TORGO-specific: handles variable breath support)
        waveform_1d = waveform.squeeze(0)
        if waveform_1d.abs().max() > 0:
            waveform_1d = waveform_1d / waveform_1d.abs().max()
            
        return waveform_1d  # [T]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        # 1. Neural Input: Audio (from HF dataset matched by sample_id)
        waveform = self._load_audio(row)

        # Process via Wav2Vec2 Feature Extractor
        audio_features = self.processor(
            waveform, 
            sampling_rate=self.sampling_rate, 
            return_tensors="pt"
        ).input_values.squeeze(0)

        # 2. Symbolic Target: Phoneme IDs
        phn_sequence = str(row['phonemes']).split()
        phn_ids = [self.phn_to_id.get(p, self.unk_id) for p in phn_sequence]

        # 3. Robustness/Explainability Metadata
        # Clean articulatory classes for the sequence
        art_features = [PHONEME_MAP.get(p.rstrip('012'), "other") for p in phn_sequence]

        return {
            "input_values": audio_features,
            "labels": torch.tensor(phn_ids, dtype=torch.long),
            "metadata": {
                "speaker": row['speaker'],
                "is_dysarthric": torch.tensor(row['label'], dtype=torch.long),
                "articulatory_features": art_features,
                "transcript": row['transcript']
            }
        }

# --- 2. Neuro-Symbolic Collator ---
class NeuroSymbolicCollator:
    """CTC-aware collator using processor's native padding for correct normalization."""
    
    def __init__(self, processor, pad_id=1):
        self.processor = processor
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, Any]]):
        # Extract raw tensors
        input_values = [item["input_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad audio features (already normalized by processor during feature extraction)
        input_padded = torch.nn.utils.rnn.pad_sequence(
            input_values, batch_first=True, padding_value=0.0
        )
        
        # Compute attention mask (1 where signal exists, 0 for padding)
        attention_mask = torch.zeros_like(input_padded, dtype=torch.long)
        for i, original_length in enumerate([len(x) for x in input_values]):
            attention_mask[i, :original_length] = 1
        
        # Pad labels with -100 (ignored by CTC loss)
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        # Aggregate metadata
        status = torch.stack([item["metadata"]["is_dysarthric"] for item in batch])
        speakers = [item["metadata"]["speaker"] for item in batch]

        return {
            "input_values": input_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "status": status,
            "speakers": speakers
        }

# --- 3. Example Usage Block ---
if __name__ == "__main__":
    MANIFEST = "./data/torgo_neuro_symbolic_manifest.csv"
    
    # Initialize Dataset
    dataset = TorgoNeuroSymbolicDataset(MANIFEST)
    
    # Initialize Collator
    collator = NeuroSymbolicCollator(dataset.processor)
    
    # Initialize DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=collator
    )
    
    # Sanity Check
    for batch in loader:
        print("\n✅ Batch Loaded Successfully")
        print(f"Input Shape (Neural): {batch['input_values'].shape}")
        print(f"Labels Shape (Symbolic): {batch['labels'].shape}")
        print(f"Dysarthric Labels: {batch['status']}")
        break
