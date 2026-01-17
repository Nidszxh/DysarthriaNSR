import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import torchaudio
import torchaudio.functional as taF
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, AutoProcessor


class TorgoNeuroSymbolicDataset(Dataset):
    """
    Dataset for dysarthric speech recognition combining neural and symbolic features.
    Loads audio waveforms and corresponding phoneme sequences from TORGO manifest.
    """
    
    def __init__(
        self, 
        manifest_path: str, 
        processor_id: str = "facebook/hubert-base-ls960", 
        sampling_rate: int = 16000
    ):
        """ Initialize TORGO dataset.
        # Args:
            manifest_path: Path to neuro-symbolic manifest CSV
            processor_id: HuggingFace model ID for feature extraction
            sampling_rate: Target audio sampling rate
        """
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
            
        self.df = pd.read_csv(manifest_path)
        initial_count = len(self.df)
        self.df = self.df[self.df["phonemes"].fillna("").str.strip() != ""].reset_index(drop=True)
        dropped = initial_count - len(self.df)
        if dropped:
            print(f"⚠️ Removed {dropped} samples with empty phoneme sequences")
        self.sampling_rate = sampling_rate
        
        # Load feature processor
        self.processor = self._load_processor(processor_id)
        
        # Build phoneme vocabulary
        self._build_vocabularies()
        
        print(f"📊 Dataset initialized: {len(self.df)} samples")
        print(f"🔤 Vocabulary size: {len(self.phn_to_id)} phonemes")
    
    def _load_processor(self, processor_id: str):
        # Load feature processor from HuggingFace.
        if "hubert" in processor_id.lower():
            print(f"🔥 Loading HuBERT feature extractor...")
            return AutoFeatureExtractor.from_pretrained(processor_id)
        
        # Try AutoProcessor first, fall back to feature extractor
        try:
            return AutoProcessor.from_pretrained(processor_id)
        except Exception as e:
            print(f"⚠️  AutoProcessor failed, using feature extractor: {e}")
            return AutoFeatureExtractor.from_pretrained(processor_id)
    
    def _build_vocabularies(self) -> None:
        # Build vocabularies for phonemes and articulatory classes.
        # Extract unique phonemes from manifest
        phonemes_list = sorted(set(
            phoneme
            for phonemes_str in self.df['phonemes'].fillna("")
            for phoneme in phonemes_str.split()
        ))
        
        # Create vocabulary with special tokens
        self.phn_to_id = {
            '<BLANK>': 0,  # CTC blank token
            '<PAD>': 1,    # Padding token
            '<UNK>': 2,    # Unknown token
        }
        
        # Add regular phonemes starting from index 3
        for i, phoneme in enumerate(phonemes_list):
            self.phn_to_id[phoneme] = i + 3
        
        # Create reverse mapping
        self.id_to_phn = {idx: phn for phn, idx in self.phn_to_id.items()}
        
        # Store special token IDs
        self.blank_id = 0
        self.pad_id = 1
        self.unk_id = 2

        # Build articulatory class vocabulary
        art_source = self.df['articulatory_classes'] if 'articulatory_classes' in self.df else pd.Series([], dtype=str)
        art_classes = sorted(set(
            art
            for art_str in art_source.fillna("")
            for art in art_str.split()
        ))
        self.art_to_id = {
            '<PAD>': 0,
            '<UNK>': 1,
        }
        for i, art in enumerate(art_classes):
            self.art_to_id[art] = i + 2
        self.id_to_art = {idx: art for art, idx in self.art_to_id.items()}
        self.art_pad_id = 0
        self.art_unk_id = 1
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio waveform.
        Args: audio_path: Path to audio file
        Returns: Normalized waveform tensor [T]
        """
        # Load audio file
        if os.path.exists(audio_path):
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
            except Exception:
                # Return silence on load failure
                waveform = torch.zeros(1, self.sampling_rate)
                sample_rate = self.sampling_rate
        else:
            # Return silence if file not found
            waveform = torch.zeros(1, self.sampling_rate)
            sample_rate = self.sampling_rate
        
        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary using functional API to avoid recreating kernels
        if sample_rate != self.sampling_rate:
            waveform = taF.resample(waveform, sample_rate, self.sampling_rate)
        
        # Normalize using peak normalization
        waveform = waveform.squeeze(0)
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val
        
        return waveform
    
    def _encode_phonemes(self, phonemes_str: str) -> List[int]:
        """
        Convert phoneme sequence to IDs.
        
        Args: phonemes_str: Space-separated phoneme string
            
        Returns: List of phoneme IDs
        """
        phoneme_list = phonemes_str.split()
        return [self.phn_to_id.get(phn, self.unk_id) for phn in phoneme_list]

    def _encode_articulatory_classes(self, art_str: str) -> List[int]:
        """Convert articulatory class sequence to IDs."""
        art_list = art_str.split()
        return [self.art_to_id.get(art, self.art_unk_id) for art in art_list]
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - input_values: Processed audio features
                - labels: Phoneme ID sequence
                - metadata: Speaker and dysarthria information
        """
        row = self.df.iloc[idx]
        
        # Load and process audio
        waveform = self._load_audio(row['path'])
        audio_features = self.processor(
            waveform, 
            sampling_rate=self.sampling_rate, 
            return_tensors="pt"
        ).input_values.squeeze(0)
        
        # Encode phoneme and articulatory sequences
        phonemes_str = str(row['phonemes'])
        phoneme_ids = self._encode_phonemes(phonemes_str)
        art_str = str(row.get('articulatory_classes', ""))
        art_ids = self._encode_articulatory_classes(art_str)

        # Align lengths defensively to avoid mismatch errors
        if not art_ids:
            art_ids = [self.art_pad_id] * len(phoneme_ids)
        seq_len = min(len(phoneme_ids), len(art_ids))
        phoneme_ids = phoneme_ids[:seq_len]
        art_ids = art_ids[:seq_len]
        
        return {
            "input_values": audio_features,
            "labels": torch.tensor(phoneme_ids, dtype=torch.long),
            "articulatory_labels": torch.tensor(art_ids, dtype=torch.long),
            "metadata": {
                "speaker": row['speaker'],
                "is_dysarthric": torch.tensor(row['label'], dtype=torch.long),
                "transcript": row['transcript']
            }
        }


class NeuroSymbolicCollator:
    """
    Collator for batching neuro-symbolic samples with proper padding.
    Handles variable-length audio and phoneme sequences for CTC training.
    """
    
    def __init__(self, processor, pad_id: int = 1):
        """
        Initialize collator.
        
        Args:
            processor: Feature processor instance
            pad_id: Padding token ID
        """
        self.processor = processor
        self.pad_id = pad_id
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples with padding.
        
        Args:
            batch: List of dataset samples
            
        Returns:
            Dictionary containing:
                - input_values: Padded audio features [batch, time]
                - attention_mask: Attention mask [batch, time]
                - labels: Padded phoneme IDs [batch, seq_len]
                - status: Dysarthria labels [batch]
                - speakers: List of speaker IDs
        """
        # Extract components
        input_values = [item["input_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        art_labels = [item["articulatory_labels"] for item in batch]
        
        # Pad audio features
        input_padded = torch.nn.utils.rnn.pad_sequence(
            input_values, batch_first=True, padding_value=0.0
        )
        
        # Create attention mask (1 for signal, 0 for padding)
        attention_mask = torch.zeros_like(input_padded, dtype=torch.long)
        for i, length in enumerate(len(x) for x in input_values):
            attention_mask[i, :length] = 1
        
        # Pad labels (use -100 to ignore in loss computation)
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        art_padded = torch.nn.utils.rnn.pad_sequence(
            art_labels, batch_first=True, padding_value=-100
        )
        
        # Aggregate metadata
        status = torch.stack([item["metadata"]["is_dysarthric"] for item in batch])
        speakers = [item["metadata"]["speaker"] for item in batch]
        
        return {
            "input_values": input_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "articulatory_labels": art_padded,
            "status": status,
            "speakers": speakers
        }


def create_dataloaders( manifest_path: str, processor_id: str = "facebook/hubert-base-ls960",
    batch_size: int = 4, num_workers: int = 4, sampling_rate: int = 16000) -> DataLoader:
    """
    Create DataLoader for TORGO dataset.
    
    Args:
        manifest_path: Path to manifest CSV
        processor_id: HuggingFace model ID
        batch_size: Batch size
        num_workers: Number of worker processes
        sampling_rate: Audio sampling rate
        
    Returns:
        DataLoader instance
    """
    dataset = TorgoNeuroSymbolicDataset(
        manifest_path=manifest_path,
        processor_id=processor_id,
        sampling_rate=sampling_rate
    )
    
    collator = NeuroSymbolicCollator(dataset.processor)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )


def main() -> None:
    manifest_path = "./data/torgo_neuro_symbolic_manifest.csv"
    
    # Create dataloader
    loader = create_dataloaders(
        manifest_path=manifest_path,
        batch_size=4,
        num_workers=4  # Use 0 for debugging
    )
    
    # Test batch loading
    for batch in loader:
        print("\n✅ Batch loaded successfully")
        print(f"Input shape (neural): {batch['input_values'].shape}")
        print(f"Labels shape (symbolic): {batch['labels'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Dysarthric labels: {batch['status']}")
        print(f"Speakers: {batch['speakers']}")
        break


if __name__ == "__main__":
    main()