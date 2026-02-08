import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as taF
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, AutoProcessor

# Import from project config (single source of truth)
try:
    from src.utils.config import ProjectPaths, normalize_phoneme
except ImportError:
    # Fallback for different execution contexts
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.config import ProjectPaths, normalize_phoneme


class TorgoNeuroSymbolicDataset(Dataset):
    
    def __init__(
        self,
        manifest_path: str,
        processor_id: str = "facebook/hubert-base-ls960",
        sampling_rate: int = 16000,
        max_audio_length: Optional[float] = None
    ):
        # Validate manifest exists
        if not Path(manifest_path).exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                f"Run 'python src/data/manifest.py' to generate it."
            )
        
        self.manifest_path = manifest_path
        self.sampling_rate = sampling_rate
        self.max_audio_samples = (
            int(max_audio_length * sampling_rate) if max_audio_length else None
        )
        
        # Load manifest and validate
        self.df = pd.read_csv(manifest_path)
        self._validate_and_clean_manifest()
        
        # Load feature processor
        self.processor = self._load_processor(processor_id)
        
        # Build vocabularies
        self._build_vocabularies()
        
        # Pre-calculate inverse-frequency phoneme weights
        self.phoneme_weights = self._calculate_phoneme_weights()
        
        print(f"✅ Dataset initialized: {len(self.df)} samples from {manifest_path}")
        print(f"   Vocabulary: {len(self.phn_to_id)} phonemes")
        print(f"   Articulatory classes: {len(self.art_to_id)}")
    
    def _validate_and_clean_manifest(self) -> None:
        """
        Validate manifest data quality and remove invalid samples.
        
        Checks:
        1. Remove samples with empty phoneme sequences
        2. Validate required columns exist
        3. Check for duplicate sample IDs
        
        Modifies self.df in-place.
        """
        initial_count = len(self.df)
        
        # Required columns
        required_cols = ['path', 'phonemes', 'speaker', 'label']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Manifest missing required columns: {missing_cols}")
        
        # Remove empty phoneme sequences
        self.df = self.df[
            self.df["phonemes"].fillna("").str.strip() != ""
        ].reset_index(drop=True)
        
        dropped = initial_count - len(self.df)
        if dropped > 0:
            warnings.warn(
                f"Dropped {dropped} samples with empty phoneme sequences "
                f"({dropped/initial_count*100:.1f}%)",
                UserWarning
            )
        
        # Check for duplicates (warn but don't remove - may be legitimate multi-mic)
        if self.df.duplicated(subset=['path']).any():
            n_duplicates = self.df.duplicated(subset=['path']).sum()
            warnings.warn(
                f"Found {n_duplicates} duplicate file paths in manifest. "
                f"This may indicate multi-microphone recordings.",
                UserWarning
            )
    
    def _load_processor(self, processor_id: str):
        if "hubert" in processor_id.lower():
            print(f"📦 Loading HuBERT feature extractor: {processor_id}")
            return AutoFeatureExtractor.from_pretrained(processor_id)
        
        # Try AutoProcessor first, fall back to feature extractor
        try:
            return AutoProcessor.from_pretrained(processor_id)
        except Exception:
            warnings.warn(
                f"AutoProcessor failed for {processor_id}, using feature extractor",
                UserWarning
            )
            return AutoFeatureExtractor.from_pretrained(processor_id)
    
    def _build_vocabularies(self) -> None:
        """
        Build phoneme and articulatory class vocabularies from manifest.
        
        Creates:
        - phn_to_id: Phoneme → ID mapping (includes special tokens)
        - id_to_phn: ID → Phoneme reverse mapping
        - art_to_id: Articulatory class → ID mapping
        - id_to_art: ID → Articulatory class reverse mapping
        
        Special tokens:
        - <BLANK> (ID 0): CTC blank token for alignment
        - <PAD> (ID 1): Padding for variable-length sequences
        - <UNK> (ID 2): Unknown/out-of-vocabulary phonemes
        """
        # Extract unique phonemes (stress-agnostic)
        phonemes_list = sorted(set(
            normalize_phoneme(phoneme)
            for phonemes_str in self.df['phonemes'].fillna("")
            for phoneme in phonemes_str.split()
            if phoneme.strip()
        ))
        
        # Create phoneme vocabulary with special tokens
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
        
        # Store special token IDs for convenience
        self.blank_id = 0
        self.pad_id = 1
        self.unk_id = 2
        
        # Build articulatory class vocabulary (if available)
        if 'articulatory_classes' in self.df.columns:
            art_classes = sorted(set(
                art
                for art_str in self.df['articulatory_classes'].fillna("")
                for art in art_str.split()
                if art.strip()
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
        else:
            # Fallback if articulatory classes not present
            self.art_to_id = {'<PAD>': 0, '<UNK>': 1}
            self.id_to_art = {0: '<PAD>', 1: '<UNK>'}
            self.art_pad_id = 0
            self.art_unk_id = 1
    
    def _calculate_phoneme_weights(self) -> torch.Tensor:
        """
        Calculate inverse-frequency weights for phoneme class balancing.
        
        Uses sqrt-damped inverse frequency to avoid extreme weights for rare phonemes.
        Formula: weight[i] = sqrt(median_freq / freq[i])
        
        Returns:
            Tensor of shape [num_phonemes] with weights aligned to phn_to_id
        """
        # Collect all phonemes from manifest
        all_phonemes: List[str] = []
        for row in self.df['phonemes'].fillna(""):
            all_phonemes.extend([
                normalize_phoneme(p) for p in str(row).split() if p.strip()
            ])
        
        # Frequency counts
        counts = pd.Series(all_phonemes).value_counts()
        median_freq = float(counts.median()) if not counts.empty else 1.0
        
        # Initialize weights to ones
        weights = torch.ones(len(self.phn_to_id), dtype=torch.float32)
        
        # Assign inverse-frequency weights (sqrt-damped)
        for phn, phn_id in self.phn_to_id.items():
            if phn in counts:
                freq = float(counts[phn])
                # sqrt damping prevents extreme weights
                weights[phn_id] = np.sqrt(median_freq / max(freq, 1.0))
            else:
                # Unseen phonemes get median weight
                weights[phn_id] = 1.0
        
        # Adjust special tokens
        # BLANK: Slightly lower weight to discourage over-insertion
        weights[self.phn_to_id['<BLANK>']] *= 0.8
        # PAD: Higher weight since ignored in loss (doesn't affect training)
        weights[self.phn_to_id['<PAD>']] *= 1.5
        
        # Clamp for numerical stability
        weights = torch.clamp(weights, min=0.5, max=5.0)
        
        return weights
    
    def get_loss_weights(self) -> torch.Tensor:
        return self.phoneme_weights
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio waveform
        
        Preprocessing steps:
        1. Load audio (handle failures gracefully)
        2. Convert stereo → mono
        3. Resample to target sampling rate
        4. Truncate to max_audio_length (memory optimization)
        5. Peak normalization (critical for dysarthric speech)
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Normalized waveform tensor [T] where T = num_samples
        
        Notes:
            - Returns silence (zeros) on load failure to avoid crashing training
            - Peak normalization accounts for variable breath support in dysarthria
        """
        # Attempt to load audio
        if Path(audio_path).exists():
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
            except Exception as e:
                warnings.warn(
                    f"Failed to load audio from {audio_path}: {e}. "
                    f"Returning silence.",
                    UserWarning
                )
                waveform = torch.zeros(1, self.sampling_rate)
                sample_rate = self.sampling_rate
        else:
            warnings.warn(
                f"Audio file not found: {audio_path}. Returning silence.",
                UserWarning
            )
            waveform = torch.zeros(1, self.sampling_rate)
            sample_rate = self.sampling_rate
        
        # Convert stereo to mono (average channels)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary (use functional API to avoid kernel recreation)
        if sample_rate != self.sampling_rate:
            waveform = taF.resample(waveform, sample_rate, self.sampling_rate)
        
        # Truncate long waveforms (VRAM optimization)
        if self.max_audio_samples is not None:
            if waveform.shape[-1] > self.max_audio_samples:
                waveform = waveform[..., :self.max_audio_samples]
        
        # Peak normalization (critical for variable dysarthric breath support)
        waveform = waveform.squeeze(0)
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val
        
        return waveform
    
    def _encode_phonemes(self, phonemes_str: str) -> List[int]:
        
        phoneme_list = [
            normalize_phoneme(phn) for phn in phonemes_str.split() if phn.strip()
        ]
        return [self.phn_to_id.get(phn, self.unk_id) for phn in phoneme_list]
    
    def _encode_articulatory_classes(self, art_str: str) -> List[int]:
        
        art_list = [a for a in art_str.split() if a.strip()]
        return [self.art_to_id.get(art, self.art_unk_id) for art in art_list]
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]: # Get dataset items by index

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
        
        # CRITICAL: Validate 1:1 alignment (phoneme ↔ articulatory class)
        if len(phoneme_ids) != len(art_ids):
            raise ValueError(
                f"⚠️  Data quality error in sample {idx} (speaker: {row['speaker']}): "
                f"Phoneme sequence length ({len(phoneme_ids)}) ≠ "
                f"Articulatory class length ({len(art_ids)}). "
                f"This indicates misalignment during manifest generation. "
                f"Phonemes: {phonemes_str[:80]}... "
                f"Articulatory: {art_str[:80]}..."
            )
        
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
    
    Handles variable-length audio and phoneme sequences for CTC training:
    - Pads audio to max batch length
    - Pads labels with -100 (ignored by CTC/CE loss)
    - Creates attention masks for valid regions
    
    Args:
        processor: Feature processor instance
        pad_id: Padding token ID (default: 1)
    """
    
    def __init__(self, processor, pad_id: int = 1):
        self.processor = processor
        self.pad_id = pad_id
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        # Extract components
        input_values = [item["input_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        art_labels = [item["articulatory_labels"] for item in batch]
        
        # Pad audio features (use 0.0 for silence)
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
        transcripts = [item["metadata"]["transcript"] for item in batch]
        
        return {
            "input_values": input_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "articulatory_labels": art_padded,
            "status": status,
            "speakers": speakers,
            "transcripts": transcripts
        }


def create_dataloaders(
    manifest_path: str,
    processor_id: str = "facebook/hubert-base-ls960",
    batch_size: int = 4,
    num_workers: int = 4,
    sampling_rate: int = 16000,
    max_audio_length: Optional[float] = None
) -> DataLoader:

    dataset = TorgoNeuroSymbolicDataset(
        manifest_path=manifest_path,
        processor_id=processor_id,
        sampling_rate=sampling_rate,
        max_audio_length=max_audio_length
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
    """Test dataloader module."""
    from src.utils.config import get_default_config
    
    config = get_default_config()
    manifest_path = config.data.manifest_path
    
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            f"Run 'python src/data/manifest.py' to generate it."
        )
    
    # Create dataloader
    loader = create_dataloaders(
        manifest_path=str(manifest_path),
        processor_id=config.model.hubert_model_id,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        sampling_rate=config.data.sampling_rate,
        max_audio_length=config.data.max_audio_length
    )
    
    # Test batch loading
    print("\nTesting DataLoader\n")
    
    for batch in loader:
        print(f"✅ Batch loaded successfully")
        print(f"   Input shape: {batch['input_values'].shape}")
        print(f"   Labels shape: {batch['labels'].shape}")
        print(f"   Attention mask shape: {batch['attention_mask'].shape}")
        print(f"   Dysarthric labels: {batch['status'].tolist()}")
        print(f"   Speakers: {batch['speakers']}")
        break
    
    
if __name__ == "__main__":
    main()