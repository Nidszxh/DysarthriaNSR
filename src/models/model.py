"""
Neuro-Symbolic ASR Model for Dysarthric Speech Recognition

Combines HuBERT encoder (self-supervised neural representations) with symbolic
phoneme constraints for robust and explainable dysarthric speech recognition.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel

try:
    from utils.config import ModelConfig, SymbolicConfig
except ImportError:
    # Handle imports when called from different locations
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.config import ModelConfig, SymbolicConfig


class ArticulatoryFeatureEncoder:
    """Encodes phonemes into articulatory feature vectors for symbolic reasoning."""
    
    PHONEME_FEATURES = {
        # Stops
        'P': {'manner': 'stop', 'place': 'bilabial', 'voice': 'voiceless'},
        'B': {'manner': 'stop', 'place': 'bilabial', 'voice': 'voiced'},
        'T': {'manner': 'stop', 'place': 'alveolar', 'voice': 'voiceless'},
        'D': {'manner': 'stop', 'place': 'alveolar', 'voice': 'voiced'},
        'K': {'manner': 'stop', 'place': 'velar', 'voice': 'voiceless'},
        'G': {'manner': 'stop', 'place': 'velar', 'voice': 'voiced'},
        
        # Fricatives
        'F': {'manner': 'fricative', 'place': 'labiodental', 'voice': 'voiceless'},
        'V': {'manner': 'fricative', 'place': 'labiodental', 'voice': 'voiced'},
        'TH': {'manner': 'fricative', 'place': 'dental', 'voice': 'voiceless'},
        'DH': {'manner': 'fricative', 'place': 'dental', 'voice': 'voiced'},
        'S': {'manner': 'fricative', 'place': 'alveolar', 'voice': 'voiceless'},
        'Z': {'manner': 'fricative', 'place': 'alveolar', 'voice': 'voiced'},
        'SH': {'manner': 'fricative', 'place': 'postalveolar', 'voice': 'voiceless'},
        'ZH': {'manner': 'fricative', 'place': 'postalveolar', 'voice': 'voiced'},
        'HH': {'manner': 'fricative', 'place': 'glottal', 'voice': 'voiceless'},
        
        # Affricates
        'CH': {'manner': 'affricate', 'place': 'postalveolar', 'voice': 'voiceless'},
        'JH': {'manner': 'affricate', 'place': 'postalveolar', 'voice': 'voiced'},
        
        # Nasals
        'M': {'manner': 'nasal', 'place': 'bilabial', 'voice': 'voiced'},
        'N': {'manner': 'nasal', 'place': 'alveolar', 'voice': 'voiced'},
        'NG': {'manner': 'nasal', 'place': 'velar', 'voice': 'voiced'},
        
        # Liquids
        'L': {'manner': 'liquid', 'place': 'alveolar', 'voice': 'voiced'},
        'R': {'manner': 'liquid', 'place': 'alveolar', 'voice': 'voiced'},
        
        # Glides
        'W': {'manner': 'glide', 'place': 'labio-velar', 'voice': 'voiced'},
        'Y': {'manner': 'glide', 'place': 'palatal', 'voice': 'voiced'},
        
        # Vowels
        'IY': {'manner': 'vowel', 'place': 'front', 'voice': 'voiced'},
        'IH': {'manner': 'vowel', 'place': 'front', 'voice': 'voiced'},
        'EH': {'manner': 'vowel', 'place': 'front', 'voice': 'voiced'},
        'EY': {'manner': 'vowel', 'place': 'front', 'voice': 'voiced'},
        'AE': {'manner': 'vowel', 'place': 'front', 'voice': 'voiced'},
        'AA': {'manner': 'vowel', 'place': 'back', 'voice': 'voiced'},
        'AO': {'manner': 'vowel', 'place': 'back', 'voice': 'voiced'},
        'OW': {'manner': 'vowel', 'place': 'back', 'voice': 'voiced'},
        'UH': {'manner': 'vowel', 'place': 'back', 'voice': 'voiced'},
        'UW': {'manner': 'vowel', 'place': 'back', 'voice': 'voiced'},
        'AH': {'manner': 'vowel', 'place': 'central', 'voice': 'voiced'},
        'ER': {'manner': 'vowel', 'place': 'central', 'voice': 'voiced'},
        'AX': {'manner': 'vowel', 'place': 'central', 'voice': 'voiced'},
        
        # Diphthongs
        'AY': {'manner': 'diphthong', 'place': 'front', 'voice': 'voiced'},
        'AW': {'manner': 'diphthong', 'place': 'back', 'voice': 'voiced'},
        'OY': {'manner': 'diphthong', 'place': 'back', 'voice': 'voiced'},
    }
    
    @classmethod
    def compute_distance(cls, ph1: str, ph2: str, weights: Dict[str, float]) -> float:
        """
        Compute articulatory distance between two phonemes.
        
        Args:
            ph1: First phoneme symbol
            ph2: Second phoneme symbol
            weights: Dictionary with 'manner', 'place', 'voice' weights
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        if ph1 not in cls.PHONEME_FEATURES or ph2 not in cls.PHONEME_FEATURES:
            return 1.0  # Maximum distance for unknown phonemes
        
        f1, f2 = cls.PHONEME_FEATURES[ph1], cls.PHONEME_FEATURES[ph2]
        
        # Categorical distance (0 if same, 1 if different)
        manner_dist = 0.0 if f1['manner'] == f2['manner'] else 1.0
        place_dist = 0.0 if f1['place'] == f2['place'] else 1.0
        voice_dist = 0.0 if f1['voice'] == f2['voice'] else 1.0
        
        # Weighted sum
        total_dist = (
            weights['manner'] * manner_dist +
            weights['place'] * place_dist +
            weights['voice'] * voice_dist
        )
        
        # Convert distance to similarity (exponential decay)
        similarity = np.exp(-3.0 * total_dist)
        return similarity


class SymbolicConstraintLayer(nn.Module):
    """
    Neuro-symbolic layer applying dysarthric phoneme confusion rules.
    Fuses neural predictions with symbolic knowledge.
    """
    
    def __init__(
        self, 
        num_phonemes: int,
        phn_to_id: Dict[str, int],
        id_to_phn: Dict[int, str],
        symbolic_config: SymbolicConfig,
        constraint_weight: float = 0.3,
        learnable: bool = True
    ):
        super().__init__()
        self.num_phonemes = num_phonemes
        self.phn_to_id = phn_to_id
        self.id_to_phn = id_to_phn
        self.config = symbolic_config
        
        # Learnable fusion weight β
        if learnable:
            self.beta = nn.Parameter(torch.tensor(constraint_weight))
        else:
            self.register_buffer('beta', torch.tensor(constraint_weight))
        
        # Build constraint matrix C[i,j] = similarity/substitution probability
        self.register_buffer('constraint_matrix', self._build_constraint_matrix())
    
    def _build_constraint_matrix(self) -> torch.Tensor:
        """
        Build symbolic constraint matrix C where:
        - C[i,i] = 1.0 (identity - correct phoneme)
        - C[i,j] = substitution_rules[(i,j)] if rule exists
        - C[i,j] = articulatory_distance(i,j) otherwise
        
        Returns:
            Constraint matrix of shape [num_phonemes, num_phonemes]
        """
        C = torch.eye(self.num_phonemes)
        
        # Articulatory distance weights
        weights = {
            'manner': self.config.manner_weight,
            'place': self.config.place_weight,
            'voice': self.config.voice_weight
        }
        
        for i in range(self.num_phonemes):
            for j in range(self.num_phonemes):
                if i == j:
                    continue
                
                ph_i = self.id_to_phn.get(i, '<UNK>')
                ph_j = self.id_to_phn.get(j, '<UNK>')
                
                # Remove stress markers (0, 1, 2) from ARPABET phonemes
                ph_i_clean = ph_i.rstrip('012')
                ph_j_clean = ph_j.rstrip('012')
                
                # Check for explicit substitution rule
                if (ph_i_clean, ph_j_clean) in self.config.substitution_rules:
                    C[i, j] = self.config.substitution_rules[(ph_i_clean, ph_j_clean)]
                else:
                    # Use articulatory similarity
                    similarity = ArticulatoryFeatureEncoder.compute_distance(
                        ph_i_clean, ph_j_clean, weights
                    )
                    C[i, j] = similarity
        
        return C
    
    def forward(
        self, 
        logits: torch.Tensor,
        speaker_severity: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply symbolic constraints to neural phoneme predictions.
        
        Args:
            logits: Neural logits [batch, time, num_phonemes]
            speaker_severity: Speaker severity scores (0-5) [batch] (optional)
        
        Returns:
            Dictionary containing:
                - logits: Constrained logits
                - beta: Current constraint weight
                - P_neural: Neural predictions (probabilities)
                - P_constrained: Constrained predictions (probabilities)
        """
        # Neural predictions
        P_neural = F.softmax(logits, dim=-1)  # [batch, time, num_phonemes]
        
        # Apply symbolic constraints via matrix multiplication
        # P_constrained[b,t,j] = Σ_i P_neural[b,t,i] * C[i,j]
        P_constrained = torch.matmul(P_neural, self.constraint_matrix)
        
        # Adaptive fusion based on speaker severity (if provided)
        if speaker_severity is not None:
            beta_adaptive = self._compute_adaptive_beta(speaker_severity)
        else:
            beta_adaptive = self.beta
        
        # Weighted fusion
        P_final = beta_adaptive * P_constrained + (1 - beta_adaptive) * P_neural
        
        # Convert back to logits for loss computation
        logits_constrained = torch.log(P_final + 1e-8)
        
        return {
            'logits': logits_constrained,
            'beta': beta_adaptive.mean() if speaker_severity is not None else self.beta,
            'P_neural': P_neural,
            'P_constrained': P_constrained,
        }
    
    def _compute_adaptive_beta(
        self, 
        speaker_severity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive constraint weight based on speaker severity.
        Higher severity → rely more on symbolic constraints.
        
        Args:
            speaker_severity: Speaker severity scores [batch]
            
        Returns:
            Adaptive beta weights
        """
        # β_adaptive = β_base + 0.1 * (severity / 5.0)
        severity_normalized = speaker_severity.float() / 5.0
        beta_adaptive = self.beta + 0.1 * severity_normalized
        beta_adaptive = torch.clamp(beta_adaptive, 0.0, 0.8)  # Cap at 0.8
        return beta_adaptive.view(-1, 1, 1)  # [batch, 1, 1]


class PhonemeClassifier(nn.Module):
    """Phoneme prediction head."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Multi-layer classifier
        self.projection = nn.Linear(768, config.hidden_dim)  # HuBERT hidden = 768
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_dim, config.num_phonemes)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through phoneme classifier.
        
        Args:
            hidden_states: HuBERT hidden states [batch, time, 768]
        
        Returns:
            Phoneme logits [batch, time, num_phonemes]
        """
        x = self.projection(hidden_states)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


class NeuroSymbolicASR(nn.Module):
    """
    Complete Neuro-Symbolic ASR model combining:
    1. HuBERT encoder (self-supervised neural representations)
    2. Phoneme classifier (neural prediction head)
    3. Symbolic constraint layer (symbolic reasoning)
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        symbolic_config: SymbolicConfig,
        phn_to_id: Dict[str, int],
        id_to_phn: Dict[int, str]
    ):
        super().__init__()
        self.model_config = model_config
        self.symbolic_config = symbolic_config
        
        # Update num_phonemes from vocabulary
        model_config.num_phonemes = len(phn_to_id)
        
        # HuBERT Encoder
        print(f"🧠 Loading HuBERT: {model_config.hubert_model_id}")
        self.hubert = HubertModel.from_pretrained(model_config.hubert_model_id)
        # Enable gradient checkpointing to reduce activation memory
        try:
            self.hubert.gradient_checkpointing_enable()
        except Exception:
            # Fallback for older transformers versions
            if hasattr(self.hubert, "config"):
                setattr(self.hubert.config, "gradient_checkpointing", True)
        
        # Configure frozen layers
        self._configure_frozen_layers()
        
        # Phoneme Classifier Head
        self.phoneme_classifier = PhonemeClassifier(model_config)
        
        # Symbolic Constraint Layer
        self.symbolic_layer = SymbolicConstraintLayer(
            num_phonemes=model_config.num_phonemes,
            phn_to_id=phn_to_id,
            id_to_phn=id_to_phn,
            symbolic_config=symbolic_config,
            constraint_weight=model_config.constraint_weight_init,
            learnable=model_config.constraint_learnable
        )
        
        print(f"✅ Model initialized: {self.count_parameters():,} trainable parameters")
    
    def _configure_frozen_layers(self) -> None:
        """Configure which HuBERT layers to freeze."""
        # Freeze feature extractor (CNN layers)
        if self.model_config.freeze_feature_extractor:
            for param in self.hubert.feature_extractor.parameters():
                param.requires_grad = False
        
        # Freeze specific encoder layers
        for layer_idx in self.model_config.freeze_encoder_layers:
            for param in self.hubert.encoder.layers[layer_idx].parameters():
                param.requires_grad = False
    
    def count_parameters(self) -> int:
        """
        Count trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_severity: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through neuro-symbolic architecture.
        
        Args:
            input_values: Raw waveform features [batch, time]
            attention_mask: Mask for padded regions [batch, time]
            speaker_severity: Dysarthria severity scores [batch] (optional)
        
        Returns:
            Dictionary containing:
                - logits_neural: Neural predictions
                - logits_constrained: Constrained predictions
                - hidden_states: HuBERT hidden states
                - beta: Constraint weight
                - P_neural: Neural probabilities
                - P_constrained: Constrained probabilities
        """
        # HuBERT encoding
        hubert_outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract hidden states from last layer
        hidden_states = hubert_outputs.last_hidden_state  # [batch, time, 768]
        
        # Phoneme prediction (neural)
        logits_neural = self.phoneme_classifier(hidden_states)
        
        # Apply symbolic constraints
        symbolic_outputs = self.symbolic_layer(
            logits_neural,
            speaker_severity=speaker_severity
        )
        
        return {
            'logits_neural': logits_neural,
            'logits_constrained': symbolic_outputs['logits'],
            'hidden_states': hidden_states,
            'beta': symbolic_outputs['beta'],
            'P_neural': symbolic_outputs.get('P_neural'),
            'P_constrained': symbolic_outputs.get('P_constrained'),
        }
    
    def freeze_encoder(self) -> None:
        """Freeze entire HuBERT encoder for fine-tuning only the head."""
        for param in self.hubert.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self, layers: Optional[List[int]] = None) -> None:
        """
        Unfreeze specific encoder layers for gradual unfreezing.
        
        Args:
            layers: List of layer indices to unfreeze, or None for all layers
        """
        if layers is None:
            # Unfreeze all encoder layers
            for param in self.hubert.encoder.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific layers
            for layer_idx in layers:
                for param in self.hubert.encoder.layers[layer_idx].parameters():
                    param.requires_grad = True

    def _unfreeze_all_hubert(self) -> None:
        """Helper to unfreeze the full HuBERT stack before reapplying freezes."""
        for param in self.hubert.parameters():
            param.requires_grad = True

    def unfreeze_after_warmup(self) -> None:
        """Unfreeze encoder after warmup while keeping configured frozen layers."""
        self._unfreeze_all_hubert()
        self._configure_frozen_layers()


def main() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.config import get_default_config
    
    config = get_default_config()
    
    # Mock vocabulary
    phn_to_id = {'<BLANK>': 0, '<PAD>': 1, '<UNK>': 2, 'P': 3, 'B': 4, 'T': 5}
    id_to_phn = {v: k for k, v in phn_to_id.items()}
    
    model = NeuroSymbolicASR(
        model_config=config.model,
        symbolic_config=config.symbolic,
        phn_to_id=phn_to_id,
        id_to_phn=id_to_phn
    )
    
    # Test forward pass
    batch_size, seq_len = 2, 16000
    dummy_input = torch.randn(batch_size, seq_len)
    dummy_mask = torch.ones(batch_size, seq_len)
    
    outputs = model(dummy_input, attention_mask=dummy_mask)
    
    print(f"\n🔍 Output shapes:")
    print(f"Logits (neural): {outputs['logits_neural'].shape}")
    print(f"Logits (constrained): {outputs['logits_constrained'].shape}")
    beta_val = outputs['beta']
    if isinstance(beta_val, torch.Tensor):
        beta_val = beta_val.item()
    print(f"Beta (constraint weight): {beta_val:.3f}")


if __name__ == "__main__":
    main()