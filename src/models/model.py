"""
Neuro-Symbolic ASR Model for Dysarthric Speech Recognition

Combines HuBERT encoder (self-supervised neural representations) with symbolic
phoneme constraints for robust and explainable dysarthric speech recognition.

Architecture:
    1. HuBERT Encoder: Pretrained SSL model (95M params, 12 transformer layers)
    2. Phoneme Classifier: 768→512→num_phonemes projection
    3. Symbolic Constraint Layer: Articulatory-based phoneme similarity matrix
    4. Constraint Aggregation: Learnable neural-symbolic fusion (β parameter)
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel

try:
    from src.utils.config import ModelConfig, SymbolicConfig, normalize_phoneme
except ImportError:
    # Handle imports when called from different locations
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.config import ModelConfig, SymbolicConfig, normalize_phoneme


class ArticulatoryFeatureEncoder:
    """
    Encodes phonemes into articulatory feature vectors for symbolic reasoning.
    
    Features based on phonetic theory:
    - Manner of articulation: How airflow is obstructed (stop, fricative, nasal, etc.)
    - Place of articulation: Where obstruction occurs (bilabial, alveolar, velar, etc.)
    - Voicing: Vocal fold vibration (voiced vs voiceless)
    
    These features are preserved in dysarthric substitutions more than random phoneme
    changes, enabling principled constraint matrices.
    """
    
    PHONEME_FEATURES = {
        # STOPS
        'P': {'manner': 'stop', 'place': 'bilabial', 'voice': 'voiceless'},
        'B': {'manner': 'stop', 'place': 'bilabial', 'voice': 'voiced'},
        'T': {'manner': 'stop', 'place': 'alveolar', 'voice': 'voiceless'},
        'D': {'manner': 'stop', 'place': 'alveolar', 'voice': 'voiced'},
        'K': {'manner': 'stop', 'place': 'velar', 'voice': 'voiceless'},
        'G': {'manner': 'stop', 'place': 'velar', 'voice': 'voiced'},
        
        # FRICATIVES
        'F': {'manner': 'fricative', 'place': 'labiodental', 'voice': 'voiceless'},
        'V': {'manner': 'fricative', 'place': 'labiodental', 'voice': 'voiced'},
        'TH': {'manner': 'fricative', 'place': 'dental', 'voice': 'voiceless'},
        'DH': {'manner': 'fricative', 'place': 'dental', 'voice': 'voiced'},
        'S': {'manner': 'fricative', 'place': 'alveolar', 'voice': 'voiceless'},
        'Z': {'manner': 'fricative', 'place': 'alveolar', 'voice': 'voiced'},
        'SH': {'manner': 'fricative', 'place': 'postalveolar', 'voice': 'voiceless'},
        'ZH': {'manner': 'fricative', 'place': 'postalveolar', 'voice': 'voiced'},
        'HH': {'manner': 'fricative', 'place': 'glottal', 'voice': 'voiceless'},
        
        # AFFRICATES
        'CH': {'manner': 'affricate', 'place': 'postalveolar', 'voice': 'voiceless'},
        'JH': {'manner': 'affricate', 'place': 'postalveolar', 'voice': 'voiced'},
        
        # NASALS
        'M': {'manner': 'nasal', 'place': 'bilabial', 'voice': 'voiced'},
        'N': {'manner': 'nasal', 'place': 'alveolar', 'voice': 'voiced'},
        'NG': {'manner': 'nasal', 'place': 'velar', 'voice': 'voiced'},
        
        # LIQUIDS
        'L': {'manner': 'liquid', 'place': 'alveolar', 'voice': 'voiced'},
        'R': {'manner': 'liquid', 'place': 'alveolar', 'voice': 'voiced'},
        
        # GLIDES
        'W': {'manner': 'glide', 'place': 'labio-velar', 'voice': 'voiced'},
        'Y': {'manner': 'glide', 'place': 'palatal', 'voice': 'voiced'},
        
        # VOWELS (Monophthongs)
        'IY': {'manner': 'vowel', 'place': 'front', 'voice': 'voiced', 'height': 'high'},
        'IH': {'manner': 'vowel', 'place': 'front', 'voice': 'voiced', 'height': 'high'},
        'EH': {'manner': 'vowel', 'place': 'front', 'voice': 'voiced', 'height': 'mid'},
        'EY': {'manner': 'vowel', 'place': 'front', 'voice': 'voiced', 'height': 'mid'},
        'AE': {'manner': 'vowel', 'place': 'front', 'voice': 'voiced', 'height': 'low'},
        'AA': {'manner': 'vowel', 'place': 'back', 'voice': 'voiced', 'height': 'low'},
        'AO': {'manner': 'vowel', 'place': 'back', 'voice': 'voiced', 'height': 'mid'},
        'OW': {'manner': 'vowel', 'place': 'back', 'voice': 'voiced', 'height': 'mid'},
        'UH': {'manner': 'vowel', 'place': 'back', 'voice': 'voiced', 'height': 'high'},
        'UW': {'manner': 'vowel', 'place': 'back', 'voice': 'voiced', 'height': 'high'},
        'AH': {'manner': 'vowel', 'place': 'central', 'voice': 'voiced', 'height': 'mid'},
        'ER': {'manner': 'vowel', 'place': 'central', 'voice': 'voiced', 'height': 'mid'},
        'AX': {'manner': 'vowel', 'place': 'central', 'voice': 'voiced', 'height': 'mid'},
        
        # DIPHTHONGS
        'AY': {'manner': 'diphthong', 'place': 'front', 'voice': 'voiced'},
        'AW': {'manner': 'diphthong', 'place': 'back', 'voice': 'voiced'},
        'OY': {'manner': 'diphthong', 'place': 'back', 'voice': 'voiced'},
    }
    
    @classmethod
    def compute_distance(
        cls, 
        ph1: str, 
        ph2: str, 
        weights: Dict[str, float],
        decay_factor: float = 3.0
    ) -> float:
        """
        Compute articulatory similarity between two phonemes.
        
        Uses weighted Euclidean distance in articulatory feature space,
        then converts to similarity via exponential decay.
        
        Args:
            ph1: First phoneme symbol (stress-normalized)
            ph2: Second phoneme symbol (stress-normalized)
            weights: Feature weights dict with 'manner', 'place', 'voice' keys
            decay_factor: Exponential decay rate (higher = sharper falloff)
        
        Returns:
            Similarity score in [0, 1], where:
                - 1.0 = identical phonemes
                - 0.5 = same manner, different place/voice
                - <0.1 = articulatorily dissimilar
        
        Formula:
            distance = sqrt(Σ w_i * (f1_i ≠ f2_i)^2)
            similarity = exp(-decay_factor * distance)
        """
        if ph1 not in cls.PHONEME_FEATURES or ph2 not in cls.PHONEME_FEATURES:
            return 1.0  # Maximum distance for unknown phonemes
        
        f1, f2 = cls.PHONEME_FEATURES[ph1], cls.PHONEME_FEATURES[ph2]
        
        # Categorical distance (0 if same, 1 if different)
        manner_dist = 0.0 if f1['manner'] == f2['manner'] else 1.0
        place_dist = 0.0 if f1['place'] == f2['place'] else 1.0
        voice_dist = 0.0 if f1['voice'] == f2['voice'] else 1.0
        
        # Weighted Euclidean distance
        total_dist = np.sqrt(
            weights['manner'] ** 2 * manner_dist +
            weights['place'] ** 2 * place_dist +
            weights['voice'] ** 2 * voice_dist
        )
        
        # Convert distance to similarity (exponential decay)
        similarity = np.exp(-decay_factor * total_dist)
        return float(similarity)


class SymbolicConstraintLayer(nn.Module):
    """
    Neuro-symbolic layer applying dysarthric phoneme confusion rules.
    
    This layer fuses neural predictions with symbolic knowledge via a
    constraint matrix C that encodes:
    1. Explicit dysarthric substitution rules (from clinical literature)
    2. Articulatory similarity (for phoneme pairs without explicit rules)
    
    The fusion is controlled by learnable weight β:
        P_final = β * P_constrained + (1-β) * P_neural
    where β adapts based on speaker severity.
    
    Attributes:
        constraint_matrix: [num_phonemes, num_phonemes] similarity matrix
        beta: Learnable fusion weight (initialized to config value)
        rule_activations: List tracking which rules fired (if tracking enabled)
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
        
        # Rule activation tracking (for explainability)
        self.rule_activations = [] if symbolic_config.track_rule_activations else None
    
    def _build_constraint_matrix(self) -> torch.Tensor:
        """
        Build symbolic constraint matrix C.
        
        Matrix construction:
            C[i, i] = 1.0 (identity - correct phoneme)
            C[i, j] = substitution_rules[(i,j)] if rule exists
            C[i, j] = articulatory_distance(i,j) otherwise
        
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
                
                # Normalize to stress-agnostic form
                ph_i_clean = normalize_phoneme(ph_i)
                ph_j_clean = normalize_phoneme(ph_j)
                
                # Check for explicit substitution rule
                if (ph_i_clean, ph_j_clean) in self.config.substitution_rules:
                    C[i, j] = self.config.substitution_rules[(ph_i_clean, ph_j_clean)]
                else:
                    # Use articulatory similarity
                    similarity = ArticulatoryFeatureEncoder.compute_distance(
                        ph_i_clean, 
                        ph_j_clean, 
                        weights,
                        decay_factor=self.config.distance_decay_factor
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
                - logits: Constrained logits [batch, time, num_phonemes]
                - beta: Current constraint weight (scalar or [batch, 1, 1])
                - P_neural: Neural predictions (probabilities)
                - P_constrained: Constrained predictions (probabilities)
                - rule_activations: List of activated rules (if tracking enabled)
        
        Forward Pass Logic:
            1. Convert logits to probabilities: P_neural = softmax(logits)
            2. Apply constraints: P_constrained = C @ P_neural
            3. Adaptive fusion: P_final = β(severity) * P_c + (1-β) * P_n
            4. Convert back to logits for loss computation
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
        
        # Track rule activations (for explainability)
        if self.rule_activations is not None:
            self._track_activations(P_neural, P_final, beta_adaptive)
        
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
        
        Hypothesis: Higher dysarthria severity benefits more from symbolic
        constraints (articulatory patterns preserved even in severe impairment).
        
        Formula:
            β_adaptive = β_base + 0.1 * (severity / 5.0)
            Clamped to [0.0, 0.8] to prevent over-reliance on rules
        """
        # Normalize severity to [0, 1]
        severity_normalized = speaker_severity.float() / 5.0
        
        # Adaptive adjustment: higher severity → higher β
        beta_adaptive = self.beta + 0.1 * severity_normalized
        
        # Clamp to prevent extreme values
        beta_adaptive = torch.clamp(beta_adaptive, 0.0, 0.8)
        
        return beta_adaptive.view(-1, 1, 1)  # [batch, 1, 1] for broadcasting
    
    def _track_activations(
        self,
        P_neural: torch.Tensor,
        P_final: torch.Tensor,
        beta: torch.Tensor
    ) -> None:
        """
        Track which symbolic rules were activated during forward pass.
        
        A rule is "activated" when the constrained prediction differs
        significantly from the neural prediction.
        
        Args:
            P_neural: Neural probabilities [batch, time, num_phonemes]
            P_final: Final probabilities [batch, time, num_phonemes]
            beta: Fusion weight [batch, 1, 1] or scalar
        
        Stores:
            rule_activations: List of dicts with:
                - timestep: Frame index
                - neural_pred: Neural top-1 phoneme
                - final_pred: Final top-1 phoneme
                - beta: Constraint weight used
                - prob_shift: Change in probability
        """
        if self.rule_activations is None:
            return
        
        # Get top-1 predictions
        neural_preds = torch.argmax(P_neural, dim=-1)  # [batch, time]
        final_preds = torch.argmax(P_final, dim=-1)    # [batch, time]
        
        # Find frames where predictions changed
        changed = neural_preds != final_preds
        
        # Extract beta value (handle both tensor and scalar)
        beta_val = beta.item() if isinstance(beta, torch.Tensor) and beta.numel() == 1 else beta
        
        # Log activations (limit to prevent memory bloat)
        if len(self.rule_activations) < 10000:  # Max 10K activations
            for b in range(P_neural.size(0)):
                for t in range(P_neural.size(1)):
                    if changed[b, t]:
                        neural_id = neural_preds[b, t].item()
                        final_id = final_preds[b, t].item()
                        
                        self.rule_activations.append({
                            'batch': b,
                            'timestep': t,
                            'neural_pred': self.id_to_phn.get(neural_id, '<UNK>'),
                            'final_pred': self.id_to_phn.get(final_id, '<UNK>'),
                            'beta': float(beta_val) if not isinstance(beta_val, torch.Tensor) else float(beta_val[b].item()),
                            'prob_shift': float((P_final[b, t, final_id] - P_neural[b, t, final_id]).item())
                        })
    
    def get_rule_statistics(self) -> Dict[str, any]:
        """
        Get statistics about rule activations for analysis.
        
        Returns:
            Dictionary with:
                - total_activations: Total number of rule firings
                - unique_rules: Set of unique (neural→final) substitutions
                - top_rules: Most frequent substitution patterns
                - avg_beta: Average β value when rules fired
        """
        if self.rule_activations is None or len(self.rule_activations) == 0:
            return {
                'total_activations': 0,
                'unique_rules': set(),
                'top_rules': [],
                'avg_beta': 0.0
            }
        
        # Count substitution patterns
        substitutions = defaultdict(int)
        betas = []
        
        for activation in self.rule_activations:
            key = (activation['neural_pred'], activation['final_pred'])
            substitutions[key] += 1
            betas.append(activation['beta'])
        
        # Get top-10 most frequent rules
        top_rules = sorted(substitutions.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_activations': len(self.rule_activations),
            'unique_rules': set(substitutions.keys()),
            'top_rules': top_rules,
            'avg_beta': float(np.mean(betas)) if betas else 0.0
        }
    
    def clear_activations(self) -> None:
        """Clear rule activation history (call between epochs)."""
        if self.rule_activations is not None:
            self.rule_activations = []


class PhonemeClassifier(nn.Module):
    """
    Phoneme prediction head for frame-level classification.
    
    Architecture: 768 → 512 → num_phonemes
        - Projection layer reduces HuBERT dimension
        - LayerNorm for training stability
        - GELU activation (smooth, non-saturating)
        - Dropout for regularization
        - Linear classifier to phoneme vocabulary
    """
    
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
    
    This architecture enables:
    - Robust phoneme recognition via SSL pretraining
    - Explainable predictions via articulatory constraints
    - Adaptive fusion based on dysarthria severity
    
    Training Strategy:
    - Freeze encoder initially (warmup epochs)
    - Unfreeze upper layers progressively
    - Multi-task learning: CTC + frame-level CE
    
    Args:
        model_config: Neural architecture configuration
        symbolic_config: Symbolic reasoning configuration
        phn_to_id: Phoneme → ID mapping
        id_to_phn: ID → Phoneme reverse mapping
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
        
        # Enable gradient checkpointing (reduces activation memory by ~40%)
        try:
            self.hubert.gradient_checkpointing_enable()
            print("   ✅ Gradient checkpointing enabled")
        except Exception:
            # Fallback for older transformers versions
            if hasattr(self.hubert, "config"):
                setattr(self.hubert.config, "gradient_checkpointing", True)
                print("   ✅ Gradient checkpointing enabled (legacy mode)")
        
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
        
        trainable_params = self.count_parameters()
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ Model initialized:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    def _configure_frozen_layers(self) -> None:
        """
        Configure which HuBERT layers to freeze.
        
        Freezing Strategy:
        - Feature extractor (CNN): Always frozen (low-level acoustics)
        - Encoder layers 0-7: Frozen initially (generic phonetics)
        - Encoder layers 8-11: Trainable (dysarthria-specific adaptation)
        
        Rationale:
            Lower layers learn universal acoustic features (formants, VOT, etc.)
            that are shared across typical and dysarthric speech. Upper layers
            need to adapt to dysarthric substitution patterns.
        """
        # Freeze feature extractor (CNN layers)
        if self.model_config.freeze_feature_extractor:
            for param in self.hubert.feature_extractor.parameters():
                param.requires_grad = False
            print(f"   🧊 Froze feature extractor (CNN layers)")
        
        # Freeze specific encoder layers
        for layer_idx in self.model_config.freeze_encoder_layers:
            for param in self.hubert.encoder.layers[layer_idx].parameters():
                param.requires_grad = False
        
        if self.model_config.freeze_encoder_layers:
            print(f"   🧊 Froze encoder layers {self.model_config.freeze_encoder_layers}")
    
    def count_parameters(self) -> int:
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_severity: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through neuro-symbolic architecture.
        
        Forward Flow:
            Waveform → HuBERT → Phoneme Classifier → Symbolic Constraints → Logits
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
        """Freeze entire HuBERT encoder for warmup training."""
        for param in self.hubert.parameters():
            param.requires_grad = False
        print("🧊 Froze entire HuBERT encoder")
    
    def unfreeze_encoder(self, layers: Optional[List[int]] = None) -> None:
        """
        Unfreeze specific encoder layers for progressive fine-tuning.
        
        Args:
            layers: List of layer indices to unfreeze, or None for all layers
        """
        if layers is None:
            # Unfreeze all encoder layers
            for param in self.hubert.encoder.parameters():
                param.requires_grad = True
            print("🔥 Unfroze all encoder layers")
        else:
            # Unfreeze specific layers
            for layer_idx in layers:
                for param in self.hubert.encoder.layers[layer_idx].parameters():
                    param.requires_grad = True
            print(f"🔥 Unfroze encoder layers {layers}")
    
    def _unfreeze_all_hubert(self) -> None:
        """Helper to unfreeze the full HuBERT stack before reapplying freezes."""
        for param in self.hubert.parameters():
            param.requires_grad = True
    
    def unfreeze_after_warmup(self) -> None:
        """
        Unfreeze encoder after warmup while keeping configured frozen layers.
        
        This implements progressive unfreezing:
        1. Unfreeze all HuBERT layers
        2. Re-apply configured freezes (feature extractor + specified layers)
        
        Effect: Unfreezes upper encoder layers while keeping lower layers frozen.
        """
        self._unfreeze_all_hubert()
        self._configure_frozen_layers()
        print("🔥 Unfroze HuBERT encoder (keeping configured frozen layers)")
    
    def get_rule_statistics(self) -> Dict:
        """Get symbolic layer rule activation statistics."""
        return self.symbolic_layer.get_rule_statistics()
    
    def clear_rule_activations(self) -> None:
        """Clear symbolic layer rule activation history."""
        self.symbolic_layer.clear_activations()


def main() -> None:
    """Test model module."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.config import get_default_config
    
    config = get_default_config()
    
    # Mock vocabulary
    phn_to_id = {'<BLANK>': 0, '<PAD>': 1, '<UNK>': 2, 'P': 3, 'B': 4, 'T': 5, 'D': 6}
    id_to_phn = {v: k for k, v in phn_to_id.items()}
    
    print("\nTesting NeuroSymbolicASR Model")
    
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
    dummy_severity = torch.tensor([0.0, 5.0])  # Control vs severe
    
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(
            dummy_input, 
            attention_mask=dummy_mask,
            speaker_severity=dummy_severity
        )
    
    print(f"\n✅ Forward pass successful!")
    print(f"   Logits (neural) shape: {outputs['logits_neural'].shape}")
    print(f"   Logits (constrained) shape: {outputs['logits_constrained'].shape}")
    print(f"   Hidden states shape: {outputs['hidden_states'].shape}")
    
    beta_val = outputs['beta']
    if isinstance(beta_val, torch.Tensor):
        if beta_val.numel() > 1:
            print(f"   Beta (per sample): {beta_val.squeeze().tolist()}")
        else:
            print(f"   Beta: {beta_val.item():.3f}")
    else:
        print(f"   Beta: {beta_val:.3f}")
    
    # Test rule statistics
    print("\nRule activation statistics:")
    stats = model.get_rule_statistics()
    print(f"   Total activations: {stats['total_activations']}")
    print(f"   Unique rules: {len(stats['unique_rules'])}")
    print(f"   Average beta: {stats['avg_beta']:.3f}")

if __name__ == "__main__":
    main()
