# Implementation Status & Architecture (Feb 2026)

## Current Status
- Core pipeline is implemented and runs end-to-end: download -> manifest -> dataloader -> training -> evaluation.
- Training uses PyTorch Lightning with MLflow logging and saves results under `checkpoints/` and `results/`.
- Baseline results (baseline_v1, Jan 2026) are recorded under `results/baseline_v1/`.

**Baseline Performance (baseline_v1 - Jan 2026)**
- Test PER: 0.567 ± 0.365 on 3,553 samples (3 TORGO speakers)
- Dysarthric PER: 0.541 | Control PER: 0.575
- Model: 94.8M params (416K trainable), HuBERT encoder with symbolic constraint layer

## Architecture and Data Flow
```
download.py  -> data/raw/audio + data/processed/raw_extraction_map.csv
manifest.py  -> data/processed/torgo_neuro_symbolic_manifest.csv
dataloader.py -> train.py -> evaluate.py (utilities)
```

## Data Pipeline (Implemented)
- `download.py`: pulls `abnerh/TORGO-database`, writes audio under `data/raw/audio/` using `{speaker}_{hash}_{original_name}` naming, and logs a raw extraction map.
- `manifest.py`: loads HF metadata with `Audio(decode=False)`, matches local audio, runs `g2p_en`, and writes phonemes + articulatory classes + audio metrics into the manifest.
- `dataloader.py`: loads audio from local paths, peak-normalizes, truncates to `max_audio_length`, and pads labels with `-100` for CTC/CE.

## Model (Implemented)
- HuBERT base (`facebook/hubert-base-ls960`) with gradient checkpointing.
- Feature extractor is always frozen; encoder layers in `ModelConfig.freeze_encoder_layers` stay frozen after warmup.
- Phoneme classifier head: 768 -> 512 -> vocab.
- Symbolic layer blends neural probabilities with articulatory similarity + substitution rules. Adaptive beta is severity-aware and clamped to 0.8.
- Optional articulatory heads (manner/place/voice) are trained when vocab is present.

## Training (Implemented)
- Losses: CTC + frame-level CE + articulatory CE (manner/place/voice).
- Default weights: `lambda_ctc=0.8`, `lambda_ce=0.2`, `lambda_articulatory=0.1`.
- Class balancing: inverse-frequency phoneme weights for CE + `WeightedRandomSampler` for dysarthric/control.
- CTC length guard drops samples where `label_lengths > input_lengths`.
- MLflow logging with flattened config parameters.

**Current defaults (see `src/utils/config.py`)**
- Learning rate: 3e-5, OneCycleLR warmup ratio 0.05
- Batch size: 4, gradient accumulation: 16
- Max audio length: 8.0s, mixed precision `16-mixed`
- Encoder warmup epochs: 5

## Evaluation (Implemented)
- `train.py` runs `evaluate_model()` after fitting and writes outputs to `results/{run_name}/`.
- `evaluate.py` provides PER/WER, bootstrap CI, confusion matrices, and beam search decoding utilities.

## Known Issues (Observed)
- High insertion rate in baseline evaluation suggests blank suppression in CTC.
- Dysarthric vs control PER differences likely need length-stratified analysis.

## Short-Term Diagnostic Plan (Actionable)
- Inspect blank posterior statistics (mean, histogram, blank vs non-blank ratio).
- Compare blank probability distributions for dysarthric vs control.
- Add insertion-sensitive decoding (length penalty or insertion penalty).
- Test blank-regularization (blank-weighted CE or blank prior KL).
- Run ablations: neural-only, symbolic-only, fixed beta sweeps.

**CTC Loss Details:**
- Allows variable-length alignment between input frames and phoneme sequence
- Handles repeated phonemes and blank insertions
- Beam search decoding with width=10 during inference

**Focal Loss Formulation:**
```
FL(p_t) = -α_t(1 - p_t)^γ log(p_t)

Where:
- p_t = predicted probability of true class
- α_t = class weight (inversely proportional to frequency)
- γ = focusing parameter (γ=2)
```

### 3.4 Phoneme-to-Frame Alignment

**Training Alignment:**
- Use forced alignment timestamps from TORGO
- Generate frame-level phoneme labels:
  ```
  Frame duration = 20ms (50Hz rate)
  Phoneme duration = {start_time, end_time}
  Frame labels = [phoneme_id] for all frames in [start_time, end_time]
  ```

**Inference Alignment:**
- CTC prefix beam search produces phoneme sequence + timestamps
- Post-process to merge repeated phonemes
- Smooth boundaries using median filtering

---

## 4. Neuro-Symbolic Constraint Layer

### 4.1 Symbolic Knowledge Representation

**Dysarthric Phoneme Confusion Rules:**

Based on linguistic research, encode common dysarthric substitution patterns:

```python
# Articulatory Feature Matrix
phoneme_features = {
    'B': {'manner': 'stop', 'place': 'bilabial', 'voice': 'voiced'},
    'P': {'manner': 'stop', 'place': 'bilabial', 'voice': 'voiceless'},
    'D': {'manner': 'stop', 'place': 'alveolar', 'voice': 'voiced'},
    # ... all 44 phonemes
}

# Dysarthric Substitution Rules
confusion_matrix = {
    ('B', 'P'): 0.85,  # High probability B→P (devoicing)
    ('D', 'T'): 0.82,  # High probability D→T (devoicing)
    ('S', 'TH'): 0.65, # Moderate S→TH (fronting)
    # ... derived from clinical phonetics literature
}

# Articulatory Similarity Function
def articulatory_distance(ph1, ph2):
    diff = sum([
        feature_diff(ph1['manner'], ph2['manner']) * w_manner,
        feature_diff(ph1['place'], ph2['place']) * w_place,
        feature_diff(ph1['voice'], ph2['voice']) * w_voice
    ])
    return exp(-diff)  # Convert to similarity
```

### 4.2 Constraint Integration Mechanism

**Architecture:**
```
Neural Phoneme Logits [batch, frames, 44]
    ↓
Softmax → P_neural
    ↓
    ├─────────────────────┐
    │                     │
    ↓                     ↓
Symbolic Constraint    Identity
Matrix [44, 44]         Path
    ↓                     │
P_constrained           │
    ↓                     ↓
    └──── Weighted Fusion ────┘
              ↓
    Final Phoneme Probs
```

**Mathematical Formulation:**

```python
# Step 1: Neural predictions
P_neural = softmax(logits)  # [batch, frames, 44]

# Step 2: Apply symbolic constraints
P_constrained = P_neural @ C  # Matrix multiplication with constraint matrix C

Where C[i,j] represents:
- C[i,i] = 1.0 (identity - correct phoneme)
- C[i,j] = confusion_matrix[(i,j)] if (i,j) in known confusions
- C[i,j] = articulatory_distance(i,j) otherwise

# Step 3: Fusion
P_final = β × P_constrained + (1-β) × P_neural

Where β (constraint weight) is:
- Learnable parameter (initialized to 0.3)
- Can be speaker-adaptive (higher for severe dysarthria)
```

### 4.3 Rule-Based Corrections

**Post-Processing Symbolic Rules:**

```python
def apply_phonetic_rules(phoneme_sequence, speaker_severity):
    rules = [
        # Rule 1: Voicing correction
        {
            'pattern': ['P', 'vowel', 'P'],
            'correction': ['B', 'vowel', 'B'],
            'condition': lambda: inter_vowel_position(),
            'confidence': 0.7
        },
        # Rule 2: Cluster simplification
        {
            'pattern': ['consonant_cluster'],
            'correction': ['single_consonant'],
            'condition': lambda: severity > 3,
            'confidence': 0.6
        },
        # ... more rules from clinical phonology
    ]
    
    for rule in rules:
        if match(phoneme_sequence, rule['pattern']) and rule['condition']():
            apply_correction(phoneme_sequence, rule, speaker_severity)
```

---

## 5. ASR Decoding Module

### 5.1 Phoneme-to-Word Decoder

**Architecture Choice: Hybrid CTC + Attention**

```
Phoneme Sequence (from phoneme model)
    ↓
Pronunciation Lexicon Lookup
    ↓
Word Hypotheses Generation
    ↓
Language Model Rescoring
    ↓
Final Word Sequence
```

### 5.2 Pronunciation Lexicon

**Dysarthria-Augmented Lexicon:**

```python
# Standard lexicon
lexicon = {
    'HELLO': ['HH', 'AH', 'L', 'OW'],
    'WORLD': ['W', 'ER', 'L', 'D'],
}

# Augmented with dysarthric variants
dysarthric_variants = {
    'HELLO': [
        ['HH', 'AH', 'L', 'OW'],      # Standard
        ['HH', 'AH', 'W', 'OW'],      # L→W substitution
        ['AH', 'L', 'OW'],            # Initial H deletion
        ['HH', 'AH', 'OW'],           # Cluster reduction (LL→L)
    ]
}
```

**Lexicon Construction:**
- Base: CMU Pronouncing Dictionary (125k words)
- Augmentation: Generate variants using confusion rules
- Weighting: Assign probabilities based on dysarthria severity

### 5.3 Decoding Algorithm

**Weighted Finite State Transducer (WFST) Decoding:**

```
Components:
1. H (HMM): Phoneme acoustic model
2. C (Context): Phoneme context dependencies
3. L (Lexicon): Phoneme-to-word mapping (augmented)
4. G (Grammar): N-gram language model

Composition: HCLG = H ∘ C ∘ L ∘ G

Search: Token-passing beam search
- Beam width: 16
- Acoustic weight: 0.7
- LM weight: 0.3 (adjustable)
```

**Beam Search Implementation:**

```python
def beam_search_decode(phoneme_probs, lexicon, lm, beam_width=16):
    """
    phoneme_probs: [frames, 44] probability matrix
    """
    beams = [{'sequence': [], 'score': 0.0, 'phonemes': []}]
    
    for t in range(len(phoneme_probs)):
        candidates = []
        for beam in beams:
            for ph_id, ph_prob in enumerate(phoneme_probs[t]):
                # Generate candidate
                new_seq = beam['phonemes'] + [ph_id]
                
                # Compute scores
                acoustic_score = log(ph_prob)
                lm_score = lm.score(beam['sequence'], new_seq, lexicon)
                total_score = beam['score'] + acoustic_score + lm_score
                
                candidates.append({
                    'sequence': update_word_sequence(beam, new_seq, lexicon),
                    'score': total_score,
                    'phonemes': new_seq
                })
        
        # Keep top-k beams
        beams = sorted(candidates, key=lambda x: x['score'], reverse=True)[:beam_width]
    
    return beams[0]['sequence']  # Return best hypothesis
```

### 5.4 Language Model Integration

**N-gram LM Training:**
- Train on general text corpus (LibriSpeech transcripts)
- Fine-tune on TORGO transcripts (limited data → importance of general LM)
- Model: 3-gram with Kneser-Ney smoothing

**Rescoring with Neural LM (Optional):**
- Generate N-best hypotheses (N=10) from beam search
- Rescore with LSTM or Transformer LM
- Select best hypothesis based on combined score

---

## 6. Explainability Module

### 6.1 Phoneme Attribution Analysis

**Goal:** Identify which phonemes caused ASR errors

**Method 1: Alignment-Based Attribution**
```python
def phoneme_error_attribution(predicted_phonemes, ground_truth_phonemes):
    # Step 1: Align sequences using edit distance
    alignment = levenshtein_alignment(predicted_phonemes, ground_truth_phonemes)
    
    # Step 2: Categorize errors
    errors = {
        'substitutions': [],  # [(predicted, expected, position)]
        'deletions': [],      # [(expected, position)]
        'insertions': []      # [(predicted, position)]
    }
    
    for op, ph_pred, ph_true, pos in alignment:
        if op == 'substitute':
            errors['substitutions'].append((ph_pred, ph_true, pos))
        elif op == 'delete':
            errors['deletions'].append((ph_true, pos))
        elif op == 'insert':
            errors['insertions'].append((ph_pred, pos))
    
    # Step 3: Map to articulatory features
    feature_errors = analyze_feature_patterns(errors)
    return errors, feature_errors
```

**Method 2: Attention-Based Attribution**
```python
# Extract attention weights from transformer layers
attention_weights = model.get_attention_weights()  # [layers, heads, frames, frames]

# Aggregate attention to phoneme boundaries
phoneme_attention = aggregate_attention_to_phonemes(
    attention_weights, 
    phoneme_boundaries
)

# Identify high-attention phonemes in error regions
error_phoneme_attention = phoneme_attention[error_frames]
```

### 6.2 Symbolic Rule Activation Tracking

**Rule Logging System:**

```python
class SymbolicRuleTracker:
    def __init__(self):
        self.activations = []
    
    def log_rule_activation(self, rule_id, input_context, output_correction, confidence):
        self.activations.append({
            'rule_id': rule_id,
            'input': input_context,
            'output': output_correction,
            'confidence': confidence,
            'timestamp': current_frame()
        })
    
    def generate_explanation(self):
        explanation = {
            'total_rules_fired': len(self.activations),
            'high_confidence_corrections': [
                a for a in self.activations if a['confidence'] > 0.7
            ],
            'rule_frequency': Counter([a['rule_id'] for a in self.activations])
        }
        return explanation
```

### 6.3 Feature-Level Error Analysis

**Articulatory Feature Confusion Matrix:**

```python
# Build confusion matrix for each feature dimension
feature_dims = ['manner', 'place', 'voice']

for feature in feature_dims:
    confusion[feature] = defaultdict(lambda: defaultdict(int))
    
    for (pred_ph, true_ph, pos) in substitution_errors:
        pred_val = phoneme_features[pred_ph][feature]
        true_val = phoneme_features[true_ph][feature]
        confusion[feature][true_val][pred_val] += 1

# Visualization: Heatmap for each feature dimension
plot_confusion_heatmap(confusion['manner'], title='Manner of Articulation Errors')
plot_confusion_heatmap(confusion['place'], title='Place of Articulation Errors')
plot_confusion_heatmap(confusion['voice'], title='Voicing Errors')
```

### 6.4 Explainable Output Format

**Structured Explanation JSON:**

```json
{
  "utterance_id": "M01_session1_sentence5",
  "ground_truth": "THE QUICK BROWN FOX",
  "prediction": "THE QICK BOWN FOX",
  "wer": 0.25,
  "phoneme_analysis": {
    "errors": [
      {
        "type": "deletion",
        "position": 5,
        "expected_phoneme": "W",
        "context": "K_IH_[W]_IH_K",
        "articulatory_features": {
          "manner": "approximant",
          "place": "labio-velar",
          "voice": "voiced"
        },
        "probable_cause": "Cluster simplification (KW→K)",
        "symbolic_rule_activated": "rule_42_cluster_reduction"
      },
      {
        "type": "substitution",
        "position": 12,
        "predicted_phoneme": "W",
        "expected_phoneme": "R",
        "feature_differences": {
          "place": "labio-velar vs. alveolar"
        },
        "probable_cause": "Liquid gliding (R→W)",
        "neural_confidence": 0.62,
        "symbolic_correction_applied": true
      }
    ]
  },
  "symbolic_rules_summary": {
    "total_fired": 3,
    "rule_details": [
      {"rule_id": 42, "name": "cluster_reduction", "confidence": 0.85},
      {"rule_id": 17, "name": "liquid_gliding", "confidence": 0.70}
    ]
  },
  "attention_visualization": "base64_encoded_heatmap"
}
```

---

## 7. Training Pipeline

### 7.1 Training Data Split

```
TORGO Dataset Split:
├── Train: 70% (stratified by speaker severity)
├── Validation: 15% (different speakers from train)
└── Test: 15% (held-out speakers, unseen during training)

Cross-Validation Strategy:
- Leave-One-Speaker-Out (LOSO) for generalization testing
- Ensures model doesn't overfit to specific dysarthric patterns
```

### 7.2 Training Hyperparameters

```python
training_config = {
    # Optimization
    'optimizer': 'AdamW',
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'lr_scheduler': 'OneCycleLR',
    'warmup_ratio': 0.1,  # 10% of total steps
    
    # Training dynamics (VRAM-optimized for RTX 4060 8GB)
    'batch_size': 1,  # Further reduced from 2
    'gradient_accumulation': 8,  # Effective batch size = 8
    'max_epochs': 5,  # For experimentation; increase for convergence
    'early_stopping_patience': 10,
    
    # Regularization
    'dropout': 0.1,
    'layer_dropout': 0.05,  # Drop entire transformer layers
    'label_smoothing': 0.1,
    
    # Loss weights (fixed, not dynamic)
    'lambda_ctc': 0.7,  # Primary: phoneme alignment
    'lambda_ce': 0.3,   # Auxiliary: frame-level classification
    
    # Precision & Memory
    'precision': '16-mixed',  # Mixed precision for GPU efficiency
    'gradient_checkpointing': True,  # Trade compute for memory
    'max_audio_length': 8.0,  # Truncate audio to 8 seconds
}
```

**Memory Optimizations in Detail**:
- **Gradient Checkpointing** ([src/models/model.py](src/models/model.py#L325)): Recompute activations during backward pass instead of storing them. Reduces memory by ~40% at cost of ~20% slower training.
- **Audio Truncation** ([src/data/dataloader.py](src/data/dataloader.py#L168)): Limit input length to 8 seconds (~128k samples at 16kHz). Prevents pathological long-sequence memory spikes.
- **Frozen Encoder Layers**: Freeze first 8 of 12 HuBERT layers ([src/models/model.py](src/models/model.py#L352)). Low-level acoustic features don't need fine-tuning for dysarthria.
- **Mixed Precision (fp16)**: Use 16-bit floats for activations/weights, 32-bit for loss computation. PyTorch Lightning handles this automatically.
- **CUDA Fragmentation Mitigation** ([train.py](train.py#L551)): Set `PYTORCH_ALLOC_CONF=expandable_segments:True` to allow dynamic memory pool resizing.

### 7.3 Training Loop

```python
for epoch in range(max_epochs):
    for batch in train_loader:
        # Forward pass
        audio, phoneme_labels, word_labels = batch
        
        # 1. HuBERT encoding
        hidden_states = hubert_model(audio)
        
        # 2. Phoneme prediction
        phoneme_logits = phoneme_classifier(hidden_states)
        
        # 3. Apply symbolic constraints
        phoneme_probs = symbolic_layer(phoneme_logits, speaker_severity)
        
        # 4. Compute multi-task loss
        loss_ctc = ctc_loss(phoneme_probs, phoneme_labels)
        loss_ce = cross_entropy(phoneme_logits, phoneme_labels)
        loss_focal = focal_loss(phoneme_logits, phoneme_labels)
        loss_symbolic = constraint_violation_loss(phoneme_probs)
        
        total_loss = (lambda_ctc * loss_ctc + 
                      lambda_ce * loss_ce + 
                      lambda_focal * loss_focal +
                      lambda_symbolic * loss_symbolic)
        
        # 5. Backward pass
        total_loss.backward()
        
        # 6. Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 7. Optimizer step
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation
    val_metrics = evaluate(model, val_loader)
    
    # Early stopping
    if val_metrics['wer'] < best_wer:
        best_wer = val_metrics['wer']
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            break
```

---

## 8. Evaluation Metrics

### 8.1 ASR Performance Metrics

**Word Error Rate (WER):**
```python
WER = (S + D + I) / N

Where:
- S = number of substitutions
- D = number of deletions
- I = number of insertions
- N = total words in reference
```

**Phoneme Error Rate (PER):**
```python
PER = (S_ph + D_ph + I_ph) / N_ph
```

**Character Error Rate (CER):**
- Useful for partial credit in dysarthric speech

### 8.2 Dysarthria-Specific Metrics

**Severity-Stratified WER:**
```python
wer_by_severity = {
    'mild': compute_wer(predictions[severity=='mild']),
    'moderate': compute_wer(predictions[severity=='moderate']),
    'severe': compute_wer(predictions[severity=='severe'])
}
```

**Intelligibility Correlation:**
- Pearson correlation between predicted WER and clinical intelligibility scores

### 8.3 Explainability Metrics

**Rule Precision:**
```python
rule_precision = (correct_rule_applications) / (total_rule_applications)
```

**Attribution Consistency:**
- Inter-rater agreement between model explanations and expert phonetic analysis

### 8.4 Stratified Evaluation and Statistical Testing

**Length-Stratified PER:**
- Buckets: 0-5, 6-10, 11-20, 21+ phonemes
- Report mean PER with bootstrap confidence intervals per bucket

**Speaker-Stratified PER:**
- Per-speaker PER with mean and std
- Compare dysarthric vs control with statistical testing

**Statistical Tests:**
- Paired tests for model comparisons (paired t-test or Wilcoxon)
- Group tests for dysarthric vs control (Welch t-test or Mann-Whitney U)
- Correct for multiple comparisons (Holm or Bonferroni)

---

## 9. System Integration & Deployment

### 9.1 Inference Pipeline

```python
def inference_pipeline(audio_file, speaker_severity=None):
    # 1. Preprocessing
    audio = load_audio(audio_file, target_sr=16000)
    audio = normalize(audio)
    
    # 2. HuBERT encoding
    with torch.no_grad():
        hidden_states = hubert_model(audio)
    
    # 3. Phoneme prediction
    phoneme_logits = phoneme_classifier(hidden_states)
    phoneme_probs = symbolic_layer(phoneme_logits, speaker_severity)
    
    # 4. CTC decoding → phoneme sequence
    phoneme_sequence = ctc_decode(phoneme_probs)
    
    # 5. Lexicon lookup + beam search
    word_sequence = beam_search_decode(phoneme_sequence, lexicon, lm)
    
    # 6. Generate explanations
    explanations = explainability_module.analyze(
        phoneme_sequence, 
        ground_truth=None,  # If available
        attention_weights=hidden_states
    )
    
    return {
        'transcription': ' '.join(word_sequence),
        'phonemes': phoneme_sequence,
        'explanations': explanations
    }
```

### 9.2 Model Serving Architecture

```
Client Request (Audio)
    ↓
API Gateway (FastAPI)
    ↓

### 9.3 ONNX Export Checklist

1. Export HuBERT encoder + classifier + symbolic layer with dynamic axes.
2. Validate ONNX parity on a small batch (logits + PER spot check).
3. Run ONNX Runtime CPU inference for a short sample.
4. Optional: keep CTC decoding outside ONNX to avoid dynamic control flow.
5. Package model and inference config for deployment.
Preprocessing Service
    ↓
GPU Inference Service (HuBERT + Phoneme Model)
    ↓
Decoding Service (CPU-based beam search)
    ↓
Explainability Service
    ↓
Response (Transcription + Explanations)
```

---

## Implementation Checklist

### ✅ Data & Preprocessing
- [x] TORGO dataset download via HuggingFace ([download.py](download.py))
- [x] Manifest generation with phoneme extraction ([manifest.py](manifest.py))
- [x] Articulatory feature mapping (stops, fricatives, nasals, liquids, glides, vowels)
- [x] RMS energy calculation for signal quality assessment
- [x] Speaker-prefixed sample matching

### ✅ Neural Component
- [x] HuBERT encoder initialization (facebook/hubert-base-ls960)
- [x] Selective layer freezing (layers 0-5 frozen, 6-11 fine-tuned)
- [x] Projection layer (768 → 512-dim)
- [x] Phoneme classifier with dropout

### ✅ Symbolic Component
- [x] ArticulatoryFeatureEncoder (manner, place, voicing features)
- [x] SymbolicConstraintMatrix (euclidean distance-based similarity)
- [x] ConstraintAggregation (weighted combination of neural + symbolic logits)
- [x] Dysarthria-aware weighting (severity-dependent interpolation)

### ✅ Training
- [x] PyTorch Lightning trainer setup ([train.py](train.py))
- [x] Multi-task learning (CTC + Focal + KL constraint losses)
- [x] Gradient accumulation & VRAM optimization
- [x] Learning rate scheduling (cosine annealing + warmup)
- [x] Callbacks (EarlyStopping, ModelCheckpoint, LearningRateMonitor)
- [x] MLflow experiment tracking with safe parameter flattening

### ✅ Evaluation
- [x] PER (Phoneme Error Rate) computation ([evaluate.py](evaluate.py))
- [x] WER (Word Error Rate) with jiwer
- [x] Phoneme alignment & confusion matrices
- [x] Per-speaker and per-articulatory-class error analysis
- [x] Visualization utilities

### ✅ Data Management
- [x] PyTorch Dataset & DataLoader ([dataloader.py](dataloader.py))
- [x] CTC-compatible batching (attention masks, label padding)
- [x] Peak normalization for dysarthric speech variability
- [x] Collator for flexible sequence handling

### ✅ Configuration
- [x] Dataclass-based config ([config.py](config.py))
- [x] ModelConfig, TrainingConfig, SymbolicConfig, DataConfig
- [x] Safe YAML loading & parameter override

---

## 📋 Not Yet Implemented

### Explainability & Analysis
- [ ] Interactive dashboard (phoneme confusion heatmaps, rule activation maps)
- [ ] Per-speaker error attribution with clinical interpretability
- [ ] Rule activation tracking for symbolic constraint layer
- [ ] Attention visualization for HuBERT layer analysis

### Model Deployment & Checkpoints
- [ ] Pretrained model checkpoints (dysarthric/control baselines)
- [ ] ONNX export for inference optimization
- [ ] TorchScript export for production deployment
- [ ] Streaming inference API (FastAPI)

### Advanced Features
- [ ] Automatic dysarthric substitution rule discovery (from confusion matrices)
- [ ] Speaker adaptation fine-tuning
- [ ] Confidence scoring & uncertainty estimation
- [ ] Real-time feedback system for speech therapy

### Research Extensions
- [ ] Ablation studies (neural vs. symbolic contribution quantification)
- [ ] Cross-dataset evaluation (on non-TORGO dysarthric corpora)
- [ ] Comparison with baseline ASR systems
- [ ] Analysis of constraint matrix learned weights

---

## Key Design Decisions & Rationale

### Why HuBERT over Wav2Vec2?
- HuBERT uses discrete clustering targets (more phoneme-aligned)
- Better performance on dysarthric speech in preliminary experiments
- Smaller parameter footprint for VRAM constraints

### Why Symbolic Constraints?
- Dysarthric substitutions follow articulatory patterns (preserve place/manner)
- Symbolic layer provides interpretability (which rules fired?)
- Modular design allows separate ablation of neural vs. symbolic components

### Why Multi-Task Learning?
- Dysarthria classification as auxiliary task improves speaker-level discrimination
- Focal loss handles class imbalance (fewer dysarthric speakers)
- KL constraint loss provides soft regularization (not hard constraints)

### Why Gradient Accumulation?
- RTX 4060 (8GB VRAM) cannot fit batch_size > 2
- Accumulation steps=12 → effective batch=24 (better gradient estimates)
- Maintains training stability without reducing effective batch size

### Why Speaker-Independent Splits?
- TORGO has only ~15 speakers; speaker-dependent splits would leak test information
- Generalization to unseen dysarthric speakers is the research goal
- Per-speaker evaluation metrics computed separately

---

## Integration Points & Cross-Component Communication

### Data → Model Flow
```
manifest.csv (speaker, phonemes, articulatory_classes)
    ↓
TorgoNeuroSymbolicDataset (streams from HF + manifest)
    ↓
NeuroSymbolicCollator (batches with attention masks, label padding=-100)
    ↓
DataLoader (shuffle, variable-length batching)
    ↓
NeuroSymbolicASR (forward pass)
```

### Model Inference Flow
```
Audio Waveform (16kHz)
    ↓
HuBERT Feature Extraction (768-dim, 50Hz frame rate)
    ↓
Projection (768 → 512)
    ↓
Phoneme Classifier → Neural Logits
    ↓
SymbolicConstraintMatrix (precomputed from phoneme features)
    ↓
ConstraintAggregation (α * neural + (1-α) * (C @ neural))
    ↓
Constrained Logits
    ↓
CTC Decoder (greedy or beam search)
    ↓
Phoneme Sequence → Transcript
```

### Training Loop
```
Batch from DataLoader
    ↓
Forward pass → neural logits + constrained logits
    ↓
CTC Loss (alignment) + Focal Loss (dysarthria) + KL Loss (constraint)
    ↓
Backward pass (gradient accumulation every N steps)
    ↓
Learning rate scheduler update
    ↓
Validation: PER computation on val set
    ↓
Early stopping (metric: val_per, patience=5)
```

---

## Common Debugging Patterns

**Issue**: Audio appears silent in batch
- **Cause**: Peak normalization not applied during loading
- **Solution**: Verify `_load_audio()` normalizes by `max(abs(waveform))`

**Issue**: CTC loss NaN or infinite
- **Cause**: Input lengths > output lengths (feature extraction stride issue)
- **Solution**: Check HuBERT feature rate: ~50Hz for 16kHz audio (stride=320)

**Issue**: Model not learning (loss plateaus)
- **Cause**: Symbolic constraint too strong (α too low), masking neural learning
- **Solution**: Increase α (start 0.8, decrease to 0.5 for dysarthric cohorts)

**Issue**: Validation PER very high
- **Cause**: Speaker-prefixed sample IDs not matching HF dataset keys
- **Solution**: Verify manifest `sample_id` format: `f"{speaker}_{filename}"`
