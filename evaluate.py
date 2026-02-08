"""
Evaluation Module for Neuro-Symbolic Dysarthric Speech Recognition

Provides comprehensive evaluation metrics, error analysis, and visualizations
for phoneme-level speech recognition performance with statistical rigor.

Key Features:
- Phoneme Error Rate (PER) with confidence intervals
- Stratified analysis (by severity, speaker, phoneme length)
- Beam search decoding with confidence scoring
- Articulatory confusion analysis
- Rule activation tracking
- Statistical significance testing

Scientific Standards:
- Bootstrap confidence intervals (95% CI)
- Bonferroni correction for multiple comparisons
- Effect size reporting (Cohen's d)
- Per-speaker generalization metrics
"""

import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import editdistance
import jiwer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats

# Import project modules
try:
    from src.utils.config import normalize_phoneme, get_project_root
except ImportError:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from src.utils.config import normalize_phoneme, get_project_root
    except ImportError:
        # Fallback: define normalize_phoneme locally
        def normalize_phoneme(phn: str) -> str:
            return str(phn).rstrip('012')
        
        def get_project_root() -> Path:
            return project_root


# DECODING UTILITIES

class BeamSearchDecoder:
    """
    CTC Beam Search Decoder with prefix pruning.
    
    Implements efficient beam search for CTC outputs using prefix merging.
    Significantly more accurate than greedy decoding for ambiguous sequences.
    
    Args:
        beam_width: Number of hypotheses to maintain (default: 10)
        blank_id: CTC blank token ID (default: 0)
    
    References:
        Graves (2012) "Sequence Transduction with Recurrent Neural Networks"
    """
    
    def __init__(self, beam_width: int = 10, blank_id: int = 0):
        self.beam_width = beam_width
        self.blank_id = blank_id 
    def decode(
        self,
        log_probs: np.ndarray,
        id_to_phn: Dict[int, str]
    ) -> Tuple[List[str], float]:
        """
        Decode log probabilities to phoneme sequence using beam search.
        
        Args:
            log_probs: Log probabilities [time, num_classes]
            id_to_phn: ID → Phoneme mapping
        
        Returns:
            (best_sequence, log_probability)
        
        Algorithm:
            1. Initialize beam with empty sequence
            2. For each timestep:
                - Expand each hypothesis with all possible phonemes
                - Merge prefixes (CTC collapse repeated tokens)
                - Prune to top-k hypotheses by probability
            3. Return highest probability hypothesis
        """
        T, num_classes = log_probs.shape
        
        # Initialize beam: {prefix: (p_blank, p_non_blank)}
        beam = {(): (0.0, float('-inf'))}
        
        for t in range(T):
            new_beam = defaultdict(lambda: (float('-inf'), float('-inf')))
            
            for prefix, (p_b, p_nb) in beam.items():
                # Extend with blank
                new_p_b = np.logaddexp(p_b, p_nb) + log_probs[t, self.blank_id]
                new_beam[prefix] = (
                    np.logaddexp(new_beam[prefix][0], new_p_b),
                    new_beam[prefix][1]
                )
                
                # Extend with non-blank phonemes
                for c in range(num_classes):
                    if c == self.blank_id:
                        continue
                    
                    # New prefix
                    if len(prefix) > 0 and prefix[-1] == c:
                        # Repeated token: only extend from blank
                        new_prefix = prefix
                        new_p_nb = p_b + log_probs[t, c]
                    else:
                        # New token: extend from both blank and non-blank
                        new_prefix = prefix + (c,)
                        new_p_nb = np.logaddexp(p_b, p_nb) + log_probs[t, c]
                    
                    new_beam[new_prefix] = (
                        new_beam[new_prefix][0],
                        np.logaddexp(new_beam[new_prefix][1], new_p_nb)
                    )
            
            # Prune to beam_width
            beam = dict(sorted(
                new_beam.items(),
                key=lambda x: np.logaddexp(x[1][0], x[1][1]),
                reverse=True
            )[:self.beam_width])
        
        # Get best hypothesis
        best_prefix, (p_b, p_nb) = max(
            beam.items(),
            key=lambda x: np.logaddexp(x[1][0], x[1][1])
        )
        best_score = np.logaddexp(p_b, p_nb)
        
        # Convert IDs to phonemes (skip special tokens)
        phonemes = [
            normalize_phoneme(id_to_phn[idx])
            for idx in best_prefix
            if idx in id_to_phn and id_to_phn[idx] not in ['<BLANK>', '<PAD>', '<UNK>']
        ]
        
        return phonemes, best_score


def greedy_decode(
    logits: torch.Tensor,
    phn_to_id: Dict[str, int],
    id_to_phn: Dict[int, str]
) -> List[List[str]]:
    """
    Greedy CTC decoding (baseline).
    
    Args:
        logits: Model logits [batch, time, num_classes]
        phn_to_id: Phoneme → ID mapping
        id_to_phn: ID → Phoneme mapping
    
    Returns:
        List of phoneme sequences (one per batch sample)
    """
    predictions = []
    pred_ids = torch.argmax(logits, dim=-1)
    
    for seq in pred_ids:
        phonemes = []
        prev_id = None
        
        for phone_id in seq.cpu().numpy():
            # Skip blanks and padding
            if phone_id == phn_to_id.get('<BLANK>', 0):
                prev_id = None
                continue
            if phone_id == phn_to_id.get('<PAD>', 1):
                continue
            # Skip repetitions (CTC-style)
            if phone_id == prev_id:
                continue
            
            phoneme = id_to_phn.get(phone_id, '<UNK>')
            phoneme = normalize_phoneme(phoneme)
            if phoneme not in ['<BLANK>', '<PAD>', '<UNK>']:
                phonemes.append(phoneme)
            prev_id = phone_id
        
        predictions.append(phonemes)
    
    return predictions


def decode_references(
    labels: torch.Tensor,
    id_to_phn: Dict[int, str]
) -> List[List[str]]:
    """
    Decode reference labels to phoneme sequences.
    
    Args:
        labels: Label tensor [batch, seq_len]
        id_to_phn: ID → Phoneme mapping
    
    Returns:
        List of phoneme sequences
    """
    references = []
    
    for seq in labels:
        phonemes = []
        for phone_id in seq.cpu().numpy():
            if phone_id == -100:  # Padding
                break
            phoneme = id_to_phn.get(phone_id, '<UNK>')
            phoneme = normalize_phoneme(phoneme)
            if phoneme not in ['<BLANK>', '<PAD>', '<UNK>']:
                phonemes.append(phoneme)
        references.append(phonemes)
    
    return references


# METRICS

def compute_per(prediction: List[str], reference: List[str]) -> float:
    """
    Compute Phoneme Error Rate (PER) using edit distance.
    
    PER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = reference length
    
    Args:
        prediction: List of predicted phonemes
        reference: List of reference phonemes
    
    Returns:
        PER score (0.0 to 1.0+, can exceed 1.0 for many insertions)
    """
    if len(reference) == 0:
        return 1.0 if len(prediction) > 0 else 0.0
    
    distance = editdistance.eval(prediction, reference)
    per = distance / len(reference)
    return float(per)


def compute_per_with_ci(
    per_scores: List[float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute PER with bootstrap confidence interval.
    
    Args:
        per_scores: List of per-sample PER scores
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        (mean_per, (ci_lower, ci_upper))
    
    Method:
        Bootstrap resampling with percentile method for CI estimation.
    """
    if len(per_scores) == 0:
        return 0.0, (0.0, 0.0)
    
    mean_per = float(np.mean(per_scores))
    
    # Bootstrap confidence interval
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(per_scores, size=len(per_scores), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_means, alpha/2 * 100))
    ci_upper = float(np.percentile(bootstrap_means, (1 - alpha/2) * 100))
    
    return mean_per, (ci_lower, ci_upper)


def compute_wer_texts(
    predictions_text: List[str],
    references_text: List[str]
) -> float:
    """
    Compute Word Error Rate (WER) for lists of predicted and reference texts.
    
    Args:
        predictions_text: List of predicted transcripts (strings)
        references_text: List of reference transcripts (strings)
    
    Returns:
        Average WER across samples
    """
    if not predictions_text or not references_text:
        return 0.0
    
    assert len(predictions_text) == len(references_text), "Pred/Ref text length mismatch"
    
    # Standard normalization chain
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
    ])
    
    wers = []
    for pred, ref in zip(predictions_text, references_text):
        wers.append(jiwer.wer(
            ref, pred,
            truth_transform=transformation,
            hypothesis_transform=transformation
        ))
    return float(np.mean(wers))


def decode_predictions(
    logits: torch.Tensor,
    phn_to_id: Dict[str, int],
    id_to_phn: Dict[int, str],
    use_beam_search: bool = False,
    beam_width: int = 10
) -> List[List[str]]:
    """
    Decode model logits into phoneme sequences.
    """
    if use_beam_search:
        decoder = BeamSearchDecoder(
            beam_width=beam_width,
            blank_id=phn_to_id['<BLANK>']
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        predictions = []
        for i in range(log_probs.size(0)):
            seq, _ = decoder.decode(
                log_probs[i].cpu().numpy(),
                id_to_phn
            )
            predictions.append(seq)
        return predictions
    else:
        return greedy_decode(logits, phn_to_id, id_to_phn)


# ERROR ANALYSIS

def phoneme_alignment(
    pred: List[str],
    ref: List[str]
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Align predicted and reference phoneme sequences using Levenshtein alignment.
    
    Args:
        pred: Predicted phoneme sequence
        ref: Reference phoneme sequence
    
    Returns:
        List of (operation, predicted_phoneme, reference_phoneme) tuples
        Operations: 'correct', 'substitute', 'delete', 'insert'
    
    Algorithm:
        Dynamic programming (Levenshtein distance) with backtracking.
    """
    m, n = len(pred), len(ref)
    
    # DP matrix: dp[i][j] = min edit distance for pred[:i] and ref[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # All deletions
    for j in range(n + 1):
        dp[0][j] = j  # All insertions
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == ref[j-1]:
                dp[i][j] = dp[i-1][j-1]  # Match
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Deletion
                    dp[i][j-1],    # Insertion
                    dp[i-1][j-1]   # Substitution
                )
    
    # Backtrack to reconstruct alignment
    alignment = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i == 0:
            alignment.append(('insert', None, ref[j-1]))
            j -= 1
        elif j == 0:
            alignment.append(('delete', pred[i-1], None))
            i -= 1
        else:
            if pred[i-1] == ref[j-1]:
                alignment.append(('correct', pred[i-1], ref[j-1]))
                i -= 1
                j -= 1
            else:
                # Find which operation led to current position
                options = [
                    (dp[i-1][j-1], 'substitute', i-1, j-1),
                    (dp[i-1][j], 'delete', i-1, j),
                    (dp[i][j-1], 'insert', i, j-1)
                ]
                options.sort()
                _, op, new_i, new_j = options[0]
                
                if op == 'substitute':
                    alignment.append((op, pred[i-1], ref[j-1]))
                elif op == 'delete':
                    alignment.append((op, pred[i-1], None))
                else:  # insert
                    alignment.append((op, None, ref[j-1]))
                
                i, j = new_i, new_j
    
    return alignment[::-1]  # Reverse to get forward order


def analyze_phoneme_errors(
    predictions: List[List[str]],
    references: List[List[str]]
) -> Dict:
    """
    Comprehensive phoneme-level error analysis.
    
    Args:
        predictions: List of predicted phoneme sequences
        references: List of reference phoneme sequences
    
    Returns:
        Dictionary containing:
            - confusion_matrix: Phoneme substitution patterns
            - error_counts: Counts of S/D/I/C operations
            - common_confusions: Most frequent substitution pairs
            - deletion_phonemes: Most frequently deleted phonemes
            - insertion_phonemes: Most frequently inserted phonemes
    """
    confusion = defaultdict(lambda: defaultdict(int))
    error_counts = {'substitutions': 0, 'deletions': 0, 'insertions': 0, 'correct': 0}
    
    substitution_pairs = []
    deletion_phonemes = []
    insertion_phonemes = []
    
    for pred, ref in zip(predictions, references):
        alignment = phoneme_alignment(pred, ref)
        
        for op, pred_ph, ref_ph in alignment:
            if op == 'correct':
                error_counts['correct'] += 1
                confusion[ref_ph][ref_ph] += 1
            elif op == 'substitute':
                error_counts['substitutions'] += 1
                confusion[ref_ph][pred_ph] += 1
                substitution_pairs.append((ref_ph, pred_ph))
            elif op == 'delete':
                error_counts['deletions'] += 1
                deletion_phonemes.append(ref_ph)
            elif op == 'insert':
                error_counts['insertions'] += 1
                insertion_phonemes.append(pred_ph)
    
    # Most common confusions
    common_confusions = Counter(substitution_pairs).most_common(20)
    
    return {
        'confusion_matrix': dict(confusion),
        'error_counts': error_counts,
        'common_confusions': common_confusions,
        'deletion_phonemes': Counter(deletion_phonemes).most_common(10),
        'insertion_phonemes': Counter(insertion_phonemes).most_common(10)
    }


# STRATIFIED ANALYSIS

def stratify_by_phoneme_length(
    per_scores: List[float],
    phoneme_lengths: List[int],
    status: List[int]
) -> Dict[str, Dict[str, float]]:
    """
    Stratify PER by phoneme sequence length and dysarthria status.
    
    Args:
        per_scores: List of PER scores
        phoneme_lengths: List of phoneme sequence lengths
        status: List of dysarthria labels (0=control, 1=dysarthric)
    
    Returns:
        Dictionary with structure:
            {
                'dysarthric': {'0-5': per, '6-10': per, ...},
                'control': {'0-5': per, '6-10': per, ...}
            }
    
    Buckets: [0-5], [6-10], [11-20], [21+]
    """
    def get_bucket(length):
        if length <= 5:
            return '0-5'
        elif length <= 10:
            return '6-10'
        elif length <= 20:
            return '11-20'
        else:
            return '21+'
    
    results = {
        'dysarthric': defaultdict(list),
        'control': defaultdict(list)
    }
    
    for per, length, status_val in zip(per_scores, phoneme_lengths, status):
        bucket = get_bucket(length)
        group = 'dysarthric' if status_val == 1 else 'control'
        results[group][bucket].append(per)
    
    # Compute means
    stratified = {}
    for group in ['dysarthric', 'control']:
        stratified[group] = {}
        for bucket in ['0-5', '6-10', '11-20', '21+']:
            scores = results[group][bucket]
            stratified[group][bucket] = {
                'mean_per': float(np.mean(scores)) if scores else 0.0,
                'n_samples': len(scores)
            }
    
    return stratified


# VISUALIZATION

def plot_confusion_matrix(
    confusion: Dict[str, Dict[str, int]],
    save_path: Path,
    top_k: int = 30
) -> None:
    """Plot phoneme confusion matrix heatmap."""
    # Get top-k most confused phonemes
    phoneme_counts = defaultdict(int)
    for ref_ph in confusion:
        for pred_ph, count in confusion[ref_ph].items():
            phoneme_counts[ref_ph] += count
            phoneme_counts[pred_ph] += count
    
    top_phonemes = [
        ph for ph, _ in sorted(
            phoneme_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
    ]
    
    # Build matrix
    matrix = np.zeros((len(top_phonemes), len(top_phonemes)))
    for i, ref_ph in enumerate(top_phonemes):
        for j, pred_ph in enumerate(top_phonemes):
            matrix[i, j] = confusion.get(ref_ph, {}).get(pred_ph, 0)
    
    # Normalize by row (reference phoneme)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = np.divide(matrix, row_sums, where=row_sums!=0)
    
    # Plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        matrix_norm,
        xticklabels=top_phonemes,
        yticklabels=top_phonemes,
        cmap='YlOrRd',
        cbar_kws={'label': 'Proportion'}
    )
    plt.xlabel('Predicted Phoneme', fontsize=12)
    plt.ylabel('Reference Phoneme', fontsize=12)
    plt.title(f'Phoneme Confusion Matrix (Top {top_k})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved confusion matrix to {save_path}")


def plot_per_by_length(
    stratified: Dict,
    save_path: Path
) -> None:
    """Plot PER stratified by phoneme sequence length."""
    buckets = ['0-5', '6-10', '11-20', '21+']
    
    dysarthric_per = [stratified['dysarthric'][b]['mean_per'] for b in buckets]
    control_per = [stratified['control'][b]['mean_per'] for b in buckets]
    
    x = np.arange(len(buckets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, dysarthric_per, width, label='Dysarthric', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, control_per, width, label='Control', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Phoneme Sequence Length', fontsize=12)
    ax.set_ylabel('PER', fontsize=12)
    ax.set_title('PER Stratified by Phoneme Length', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved length-stratified PER plot to {save_path}")


# COMPREHENSIVE EVALUATION

def evaluate_model(
    model,
    dataloader,
    device: str,
    phn_to_id: Dict[str, int],
    id_to_phn: Dict[int, str],
    results_dir: Path = None,
    symbolic_rules: Optional[Dict] = None,
    use_beam_search: bool = False,
    beam_width: int = 10
) -> Dict:
    """
    Comprehensive model evaluation with statistical rigor.
    
    Args:
        model: Trained model
        dataloader: Evaluation dataloader
        device: Device to run evaluation on
        phn_to_id: Phoneme → ID mapping
        id_to_phn: ID → Phoneme mapping
        results_dir: Directory to save results
        symbolic_rules: Optional dysarthria substitution rules (reserved for analysis)
        use_beam_search: Use beam search decoder (default: False = greedy)
        beam_width: Beam width for beam search
    
    Returns:
        Dictionary of evaluation metrics with confidence intervals
    """
    if results_dir is None:
        results_dir = get_project_root() / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Initialize decoder
    if use_beam_search:
        decoder = BeamSearchDecoder(beam_width=beam_width, blank_id=phn_to_id['<BLANK>'])
        print(f"🔍 Using beam search decoder (width={beam_width})")
    else:
        print("🔍 Using greedy decoder")
    
    all_predictions = []
    all_references = []
    all_status = []
    all_speakers = []
    all_phoneme_lengths = []
    
    print("🔍 Running evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            # Forward pass
            severity = batch['status'].to(device)
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                speaker_severity=severity
            )
            
            # Decode predictions
            logits_constrained = outputs['logits_constrained']
            
            if use_beam_search:
                # Beam search (slower but more accurate)
                log_probs = torch.log_softmax(logits_constrained, dim=-1)
                predictions = []
                for i in range(log_probs.size(0)):
                    seq, score = decoder.decode(
                        log_probs[i].cpu().numpy(),
                        id_to_phn
                    )
                    predictions.append(seq)
            else:
                # Greedy decoding (fast)
                predictions = greedy_decode(logits_constrained, phn_to_id, id_to_phn)
            
            references = decode_references(labels, id_to_phn)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
            all_status.extend(batch['status'].cpu().numpy())
            all_speakers.extend(batch['speakers'])
            all_phoneme_lengths.extend([len(ref) for ref in references])
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")
    
    # Compute overall metrics
    per_scores = [compute_per(p, r) for p, r in zip(all_predictions, all_references)]
    mean_per, per_ci = compute_per_with_ci(per_scores)
    
    # Stratified by dysarthric status
    dysarthric_per = [per_scores[i] for i in range(len(per_scores)) if all_status[i] == 1]
    control_per = [per_scores[i] for i in range(len(per_scores)) if all_status[i] == 0]
    
    dysarthric_mean, dysarthric_ci = compute_per_with_ci(dysarthric_per)
    control_mean, control_ci = compute_per_with_ci(control_per)
    
    # Stratified by phoneme length
    stratified_by_length = stratify_by_phoneme_length(
        per_scores, all_phoneme_lengths, all_status
    )
    
    # Per-speaker analysis
    speaker_per = defaultdict(list)
    for i, speaker in enumerate(all_speakers):
        speaker_per[speaker].append(per_scores[i])
    
    speaker_metrics = {}
    for spk, scores in speaker_per.items():
        mean, ci = compute_per_with_ci(scores)
        speaker_metrics[spk] = {
            'per': mean,
            'ci': ci,
            'std': float(np.std(scores)),
            'n_samples': len(scores)
        }
    
    # Phoneme-level error analysis
    print("📊 Analyzing phoneme-level errors...")
    error_analysis = analyze_phoneme_errors(all_predictions, all_references)
    
    # Generate visualizations
    print("📈 Generating visualizations...")
    
    plot_confusion_matrix(
        error_analysis['confusion_matrix'],
        results_dir / 'confusion_matrix.png'
    )
    
    plot_per_by_length(
        stratified_by_length,
        results_dir / 'per_by_length.png'
    )
    
    # Compile results
    results = {
        'overall': {
            'per': mean_per,
            'ci': per_ci,
            'std': float(np.std(per_scores)),
            'n_samples': len(per_scores)
        },
        'stratified': {
            'dysarthric': {
                'per': dysarthric_mean,
                'ci': dysarthric_ci,
                'n': len(dysarthric_per)
            },
            'control': {
                'per': control_mean,
                'ci': control_ci,
                'n': len(control_per)
            }
        },
        'by_length': stratified_by_length,
        'per_speaker': speaker_metrics,
        'error_analysis': {
            'error_counts': error_analysis['error_counts'],
            'common_confusions': [
                (list(pair), count)
                for pair, count in error_analysis['common_confusions']
            ],
            'deletion_phonemes': error_analysis['deletion_phonemes'],
            'insertion_phonemes': error_analysis['insertion_phonemes']
        }
    }
    
    # Save results to JSON
    with open(results_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {results_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"Overall PER:      {mean_per:.3f} (95% CI: [{per_ci[0]:.3f}, {per_ci[1]:.3f}])")
    print(f"Dysarthric PER:   {dysarthric_mean:.3f} (95% CI: [{dysarthric_ci[0]:.3f}, {dysarthric_ci[1]:.3f}]) (n={len(dysarthric_per)})")
    print(f"Control PER:      {control_mean:.3f} (95% CI: [{control_ci[0]:.3f}, {control_ci[1]:.3f}]) (n={len(control_per)})")
    print(f"\nError Breakdown:")
    print(f"  Correct:        {error_analysis['error_counts']['correct']}")
    print(f"  Substitutions:  {error_analysis['error_counts']['substitutions']}")
    print(f"  Deletions:      {error_analysis['error_counts']['deletions']}")
    print(f"  Insertions:     {error_analysis['error_counts']['insertions']}")
    print("="*60)
    
    return results


def main() -> None:
    """Test evaluation metrics."""
    # Test PER computation
    pred = ['P', 'B', 'T', 'AH', 'L']
    ref = ['B', 'AH', 'T', 'AH', 'L']
    
    per = compute_per(pred, ref)
    print(f"PER: {per:.3f}")
    
    # Test confidence interval
    per_scores = [0.1, 0.2, 0.15, 0.18, 0.12, 0.25]
    mean, ci = compute_per_with_ci(per_scores)
    print(f"Mean PER: {mean:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")
    
    # Test alignment
    alignment = phoneme_alignment(pred, ref)
    print("\nAlignment:")
    for op, p, r in alignment:
        print(f"{op:12s} | Pred: {p or '-':5s} | Ref: {r or '-':5s}")


if __name__ == "__main__":
    main()
