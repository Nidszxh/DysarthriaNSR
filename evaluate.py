import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import editdistance
from jiwer import wer as compute_wer_jiwer


def compute_per(prediction: List[str], reference: List[str]) -> float:
    """
    Compute Phoneme Error Rate (PER) using edit distance.
    
    PER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = reference length
    
    Args:
        prediction: List of predicted phonemes
        reference: List of reference phonemes
    
    Returns:
        PER score (0.0 to 1.0+)
    """
    if len(reference) == 0:
        return 1.0 if len(prediction) > 0 else 0.0
    
    distance = editdistance.eval(prediction, reference)
    per = distance / len(reference)
    return per


def compute_wer(prediction: str, reference: str) -> float:
    """
    Compute Word Error Rate (WER) using jiwer library.
    
    Args:
        prediction: Predicted sentence (string)
        reference: Reference sentence (string)
    
    Returns:
        WER score
    """
    return compute_wer_jiwer(reference, prediction)


def phoneme_alignment(pred: List[str], ref: List[str]) -> List[Tuple[str, str, str]]:
    """
    Align predicted and reference phoneme sequences to identify errors.
    
    Returns:
        List of (operation, predicted_phoneme, reference_phoneme) tuples
        Operations: 'correct', 'substitute', 'delete', 'insert'
    """
    # Use dynamic programming for alignment
    m, n = len(pred), len(ref)
    
    # DP matrix: dp[i][j] = min edit distance for pred[:i] and ref[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize
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
    
    # Backtrack to get alignment
    alignment = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i == 0:
            # Only insertions left
            alignment.append(('insert', None, ref[j-1]))
            j -= 1
        elif j == 0:
            # Only deletions left
            alignment.append(('delete', pred[i-1], None))
            i -= 1
        else:
            if pred[i-1] == ref[j-1]:
                alignment.append(('correct', pred[i-1], ref[j-1]))
                i -= 1
                j -= 1
            else:
                # Find which operation led here
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
    references: List[List[str]],
    phoneme_features: Dict[str, Dict[str, str]]
) -> Dict:
    """
    Comprehensive phoneme-level error analysis.
    
    Returns:
        Dictionary containing:
        - confusion_matrix: Phoneme substitution patterns
        - error_breakdown: Counts of S/D/I
        - feature_errors: Errors by articulatory features
        - common_confusions: Most frequent phoneme pairs
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
    
    # Analyze articulatory feature patterns
    feature_errors = analyze_feature_patterns(
        substitution_pairs, 
        phoneme_features
    )
    
    # Most common confusions
    common_confusions = Counter(substitution_pairs).most_common(20)
    
    return {
        'confusion_matrix': dict(confusion),
        'error_counts': error_counts,
        'feature_errors': feature_errors,
        'common_confusions': common_confusions,
        'deletion_phonemes': Counter(deletion_phonemes).most_common(10),
        'insertion_phonemes': Counter(insertion_phonemes).most_common(10)
    }


def analyze_feature_patterns(
    substitution_pairs: List[Tuple[str, str]],
    phoneme_features: Dict[str, Dict[str, str]]
) -> Dict:
    """
    Analyze errors by articulatory features (manner, place, voice).
    
    Returns feature-level confusion statistics.
    """
    feature_errors = {
        'manner': defaultdict(lambda: defaultdict(int)),
        'place': defaultdict(lambda: defaultdict(int)),
        'voice': defaultdict(lambda: defaultdict(int))
    }
    
    for ref_ph, pred_ph in substitution_pairs:
        # Clean phoneme symbols
        ref_clean = ref_ph.rstrip('012')
        pred_clean = pred_ph.rstrip('012')
        
        if ref_clean not in phoneme_features or pred_clean not in phoneme_features:
            continue
        
        ref_feat = phoneme_features[ref_clean]
        pred_feat = phoneme_features[pred_clean]
        
        # Track feature-level confusions
        for feature in ['manner', 'place', 'voice']:
            ref_val = ref_feat.get(feature, 'unknown')
            pred_val = pred_feat.get(feature, 'unknown')
            feature_errors[feature][ref_val][pred_val] += 1
    
    return dict(feature_errors)


def plot_confusion_matrix(
    confusion: Dict[str, Dict[str, int]],
    save_path: Path,
    top_k: int = 20
):
    """Plot phoneme confusion matrix heatmap."""
    # Get top-k most confused phonemes
    all_phonemes = set()
    for ref_ph in confusion:
        all_phonemes.add(ref_ph)
        for pred_ph in confusion[ref_ph]:
            all_phonemes.add(pred_ph)
    
    # Filter to top-k most frequent
    phoneme_counts = defaultdict(int)
    for ref_ph in confusion:
        for pred_ph, count in confusion[ref_ph].items():
            phoneme_counts[ref_ph] += count
            phoneme_counts[pred_ph] += count
    
    top_phonemes = [ph for ph, _ in sorted(
        phoneme_counts.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_k]]
    
    # Build matrix
    matrix = np.zeros((len(top_phonemes), len(top_phonemes)))
    for i, ref_ph in enumerate(top_phonemes):
        for j, pred_ph in enumerate(top_phonemes):
            matrix[i, j] = confusion.get(ref_ph, {}).get(pred_ph, 0)
    
    # Normalize by row (reference phoneme)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = np.divide(matrix, row_sums, where=row_sums!=0)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix_norm,
        xticklabels=top_phonemes,
        yticklabels=top_phonemes,
        cmap='YlOrRd',
        annot=False,
        fmt='.2f',
        cbar_kws={'label': 'Normalized Count'}
    )
    plt.xlabel('Predicted Phoneme')
    plt.ylabel('Reference Phoneme')
    plt.title('Phoneme Confusion Matrix (Normalized by Row)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model(
    model,
    dataloader,
    device: str,
    save_dir: Path,
    phoneme_features: Dict
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Returns:
        Dictionary of metrics and saves visualizations.
    """
    model.eval()
    
    all_predictions = []
    all_references = []
    all_status = []  # Dysarthric vs control
    all_speakers = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            # Forward pass
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask
            )
            
            # Decode predictions
            logits = outputs['logits_constrained']
            predictions = _decode_batch(logits, model.phn_to_id, model.id_to_phn)
            references = _decode_references(labels, model.id_to_phn)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
            all_status.extend(batch['status'].cpu().numpy())
            all_speakers.extend(batch['speakers'])
    
    # Compute overall metrics
    per_scores = [compute_per(p, r) for p, r in zip(all_predictions, all_references)]
    avg_per = np.mean(per_scores)
    
    # Stratified by status
    dysarthric_per = [per_scores[i] for i in range(len(per_scores)) if all_status[i] == 1]
    control_per = [per_scores[i] for i in range(len(per_scores)) if all_status[i] == 0]
    
    # Per-speaker analysis
    speaker_per = defaultdict(list)
    for i, speaker in enumerate(all_speakers):
        speaker_per[speaker].append(per_scores[i])
    
    speaker_metrics = {
        spk: {
            'per': np.mean(scores),
            'std': np.std(scores),
            'n_samples': len(scores)
        }
        for spk, scores in speaker_per.items()
    }
    
    # Phoneme-level error analysis
    error_analysis = analyze_phoneme_errors(
        all_predictions,
        all_references,
        phoneme_features
    )
    
    # Generate visualizations
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plot_confusion_matrix(
        error_analysis['confusion_matrix'],
        save_dir / 'confusion_matrix.png'
    )
    
    # Save detailed results
    results = {
        'overall': {
            'per': avg_per,
            'per_std': np.std(per_scores)
        },
        'stratified': {
            'dysarthric_per': np.mean(dysarthric_per) if dysarthric_per else 0.0,
            'control_per': np.mean(control_per) if control_per else 0.0
        },
        'per_speaker': speaker_metrics,
        'error_analysis': {
            'error_counts': error_analysis['error_counts'],
            'common_confusions': error_analysis['common_confusions'],
            'deletion_phonemes': error_analysis['deletion_phonemes'],
            'insertion_phonemes': error_analysis['insertion_phonemes']
        }
    }
    
    # Save to JSON
    import json
    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n📊 Evaluation Results:")
    print(f"Overall PER: {avg_per:.3f}")
    print(f"Dysarthric PER: {results['stratified']['dysarthric_per']:.3f}")
    print(f"Control PER: {results['stratified']['control_per']:.3f}")
    
    return results


def _decode_batch(logits: torch.Tensor, phn_to_id: Dict, id_to_phn: Dict) -> List[List[str]]:
    """Decode batch of logits to phoneme sequences."""
    predictions = []
    pred_ids = torch.argmax(logits, dim=-1)
    
    for seq in pred_ids:
        phonemes = []
        prev_id = None
        
        for phone_id in seq.cpu().numpy():
            if phone_id == phn_to_id['<BLANK>']:
                prev_id = None
                continue
            if phone_id == phn_to_id['<PAD>']:
                continue
            if phone_id == prev_id:
                continue
            
            phoneme = id_to_phn.get(phone_id, '<UNK>')
            if phoneme not in ['<BLANK>', '<PAD>', '<UNK>']:
                phonemes.append(phoneme)
            prev_id = phone_id
        
        predictions.append(phonemes)
    
    return predictions


def _decode_references(labels: torch.Tensor, id_to_phn: Dict) -> List[List[str]]:
    """Decode reference labels."""
    references = []
    
    for seq in labels:
        phonemes = []
        for phone_id in seq.cpu().numpy():
            if phone_id == -100:
                break
            phoneme = id_to_phn.get(phone_id, '<UNK>')
            if phoneme not in ['<BLANK>', '<PAD>', '<UNK>']:
                phonemes.append(phoneme)
        references.append(phonemes)
    
    return references


if __name__ == "__main__":
    # Test PER computation
    pred = ['P', 'B', 'T', 'AH', 'L']
    ref = ['B', 'AH', 'T', 'AH', 'L']
    
    per = compute_per(pred, ref)
    print(f"PER: {per:.3f}")
    
    alignment = phoneme_alignment(pred, ref)
    print("\nAlignment:")
    for op, p, r in alignment:
        print(f"{op:12s} | Pred: {p or '-':5s} | Ref: {r or '-':5s}")