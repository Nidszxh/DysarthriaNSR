"""
Evaluation Module for Neuro-Symbolic Dysarthric Speech Recognition

Provides comprehensive evaluation metrics, error analysis, and visualizations
for phoneme-level speech recognition performance.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import editdistance
import jiwer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# Optional: access default symbolic rules from project config
try:
    from src.utils.config import get_default_config
except Exception:
    # Fallback: add src to path
    sys.path.insert(0, str((Path(__file__).resolve().parent / 'src').resolve()))
    try:
        from utils.config import get_default_config
    except Exception:
        get_default_config = None


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent


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
    return per


def compute_wer_texts(predictions_text: List[str], references_text: List[str]) -> float:
    """
    Compute Word Error Rate (WER) for lists of predicted and reference texts.
    Uses jiwer with standard normalization.

    Args:
        predictions_text: List of predicted transcripts (strings)
        references_text: List of reference transcripts (strings)

    Returns:
        Average WER across samples.
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
        wers.append(jiwer.wer(ref, pred, truth_transform=transformation, hypothesis_transform=transformation))
    return float(np.mean(wers))


def phoneme_alignment(pred: List[str], ref: List[str]) -> List[Tuple[str, str, str]]:
    """
    Align predicted and reference phoneme sequences to identify errors.
    
    Uses dynamic programming (Levenshtein distance) for optimal alignment.
    
    Args:
        pred: Predicted phoneme sequence
        ref: Reference phoneme sequence
    
    Returns:
        List of (operation, predicted_phoneme, reference_phoneme) tuples
        Operations: 'correct', 'substitute', 'delete', 'insert'
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


# ---- Phoneme grouping by manner of articulation ----

PHONEME_TO_MANNER = {
    # Stops
    'P': 'stop', 'B': 'stop', 'T': 'stop', 'D': 'stop', 'K': 'stop', 'G': 'stop',
    # Fricatives
    'F': 'fricative', 'V': 'fricative', 'TH': 'fricative', 'DH': 'fricative',
    'S': 'fricative', 'Z': 'fricative', 'SH': 'fricative', 'ZH': 'fricative', 'HH': 'fricative',
    # Affricates
    'CH': 'affricate', 'JH': 'affricate',
    # Nasals
    'M': 'nasal', 'N': 'nasal', 'NG': 'nasal',
    # Liquids
    'L': 'liquid', 'R': 'liquid',
    # Glides
    'W': 'glide', 'Y': 'glide',
    # Vowels
    'IY': 'vowel', 'IH': 'vowel', 'EH': 'vowel', 'EY': 'vowel', 'AE': 'vowel',
    'AA': 'vowel', 'AO': 'vowel', 'OW': 'vowel', 'UH': 'vowel', 'UW': 'vowel',
    'AH': 'vowel', 'ER': 'vowel', 'AX': 'vowel',
    # Diphthongs
    'AY': 'diphthong', 'AW': 'diphthong', 'OY': 'diphthong',
}

MANNERS_ORDER = ['stop', 'fricative', 'affricate', 'nasal', 'liquid', 'glide', 'vowel', 'diphthong']


def phoneme_manner(ph: str) -> str:
    return PHONEME_TO_MANNER.get(ph, 'unknown')


def group_confusion_by_manner(confusion: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """
    Aggregate phoneme confusion into manner-of-articulation groups.
    """
    grouped = {m: {n: 0 for n in MANNERS_ORDER} for m in MANNERS_ORDER}

    for ref_ph, preds in confusion.items():
        ref_m = phoneme_manner(ref_ph)
        if ref_m not in grouped:
            continue
        for pred_ph, count in preds.items():
            pred_m = phoneme_manner(pred_ph)
            if pred_m in grouped[ref_m]:
                grouped[ref_m][pred_m] += count
    return grouped


def summarize_minor_major(confusion: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """
    Minor = same manner; Major = different manner.
    Returns counts and ratios.
    """
    minor = 0
    major = 0
    for ref_ph, preds in confusion.items():
        ref_m = phoneme_manner(ref_ph)
        for pred_ph, count in preds.items():
            pred_m = phoneme_manner(pred_ph)
            if ref_m == pred_m:
                minor += count
            else:
                major += count
    total = minor + major
    return {
        'minor_same_manner': int(minor),
        'major_cross_manner': int(major),
        'minor_ratio': float(minor / total) if total > 0 else 0.0,
        'major_ratio': float(major / total) if total > 0 else 0.0,
    }


def plot_confusion_matrix(
    confusion: Dict[str, Dict[str, int]],
    save_path: Path,
    top_k: int = 30
) -> None:
    """
    Plot phoneme confusion matrix heatmap.
    
    Args:
        confusion: Confusion matrix dictionary
        save_path: Path to save plot
        top_k: Number of most frequent phonemes to include
    """
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


def plot_confusion_by_manner(
    grouped_confusion: Dict[str, Dict[str, int]],
    save_path: Path
) -> None:
    """
    Plot manner-of-articulation confusion heatmap.
    """
    manners = MANNERS_ORDER
    matrix = np.zeros((len(manners), len(manners)))
    for i, rm in enumerate(manners):
        for j, pm in enumerate(manners):
            matrix[i, j] = grouped_confusion.get(rm, {}).get(pm, 0)

    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = np.divide(matrix, row_sums, where=row_sums!=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix_norm,
        xticklabels=manners,
        yticklabels=manners,
        cmap='YlGnBu',
        cbar_kws={'label': 'Proportion'}
    )
    plt.xlabel('Predicted Manner', fontsize=12)
    plt.ylabel('Reference Manner', fontsize=12)
    plt.title('Confusion by Manner of Articulation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved manner confusion matrix to {save_path}")


def plot_error_distribution(
    error_counts: Dict[str, int],
    save_path: Path
) -> None:
    """
    Plot distribution of error types (S/D/I/C).
    
    Args:
        error_counts: Dictionary of error counts
        save_path: Path to save plot
    """
    labels = ['Correct', 'Substitutions', 'Deletions', 'Insertions']
    counts = [
        error_counts['correct'],
        error_counts['substitutions'],
        error_counts['deletions'],
        error_counts['insertions']
    ]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    ax1.bar(labels, counts, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Error Type Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, (label, count) in enumerate(zip(labels, counts)):
        ax1.text(i, count + max(counts)*0.02, str(count), 
                ha='center', va='bottom', fontsize=10)
    
    # Pie chart (excluding correct)
    error_labels = ['Substitutions', 'Deletions', 'Insertions']
    error_counts_only = counts[1:]
    error_colors = colors[1:]
    
    ax2.pie(error_counts_only, labels=error_labels, colors=error_colors,
           autopct='%1.1f%%', startangle=90)
    ax2.set_title('Error Type Proportions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved error distribution to {save_path}")


def plot_per_by_speaker(
    speaker_metrics: Dict[str, Dict],
    save_path: Path
) -> None:
    """
    Plot PER scores by speaker.
    
    Args:
        speaker_metrics: Dictionary of speaker-level metrics
        save_path: Path to save plot
    """
    speakers = list(speaker_metrics.keys())
    per_scores = [speaker_metrics[spk]['per'] for spk in speakers]
    std_scores = [speaker_metrics[spk]['std'] for spk in speakers]
    n_samples = [speaker_metrics[spk]['n_samples'] for spk in speakers]
    
    # Sort by PER
    sorted_idx = np.argsort(per_scores)
    speakers = [speakers[i] for i in sorted_idx]
    per_scores = [per_scores[i] for i in sorted_idx]
    std_scores = [std_scores[i] for i in sorted_idx]
    n_samples = [n_samples[i] for i in sorted_idx]
    
    # Determine dysarthric vs control (assuming naming convention)
    colors = ['#e74c3c' if not spk.startswith(('FC', 'MC')) else '#3498db' for spk in speakers]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Bar chart with error bars
    bars = ax.bar(range(len(speakers)), per_scores, color=colors, alpha=0.7, edgecolor='black')
    ax.errorbar(range(len(speakers)), per_scores, yerr=std_scores, 
               fmt='none', ecolor='black', capsize=4, alpha=0.5)
    
    ax.set_xticks(range(len(speakers)))
    ax.set_xticklabels(speakers, rotation=45, ha='right')
    ax.set_ylabel('PER', fontsize=12)
    ax.set_title('Phoneme Error Rate by Speaker', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add sample count labels
    for i, (bar, n) in enumerate(zip(bars, n_samples)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_scores[i] + 0.01,
               f'n={n}', ha='center', va='bottom', fontsize=8)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.7, label='Dysarthric'),
        Patch(facecolor='#3498db', alpha=0.7, label='Control')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved speaker PER plot to {save_path}")


def plot_common_confusions(
    common_confusions: List[Tuple[Tuple[str, str], int]],
    save_path: Path,
    top_k: int = 15
) -> None:
    """
    Plot most common phoneme substitution pairs.
    
    Args:
        common_confusions: List of ((ref, pred), count) tuples
        save_path: Path to save plot
        top_k: Number of top confusions to plot
    """
    if not common_confusions:
        return
    
    # Take top-k
    confusions = common_confusions[:top_k]
    labels = [f"{ref} → {pred}" for (ref, pred), _ in confusions]
    counts = [count for _, count in confusions]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(range(len(labels)), counts, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title(f'Top {top_k} Most Common Phoneme Substitutions', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2.,
               str(count), ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved common confusions to {save_path}")


def decode_predictions(logits: torch.Tensor, phn_to_id: Dict, id_to_phn: Dict) -> List[List[str]]:
    """
    Decode batch of logits to phoneme sequences using greedy decoding.
    
    Args:
        logits: Model logits [batch, time, num_classes]
        phn_to_id: Phoneme to ID mapping
        id_to_phn: ID to phoneme mapping
    
    Returns:
        List of phoneme sequences
    """
    predictions = []
    pred_ids = torch.argmax(logits, dim=-1)
    
    for seq in pred_ids:
        phonemes = []
        prev_id = None
        
        for phone_id in seq.cpu().numpy():
            # Skip blanks and padding
            if phone_id == phn_to_id['<BLANK>']:
                prev_id = None
                continue
            if phone_id == phn_to_id['<PAD>']:
                continue
            # Skip repetitions (CTC-style)
            if phone_id == prev_id:
                continue
            
            phoneme = id_to_phn.get(phone_id, '<UNK>')
            # Stress-agnostic comparison: strip ARPABET stress digits
            phoneme = str(phoneme).rstrip('012')
            if phoneme not in ['<BLANK>', '<PAD>', '<UNK>']:
                phonemes.append(phoneme)
            prev_id = phone_id
        
        predictions.append(phonemes)
    
    return predictions


def decode_references(labels: torch.Tensor, id_to_phn: Dict) -> List[List[str]]:
    """
    Decode reference labels to phoneme sequences.
    
    Args:
        labels: Label tensor [batch, seq_len]
        id_to_phn: ID to phoneme mapping
    
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
            # Stress-agnostic comparison: strip ARPABET stress digits
            phoneme = str(phoneme).rstrip('012')
            if phoneme not in ['<BLANK>', '<PAD>', '<UNK>']:
                phonemes.append(phoneme)
        references.append(phonemes)
    
    return references


def evaluate_model(
    model,
    dataloader,
    device: str,
    phn_to_id: Dict[str, int],
    id_to_phn: Dict[int, str],
    results_dir: Path = None,
    symbolic_rules: Dict[Tuple[str, str], float] = None
) -> Dict:
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Args:
        model: Trained model
        dataloader: Evaluation dataloader
        device: Device to run evaluation on
        phn_to_id: Phoneme to ID mapping
        id_to_phn: ID to phoneme mapping
        results_dir: Directory to save results
    
    Returns:
        Dictionary of evaluation metrics
    """
    if results_dir is None:
        results_dir = get_project_root() / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    all_predictions = []
    all_predictions_neural = []
    all_references = []
    all_status = []
    all_speakers = []
    all_beta_values = []
    
    print("🔍 Running evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            # Forward pass
            # Provide severity proxy (0=control, 1=dysarthric)
            severity = batch['status'].to(device)
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                speaker_severity=severity
            )
            
            # Decode predictions
            logits_constrained = outputs['logits_constrained']
            logits_neural = outputs['logits_neural']
            predictions = decode_predictions(logits_constrained, phn_to_id, id_to_phn)
            predictions_neural = decode_predictions(logits_neural, phn_to_id, id_to_phn)
            references = decode_references(labels, id_to_phn)
            
            all_predictions.extend(predictions)
            all_predictions_neural.extend(predictions_neural)
            all_references.extend(references)
            all_status.extend(batch['status'].cpu().numpy())
            all_speakers.extend(batch['speakers'])

            # Compute per-sample adaptive beta (replicate model logic)
            try:
                base_beta = float(model.symbolic_layer.beta.item())
            except Exception:
                base_beta = float(model.symbolic_layer.beta)
            # Severity normalization: use status (0 or 1) mapped to {0, 5}
            sev_np = batch['status'].cpu().numpy()
            for s in sev_np:
                sev_norm = (5.0 if s == 1 else 0.0) / 5.0
                beta_val = np.clip(base_beta + 0.1 * sev_norm, 0.0, 0.8)
                all_beta_values.append(float(beta_val))
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")
    
    # Compute overall metrics
    per_scores = [compute_per(p, r) for p, r in zip(all_predictions, all_references)]
    avg_per = np.mean(per_scores)
    
    # Stratified by dysarthric status
    dysarthric_per = [per_scores[i] for i in range(len(per_scores)) if all_status[i] == 1]
    control_per = [per_scores[i] for i in range(len(per_scores)) if all_status[i] == 0]
    
    # Per-speaker analysis
    speaker_per = defaultdict(list)
    for i, speaker in enumerate(all_speakers):
        speaker_per[speaker].append(per_scores[i])
    
    speaker_metrics = {
        spk: {
            'per': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'n_samples': len(scores)
        }
        for spk, scores in speaker_per.items()
    }
    
    # Phoneme-level error analysis
    print("📊 Analyzing phoneme-level errors...")
    error_analysis = analyze_phoneme_errors(all_predictions, all_references)
    manner_summary = summarize_minor_major(error_analysis['confusion_matrix'])
    grouped_conf = group_confusion_by_manner(error_analysis['confusion_matrix'])

    # Rule hit-rate analysis (optional)
    rule_counts = {}
    if symbolic_rules is None and get_default_config is not None:
        cfg = get_default_config()
        symbolic_rules = cfg.symbolic.substitution_rules
    if symbolic_rules:
        symbolic_rules = normalize_rule_keys(symbolic_rules)
        rule_counts = compute_rule_hit_counts(
            all_predictions_neural,
            all_predictions,
            all_references,
            symbolic_rules
        )
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("📈 Generating visualizations...")
    
    plot_confusion_matrix(
        error_analysis['confusion_matrix'],
        results_dir / 'confusion_matrix.png'
    )
    
    plot_error_distribution(
        error_analysis['error_counts'],
        results_dir / 'error_distribution.png'
    )
    
    plot_per_by_speaker(
        speaker_metrics,
        results_dir / 'per_by_speaker.png'
    )
    
    plot_common_confusions(
        error_analysis['common_confusions'],
        results_dir / 'common_confusions.png'
    )

    plot_confusion_by_manner(
        grouped_conf,
        results_dir / 'confusion_by_manner.png'
    )

    # Plot Beta vs PER (per-speaker averages)
    try:
        plot_beta_vs_per(
            per_scores,
            all_speakers,
            all_beta_values,
            results_dir / 'beta_vs_per.png'
        )
    except Exception as e:
        print(f"⚠️ Beta vs PER plot skipped: {e}")

    # Plot Top Rules Applied (if available)
    if rule_counts:
        plot_top_rules_applied(rule_counts, results_dir / 'top_rules_applied.png', top_k=10)
    
    # Compile results
    results = {
        'overall': {
            'per': float(avg_per),
            'per_std': float(np.std(per_scores)),
            'n_samples': len(per_scores)
        },
        'stratified': {
            'dysarthric_per': float(np.mean(dysarthric_per)) if dysarthric_per else 0.0,
            'dysarthric_n': len(dysarthric_per),
            'control_per': float(np.mean(control_per)) if control_per else 0.0,
            'control_n': len(control_per)
        },
        'per_speaker': speaker_metrics,
        'error_analysis': {
            'error_counts': error_analysis['error_counts'],
            'common_confusions': [(list(pair), count) for pair, count in error_analysis['common_confusions']],
            'deletion_phonemes': error_analysis['deletion_phonemes'],
            'insertion_phonemes': error_analysis['insertion_phonemes'],
            'manner_summary': manner_summary,
            'rule_counts': {f"{k[0]}→{k[1]}": v for k, v in (rule_counts.items() if rule_counts else [])}
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
    print(f"Overall PER:      {avg_per:.3f} ± {np.std(per_scores):.3f}")
    print(f"Dysarthric PER:   {results['stratified']['dysarthric_per']:.3f} (n={len(dysarthric_per)})")
    print(f"Control PER:      {results['stratified']['control_per']:.3f} (n={len(control_per)})")
    print(f"\nError Breakdown:")
    print(f"  Correct:        {error_analysis['error_counts']['correct']}")
    print(f"  Substitutions:  {error_analysis['error_counts']['substitutions']}")
    print(f"  Deletions:      {error_analysis['error_counts']['deletions']}")
    print(f"  Insertions:     {error_analysis['error_counts']['insertions']}")
    print("="*60)
    
    return results


# ---- Explainability helpers ----

def _alignment_ref_map(alignment: List[Tuple[str, Optional[str], Optional[str]]]) -> List[Optional[str]]:
    """
    Build list mapping each reference position to the predicted phoneme
    (or None if inserted/missing).
    """
    ref_map: List[Optional[str]] = []
    for op, pred_ph, ref_ph in alignment:
        if op in ('correct', 'substitute'):
            ref_map.append(pred_ph if pred_ph is not None else ref_ph)
        elif op == 'insert':
            ref_map.append(None)
        # 'delete' does not advance reference; ignore
    return ref_map


def compute_rule_hit_counts(
    preds_neural: List[List[str]],
    preds_constrained: List[List[str]],
    references: List[List[str]],
    rules: Dict[Tuple[str, str], float]
) -> Dict[Tuple[str, str], int]:
    """
    Count times when the constrained prediction corrected the neural prediction
    to match a known rule direction (constrained, neural).
    """
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for p_n, p_c, ref in zip(preds_neural, preds_constrained, references):
        aln_n = phoneme_alignment(p_n, ref)
        aln_c = phoneme_alignment(p_c, ref)
        map_n = _alignment_ref_map(aln_n)
        map_c = _alignment_ref_map(aln_c)
        L = min(len(ref), len(map_n), len(map_c))
        for i in range(L):
            pred_n = map_n[i]
            pred_c = map_c[i]
            if pred_n is None or pred_c is None:
                continue
            # Rule direction: (constrained, neural)
            key = (pred_c, pred_n)
            if key in rules:
                counts[key] += 1
    return dict(counts)


def normalize_rule_keys(rules: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    """Strip stress markers from rule keys to match stress-agnostic decoding."""
    cleaned: Dict[Tuple[str, str], float] = {}
    for (a, b), prob in rules.items():
        a_clean = str(a).rstrip('012')
        b_clean = str(b).rstrip('012')
        cleaned[(a_clean, b_clean)] = prob
    return cleaned


def plot_top_rules_applied(
    rule_counts: Dict[Tuple[str, str], int],
    save_path: Path,
    top_k: int = 10
) -> None:
    if not rule_counts:
        return
    items = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    labels = [f"{k[0]} → {k[1]}" for k, _ in items]
    counts = [v for _, v in items]
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(labels)), counts, color='#8e44ad', alpha=0.8, edgecolor='black')
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Count', fontsize=12)
    plt.title('Top Rules Applied by Symbolic Layer', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.gca().invert_yaxis()
    for bar, c in zip(bars, counts):
        plt.text(c + max(counts) * 0.02, bar.get_y() + bar.get_height()/2., str(c), va='center')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved top rules applied to {save_path}")


def plot_beta_vs_per(
    per_scores: List[float],
    speakers: List[str],
    beta_values: List[float],
    save_path: Path
) -> None:
    """
    Scatter: Speaker PER (X) vs Average Beta (Y).
    """
    # Aggregate per speaker
    spk_to_per: Dict[str, List[float]] = defaultdict(list)
    spk_to_beta: Dict[str, List[float]] = defaultdict(list)
    for spk, per, beta in zip(speakers, per_scores, beta_values):
        spk_to_per[spk].append(per)
        spk_to_beta[spk].append(beta)

    spks = list(spk_to_per.keys())
    x = [float(np.mean(spk_to_per[s])) for s in spks]
    y = [float(np.mean(spk_to_beta[s])) for s in spks]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c='#2c3e50', alpha=0.7, edgecolors='white')
    plt.xlabel('Speaker PER (avg)', fontsize=12)
    plt.ylabel('Average β (constraint weight)', fontsize=12)
    plt.title('β Activation vs. PER', fontsize=14, fontweight='bold')
    # Optional trend line
    if len(x) > 1:
        coeffs = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 100)
        ys = coeffs[0] * xs + coeffs[1]
        plt.plot(xs, ys, color='#e74c3c', linestyle='--', linewidth=1.5, label='Trend')
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved β vs PER scatter to {save_path}")


def main() -> None:
    # Test PER computation
    pred = ['P', 'B', 'T', 'AH', 'L']
    ref = ['B', 'AH', 'T', 'AH', 'L']
    
    per = compute_per(pred, ref)
    print(f"PER: {per:.3f}")
    
    alignment = phoneme_alignment(pred, ref)
    print("\nAlignment:")
    for op, p, r in alignment:
        print(f"{op:12s} | Pred: {p or '-':5s} | Ref: {r or '-':5s}")


if __name__ == "__main__":
    main()