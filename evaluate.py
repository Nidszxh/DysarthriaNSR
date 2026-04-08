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
"""

import json
import sys
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

try:
    from rapidfuzz.distance import Levenshtein as _RFLevenshtein
except Exception:
    _RFLevenshtein = None

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

try:
    from src.utils.sequence_utils import align_labels_to_logits as _align_labels_to_logits
except ImportError:
    # Inline fallback — should not happen in a correctly installed package
    def _align_labels_to_logits(labels: torch.Tensor, time_steps_logits: int) -> torch.Tensor:
        batch_size, t = labels.shape
        if t == time_steps_logits:
            return labels
        indices = (
            torch.arange(time_steps_logits, device=labels.device, dtype=torch.float32)
            * (t / float(time_steps_logits))
        ).long().clamp(0, t - 1)
        aligned = labels[:, indices]
        pad_mask = (labels == -100)
        if pad_mask.any():
            pad_fraction = pad_mask.float().mean(dim=1)
            n_pad = (pad_fraction * time_steps_logits).long()
            for b in range(batch_size):
                n = int(n_pad[b].item())
                if n > 0:
                    aligned[b, time_steps_logits - n:] = -100
        return aligned


# DECODING UTILITIES

class BigramLMScorer:
    """Phoneme bigram language model for CTC beam search shallow fusion.

    Built from training phoneme sequences using add-k (Laplace) smoothing.
    Provides log P(next_id | prev_id) for use as a beam-search bonus:

        combined_score = acoustic_score + lm_weight * lm_score

    Args:
        k: Laplace smoothing constant (default 0.5 — softer than add-1).
    """

    def __init__(self, k: float = 0.5) -> None:
        self.k = k
        self._bigram_counts: Dict = defaultdict(int)   # (prev_id, next_id) → count
        self._unigram_counts: Dict = defaultdict(int)  # prev_id → count
        self._vocab_size: int = 0
        self._log_prob_matrix: Optional[np.ndarray] = None
        self._built: bool = False

    def fit(
        self,
        phoneme_id_seqs: List[List[int]],
        vocab_size: int,
    ) -> None:
        """Count bigrams from training sequences and set vocab size for smoothing.

        Args:
            phoneme_id_seqs: List of phoneme-ID sequences (from training labels).
            vocab_size:      Total vocabulary size (len(phn_to_id)).
        """
        self._bigram_counts.clear()
        self._unigram_counts.clear()

        for seq in phoneme_id_seqs:
            clean = [t for t in seq if t >= 0]  # drop -100 padding
            for i in range(len(clean) - 1):
                self._bigram_counts[(clean[i], clean[i + 1])] += 1
                self._unigram_counts[clean[i]] += 1
            if clean:
                self._unigram_counts[clean[-1]] += 1

        self._vocab_size = vocab_size
        self._log_prob_matrix = np.empty((vocab_size, vocab_size), dtype=np.float32)

        for prev_id in range(vocab_size):
            uni = self._unigram_counts.get(prev_id, 0)
            den = max(uni + self.k * vocab_size, 1e-10)
            base = np.log(max(self.k / den, 1e-10))
            self._log_prob_matrix[prev_id, :] = base

        for (prev_id, next_id), cnt in self._bigram_counts.items():
            uni = self._unigram_counts.get(prev_id, 0)
            den = max(uni + self.k * vocab_size, 1e-10)
            self._log_prob_matrix[prev_id, next_id] = np.log(max((cnt + self.k) / den, 1e-10))

        self._built = True

    def log_prob(self, prev_id: int, next_id: int) -> float:
        """Return log P(next_id | prev_id) with add-k smoothing."""
        if not self._built:
            return 0.0
        if self._log_prob_matrix is not None:
            return float(self._log_prob_matrix[prev_id, next_id])
        num = self._bigram_counts.get((prev_id, next_id), 0) + self.k
        den = self._unigram_counts.get(prev_id, 0) + self.k * self._vocab_size
        return float(np.log(max(num / max(den, 1e-10), 1e-10)))

    def score_prefix(self, prefix: tuple) -> float:
        """Sum log-bigram probs for all consecutive pairs in prefix."""
        if len(prefix) < 2:
            return 0.0
        return sum(
            self.log_prob(prefix[i], prefix[i + 1])
            for i in range(len(prefix) - 1)
        )


class BeamSearchDecoder:
    """
    CTC Beam Search Decoder with prefix pruning.
    
    Implements efficient beam search for CTC outputs using prefix merging.
    Significantly more accurate than greedy decoding for ambiguous sequences.
    """
    
    def __init__(
        self,
        beam_width: int = 10,
        blank_id: int = 0,
        length_norm_alpha: float = 0.0,
        lm_scorer: Optional["BigramLMScorer"] = None,
        lm_weight: float = 0.0,
    ) -> None:
        self.beam_width        = beam_width
        self.blank_id          = blank_id
        self.length_norm_alpha = length_norm_alpha
        self.lm_scorer         = lm_scorer    # Optional bigram LM for shallow fusion
        self.lm_weight         = lm_weight    # λ; 0.0 = acoustic-only
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
        # Use float64 in numpy beam arithmetic for stability (especially when
        # model outputs come from AMP/bfloat16).  We keep CTC semantics strict:
        # only BLANK contributes to p_blank, while PAD/UNK are excluded from
        # search-space normalization entirely.
        log_probs = np.asarray(log_probs, dtype=np.float64)
        T, num_classes = log_probs.shape

        blank_id = int(self.blank_id)
        special_ids = {
            int(idx)
            for idx, tok in id_to_phn.items()
            if tok in ('<BLANK>', '<PAD>', '<UNK>')
        }
        if blank_id not in special_ids:
            special_ids.add(blank_id)

        emit_ids = [
            i for i in range(num_classes)
            if (i not in special_ids)
        ]
        allowed_ids = [blank_id] + emit_ids

        # Beam state: {prefix: (p_blank, p_non_blank, lm_for_blank, lm_for_non_blank)}
        # p_blank and p_non_blank are PURE acoustic log-probs (CTC semantics).
        # Separate LM accumulators keep LM scores path-consistent with the
        # acoustic branch that produced each hypothesis component.
        beam: Dict[tuple, tuple] = {(): (0.0, float('-inf'), 0.0, 0.0)}

        use_lm = (self.lm_scorer is not None and self.lm_weight > 0.0)

        for t in range(T):
            new_beam: Dict[tuple, tuple] = defaultdict(
                lambda: (float('-inf'), float('-inf'), 0.0, 0.0)
            )

            # Renormalize each frame over CTC-valid IDs only:
            # {BLANK + lexical phonemes}. This prevents PAD/UNK probability mass
            # from artificially boosting blank transitions.
            frame_allowed = log_probs[t, allowed_ids]
            frame_log_z = np.logaddexp.reduce(frame_allowed)
            blank_lp = log_probs[t, blank_id] - frame_log_z

            for prefix, (p_b, p_nb, lm_b_acc, lm_nb_acc) in beam.items():
                # --- Extend with true CTC blank only ---------------------------
                new_p_b = np.logaddexp(p_b, p_nb) + blank_lp
                src_lm_for_blank = lm_b_acc if p_b >= p_nb else lm_nb_acc
                prev_b, prev_nb, prev_lm_b, prev_lm_nb = new_beam[prefix]
                new_beam[prefix] = (
                    np.logaddexp(prev_b, new_p_b),
                    prev_nb,
                    src_lm_for_blank if new_p_b > prev_b else prev_lm_b,
                    prev_lm_nb,
                )

                # --- Extend with non-blank phonemes ---------------------------
                for c in emit_ids:
                    c_lp = log_probs[t, c] - frame_log_z

                    if len(prefix) > 0 and prefix[-1] == c:
                        # Repeated token (CTC collapse): same labeling prefix.
                        # Include BOTH p_blank and p_non_blank predecessors.
                        # Omitting p_non_blank underestimates long same-token
                        # runs and can bias beam search toward blank/deletions.
                        new_prefix = prefix
                        new_p_nb   = np.logaddexp(p_b, p_nb) + c_lp
                        new_lm_acc = lm_nb_acc if p_nb >= p_b else lm_b_acc
                    else:
                        # New token → extend prefix, update LM accumulator
                        new_prefix = prefix + (c,)
                        new_p_nb   = np.logaddexp(p_b, p_nb) + c_lp
                        src_lm_for_nb = lm_nb_acc if p_nb >= p_b else lm_b_acc
                        if use_lm and len(prefix) > 0:
                            # Incremental bigram score for this single new token
                            new_lm_acc = src_lm_for_nb + self.lm_scorer.log_prob(prefix[-1], c)
                        else:
                            new_lm_acc = src_lm_for_nb

                    prev_b, prev_nb, prev_lm_b, prev_lm_nb = new_beam[new_prefix]
                    new_beam[new_prefix] = (
                        prev_b,
                        np.logaddexp(prev_nb, new_p_nb),
                        prev_lm_b,
                        new_lm_acc if new_p_nb > prev_nb else prev_lm_nb,
                    )

            # Prune to beam_width — rank by acoustic + LM together
            def _beam_rank(item: tuple) -> float:
                pfx, (pb, pnb, lm_b, lm_nb) = item
                lm_s = lm_nb if pnb >= pb else lm_b
                return np.logaddexp(pb, pnb) + (self.lm_weight * lm_s if use_lm else 0.0)

            beam = dict(sorted(new_beam.items(), key=_beam_rank, reverse=True)[:self.beam_width])

        # Final selection: acoustic / len^α + λ * lm_cumulative
        # §3.3 fix: length norm applied to acoustic score only; LM bonus is
        # per-token so it should NOT be divided by sequence length.
        def _norm_score(prefix: tuple, p_b: float, p_nb: float, lm_b: float, lm_nb: float) -> float:
            acoustic  = np.logaddexp(p_b, p_nb)
            lm_s = lm_nb if p_nb >= p_b else lm_b
            lm_bonus  = (self.lm_weight * lm_s) if use_lm else 0.0
            length    = max(len(prefix), 1)
            return acoustic / (length ** self.length_norm_alpha) + lm_bonus

        best_prefix, (p_b, p_nb, _lm_b, _lm_nb) = max(
            beam.items(),
            key=lambda x: _norm_score(x[0], x[1][0], x[1][1], x[1][2], x[1][3])
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
    id_to_phn: Dict[int, str],
    output_lengths: Optional[torch.Tensor] = None,
) -> List[List[str]]:
    """
    Greedy CTC decoding with correct collapse and padding-frame masking.

    CTC collapse rules applied in order per frame:
      1. Blank token  → reset prev_id; skip frame.
      2. PAD token    → reset prev_id; skip frame (same semantics as blank).
      3. Repeat token → skip frame (consecutive duplicate).
      4. Real token   → emit and update prev_id.

    Args:
        logits:         Model output [batch, time, num_classes].
                        May be raw logits or log-probs — argmax is identical.
        phn_to_id:      Phoneme → ID mapping.
        id_to_phn:      ID → Phoneme mapping.
        output_lengths: Valid frame count per sample [batch].  When supplied,
                        frames beyond output_lengths[i] are not decoded,
                        preventing padding-frame noise from being emitted as
                        spurious phoneme insertions.

    Returns:
        List of phoneme sequences (one per batch sample).
    """
    blank_id = phn_to_id.get('<BLANK>', 0)
    pad_id   = phn_to_id.get('<PAD>',   1)

    predictions = []
    pred_ids = torch.argmax(logits, dim=-1)  # [B, T]

    for i, seq in enumerate(pred_ids):
        # Truncate to valid (non-padded) frames only.
        # Without this, zero-padded audio frames produce random non-blank
        # predictions that inflate the insertion count dramatically.
        if output_lengths is not None:
            valid_len = int(output_lengths[i].item())
            seq = seq[:valid_len]

        phonemes: List[str] = []
        prev_id: Optional[int] = None

        for phone_id in seq.cpu().numpy():
            phone_id = int(phone_id)

            # Blank: separator in CTC — resets repeat-detector, emits nothing.
            if phone_id == blank_id:
                prev_id = None
                continue

            # PAD: treat identically to blank (resets repeat-detector).
            if phone_id == pad_id:
                prev_id = None
                continue

            # Consecutive duplicate without an intervening blank → collapse.
            if phone_id == prev_id:
                continue

            phoneme = id_to_phn.get(phone_id, '<UNK>')
            phoneme = normalize_phoneme(phoneme)
            if phoneme not in ('<BLANK>', '<PAD>', '<UNK>'):
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


def bootstrap_paired_per_delta(
    constrained_per_scores: List[float],
    neural_per_scores: List[float],
    n_bootstrap: int = 10000,
) -> Dict[str, float]:
    """Estimate paired PER delta significance via bootstrap resampling.

    Delta is defined as ``mean(constrained - neural)`` so positive values imply
    constrained decoding is worse.

    Returns:
        Dict with ``delta_mean``, ``ci_95_low``, ``ci_95_high``, and
        a two-sided empirical ``p_value_two_sided`` for H0: delta == 0.
    """
    if len(constrained_per_scores) == 0 or len(constrained_per_scores) != len(neural_per_scores):
        return {
            'delta_mean': float('nan'),
            'ci_95_low': float('nan'),
            'ci_95_high': float('nan'),
            'p_value_two_sided': float('nan'),
        }

    c = np.asarray(constrained_per_scores, dtype=float)
    n = np.asarray(neural_per_scores, dtype=float)
    deltas = c - n
    delta_mean = float(np.mean(deltas))

    boot = []
    m = len(deltas)
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, m, size=m)
        boot.append(float(np.mean(deltas[idx])))
    boot_arr = np.asarray(boot, dtype=float)

    ci_low = float(np.percentile(boot_arr, 2.5))
    ci_high = float(np.percentile(boot_arr, 97.5))

    # Two-sided empirical p-value around zero.
    p_left = float(np.mean(boot_arr <= 0.0))
    p_right = float(np.mean(boot_arr >= 0.0))
    p_two_sided = float(min(1.0, 2.0 * min(p_left, p_right)))

    return {
        'delta_mean': delta_mean,
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
        'p_value_two_sided': p_two_sided,
    }


def compute_wer_texts(
    predictions_text: List[str],
    references_text: List[str]
) -> float:
    """
    Compute Word Error Rate (WER) for lists of predicted and reference texts.

    Compatible with jiwer ≥ 4.0 (list-level batch API).
    When phoneme sequences are passed as space-separated strings the
    "words" are individual phoneme tokens, so WER == PER at the token level.

    Args:
        predictions_text: List of predicted transcripts (strings)
        references_text: List of reference transcripts (strings)

    Returns:
        Corpus-level WER (float in [0, ∞))
    """
    if not predictions_text or not references_text:
        return 0.0

    assert len(predictions_text) == len(references_text), "Pred/Ref text length mismatch"

    # Normalise to lowercase, strip extra whitespace
    def _norm(s: str) -> str:
        return ' '.join(s.lower().split())

    refs  = [_norm(r) for r in references_text]
    hyps  = [_norm(p) for p in predictions_text]

    # jiwer 4.x can compute corpus WER directly from two lists of strings
    return float(jiwer.wer(refs, hyps))


def decode_predictions(
    log_probs: torch.Tensor,
    phn_to_id: Dict[str, int],
    id_to_phn: Dict[int, str],
    use_beam_search: bool = False,
    beam_width: int = 10,
    output_lengths: Optional[torch.Tensor] = None,
    lm_scorer: Optional[BigramLMScorer] = None,
    lm_weight: float = 0.0,
) -> List[List[str]]:
    """
    Decode model logits into phoneme sequences.

    Args:
        log_probs:      Model output [batch, time, num_classes].
        phn_to_id:      Phoneme → ID mapping.
        id_to_phn:      ID → Phoneme mapping.
        use_beam_search: If True use CTC beam search; otherwise greedy.
        beam_width:     Beam width for beam search.
        output_lengths: Valid frame count per sample [batch] (from model).
                        Prevents padding-frame noise from becoming insertions.
        lm_scorer:      Optional BigramLMScorer for shallow-fusion LM rescoring.
        lm_weight:      LM weight λ for shallow fusion (0.0 = disabled).
    """
    if use_beam_search:
        # §3.1 fix: `config` is not in scope here; use hardcoded default 0.6
        # (caller can override indirectly through BeamSearchDecoder constructor).
        _alpha = 0.6
        decoder = BeamSearchDecoder(
            beam_width=beam_width,
            blank_id=phn_to_id['<BLANK>'],
            length_norm_alpha=_alpha,
            lm_scorer=lm_scorer,
            lm_weight=lm_weight,
        )
        predictions = []
        for i in range(log_probs.size(0)):
            # Truncate to valid frames before handing to beam search.
            valid_len = (
                int(output_lengths[i].item())
                if output_lengths is not None
                else log_probs.size(1)
            )
            seq, _ = decoder.decode(
                log_probs[i, :valid_len].float().cpu().numpy(),
                id_to_phn
            )
            predictions.append(seq)
        return predictions
    else:
        return greedy_decode(log_probs, phn_to_id, id_to_phn, output_lengths=output_lengths)


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

    # O-5: Fast C-extension path via rapidfuzz (fallback to DP if unavailable).
    if _RFLevenshtein is not None:
        try:
            ops = _RFLevenshtein.editops(pred, ref)
            alignment: List[Tuple[str, Optional[str], Optional[str]]] = []
            i = j = 0

            for op in ops:
                src = int(op.src_pos)
                dst = int(op.dest_pos)

                # Emit exact matches before the next edit operation.
                while i < src and j < dst:
                    alignment.append(('correct', pred[i], ref[j]))
                    i += 1
                    j += 1

                if op.tag == 'replace':
                    alignment.append(('substitute', pred[src], ref[dst]))
                    i = src + 1
                    j = dst + 1
                elif op.tag == 'delete':
                    alignment.append(('delete', pred[src], None))
                    i = src + 1
                    j = dst
                elif op.tag == 'insert':
                    alignment.append(('insert', None, ref[dst]))
                    i = src
                    j = dst + 1

            # Flush trailing exact matches and residual tails.
            while i < m and j < n:
                alignment.append(('correct', pred[i], ref[j]))
                i += 1
                j += 1
            while i < m:
                alignment.append(('delete', pred[i], None))
                i += 1
            while j < n:
                alignment.append(('insert', None, ref[j]))
                j += 1

            return alignment
        except Exception:
            # Safety-first: keep original pure-Python path on any unexpected error.
            pass
    
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
    references: List[List[str]],
    alignments: Optional[List[List[Tuple]]] = None,
) -> Dict:
    """
    Comprehensive phoneme-level error analysis.
    
    Args:
        predictions: List of predicted phoneme sequences
        references: List of reference phoneme sequences
        alignments: Optional pre-computed alignments from ``phoneme_alignment()``.
                    When provided, skips internal alignment (4× speedup when shared
                    across multiple analysis functions).
    
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
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        alignment = alignments[i] if alignments is not None else phoneme_alignment(pred, ref)
        
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


def compute_per_phoneme_breakdown(
    predictions: List[List[str]],
    references: List[List[str]],
    alignments: Optional[List[List[Tuple]]] = None,
) -> Dict[str, Dict]:
    """
    Compute per-phoneme substitution, deletion, insertion counts and PER.

    For each phoneme that appears in the reference, computes:
      - n_ref : total occurrences in reference
      - n_sub : times it was substituted by a different phoneme
      - n_del : times it was deleted
      - n_ins : times it was spuriously inserted as that phoneme token
      - per   : (n_sub + n_del) / n_ref

    Args:
        predictions: List of predicted phoneme sequences.
        references:  List of reference phoneme sequences.
        alignments:  Optional pre-computed alignments. When provided, skips
                     internal ``phoneme_alignment()`` calls (4× speedup).

    Returns:
        Dict mapping phoneme → {n_ref, n_sub, n_del, n_ins, per},
        sorted by PER descending.
    """
    from collections import defaultdict as _dd
    counts: Dict[str, Dict[str, int]] = _dd(lambda: {'n_ref': 0, 'n_sub': 0, 'n_del': 0, 'n_ins': 0})

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        alignment = alignments[i] if alignments is not None else phoneme_alignment(pred, ref)
        for op, pred_ph, ref_ph in alignment:
            if op == 'correct':
                # ref_ph correctly predicted
                counts[ref_ph]['n_ref'] += 1
            elif op == 'substitute':
                # ref_ph present in ref but wrongly predicted as pred_ph
                counts[ref_ph]['n_ref'] += 1
                counts[ref_ph]['n_sub'] += 1
                if pred_ph:
                    counts[pred_ph]['n_ins'] += 1  # spurious prediction
            elif op == 'insert':
                # ref has ref_ph but prediction missed it → deletion from ASR view
                # (pred_ph is None in this alignment op)
                if ref_ph:
                    counts[ref_ph]['n_ref'] += 1
                    counts[ref_ph]['n_del'] += 1
            elif op == 'delete':
                # pred has pred_ph not present in ref → spurious insertion from ASR view
                # (ref_ph is None in this alignment op)
                if pred_ph:
                    counts[pred_ph]['n_ins'] += 1

    result: Dict[str, Dict] = {}
    for ph, c in counts.items():
        n_ref = c['n_ref']
        per_ph = (c['n_sub'] + c['n_del']) / n_ref if n_ref > 0 else 0.0
        result[ph] = {
            'n_ref': n_ref,
            'n_sub': c['n_sub'],
            'n_del': c['n_del'],
            'n_ins': c['n_ins'],
            'per': round(float(per_ph), 4),
        }

    # Sort by PER descending for human-readable JSON output
    return dict(sorted(result.items(), key=lambda x: x[1]['per'], reverse=True))


def compute_articulatory_stratified_per(
    predictions: List[List[str]],
    references: List[List[str]],
    alignments: Optional[List[List[Tuple]]] = None,
) -> Dict[str, Dict]:
    """
    Compute PER stratified by manner-of-articulation class (E-02).

    Groups reference phonemes by their manner class (stop, fricative, nasal,
    liquid, glide, vowel, affricate, diphthong) and reports per-class error
    counts and PER.  Phonemes not present in PHONEME_FEATURES are grouped
    under the key ``"other"``.

    Args:
        predictions: List of predicted phoneme sequences.
        references:  List of reference phoneme sequences.
        alignments:  Optional pre-computed alignments. When provided, skips
                     internal ``phoneme_alignment()`` calls (4× speedup).

    Returns:
        Dict mapping manner_class → {per, n_ref, n_sub, n_del}, sorted by
        PER descending.
    """
    from src.models.model import ArticulatoryFeatureEncoder

    phoneme_features = ArticulatoryFeatureEncoder.PHONEME_FEATURES
    from collections import defaultdict as _dd
    counts: Dict[str, Dict[str, int]] = _dd(lambda: {'n_ref': 0, 'n_sub': 0, 'n_del': 0})

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        alignment = alignments[i] if alignments is not None else phoneme_alignment(pred, ref)
        for op, _pred_ph, ref_ph in alignment:
            if op in ('correct', 'substitute', 'insert'):
                if ref_ph:
                    manner = phoneme_features.get(ref_ph, {}).get('manner', 'other')
                    counts[manner]['n_ref'] += 1
                    if op == 'substitute':
                        counts[manner]['n_sub'] += 1
                    elif op == 'insert':  # deletion from ASR perspective
                        counts[manner]['n_del'] += 1

    result: Dict[str, Dict] = {}
    for manner, c in counts.items():
        n_ref = c['n_ref']
        per_m = (c['n_sub'] + c['n_del']) / n_ref if n_ref > 0 else 0.0
        result[manner] = {
            'per': round(float(per_m), 4),
            'n_ref': n_ref,
            'n_sub': c['n_sub'],
            'n_del': c['n_del'],
        }

    return dict(sorted(result.items(), key=lambda x: x[1]['per'], reverse=True))


def compute_rule_pair_confusion(
    predictions_neural: List[List[str]],
    predictions_constrained: List[List[str]],
    references: List[List[str]],
    substitution_rules: Dict,
    neural_alignments: Optional[List[List[Tuple]]] = None,
    constrained_alignments: Optional[List[List[Tuple]]] = None,
) -> Dict[str, Dict]:
    """
    Compare per-rule-pair substitution counts between the neural and constrained
    decoders (E-03).

    For each symbolic substitution rule (e.g. B→P), counts how many times that
    substitution appears in the neural predictions vs. the constrained predictions
    and reports the delta.  A positive delta means the constraint *reduced* that
    substitution (helpful); a negative delta means it *increased* it (harmful).

    Args:
        predictions_neural:      Neural-only decoded phoneme sequences.
        predictions_constrained: Constrained decoded phoneme sequences.
        references:              Ground-truth phoneme sequences.
        substitution_rules:      Dict with Tuple[str, str] keys, e.g. {('B','P'): 0.85}.
        neural_alignments:       Optional pre-computed alignments for neural predictions.
                                 When provided, skips internal ``phoneme_alignment()``
                                 calls (4× speedup when shared across functions).
        constrained_alignments:  Optional pre-computed alignments for constrained predictions.

    Returns:
        Dict mapping ``"SRC->TGT"`` → {neural_count, constrained_count, delta,
        rule_weight}, sorted by |delta| descending.
    """
    from collections import defaultdict as _dd

    def _count_subs(
        preds: List[List[str]],
        refs: List[List[str]],
        cached: Optional[List[List[Tuple]]] = None,
    ) -> Dict[str, int]:
        sub_counts: Dict[str, int] = _dd(int)
        for i, (pred, ref) in enumerate(zip(preds, refs)):
            alignment = cached[i] if cached is not None else phoneme_alignment(pred, ref)
            for op, pred_ph, ref_ph in alignment:
                if op == 'substitute' and ref_ph and pred_ph:
                    sub_counts[f"{ref_ph}->{pred_ph}"] += 1
        return dict(sub_counts)

    neural_subs      = _count_subs(predictions_neural,      references, neural_alignments)
    constrained_subs = _count_subs(predictions_constrained, references, constrained_alignments)

    result: Dict[str, Dict] = {}
    for (src, tgt), weight in substitution_rules.items():
        key = f"{src}->{tgt}"
        n_neural = neural_subs.get(key, 0)
        n_const  = constrained_subs.get(key, 0)
        result[key] = {
            'neural_count':      n_neural,
            'constrained_count': n_const,
            'delta':             n_neural - n_const,  # positive = constraint reduced it
            'rule_weight':       float(weight),
        }

    return dict(sorted(result.items(), key=lambda x: abs(x[1]['delta']), reverse=True))


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


def compute_severity_stratified_per(
    per_by_speaker: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Stratify per-speaker PER into TORGO clinical severity buckets.

    Buckets are defined from continuous severity scores derived from TORGO
    intelligibility estimates (Rudzicz et al., 2012):
    - ``normal``   : severity = 0.0 (control speakers)
    - ``mild``     : 0.0 < severity < 3.0
    - ``moderate`` : 3.0 <= severity < 4.0
    - ``severe``   : 4.0 <= severity < 4.7
    - ``profound`` : severity >= 4.7

    Args:
        per_by_speaker: Mapping of speaker_id -> mean PER.

    Returns:
        Mapping bucket -> summary metrics.
    """
    from src.utils.config import get_speaker_severity

    bucket_order = ('normal', 'mild', 'moderate', 'severe', 'profound')
    buckets: Dict[str, List[float]] = {k: [] for k in bucket_order}
    bucket_speakers: Dict[str, List[str]] = {k: [] for k in bucket_order}

    def _bucket_from_severity(sev: float) -> str:
        if sev <= 0.0:
            return 'normal'
        if sev < 3.0:
            return 'mild'
        if sev < 4.0:
            return 'moderate'
        if sev < 4.7:
            return 'severe'
        return 'profound'

    for spk, per in per_by_speaker.items():
        sev = get_speaker_severity(spk)
        bucket = _bucket_from_severity(sev)
        buckets[bucket].append(per)
        bucket_speakers[bucket].append(spk)

    result = {}
    for b in bucket_order:
        scores = buckets[b]
        if scores:
            mean_val, ci = compute_per_with_ci(scores)
        else:
            mean_val, ci = 0.0, [0.0, 0.0]
        result[b] = {
            'mean_per':   float(mean_val),
            'ci':         ci,
            'n_speakers': len(scores),
            'speakers':   bucket_speakers[b],
        }
    return result


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
    matrix_norm = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    
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


def plot_rule_impact(
    stats: Dict,
    save_path: Path,
    model=None,
    id_to_phn: Optional[Dict[int, str]] = None,
) -> None:
    """Visualize the most active symbolic corrections.

    E6: When no rule activations have been recorded (``top_rules`` is empty),
    fall back to displaying the current constraint matrix as a log-scale heatmap.
    This gives reviewers a symbolic-layer visualisation even when
    ``SymbolicRuleTracker`` returns zero activations.

    Args:
        stats:      Rule statistics dict from ``model.get_rule_statistics()``.
        save_path:  PNG output path.
        model:      Optional model reference; used for fallback heatmap.
        id_to_phn:  Optional ID→phoneme mapping; used to label heatmap axes.
    """
    if stats and stats.get('top_rules'):
        # Normal path: bar chart of rule activation frequencies
        rules  = [f"{r[0][0]}->{r[0][1]}" for r in stats['top_rules']]
        counts = [r[1] for r in stats['top_rules']]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=counts, y=rules, palette="viridis")
        plt.title("Top Symbolic Corrections (Rule Activations)")
        plt.xlabel("Frequency of Activation")
        plt.ylabel("Phoneme Substitution (Neural -> Symbolic)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    # E6 fallback: constraint matrix heatmap 
    if model is None:
        return  # Nothing to show
    try:
        C = model.get_constraint_matrix()  # [V, V] tensor (or numpy array)
        if hasattr(C, 'cpu'):
            C = C.cpu().numpy()
        C = np.array(C, dtype=float)

        # Use log scale to reveal small non-zero entries
        C_log = np.log1p(C)

        # Build tick labels from id_to_phn if available
        n = C.shape[0]
        if id_to_phn and len(id_to_phn) == n:
            labels = [id_to_phn.get(i, str(i)) for i in range(n)]
        else:
            labels = None

        fig, ax = plt.subplots(figsize=(max(10, n // 3), max(8, n // 3)))
        im = ax.imshow(C_log, aspect='auto', cmap='viridis', origin='upper')
        plt.colorbar(im, ax=ax, label='log(1 + constraint weight)')
        if labels:
            tick_step = max(1, n // 20)
            ticks = list(range(0, n, tick_step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([labels[t] for t in ticks], rotation=90, fontsize=6)
            ax.set_yticklabels([labels[t] for t in ticks], fontsize=6)
        ax.set_xlabel('Target phoneme (column)')
        ax.set_ylabel('Source phoneme (row)')
        ax.set_title('Symbolic Constraint Matrix (E6 fallback — no rule activations recorded)')
        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print("✅ Constraint matrix heatmap saved (E6 fallback)")
    except Exception as _e6_exc:
        print(f"⚠️  Constraint matrix heatmap failed (non-fatal): {_e6_exc}")


def plot_blank_histogram(blank_probs: List[float], save_path: Path, target_prob: float = 0.75) -> None:
    """Histogram of per-utterance mean blank probabilities (I2 diagnostic).

    A healthy CTC model should have a blank probability distribution centred
    around the ``blank_target_prob`` configured in ``TrainingConfig``.
    A distribution peaking near 1.0 indicates the insertion bias pathology.

    Args:
        blank_probs:  List of per-utterance mean blank probabilities.
        save_path:    PNG output path.
        target_prob:  Target blank probability drawn as a reference line
                      (should match ``TrainingConfig.blank_target_prob``, default 0.75).
    """
    if not blank_probs:
        return
    arr = np.array(blank_probs, dtype=float)
    mean_bp = float(arr.mean())
    median_bp = float(np.median(arr))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(mean_bp,   color='crimson',  linestyle='--', linewidth=1.5,
               label=f'mean={mean_bp:.3f}')
    ax.axvline(median_bp, color='darkorange', linestyle=':', linewidth=1.5,
               label=f'median={median_bp:.3f}')
    ax.axvline(target_prob, color='green', linestyle='-', linewidth=1.2, alpha=0.6,
               label=f'target={target_prob:.2f}')
    ax.set_xlabel('Mean blank probability per utterance')
    ax.set_ylabel('Count')
    ax.set_title('Blank Probability Distribution (I2 — CTC Insertion Diagnostic)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_clinical_gap(
    dysarthric_mean: float,
    control_mean: float,
    save_path: Path
) -> None:
    """Plot performance gap between dysarthric and control speakers."""
    labels = ["Dysarthric", "Control"]
    values = [dysarthric_mean, control_mean]

    plt.figure(figsize=(6, 4))
    # Seaborn 0.14+ requires hue when passing a palette.
    sns.barplot(
        x=labels,
        y=values,
        hue=labels,
        palette=["#e74c3c", "#3498db"],
        legend=False,
    )
    plt.ylabel("PER")
    plt.title("Clinical Diagnostic: Dysarthric vs Control")
    plt.ylim(0.0, max(values) * 1.1 if values else 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_phoneme_breakdown(
    breakdown: Dict[str, Dict],
    save_path: Path,
    min_refs: int = 10,
    top_k: int = 30,
) -> None:
    """
    Bar chart of per-phoneme PER for the most frequently occurring phonemes.

    Phonemes with fewer than ``min_refs`` reference occurrences are excluded
    to avoid noisy estimates from rare tokens.

    Args:
        breakdown : Output of compute_per_phoneme_breakdown().
        save_path : Path at which to write the PNG.  Parent is created if needed.
        min_refs  : Minimum reference Count (default 10).  Lower → more phonemes shown.
        top_k     : Maximum bars to display (default 30, sorted by PER desc).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter rare phonemes and take top_k by PER
    filtered = {
        ph: v for ph, v in breakdown.items()
        if v['n_ref'] >= min_refs
    }
    sorted_items = sorted(filtered.items(), key=lambda x: x[1]['per'], reverse=True)[:top_k]

    if not sorted_items:
        print(f"⚠️  No phonemes with ≥{min_refs} reference occurrences — skipping per-phoneme plot.")
        return

    phonemes = [item[0] for item in sorted_items]
    per_vals  = [item[1]['per'] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(max(10, len(phonemes) * 0.45), 6))
    colors = [
        '#e74c3c' if v > 0.5 else
        '#f39c12' if v > 0.25 else
        '#27ae60'
        for v in per_vals
    ]
    ax.bar(phonemes, per_vals, color=colors, alpha=0.85)
    ax.set_xlabel('Phoneme', fontsize=12)
    ax.set_ylabel('PER  (sub+del / n_ref)', fontsize=12)
    ax.set_title(
        f'Per-Phoneme Error Rate  (top {len(phonemes)}, min_refs={min_refs})',
        fontsize=13, fontweight='bold',
    )
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.50, color='red',    linestyle='--', linewidth=0.9, alpha=0.6, label='PER=0.50')
    ax.axhline(0.25, color='orange', linestyle='--', linewidth=0.9, alpha=0.6, label='PER=0.25')
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved per-phoneme PER chart to {save_path}")


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
    beam_width: int = 10,
    generate_explanations: bool = False,
    compute_uncertainty: bool = False,
    uncertainty_n_samples: int = 20,
    lm_scorer: Optional[BigramLMScorer] = None,
    lm_weight: float = 0.0,
    ablation_mode: str = "full",
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
        generate_explanations: Run explainability pipeline and save artifacts (default: False)
        compute_uncertainty: Enable MC-Dropout uncertainty estimation via
            UncertaintyAwareDecoder (default: False)
        uncertainty_n_samples: Number of MC-Dropout forward passes (default: 20)
        lm_scorer: Optional BigramLMScorer for beam-search shallow fusion.
        lm_weight: LM weight λ (0.0 = acoustic-only; typical range 0.1–0.5).

    Returns:
        Dictionary of evaluation metrics with confidence intervals
    """
    if results_dir is None:
        results_dir = get_project_root() / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Ensure model parameters and input tensors live on the same device.
    # This is critical for LOSO resume paths where checkpoints may be loaded
    # on CPU and evaluated directly without a Trainer.fit() call.
    device = torch.device(device)
    model = model.to(device)
    if hasattr(model, 'symbolic_kl_loss') and getattr(model, 'symbolic_kl_loss') is not None:
        model.symbolic_kl_loss = model.symbolic_kl_loss.to(device)

    model.eval()

    # Initialize decoder
    if use_beam_search:
        decoder = BeamSearchDecoder(
            beam_width=beam_width,
            blank_id=phn_to_id['<BLANK>'],
            length_norm_alpha=0.6,
            lm_scorer=lm_scorer,
            lm_weight=lm_weight,
        )
        _lm_tag = f", LM λ={lm_weight:.2f}" if (lm_scorer is not None and lm_weight > 0) else ""
        print(f"🔍 Using beam search decoder (width={beam_width}{_lm_tag})")
    else:
        print("🔍 Using greedy decoder")
    
    all_predictions = []
    all_references = []
    all_predictions_neural = []
    all_status = []
    all_speakers = []
    all_phoneme_lengths = []
    all_blank_probs: List[float] = []   # I2: per-utterance mean blank probability
    articulatory_results = {"manner": [], "place": [], "voice": []}

    # Uncertainty estimation (ROADMAP §9, UncertaintyAwareDecoder)
    uncertainty_decoder = None
    all_utterance_uncertainty: list = []
    all_confidence_scores: list = []
    if compute_uncertainty:
        try:
            from src.models.uncertainty import UncertaintyAwareDecoder
            uncertainty_decoder = UncertaintyAwareDecoder(
                model, n_samples=uncertainty_n_samples
            )
            print(f"🎲 MC-Dropout uncertainty enabled ({uncertainty_n_samples} samples)")
        except Exception as exc:
            print(f"⚠️  UncertaintyAwareDecoder unavailable (non-fatal): {exc}")

    print("🔍 Running evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            # Forward pass — scale severity to [0, 5] to match training regime (bug fix S3)
            speakers_batch = batch.get('speakers', [])
            if speakers_batch and isinstance(speakers_batch[0], str):
                from src.utils.config import get_speaker_severity
                severity = torch.tensor(
                    [get_speaker_severity(s) for s in speakers_batch],
                    dtype=torch.float32, device=device
                )
            else:
                severity = batch['status'].float().to(device) * 5.0  # 0/1 → 0.0/5.0

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                speaker_severity=severity,
                output_attentions=generate_explanations,
                ablation_mode=ablation_mode,
            )

            # Per-utterance uncertainty (MC-Dropout, optional)
            if uncertainty_decoder is not None:
                try:
                    with torch.enable_grad():
                        unc_out = uncertainty_decoder.predict_with_uncertainty(
                            input_values, attention_mask, severity
                        )
                    batch_unc = unc_out.get('utterance_uncertainty')
                    if batch_unc is not None:
                        if hasattr(batch_unc, 'cpu'):
                            all_utterance_uncertainty.extend(
                                batch_unc.cpu().tolist()
                            )
                        elif isinstance(batch_unc, (list, tuple)):
                            all_utterance_uncertainty.extend(list(batch_unc))
                    # Collect per-utterance confidence scores (I4)
                    conf = unc_out.get('confidence_scores')
                    if conf is not None:
                        if hasattr(conf, 'cpu'):
                            all_confidence_scores.extend(conf.cpu().tolist())
                        elif isinstance(conf, (list, tuple)):
                            all_confidence_scores.extend(list(conf))
                except Exception:
                    pass  # Non-fatal; uncertainty simply not recorded for this batch

            # Decode predictions
            log_probs_constrained = outputs.get('log_probs_constrained', outputs['logits_constrained'])
            logits_neural = outputs.get('logits_neural')
            # output_lengths: valid CTC frame count per sample (audio-pad mask
            # converted to frame space by the model).  Guards both greedy and
            # beam decoders from emitting padding-frame noise as insertions.
            output_lengths = outputs.get('output_lengths')

            # I2: Track per-utterance mean blank probability for insertion diagnostic
            try:
                blank_id = phn_to_id.get('<BLANK>', 0)
                blank_log_probs = log_probs_constrained[:, :, blank_id]  # [B, T]
                blank_probs_batch = blank_log_probs.exp()  # [B, T]
                if output_lengths is not None:
                    for _s in range(blank_probs_batch.size(0)):
                        _valid_t = int(output_lengths[_s].item())
                        _bp = blank_probs_batch[_s, :_valid_t].mean().item()
                        all_blank_probs.append(float(_bp))
                else:
                    all_blank_probs.extend(blank_probs_batch.mean(dim=1).cpu().tolist())
            except Exception:
                pass  # Non-fatal

            if use_beam_search:
                # Beam search (slower but more accurate)
                predictions = []
                for i in range(log_probs_constrained.size(0)):
                    # Truncate to valid frames to exclude padding noise.
                    valid_len = (
                        int(output_lengths[i].item())
                        if output_lengths is not None
                        else log_probs_constrained.size(1)
                    )
                    seq, score = decoder.decode(
                        log_probs_constrained[i, :valid_len].float().cpu().numpy(),
                        id_to_phn
                    )
                    predictions.append(seq)
            else:
                # Greedy decoding (fast)
                predictions = greedy_decode(
                    log_probs_constrained, phn_to_id, id_to_phn,
                    output_lengths=output_lengths,
                )

            if logits_neural is not None:
                predictions_neural = greedy_decode(
                    logits_neural, phn_to_id, id_to_phn,
                    output_lengths=output_lengths,
                )
                all_predictions_neural.extend(predictions_neural)
            
            references = decode_references(labels, id_to_phn)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
            all_status.extend(batch['status'].cpu().numpy())
            all_speakers.extend(batch['speakers'])
            all_phoneme_lengths.extend([len(ref) for ref in references])

            if outputs.get('logits_manner') is not None and 'articulatory_labels' in batch:
                for key in ['manner', 'place', 'voice']:
                    logits = outputs[f'logits_{key}']
                    labels_art = batch['articulatory_labels'][key].to(device)
                    if logits.dim() == 2:
                        # I5 utterance-level path: logits [B, C], derive mode label
                        batch_sz = labels_art.size(0)
                        utt_labels = torch.full(
                            (batch_sz,), -100, dtype=torch.long, device=device
                        )
                        valid_mask_art = labels_art != -100
                        for _i in range(batch_sz):
                            valid_seq = labels_art[_i][valid_mask_art[_i]]
                            if valid_seq.numel() > 0:
                                utt_labels[_i] = torch.mode(valid_seq).values
                        mask = utt_labels != -100
                        if mask.any():
                            preds = torch.argmax(logits[mask], dim=-1)
                            acc = (preds == utt_labels[mask]).float().mean().item()
                            articulatory_results[key].append(acc)
                    else:
                        # Legacy frame-level path: logits [B, T, C]
                        labels_art = _align_labels_to_logits(
                            labels_art, logits.size(1)
                        )
                        mask = labels_art != -100
                        if mask.any():
                            preds = torch.argmax(logits, dim=-1)
                            acc = (preds[mask] == labels_art[mask]).float().mean().item()
                            articulatory_results[key].append(acc)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")
    
    # Compute overall metrics
    per_scores = [compute_per(p, r) for p, r in zip(all_predictions, all_references)]

    # Corpus-level WER (I1 / E2) — join phoneme sequences as space-separated tokens
    print("📝 Computing corpus-level WER...")
    predictions_text = [' '.join(p) for p in all_predictions]
    references_text  = [' '.join(r) for r in all_references]
    corpus_wer = compute_wer_texts(predictions_text, references_text)
    print(f"   WER: {corpus_wer:.3f}")

    # ── Macro-average PER over speakers (audit fix S4) ────────────────────────
    # Group per-scores by speaker, compute per-speaker mean, then macro-average
    speaker_per_raw = defaultdict(list)
    speaker_status_map = {}
    for i, spk in enumerate(all_speakers):
        speaker_per_raw[spk].append(per_scores[i])
        speaker_status_map[spk] = int(all_status[i])

    per_by_speaker = {spk: float(np.mean(v)) for spk, v in speaker_per_raw.items()}
    macro_per_scores = list(per_by_speaker.values())
    mean_per, per_ci = compute_per_with_ci(macro_per_scores)

    # Also report sample-level mean (for consistency with prior results)
    sample_mean_per, _ = compute_per_with_ci(per_scores)

    neural_per_scores = []
    if all_predictions_neural:
        neural_per_scores = [
            compute_per(p, r) for p, r in zip(all_predictions_neural, all_references)
        ]
    mean_per_neural = float(np.mean(neural_per_scores)) if neural_per_scores else None

    # Constraint precision: per-utterance rate at which the constraint improved,
    # was neutral, or degraded recognition vs. the neural-only path (audit Phase 8).
    # "Helpful" = per_constrained < per_neural for that utterance; metric is
    # analogous to rule precision without requiring per-frame forced alignment.
    constraint_precision: Optional[Dict[str, float]] = None
    helpful_count = neutral_count = harmful_count = 0
    if all_predictions_neural and len(all_predictions_neural) == len(per_scores):
        for p_c, p_n in zip(per_scores, neural_per_scores):
            if p_c < p_n - 1e-6:
                helpful_count += 1
            elif p_c > p_n + 1e-6:
                harmful_count += 1
            else:
                neutral_count += 1
        total = len(per_scores)
        constraint_precision = {
            'helpful_rate':  helpful_count / total,
            'neutral_rate':  neutral_count / total,
            'harmful_rate':  harmful_count / total,
            'rule_precision': helpful_count / total,
            'n_utterances':  total,
        }
        print(
            f"   Constraint precision  — helpful: {helpful_count/total:.1%}  "
            f"neutral: {neutral_count/total:.1%}  harmful: {harmful_count/total:.1%}"
        )

    paired_delta_stats = bootstrap_paired_per_delta(per_scores, neural_per_scores)

    # Stratified by dysarthric status (sample-level for clinical reporting)
    dysarthric_per = [per_scores[i] for i in range(len(per_scores)) if all_status[i] == 1]
    control_per    = [per_scores[i] for i in range(len(per_scores)) if all_status[i] == 0]

    dysarthric_mean, dysarthric_ci = compute_per_with_ci(dysarthric_per)
    control_mean, control_ci       = compute_per_with_ci(control_per)

    # Speaker-macro stratification
    dysarthric_spk_per = [v for spk, v in per_by_speaker.items() if speaker_status_map.get(spk) == 1]
    control_spk_per    = [v for spk, v in per_by_speaker.items() if speaker_status_map.get(spk) == 0]

    # Stratified by phoneme length
    stratified_by_length = stratify_by_phoneme_length(
        per_scores, all_phoneme_lengths, all_status
    )

    # Stratified by clinical severity bucket (mild / moderate / severe)
    stratified_by_severity = compute_severity_stratified_per(per_by_speaker)

    # Per-speaker metrics
    speaker_metrics = {}
    for spk, scores in speaker_per_raw.items():
        s_mean, s_ci = compute_per_with_ci(scores)
        speaker_metrics[spk] = {
            'per': s_mean,
            'ci': s_ci,
            'std': float(np.std(scores)),
            'n_samples': len(scores),
            'status': speaker_status_map.get(spk, -1),
        }

    # Keep full alignments only when needed; otherwise stream computation to
    # reduce peak memory on large LOSO evaluation sets.
    _need_full_alignments = bool(generate_explanations or symbolic_rules)
    _all_alignments: Optional[List[List[Tuple]]] = None
    _neural_alignments: Optional[List[List[Tuple]]] = None
    if _need_full_alignments:
        print("🔍 Pre-computing phoneme alignments (shared cache for all analysis functions)...")
        _all_alignments = [
            phoneme_alignment(p, r) for p, r in zip(all_predictions, all_references)
        ]
        if all_predictions_neural and len(all_predictions_neural) == len(all_references):
            _neural_alignments = [
                phoneme_alignment(p, r) for p, r in zip(all_predictions_neural, all_references)
            ]

    # Phoneme-level error analysis
    print("📊 Analyzing phoneme-level errors...")
    error_analysis = analyze_phoneme_errors(all_predictions, all_references, alignments=_all_alignments)

    # Per-phoneme PER breakdown (E3)
    print("📊 Computing per-phoneme PER breakdown...")
    per_phoneme_breakdown = compute_per_phoneme_breakdown(all_predictions, all_references, alignments=_all_alignments)
    _breakdown_path = results_dir / 'per_phoneme_per.json'
    with open(_breakdown_path, 'w') as _f:
        json.dump(per_phoneme_breakdown, _f, indent=2)
    print(f"✅ Saved per_phoneme_per.json ({len(per_phoneme_breakdown)} phonemes)")
    plot_per_phoneme_breakdown(per_phoneme_breakdown, results_dir / 'per_phoneme_per.png')

    # Articulatory-stratified PER (H4 / E-02): PER broken down by manner class
    print("📊 Computing articulatory-stratified PER (E-02)...")
    try:
        per_by_manner = compute_articulatory_stratified_per(all_predictions, all_references, alignments=_all_alignments)
        with open(results_dir / 'per_by_manner.json', 'w') as _f:
            json.dump(per_by_manner, _f, indent=2)
        print(f"✅ Saved per_by_manner.json ({len(per_by_manner)} manner classes)")
    except Exception as _e:
        per_by_manner = {}
        print(f"⚠️  per_by_manner failed (non-fatal): {_e}")

    # Rule-pair confusion analysis (H4 / E-03): neural vs. constrained sub counts per rule
    print("📊 Computing rule-pair confusion analysis (E-03)...")
    rule_pair_confusion: Dict = {}
    if all_predictions_neural and len(all_predictions_neural) == len(all_references):
        _rules_for_analysis = symbolic_rules or {}
        if _rules_for_analysis:
            try:
                rule_pair_confusion = compute_rule_pair_confusion(
                    all_predictions_neural,
                    all_predictions,
                    all_references,
                    _rules_for_analysis,
                    neural_alignments=_neural_alignments,
                    constrained_alignments=_all_alignments,
                )
                with open(results_dir / 'rule_pair_confusion.json', 'w') as _f:
                    json.dump(rule_pair_confusion, _f, indent=2)
                print(f"✅ Saved rule_pair_confusion.json ({len(rule_pair_confusion)} rule pairs)")
            except Exception as _e:
                print(f"⚠️  rule_pair_confusion failed (non-fatal): {_e}")
        else:
            print("ℹ️  rule_pair_confusion skipped: no symbolic_rules provided")
    else:
        print("ℹ️  rule_pair_confusion skipped: neural predictions not available")

    # Generate visualizations
    print("📈 Generating visualizations...")
    plot_confusion_matrix(error_analysis['confusion_matrix'], results_dir / 'confusion_matrix.png')
    plot_per_by_length(stratified_by_length, results_dir / 'per_by_length.png')
    plot_clinical_gap(dysarthric_mean, control_mean, results_dir / 'clinical_gap.png')

    # I2: Blank probability histogram (always generated; key insertion-bias diagnostic)
    if all_blank_probs:
        _blank_target = 0.75
        if hasattr(model, "config") and hasattr(model.config, "training"):
            _blank_target = float(getattr(model.config.training, "blank_target_prob", _blank_target))
        plot_blank_histogram(all_blank_probs, results_dir / 'blank_probability_histogram.png',
                             target_prob=_blank_target)
        print(f"✅ Blank probability histogram saved "
              f"(mean={float(np.mean(all_blank_probs)):.3f})")

    # C-4: Severity vs PER scatter plot (key SPCOM figure)
    try:
        from src.visualization.experiment_plots import plot_severity_vs_per
        from src.utils.config import TORGO_SEVERITY_MAP
        _sev_plot_path = plot_severity_vs_per(
            speaker_metrics, TORGO_SEVERITY_MAP, results_dir / 'severity_vs_per.png'
        )
        print(f"✅ Severity vs PER scatter saved → {_sev_plot_path.name}")
    except Exception as _sev_exc:
        print(f"⚠️  Severity vs PER scatter failed (non-fatal): {_sev_exc}")

    # C-5: Rule-pair confusion bar chart (symbolic constraint impact per rule)
    if rule_pair_confusion:
        try:
            from src.visualization.experiment_plots import plot_rule_pair_confusion as _plot_rpc
            _rpc_path = _plot_rpc(rule_pair_confusion, results_dir / 'rule_pair_confusion.png')
            print(f"✅ Rule-pair confusion chart saved → {_rpc_path.name}")
        except Exception as _rpc_exc:
            print(f"⚠️  Rule-pair confusion chart failed (non-fatal): {_rpc_exc}")

    rule_stats = None
    if hasattr(model, 'get_rule_statistics'):
        rule_stats = model.get_rule_statistics()
        if rule_stats is not None:
            if isinstance(rule_stats.get('unique_rules'), set):
                rule_stats['unique_rules'] = [list(item) for item in rule_stats['unique_rules']]
            if isinstance(rule_stats.get('top_rules'), list):
                rule_stats['top_rules'] = [
                    [list(pair), count] if isinstance(pair, tuple) else [pair, count]
                    for pair, count in rule_stats['top_rules']
                ]
            # N7: add utterance-level proxy for rule precision when paired
            # constrained-vs-neural comparison is available.
            if constraint_precision is not None:
                total_utt = int(constraint_precision.get('n_utterances', 0))
                if total_utt > 0:
                    rule_stats['rule_precision_proxy'] = float(helpful_count / total_utt)
                    rule_stats['rule_harm_rate_proxy'] = float(harmful_count / total_utt)
        plot_rule_impact(rule_stats, results_dir / 'rule_impact.png',
                         model=model, id_to_phn=id_to_phn)

    # ── Statistical tests (audit Phase 6 / ROADMAP §8) 
    # Welch t-test dysarthric vs. control (sample-level)
    t_stat, p_val_welch = (
        stats.ttest_ind(dysarthric_per, control_per, equal_var=False)
        if dysarthric_per and control_per else (0.0, 1.0)
    )

    # Wilcoxon rank-sum test (more robust for non-normal small samples)
    wilcox_stat, p_val_wilcox = (
        stats.mannwhitneyu(dysarthric_per, control_per, alternative='two-sided')
        if len(dysarthric_per) > 1 and len(control_per) > 1 else (0.0, 1.0)
    )

    # Holm-Bonferroni correction over the two p-values.
    # C8: statsmodels is a hard dependency (requirements.txt); remove the
    # silent Bonferroni fallback so a missing install surfaces immediately.
    from statsmodels.stats.multitest import multipletests
    p_values_raw = [float(p_val_welch), float(p_val_wilcox)]
    _, p_corrected, _, _ = multipletests(p_values_raw, alpha=0.05, method='holm')
    p_corrected = [float(p) for p in p_corrected]

    # Intelligibility correlation: per-speaker PER vs. severity score
    # E1 (revised): Always compute — even n=3 gives a useful directional signal.
    # Flag validity separately so downstream analysis can gate on it.
    # Spearman is degenerate (r=0, p=1) when ≥2 speakers share a tied rank
    # (common with multiple control speakers at severity 0.0); Pearson is still
    # meaningful.  We report both with an explicit `correlation_valid` flag.
    n_speakers_eval = len(per_by_speaker)
    correlation_valid = n_speakers_eval >= 5   # True → statistically interpretable
    try:
        from src.utils.config import get_speaker_severity
        sev_scores     = [get_speaker_severity(spk) for spk in per_by_speaker]
        per_scores_spk = [per_by_speaker[spk]        for spk in per_by_speaker]
        if n_speakers_eval >= 2:
            pearson_r,  pearson_p  = stats.pearsonr(sev_scores, per_scores_spk)
            spearman_r, spearman_p = stats.spearmanr(sev_scores, per_scores_spk)
            if not correlation_valid:
                print(f"ℹ️   Severity ↔ PER correlation computed (n={n_speakers_eval} speakers — "
                      f"descriptive only; ≥5 needed for statistical validity)")
        else:
            pearson_r = pearson_p = spearman_r = spearman_p = float('nan')
    except Exception:
        pearson_r = pearson_p = spearman_r = spearman_p = float('nan')

    # ── Articulatory confusion + Explainability (ROADMAP §6, audit Phase 5) ───
    # E5: Always build the articulatory confusion matrix by aligning predictions
    # against references — this is independent of whether full explanation JSON
    # is requested and requires no model attentions.
    _art_analyzer = None
    _attributed_errors: List[List[Dict]] = []  # reused by formatter if needed
    try:
        from src.explainability import (
            PhonemeAttributor, ArticulatoryConfusionAnalyzer,
        )
        _attributor   = PhonemeAttributor()
        _art_analyzer = ArticulatoryConfusionAnalyzer()
        for _pred_seq, _ref_seq in zip(all_predictions, all_references):
            _errors = _attributor.alignment_attribution(_pred_seq, _ref_seq)
            _subs = [
                (e['expected_phoneme'], e['predicted_phoneme'])
                for e in _errors
                if e['type'] == 'substitution'
                and e['expected_phoneme'] and e['predicted_phoneme']
            ]
            _art_analyzer.accumulate_from_errors(_subs)
            _attributed_errors.append(_errors)
        _art_analyzer.plot_feature_confusion(results_dir / 'articulatory_confusion.png')
        print("✅ Articulatory confusion matrix saved")
    except Exception as _art_exc:
        print(f"⚠️  Articulatory confusion matrix failed (non-fatal): {_art_exc}")

    if generate_explanations:
        try:
            from src.explainability import ExplainableOutputFormatter
            formatter = ExplainableOutputFormatter()
            # Re-use attributed errors computed above; fall back to re-aligning if
            # the articulatory block above failed (empty list).
            if not _attributed_errors:
                from src.explainability import PhonemeAttributor as _PA
                _fallback_attr = _PA()
                _attributed_errors = [
                    _fallback_attr.alignment_attribution(p, r)
                    for p, r in zip(all_predictions, all_references)
                ]
            for i, (pred_seq, ref_seq) in enumerate(zip(all_predictions, all_references)):
                errors = _attributed_errors[i] if i < len(_attributed_errors) else []
                utt_per = compute_per(pred_seq, ref_seq)
                spk_id  = all_speakers[i] if i < len(all_speakers) else None
                sev_val = float(get_speaker_severity(spk_id)) if spk_id else None
                explanation = formatter.format_utterance(
                    utterance_id=f"utt_{i:04d}",
                    ground_truth=' '.join(ref_seq),
                    prediction=' '.join(pred_seq),
                    errors=errors,
                    symbolic_rules_summary=rule_stats or {},
                    wer=utt_per,  # Bug B10 fix: was hardcoded 0.0; PER is the available proxy
                    per=utt_per,
                    speaker_id=spk_id,
                    severity=sev_val,
                )
                formatter.add(explanation)
            formatter.save_explanations(results_dir)
            print("✅ Explainability artifacts generated")
        except Exception as exc:
            print(f"⚠️  Explainability module failed (non-fatal): {exc}")

    # Compile results
    results = {
        'avg_per': float(mean_per),  # macro-avg (primary metric)
        'wer': float(corpus_wer),      # corpus-level WER (I1 / E2)
        'overall': {
            'per_macro_speaker': mean_per,
            'per_sample_mean':   float(sample_mean_per),
            'wer':               float(corpus_wer),
            'ci':                per_ci,
            'std':               float(np.std(per_scores)),
            'n_samples':         len(per_scores),
            'n_speakers':        len(per_by_speaker),
        },
        'symbolic_impact': {
            'per_neural':     mean_per_neural,
            'per_constrained': mean_per,
            'delta_per':      (mean_per_neural - mean_per) if mean_per_neural is not None else None,
            'paired_delta_constrained_minus_neural': paired_delta_stats,
            'ci_95_delta_per': [paired_delta_stats.get('ci_95_low'), paired_delta_stats.get('ci_95_high')],
            'p_value_neural_vs_constrained': paired_delta_stats.get('p_value_two_sided'),
        },
        'articulatory_accuracy': {
            key: float(np.mean(vals)) if vals else 0.0
            for key, vals in articulatory_results.items()
        },
        'stats': {
            'welch_t':        float(t_stat),
            'welch_p':        float(p_val_welch),
            'wilcox_u':       float(wilcox_stat),
            'wilcox_p':       float(p_val_wilcox),
            'p_holm_corrected': p_corrected,
            'significant':    bool(p_corrected[0] < 0.05 or p_corrected[1] < 0.05),
            'pearson_r_sev_per':   float(pearson_r),
            'pearson_p_sev_per':   float(pearson_p),
            'spearman_r_sev_per':  float(spearman_r),
            'spearman_p_sev_per':  float(spearman_p),
            'correlation_n_speakers': n_speakers_eval,
            'correlation_valid':   correlation_valid,  # False when n<5 (treat as descriptive)
            'p_value_neural_vs_constrained': paired_delta_stats.get('p_value_two_sided'),
            'ci_95_delta_per': [paired_delta_stats.get('ci_95_low'), paired_delta_stats.get('ci_95_high')],
        },
        'rule_impact': rule_stats,
        'stratified': {
            'dysarthric': {
                'per_sample':   dysarthric_mean,
                'per_speaker':  float(np.mean(dysarthric_spk_per)) if dysarthric_spk_per else 0.0,
                'ci':  dysarthric_ci,
                'n':   len(dysarthric_per),
            },
            'control': {
                'per_sample':  control_mean,
                'per_speaker': float(np.mean(control_spk_per)) if control_spk_per else 0.0,
                'ci': control_ci,
                'n':  len(control_per),
            },
        },
        'by_length': stratified_by_length,
        'by_severity': stratified_by_severity,
        'per_by_manner': per_by_manner,
        'rule_pair_confusion': rule_pair_confusion,
        'constraint_precision': constraint_precision,
        'per_speaker': speaker_metrics,
        'error_analysis': {
            'error_counts': error_analysis['error_counts'],
            'common_confusions': [
                (list(pair), count)
                for pair, count in error_analysis['common_confusions']
            ],
            'deletion_phonemes':  error_analysis['deletion_phonemes'],
            'insertion_phonemes': error_analysis['insertion_phonemes'],
        },
        'uncertainty': {
            'computed': compute_uncertainty and len(all_utterance_uncertainty) > 0,
            'n_samples': uncertainty_n_samples if compute_uncertainty else None,
            'entropy_mean': (
                float(np.mean(all_utterance_uncertainty))
                if all_utterance_uncertainty else None
            ),
            'entropy_std': (
                float(np.std(all_utterance_uncertainty))
                if len(all_utterance_uncertainty) > 1 else None
            ),
            'confidence_mean': (
                float(np.mean(all_confidence_scores))
                if all_confidence_scores else None
            ),
            'per_utterance': all_utterance_uncertainty if all_utterance_uncertainty else [],
        },
    }

    # Save results to JSON
    with open(results_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {results_dir}")

    # Print summary
    print("\n" + "="*65)
    print("📊 EVALUATION SUMMARY")
    print("="*65)
    print(f"Overall PER  (macro-speaker): {mean_per:.3f} (95% CI: [{per_ci[0]:.3f}, {per_ci[1]:.3f}])")
    print(f"Overall PER  (sample-mean):   {sample_mean_per:.3f}")
    print(f"Corpus WER   (phoneme-level): {corpus_wer:.3f}")
    print(f"Dysarthric   (sample-mean):   {dysarthric_mean:.3f} (n={len(dysarthric_per)})")
    print(f"Control      (sample-mean):   {control_mean:.3f} (n={len(control_per)})")
    _corr_note = f"n={n_speakers_eval}" if correlation_valid else f"n={n_speakers_eval}, descriptive only"
    print(f"Severity ↔ PER correlation:  r={pearson_r:.3f} (p={pearson_p:.4f}) [{_corr_note}]")
    print(f"Wilcoxon p (dys vs ctrl):     {p_val_wilcox:.4f} (Holm-corr: {p_corrected[1]:.4f})")
    print(f"\nError Breakdown:")
    print(f"  Correct:        {error_analysis['error_counts']['correct']}")
    print(f"  Substitutions:  {error_analysis['error_counts']['substitutions']}")
    print(f"  Deletions:      {error_analysis['error_counts']['deletions']}")
    print(f"  Insertions:     {error_analysis['error_counts']['insertions']}")
    ins = error_analysis['error_counts']['insertions']
    dels = error_analysis['error_counts']['deletions']
    ratio_str = f"{ins/max(dels,1):.1f}×" if dels > 0 else "N/A"
    print(f"  I/D ratio:      {ratio_str}  (target: <3×)")
    print("="*65)

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
