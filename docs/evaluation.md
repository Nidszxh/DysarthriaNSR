# docs/evaluation.md — Evaluation Guide

> Cross-references: [experiments.md](experiments.md) for result tables, [architecture.md](architecture.md) for model output tensors.

---

## Metrics Reference

### Phoneme Error Rate (PER)

**Formula:** `PER = (S + D + I) / N` where S = substitutions, D = deletions, I = insertions, N = reference phoneme count.

**Aggregation (B15 fix — macro-speaker PER):** Per-utterance PER scores are grouped by speaker via `defaultdict(list)`, per-speaker means are computed, then macro-averaged across speakers. The original bug (B15) logged batch-mean PER, which over-weighted high-utterance speakers. The fix groups by speaker first: `avg_per = mean([mean(speaker_scores) for speaker in speakers])`.

**Implementation:** `compute_per()` in `evaluate.py` using `editdistance.eval()`. Can exceed 1.0 when insertions > reference length.

**Validity:** Macro-speaker PER with n≥5 speakers is statistically meaningful. With n=2–3 (single split), PER is a development reference only.

---

### Word Error Rate (WER)

**Formula:** Standard WER at word level. When phoneme sequences are passed as space-separated strings, WER ≡ PER at the token level.

**Implementation:** `compute_wer_texts()` in `evaluate.py` using `jiwer.wer()` (batch API, lists of strings, jiwer 4.x compatible). Fixed in B10 — was hardcoded to `0.0` in the original `format_utterance()` call.

---

### Bootstrap Confidence Interval

**Methodology:** Non-parametric bootstrap (n=1000 resamples) with replacement over macro-speaker PER scores. The 2.5th and 97.5th percentiles give 95% CI bounds. Implemented in `compute_per_with_ci()`. With n=15 (LOSO), bootstrap CI over fold PER values is statistically appropriate. With n=3 test speakers (single split), CI covers nearly the full range [0, 1] and is meaningless.

---

### Paired Bootstrap Delta (Neural vs. Constrained)

**Function:** `bootstrap_paired_per_delta()` in `evaluate.py`. Computes `Δ = per_constrained - per_neural` for each utterance (positive = constrained is worse), bootstraps the mean delta (n=10,000 resamples), and reports a two-sided empirical p-value:
```python
p_left = mean(boot_arr <= 0.0)
p_right = mean(boot_arr >= 0.0)
p_two_sided = min(1.0, 2.0 * min(p_left, p_right))
```

`p_value=0.0` means zero bootstrap samples produced a non-negative mean delta — statistically significant but does not imply large practical effect.

---

### Welch t-test and Wilcoxon Rank-Sum

Welch t-test (`scipy.stats.ttest_ind(equal_var=False)`) and Wilcoxon rank-sum (`scipy.stats.mannwhitneyu(alternative='two-sided')`) compare dysarthric vs. control PER distributions. Wilcoxon is more robust for small, potentially non-normal samples. Holm-Bonferroni correction is applied over both p-values via `statsmodels.stats.multitest.multipletests(method='holm')` — `statsmodels` is a hard dependency; no silent fallback.

---

### Pearson/Spearman Severity Correlation

**Formula:** `pearsonr(severity_scores, per_by_speaker)` and `spearmanr(...)` from `scipy.stats`.

**Validity flag:** `correlation_valid` in `evaluation_results.json['stats']` is `True` only when `n_speakers >= 5`. With n < 5, both r and p are reported but labeled as descriptive only. Spearman is degenerate when multiple speakers share tied severity ranks (all 7 controls at 0.0 create ties); Pearson is still meaningful in this case.

---

## Greedy CTC Decoder

**Function:** `greedy_decode()` in `evaluate.py`.

**Collapse rules applied per frame in order:**
1. Blank token (ID=0) → reset `prev_id`; emit nothing
2. PAD token (ID=1) → reset `prev_id`; emit nothing (same semantics as blank — prevents probability mass inflation)
3. Consecutive duplicate of `prev_id` without intervening blank → collapse; emit nothing
4. All other real phoneme tokens → emit and update `prev_id`; call `normalize_phoneme()` to strip stress

**`output_lengths` masking:** The `output_lengths` tensor from the model (valid CTC frame count per sample, accounting for TemporalDownsampler stride) truncates each sequence before decoding. Without this, zero-padded audio frames produce spurious argmax predictions that inflate insertion count dramatically.

---

## Beam Search Decoder (`BeamSearchDecoder`)

CLI: `--beam-search --beam-width INT`. Class: `BeamSearchDecoder` in `evaluate.py`.

**Beam state:** `{prefix: (p_blank, p_non_blank, lm_cumulative_score)}` where `p_blank` and `p_non_blank` are pure acoustic log-probabilities kept separate to enable correct CTC prefix merging semantics.

**PAD/UNK exclusion:** PAD and UNK are excluded from `emit_ids`. This prevents their probability mass from inflating blank transitions. Frame log-probs are renormalized over `allowed_ids = [blank_id] + emit_ids` only.

**Repeated-token path:** When the new token equals `prefix[-1]`, both `p_blank` and `p_non_blank` predecessors are included via `new_p_nb = logaddexp(p_b, p_nb) + c_lp`. Omitting `p_non_blank` underestimates long same-token runs.

**Length normalization (B19 fix):** `score = acoustic / len(prefix)^α + lm_weight * lm_cumulative_score`. The LM bonus is added **after** length normalization — not divided by it. The original bug divided both acoustic and LM scores by `len^α`, unfairly penalizing longer hypotheses. Default α=0.6.

---

## `BigramLMScorer` (N4 fix)

Phoneme bigram LM built from training phoneme sequences. Add-k smoothing with **k=0.5** (not k=1). Rationale: k=1 (Laplace) is too aggressive for dysarthric speech — it over-smooths toward uniform distribution, reducing the LM's ability to capture canonical phoneme transitions.

```python
log_prob(prev_id, next_id) = log(
    (bigram_count(prev_id, next_id) + k) /
    (unigram_count(prev_id) + k * vocab_size)
)
```

The LM is built automatically from training phoneme sequences when `--lm-weight > 0.0`. Default `lm_weight=0.0` disables LM fusion.

---

## Symbolic Impact Analysis

**How `per_neural` and `per_constrained` are measured:** Both are greedy-decoded from sub-paths of the **same jointly-trained model** in a single forward pass. `logits_neural` is the output of `PhonemeClassifier` before `SymbolicConstraintLayer`; `log_probs_constrained` is the output after. They are **not** independent models.

**`constraint_precision` computation:** For each utterance, compare `per_constrained` vs. `per_neural` with tolerance 1e-6:
- Helpful: `per_constrained < per_neural - 1e-6`
- Neutral: `|per_constrained - per_neural| <= 1e-6`
- Harmful: `per_constrained > per_neural + 1e-6`

Rates reported as `helpful_rate`, `neutral_rate`, `harmful_rate` in `evaluation_results.json['constraint_precision']`.

---

## Phoneme Alignment

**Function:** `phoneme_alignment()` in `evaluate.py`.

**Fast path:** Uses `rapidfuzz.distance.Levenshtein.editops()` C extension when available.

**Fallback:** Pure-Python DP Levenshtein with backtracking.

**Operation semantics (prediction's perspective):**
- `'correct'`: pred_ph == ref_ph; increments correct count
- `'substitute'`: pred_ph ≠ ref_ph; both present; increments substitution count for ref_ph
- `'delete'`: pred_ph present in prediction but not in reference; increments insertion count for pred_ph
- `'insert'`: ref_ph present in reference but not in prediction; increments deletion count for ref_ph

**H-2 optimization:** `_all_alignments` is precomputed once for all (prediction, reference) pairs in `evaluate_model()`, then shared across four analysis functions: `analyze_phoneme_errors()`, `compute_per_phoneme_breakdown()`, `compute_articulatory_stratified_per()`, and `compute_rule_pair_confusion()`.

---

## Explainability Pipeline

### `PhonemeAttributor`

**Class:** `PhonemeAttributor` in `src/explainability/attribution.py`.

**Alignment-based attribution (`alignment_attribution()`):** Levenshtein alignment of predicted vs. reference phoneme sequences → per-position error type with articulatory feature differences and probable clinical cause. The `_PROBABLE_CAUSE_MAP` keys are `(dim, ref_val, pred_val)` tuples (e.g., `("voice", "voiced", "voiceless")` → `"Devoicing (voiced → voiceless)"`).

**Attention-based attribution (`attention_attribution()`):** Disabled — requires CTC forced alignment to map frame importances to phoneme positions. `output_attentions=True` must be passed to the model forward call, but forced alignment is not yet implemented.

### `ArticulatoryConfusionAnalyzer`

**Class:** `ArticulatoryConfusionAnalyzer` in `src/explainability/articulator_analysis.py`. Accumulates manner/place/voice confusion counts from substitution error pairs and plots 3-panel row-normalized heatmaps. Generated regardless of `--explain` flag (E5 fix — always runs during `evaluate_model()`).

### `ExplainableOutputFormatter`

**Class:** `ExplainableOutputFormatter` in `src/explainability/output_format.py`. Activated by `--explain` flag.

**`explanations.json` schema (per utterance):**
```json
{
  "utterance_id": "utt_0042",
  "speaker_id": "M01",
  "severity": 4.9,
  "ground_truth": "P AH T",
  "prediction": "P AH D",
  "wer": 0.333,
  "per": 0.333,
  "uncertainty": null,
  "phoneme_analysis": {
    "n_errors": 1,
    "errors": [{
      "type": "substitution",
      "position": 2,
      "predicted_phoneme": "D",
      "expected_phoneme": "T",
      "articulatory_features": { "manner": "stop", "place": "alveolar", "voice": "voiceless" },
      "feature_differences": { "voice": "voiceless → voiced" },
      "probable_cause": "Devoicing (voiced → voiceless)",
      "neural_confidence": null,
      "symbolic_correction_applied": false
    }]
  },
  "symbolic_rules_summary": {
    "total_fired": 12,
    "activation_rate": 0.08,
    "avg_blend_weight": 0.12,
    "top_rules": [{ "rule_id": "T->D", "count": 4 }]
  }
}
```

### `SymbolicRuleTracker`

**Class:** `SymbolicRuleTracker` in `src/explainability/rule_tracker.py`. Logs activations when `P_final` argmax differs from `P_neural` argmax. Capped at `_MAX_ACTIVATIONS = 50,000` entries (H-5 fix). `generate_explanation()` returns aggregated statistics.

**`rule_precision()` not wired (N7):** Implemented in `rule_tracker.py` but not called from `evaluate_model()`. An utterance-level proxy (`constraint_precision.rule_precision`) is reported instead. True rule precision requires knowing which reference phoneme appeared at each CTC frame — unavailable without forced alignment.

---

## Uncertainty Estimation (`UncertaintyAwareDecoder`)

**Class:** `UncertaintyAwareDecoder` in `src/models/uncertainty.py`. Activated by `--uncertainty` flag.

**MC-Dropout setup:** `model.train()` enables dropout; `model.hubert.eval()` is called even during MC-Dropout to preserve HuBERT LayerNorm running statistics. Without `hubert.eval()`, each stochastic pass would perturb LayerNorm stats in different directions, producing unreliable uncertainty estimates.

**Predictive entropy:** `H(y|x) = -Σ p̄_v log p̄_v` where `p̄_v = mean(softmax outputs across N samples)`. Per-frame entropy is aggregated to utterance level over valid (non-padding) frames.

**`conformal_phoneme_sets()` (M-5 documentation):** APS-like heuristic — sorts phoneme probabilities descending, includes top-k tokens until cumulative probability ≥ `coverage`. **This is NOT calibrated conformal prediction.** True conformal calibration requires fitting τ on a held-out calibration set. The current implementation uses `τ = coverage` as a conservative approximation.

**Output fields in `evaluation_results.json['uncertainty']`:**
```json
{
  "computed": true,
  "n_samples": 20,
  "entropy_mean": 1.23,
  "entropy_std": 0.45,
  "confidence_mean": 0.73,
  "per_utterance": [1.1, 0.9, 1.4]
}
```

---

## Visualization Outputs

All figures generated by `evaluate_model()` and `scripts/generate_figures.py`:

| Filename | What it shows | Key to reading | Requires |
|---|---|---|---|
| `confusion_matrix.png` | Top-30 phoneme confusion matrix, row-normalized | Diagonal = accuracy; off-diagonal hot spots reveal systematic errors (T→AH, D→AH) | Evaluation |
| `per_by_length.png` | PER stratified by utterance length × dysarthric/control | Longer utterances → higher PER; dysarthric bars should be consistently above control | Evaluation |
| `clinical_gap.png` | Dysarthric vs. control mean PER bar chart | Gap size reflects severity of recognition degradation | Evaluation |
| `rule_impact.png` | Symbolic rule activations (bar) or constraint matrix heatmap (E6 fallback) | Bar: which substitution rules fired most; Heatmap: log(1+C) — diagonal should be dominant | Evaluation |
| `blank_probability_histogram.png` | Per-utterance mean blank probability distribution | Target line at 0.75; right-shifted → deletion bias; left-shifted → insertion bias | Evaluation |
| `per_phoneme_per.png` | Per-phoneme PER (top-30 hardest, color-coded) | Red ≥ 0.70; orange 0.40–0.69; green < 0.40 | Evaluation |
| `articulatory_confusion.png` | Manner/place/voice feature confusion heatmaps (3-panel, row-normalized) | Off-diagonal = systematic articulatory errors | Evaluation (always) |
| `severity_vs_per.png` | Scatter of severity score vs. per-speaker PER with OLS (C-4) | Positive slope expected; r and p annotated; `correlation_valid` flag determines interpretability | Evaluation |
| `per_by_speaker.png` | Per-speaker PER horizontal bar, severity-sorted (M-6 fix) | Controls first (severity=0), then dysarthric in ascending severity; CI whiskers from bootstrap | Evaluation |
| `rule_pair_confusion.png` | Neural vs. constrained substitution counts per symbolic rule (C-5) | Left: absolute counts; right: Δ = neural − constrained (green = constraint reduced subs) | Evaluation + symbolic_rules provided |

Figures requiring `--explain` (i.e., `explanations.json`): `uncertainty_vs_per.png`, `uncertainty_distribution.png`. All others are generated from `evaluation_results.json` alone.
