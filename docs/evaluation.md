# docs/evaluation.md — Evaluation Guide

> Cross-references: [docs/experiments.md](experiments.md) for result tables, [docs/architecture.md](architecture.md) for model output tensors.

---

## Metrics Reference

### Phoneme Error Rate (PER)

**Formula:** `PER = (S + D + I) / N` where S = substitutions, D = deletions, I = insertions, N = reference phoneme count.

**Aggregation:** Macro-speaker PER — group utterance-level PER scores by speaker, compute per-speaker mean, then macro-average across speakers. Implemented in `evaluate_model()` in `evaluate.py`.

**Acceptable range:** LOSO macro PER of 0.28 for a mixed dysarthric/control speaker set is consistent with published TORGO baselines. Per-speaker PER for severe dysarthric speakers (M01, F01, severity ≈ 4.9) may exceed 0.5.

**Implementation:** `compute_per()` in `evaluate.py` using `editdistance.eval()`.

---

### Word Error Rate (WER)

**Formula:** Standard WER at word level. When phoneme sequences are passed as space-separated strings, WER ≡ PER at the token level.

**Aggregation:** Corpus-level WER via `jiwer.wer()` (batch API, lists of strings).

**Implementation:** `compute_wer_texts()` in `evaluate.py`. Fixed in I1 — was hardcoded to `0.0` previously (B10).

---

### Bootstrap Confidence Interval

**Methodology:** Non-parametric bootstrap (n=1000 resamples) with replacement over macro-speaker PER scores. The 2.5th and 97.5th percentiles give the 95% CI bounds. Implemented in `compute_per_with_ci()`.

**Validity caveat:** With n=3 test speakers (single split), bootstrap CI covers nearly the full range [0, 1] and is meaningless. With n=15 (LOSO), bootstrap CI over fold PER values is statistically appropriate.

---

### Welch t-test

Two-sample t-test with unequal variance (`scipy.stats.ttest_ind(equal_var=False)`) comparing dysarthric vs. control PER distributions. Used for dysarthric/control significance testing.

**Validity caveat:** With n=1–2 speakers per group in single-split settings, the test is invalid. Requires n≥5 per group for meaningful inference.

---

### Wilcoxon Rank-Sum Test

`scipy.stats.mannwhitneyu(alternative='two-sided')` — non-parametric alternative to Welch t-test. More robust for small, potentially non-normal samples.

---

### Holm-Bonferroni Correction

Applied via `statsmodels.stats.multitest.multipletests(method='holm')` over the Welch p-value and Wilcoxon p-value simultaneously. Corrected p-values are reported in `stats.p_holm_corrected`.

---

### Pearson/Spearman Severity Correlation

**Formula:** `pearsonr(severity_scores, per_by_speaker)` and `spearmanr(...)` from `scipy.stats`.

**Validity caveat:** Reported with an explicit `correlation_valid` flag in `evaluation_results.json`. The flag is `True` only when `n_speakers >= 5`. With n<5, the correlation is descriptive only and should not be cited as a statistical claim. Spearman is degenerate when multiple speakers share tied severity ranks (e.g., all 7 controls at 0.0 create ties).

---

### Paired Bootstrap Delta (Neural vs. Constrained)

**Function:** `bootstrap_paired_per_delta()` in `evaluate.py`.

Computes the paired difference `Δ = per_constrained - per_neural` (positive = constrained is worse) for each utterance, then bootstraps the mean delta with a two-sided empirical p-value. Results are reported in `symbolic_impact.paired_delta_constrained_minus_neural` and `symbolic_impact.p_value_neural_vs_constrained`.

---

## Decoding

### Greedy CTC Decoding

**Function:** `greedy_decode()` in `evaluate.py`.

**Collapse rules applied per frame (in order):**
1. Blank token (ID=0) → reset `prev_id`; emit nothing
2. PAD token (ID=1) → reset `prev_id`; emit nothing (same semantics as blank)
3. Consecutive duplicate of `prev_id` without intervening blank → collapse; emit nothing
4. All other real phoneme tokens → emit and update `prev_id`

**`output_lengths` masking:** The `output_lengths` tensor from the model (valid CTC frame count per sample, accounting for TemporalDownsampler stride) is passed to truncate each sequence before decoding. Without this, zero-padded audio frames produce spurious argmax predictions that inflate the insertion count dramatically.

---

### Beam Search Decoding (`BeamSearchDecoder`)

**Class:** `BeamSearchDecoder` in `evaluate.py`. CLI: `--beam-search --beam-width INT`.

**Algorithm:** CTC prefix beam search. Each hypothesis is tracked as `(p_blank, p_non_blank, lm_cumulative_score)`. Blank and non-blank probabilities are maintained separately to enable correct CTC prefix merging semantics (a repeated token can be reached via both p_blank and p_non_blank predecessors).

**PAD/UNK exclusion:** PAD and UNK are excluded from `emit_ids` — the set of non-blank tokens the beam can emit. This prevents the model from decoding PAD/UNK as real phoneme predictions.

**Length normalization (B19 fix):** `acoustic_score / len(prefix)^alpha + lm_weight * lm_cumulative_score`. The LM bonus is added **after** length normalization, not divided by it. The original bug divided both acoustic and LM scores by `len^alpha`, unfairly penalizing longer hypotheses that accumulated more LM bonus.

**BigramLMScorer (N4 fix):** Phoneme bigram LM with add-k smoothing (`k=0.5`). Add-k smoothing prevents `log(0)` for unseen phoneme bigrams — critical for dysarthric speech which produces non-standard phoneme sequences that may not appear in training bigrams. Built automatically from training phoneme sequences when `--lm-weight > 0`.

---

## Symbolic Impact Analysis

**How `per_neural` and `per_constrained` are computed:** Both are greedy-decoded from sub-paths of the same jointly-trained model in a single forward pass. `logits_neural` is the output of `PhonemeClassifier` before `SymbolicConstraintLayer`; `log_probs_constrained` is the output after. They are **not** independent models.

**`constraint_precision`:** Utterance-level rate at which the constraint improved (`per_constrained < per_neural - 1e-6`), was neutral, or degraded recognition. Reported as `helpful_rate`, `neutral_rate`, `harmful_rate` in `evaluation_results.json['constraint_precision']`.

**Bootstrap paired delta test:** See metrics section above. Reports `delta_mean`, `ci_95_low`, `ci_95_high`, `p_value_two_sided`.

**Interpretation:** A negative `delta_mean` (constrained < neural) means the constraint helps on average across utterances in that evaluation set. `p_value_two_sided < 0.05` indicates statistical significance of this difference.

---

## Explainability

### PhonemeAttributor

**Class:** `PhonemeAttributor` in `src/explainability/attribution.py`.

Two attribution methods:

- **Alignment-based** (`alignment_attribution()`): Levenshtein alignment of predicted vs. reference phoneme sequences → per-position error type (substitution/deletion/insertion) with articulatory feature differences and probable clinical cause (e.g., "Devoicing (voiced → voiceless)", "Liquid gliding (R/L → W/Y)").

- **Attention-based** (`attention_attribution()`): Aggregates HuBERT attention maps over the last 4 transformer layers and all heads to produce per-frame importance scores. **Requires CTC forced alignment to map frame importances to phoneme positions** — this path is not yet implemented. `output_attentions=True` must be passed to the model forward call.

### ArticulatoryConfusionAnalyzer

**Class:** `ArticulatoryConfusionAnalyzer` in `src/explainability/articulator_analysis.py`. Accumulates manner/place/voice confusion counts from substitution error pairs and plots row-normalized heatmaps. Always generated (independent of `--explain` flag).

### ExplainableOutputFormatter

**Class:** `ExplainableOutputFormatter` in `src/explainability/output_format.py`. Formats per-utterance data into the ROADMAP §6.4 JSON schema and saves to `explanations.json`. Activated by `--explain` flag.

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

### SymbolicRuleTracker

**Class:** `SymbolicRuleTracker` in `src/explainability/rule_tracker.py`. Logs activations when `P_final` argmax differs from `P_neural` argmax (i.e., the constraint changed the top prediction). Capped at 50,000 entries (H-5 fix). `generate_explanation()` returns aggregated statistics.

**Note:** `rule_precision()` is implemented in `SymbolicRuleTracker` but not directly wired into `evaluate_model()` output. An utterance-level proxy (`constraint_precision.rule_precision`) is reported instead. True rule precision requires knowing which reference phoneme appeared at each frame — unavailable without CTC forced alignment.

---

## Uncertainty Estimation

### UncertaintyAwareDecoder

**Class:** `UncertaintyAwareDecoder` in `src/models/uncertainty.py`. Activated by `--uncertainty` flag or `config.experiment.compute_uncertainty = True`.

**Method:** MC-Dropout — enables dropout at inference time (`model.train()`) while keeping HuBERT in eval mode (`model.hubert.eval()` to preserve LayerNorm statistics). Runs `n_samples` (default 20) stochastic forward passes and aggregates:

- `mean_log_probs`: [B, T, V] mean log-probabilities across samples
- `epistemic_entropy_per_frame`: [B, T] entropy of the predictive distribution `H(y|x) = -Σ p̄_v log p̄_v`
- `utterance_uncertainty`: [B] mean entropy over valid frames
- `confidence_scores`: [B] = `1 - utterance_uncertainty / log(V)`

**`conformal_phoneme_sets()`:** APS-like heuristic — sorts phoneme probabilities descending, includes top-k tokens until cumulative probability ≥ `coverage`. **This is not calibrated conformal prediction.** True conformal calibration requires fitting τ on a held-out calibration set. The current implementation uses `τ = coverage` as a conservative approximation. Documented as an APS-heuristic (M-5 fix).

**Output fields in `evaluation_results.json`:**
```json
{
  "uncertainty": {
    "computed": true,
    "n_samples": 20,
    "entropy_mean": 1.23,
    "entropy_std": 0.45,
    "confidence_mean": 0.73,
    "per_utterance": [1.1, 0.9, 1.4, ...]
  }
}
```

---

## Visualizations

All 10 figures are generated by `evaluate_model()` and `scripts/generate_figures.py`.

| Filename | What it shows | How to read it |
|---|---|---|
| `confusion_matrix.png` | Top-30 phoneme confusion matrix, row-normalized | Diagonal = accuracy; off-diagonal hot spots reveal systematic errors (e.g., T→AH, D→AH devoicing) |
| `per_by_length.png` | PER stratified by utterance length × dysarthric/control status | Longer utterances typically have higher PER; dysarthric bars should be consistently above control bars |
| `clinical_gap.png` | Dysarthric vs. control mean PER bar chart | Gap size reflects the clinical severity of recognition degradation |
| `rule_impact.png` | Symbolic rule activation frequencies (bar chart) or constraint matrix heatmap (E6 fallback) | Bar chart: which substitution rules fired most. Heatmap: log(1 + C weight) — diagonal should be dominant; off-diagonal clusters reveal learned confusions |
| `blank_probability_histogram.png` | Per-utterance mean blank probability distribution | Target line at 0.75; distribution shifted right → insertion bias; shifted left → deletion bias |
| `per_phoneme_per.png` | Per-phoneme PER (top-30 hardest, color-coded) | Red ≥ 0.70; orange 0.40–0.69; green < 0.40; high-PER phonemes identify systematic recognition failures |
| `articulatory_confusion.png` | Manner/place/voice feature confusion heatmaps (3-panel) | Row-normalized; off-diagonal entries reveal articulatory error patterns (e.g., stop→fricative, velar→alveolar fronting) |
| `severity_vs_per.png` | Scatter of severity score vs. per-speaker PER with OLS regression | Positive slope expected; r and p-value annotated; n= shown for validity assessment |
| `per_by_speaker.png` | Per-speaker PER horizontal bar chart, severity-sorted (M-6 fix) | Controls first (severity=0), then dysarthric in ascending severity order; CI whiskers from bootstrap |
| `rule_pair_confusion.png` | Neural vs. constrained substitution counts per symbolic rule | Left panel: absolute counts; right panel: Δ = neural − constrained (green = constraint reduced substitutions, red = constraint increased them) |