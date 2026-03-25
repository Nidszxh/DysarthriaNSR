# docs/experiments.md — Experiments, Results & Ablations

> Cross-references: [architecture.md](architecture.md) for component descriptions, [training.md](training.md) for exact CLI reproduction commands.

---

## Evaluation Protocol

### Phoneme Error Rate (PER)

```
PER = (S + D + I) / N
```

where S = substitutions, D = deletions, I = insertions, N = reference phoneme count. Computed using the `editdistance` C extension at phoneme level via `compute_per()` in `evaluate.py`.

**Aggregation — macro-speaker PER (B15 fix):** Per-utterance PER scores are grouped by speaker via `defaultdict(list)`, per-speaker means are computed, then macro-averaged across speakers. This treats each speaker equally regardless of utterance count. Macro-speaker PER is the publication metric; sample-mean PER (averaging all utterances directly) is reported for reference only. TORGO speakers have highly variable utterance counts; sample-mean would over-weight high-utterance speakers and obscure per-speaker patterns that are the clinical object of interest.

### Why LOSO is Mandatory

TORGO has 15 speakers. A 70/15/15 stratified split yields approximately 2 test speakers. With n=2, all per-speaker statistics — Pearson r, Welch t-test, bootstrap CI — are statistically invalid. Single-split Pearson r between severity and PER (n=2–3) has a 95% CI spanning nearly [−1, 1]. LOSO-CV with 15 folds provides the minimum sample size for bootstrap CI and significance testing.

### Bootstrap CI Methodology

Bootstrap 95% CI is computed over macro-speaker PER scores (one value per fold/speaker). With n=15 (LOSO), 1000 bootstrap resamples with replacement are drawn from the fold-level PER values; the 2.5th and 97.5th percentiles give the CI bounds. Implemented in `compute_per_with_ci()` in `evaluate.py`. The `correlation_valid` flag in `evaluation_results.json` is `True` only when `n_speakers >= 5`.

### Statistical Tests

Welch t-test (`scipy.stats.ttest_ind(equal_var=False)`) and Wilcoxon rank-sum (`scipy.stats.mannwhitneyu(alternative='two-sided')`) compare dysarthric vs. control PER distributions. Holm-Bonferroni correction is applied via `statsmodels.stats.multitest.multipletests(method='holm')` over the two p-values simultaneously. `statsmodels` is a hard dependency (C8 fix) — the silent Bonferroni fallback was removed.

---

## TORGO Speaker Severity Table

Severity formula: `severity = (1 - intelligibility/100) * 5.0` from Rudzicz et al. (2012).

| Speaker | Type | Approx. Intelligibility | Severity Score (`TORGO_SEVERITY_MAP`) |
|---|---|---|---|
| F01 | Dysarthric | ~2% | **4.90** |
| F03 | Dysarthric | ~7% | **4.65** |
| F04 | Dysarthric | ~39% | **3.05** |
| M01 | Dysarthric | ~2% | **4.90** |
| M02 | Dysarthric | ~4.5% | **4.78** |
| M03 | Dysarthric | ~8.2% | **4.59** |
| M04 | Dysarthric | ~43% | **2.85** |
| M05 | Dysarthric | ~58% | **2.10** |
| FC01 | Control | ~100% | 0.00 |
| FC02 | Control | ~100% | 0.00 |
| FC03 | Control | ~100% | 0.00 |
| MC01 | Control | ~100% | 0.00 |
| MC02 | Control | ~100% | 0.00 |
| MC03 | Control | ~100% | 0.00 |
| MC04 | Control | ~100% | 0.00 |

All 7 control speakers receive severity=0.0. This means all controls form a single severity bucket in ordinal contrastive loss. Mitigated by using continuous `TORGO_SEVERITY_MAP` scores for dysarthric speakers (C7 fix).

---

## Main Results — LOSO-CV (`loso_v1`)

This is the canonical publication result. All 15 folds complete. Per-fold results are in `results/loso_v1_loso_summary.json`.

| Metric | Value |
|---|---|
| Macro PER | **0.2848** |
| 95% CI (bootstrap, n=15 folds) | **[0.1921, 0.3801]** |
| Weighted PER | **0.2299** |
| Macro WER | **0.3362** |
| Weighted WER | **0.2631** |
| Folds complete | 15/15 |
| Total speakers | 15 (8 dysarthric + 7 control) |

Per-fold detail (speaker | PER | WER | n_samples | trained_epochs | elapsed_min) is available in `results/loso_v1_loso_summary.json`.

---

## Single-Split Baselines

**Caution:** All single-split results are computed on approximately 2 test speakers and are not publication-valid statistics. They are reported as development references only.

| Run name | avg_per | per_neural | per_constrained | Δ (const−neural) | Notes |
|---|---|---|---|---|---|
| `ablation_neural_only_v7` | **0.1346** | 0.1346 | N/A | N/A | Best overall single-split; no symbolic layer or SeverityAdapter |
| `baseline_v6` | 0.1372 | 0.1451 | 0.1372 | −0.0079 | Full system, post-fix; all B1–B23 applied |
| `ablation_no_constraint_matrix_v6` | 0.1444 | — | — | — | SeverityAdapter kept; constraint matrix removed |
| `baseline_v4` (historical) | 0.4748 | 0.305* | 0.4742 | — | Beam PER; insertion bias resolved (I/D=0.87×); superseded by `baseline_v6` and LOSO |
| `baseline_v2` | ~~0.215~~ | — | — | — | ⚠️ **Invalid** — B12 data leakage; manifest speaker extraction bug returned `'unknown'` for all speakers |

*`per_neural=0.305` in v4 is greedy decode of the model's internal neural sub-path in an older evaluation path, not an independent model. Not comparable to beam-decoded ablation results.

`baseline_v2` results are invalid due to B12 data leakage and must not be cited.

---

## Ablation Analysis

### Symbolic Constraint Effect (`baseline_v6`)

Within `baseline_v6`, the constrained path improves over the model's own internal neural sub-path. These measurements are of two decode paths within the **same jointly-trained model**, not two independent models.

- `per_neural` (greedy decode of logits_neural, internal): **0.1451**
- `per_constrained` (greedy decode of log_probs_constrained, internal): **0.1372**
- Δ = −0.0079 (constraint helps vs. internal neural path)
- `p_value_neural_vs_constrained` (bootstrap paired test, two-sided): **0.0** (statistically significant)
- Helpful / neutral / harmful per utterance: **9.16% / 87.06% / 3.78%**

Interpretation: The symbolic constraint meaningfully changes 12.94% of utterances (9.16% + 3.78%), and of those changes, roughly 70% are beneficial. The `p_value` of 0.0 reflects that empirically zero bootstrap samples produced a non-negative mean delta; it is statistically significant but the effect is practically small.

**Global comparison:** `ablation_neural_only_v7` (avg_per=0.1346) — which bypasses the entire symbolic layer and SeverityAdapter — still achieves a better global PER than the full system (0.1372). The overhead of the constraint path slightly hurts the utterances it does not improve. LOSO-CV stratified by dysarthric strata is required to determine whether the constraint provides meaningful advantage for severe speakers.

### Component Ablations

`ablation_no_constraint_matrix_v6` (avg_per=0.1444) removes the `LearnableConstraintMatrix` while keeping `SeverityAdapter` and articulatory heads. Its PER is worse than `baseline_v6` (0.1444 vs. 0.1372), confirming that the learnable constraint matrix adds value within the jointly-trained system. The `SeverityAdapter` alone (without the constraint matrix) provides no net PER benefit over removing it entirely.

### Error Profile History

Error counts from `baseline_v5` (historical, pre-LOSO, closest available reference):

| Error type | Count |
|---|---|
| Substitutions | 13,821 |
| Deletions | 4,338 |
| Insertions | 3,752 |

Insertion/Deletion ratio history demonstrating progressive insertion bias correction:

| Baseline | I/D ratio | Primary fix |
|---|---|---|
| `baseline_v1` | ~56× | None (B3 attention mask bug active) |
| `baseline_v4` | ~0.87× (final) | `BlankPriorKLLoss`, staged warmup; beam search |
| `baseline_v5` | ~0.9× | Same config, 30 epochs |

---

## Articulatory Accuracy

Evaluated utterance-level via global average pool (I5 fix: bypasses invalid frame-level CTC alignment). Target is the mode of the phoneme label sequence for the utterance. Measured on the full system (`baseline_v6`, final configuration used in paper analysis).

| Feature | Accuracy | Classes | Notes |
|---|---|---|---|
| Manner of articulation | **75.9%** | stop, fricative, affricate, nasal, liquid, glide, vowel, diphthong | Comparable across model configurations |
| Place of articulation | **82.0%** | bilabial, labiodental, dental, alveolar, postalveolar, palatal, velar, glottal, labio-velar, front, back, central | Lower than SeverityAdapter-only (`ablation_no_constraint_matrix_v6`: 88.7%), consistent with the structured label-smoothing trade-off described in the paper |
| Voicing | **94.4%** | voiced, voiceless, vowel | Strongest auxiliary axis and stable across constrained runs |

---

## Known Limitations

### Frame-CE Alignment (T-05)

`align_labels_to_logits()` in `src/utils/sequence_utils.py` uses proportional nearest-neighbor interpolation to map phoneme labels [B, L] to logit time dimension [B, T]. CTC does not provide forced alignment, so this assignment is approximate — frame 0..L-1 does not correspond to actual phoneme boundaries.

**Impact:** This is the most likely contributing cause of `per_constrained > per_neural` in early baselines (pre-v6). Frame-CE trains the phoneme classifier with near-random frame→phoneme associations, conflicting with the symbolic layer's distribution-shifting. **Mitigation applied:** `lambda_ce` reduced from 0.35 to 0.10 (C-1). **Full fix:** CTC forced alignment via `torchaudio.functional.forced_align` (future work).

### Small-N Statistical Validity

Single-split test sets contain approximately 2 speakers. All single-split statistics are invalid at n≈2. Pearson r=-0.85 with p=0.353 at n=3 reflects this: the correlation is directionally consistent but not statistically significant. LOSO-CV (15 folds) resolves this for the overall PER aggregate. Stratified dysarthric-subset LOSO analysis remains pending and is required for the primary SPCOM claim.

### Confusion Matrix Validity

Substitution/deletion/insertion statistics and confusion matrices are computed from greedy CTC decoder outputs without frame-level phoneme boundaries. Confusion statistics are indicative but not fully defensible until CTCSegmentation or Montreal Forced Aligner is applied.

---

## Per-Speaker Analysis

Per-speaker PER is available in `results/{run_name}/evaluation_results.json['per_speaker']` after any evaluation run. Each entry contains `per`, `ci` (bootstrap 95%), `std`, `n_samples`, and `status` (0=control, 1=dysarthric). For LOSO results, per-fold PER effectively corresponds to per-speaker PER since each fold holds out exactly one speaker.

---

## Reproduction Commands

```bash
# Reproduce baseline_v6 (full system, single split)
python run_pipeline.py --run-name baseline_v6

# Reproduce neural-only ablation
python run_pipeline.py --run-name ablation_neural_only_v7 --ablation neural_only

# Reproduce no-constraint-matrix ablation
python run_pipeline.py --run-name ablation_no_constraint_matrix_v6 --ablation no_constraint_matrix

# Reproduce LOSO (15/15 folds, ~32h on RTX 4060)
python run_pipeline.py --run-name loso_v1 --loso

# Resume LOSO from last completed fold (crash-safe)
python run_pipeline.py --run-name loso_v1 --loso --resume-loso

# Force-rerun specific LOSO folds (clears checkpoint and results dirs first)
python run_pipeline.py --run-name loso_v1 --loso --resume-loso \
    --loso-force-speakers M01,F01

# Eval-only with beam search and explainability on baseline_v6
python run_pipeline.py --run-name baseline_v6 --skip-train \
    --beam-search --beam-width 25 --explain --uncertainty

# Generate publication figures for LOSO result
python scripts/generate_figures.py --run-name loso_v1

# Compare runs in a single figure suite
python scripts/generate_figures.py --run-name baseline_v6 \
    --compare ablation_neural_only_v7 ablation_no_constraint_matrix_v6
```
