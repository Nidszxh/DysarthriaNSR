# docs/experiments.md — Experiments, Results & Ablations

> Cross-references: [docs/architecture.md](architecture.md) for component descriptions, [docs/training.md](training.md) for exact CLI reproduction commands.

---

## Evaluation Protocol

### Phoneme Error Rate (PER)
```
PER = (S + D + I) / N
```

where S = substitutions, D = deletions, I = insertions, N = reference phoneme count. Computed using the `editdistance` C extension at phoneme level via `compute_per()` in `evaluate.py`.

**Aggregation — macro-speaker PER:** Per-utterance PER scores are grouped by speaker, averaged within each speaker, then averaged across speakers. This treats each speaker equally regardless of utterance count. The macro-speaker aggregate is the publication metric. Sample-mean PER (averaging all utterances directly) is reported for reference only.

**Why macro, not sample-mean:** TORGO speakers have highly variable utterance counts (some speakers have 3× more recordings than others). Sample-mean PER would over-weight high-utterance speakers and obscure per-speaker performance differences that are the clinical object of interest.

### Why LOSO is Mandatory

TORGO has 15 speakers. A 70/15/15 stratified split yields approximately 2 test speakers. With n=2, all per-speaker statistics (Pearson r, Welch t-test, CI) are statistically invalid. Single-split Pearson r between severity and PER (n=2–3) has a 95% CI spanning nearly [−1, 1]. LOSO-CV with 15 folds provides the minimum sample size for bootstrap CI and significance testing.

### Bootstrap CI Methodology

Bootstrap 95% CI is computed over macro-speaker PER scores (one value per fold/speaker). With n=15 (LOSO), 1000 bootstrap resamples with replacement are drawn; the 2.5th and 97.5th percentiles give the CI bounds. Implemented in `compute_per_with_ci()` in `evaluate.py`.

---

## TORGO Speaker Severity Table

Severity formula: `severity = (1 - intelligibility/100) * 5.0`

| Speaker | Type | Approx. Intelligibility | Severity Score |
|---|---|---|---|
| F01 | Dysarthric | ~2% | 4.90 |
| F03 | Dysarthric | ~7% | 4.65 |
| F04 | Dysarthric | ~39% | 3.05 |
| M01 | Dysarthric | ~2% | 4.90 |
| M02 | Dysarthric | ~4.5% | 4.78 |
| M03 | Dysarthric | ~8.2% | 4.59 |
| M04 | Dysarthric | ~43% | 2.85 |
| M05 | Dysarthric | ~58% | 2.10 |
| FC01 | Control | ~100% | 0.00 |
| FC02 | Control | ~100% | 0.00 |
| FC03 | Control | ~100% | 0.00 |
| MC01 | Control | ~100% | 0.00 |
| MC02 | Control | ~100% | 0.00 |
| MC03 | Control | ~100% | 0.00 |
| MC04 | Control | ~100% | 0.00 |

---

## Main Results — LOSO-CV (loso_v1)

This is the canonical publication result. All 15 folds complete. Per-fold results are in `results/loso_v1_loso_summary.json`.

| Metric | Value |
|---|---|
| Macro PER | **0.2848** |
| 95% CI (bootstrap, n=15) | [0.1921, 0.3801] |
| Weighted PER | **0.2299** |
| Macro WER | **0.3362** |
| Weighted WER | **0.2631** |
| n_folds | 15 |
| Total speakers | 15 (8 dysarthric + 7 control) |

---

## Single-Split Baselines

**Caution:** All single-split results below are computed on approximately 2 test speakers and are not publication-valid statistics. They are reported as development references only. See the LOSO result above for the publication aggregate.

| Run name | avg_per | per_neural | per_constrained | Δ (const−neural) | Notes |
|---|---|---|---|---|---|
| `ablation_neural_only_v7` | **0.1346** | 0.1346 | N/A | N/A | Best overall; no symbolic layer |
| `baseline_v6` | 0.1372 | 0.1451 | 0.1372 | −0.0079 | Full system, post-fix |
| `ablation_no_constraint_matrix_v6` | 0.1444 | — | — | — | SeverityAdapter only |
| `baseline_v5` (historical) | 0.4750 | 0.305* | 0.4750 | — | Pre-fix; per_neural via greedy internal only |
| `baseline_v4` (historical) | 0.4748 | 0.305* | 0.4742 | — | Beam PER; insertion bias resolved (I/D=0.87×) |

*`per_neural=0.305` in v4/v5 is greedy decode of the model's internal neural sub-path, not an independent model. Not comparable to beam-decoded ablation results.

**Note:** `baseline_v2` results are **invalid** due to B12 data leakage (manifest speaker extraction bug `split('_')[0]` returned `'unknown'` for all speakers). All baseline_v2 metrics are inflated and must not be cited.

---

## Ablation Analysis

### Symbolic Constraint Effect

In `baseline_v6`, the constrained path improves over the model's internal neural sub-path:

- `per_neural` (greedy, internal): 0.1451
- `per_constrained` (greedy, internal): 0.1372
- Δ = −0.0079 (constraint helps versus internal neural path)
- `p_value_neural_vs_constrained`: 0.0 (bootstrap paired test, statistically significant)
- Helpful / neutral / harmful (per utterance): **9.16% / 87.06% / 3.78%**

**Interpretation:** The symbolic constraint meaningfully changes 13% of utterances, and of those changes, 9.16%/13% ≈ 70% are beneficial. However, `ablation_neural_only_v7` (which bypasses the entire symbolic layer and SeverityAdapter) achieves a better global PER (0.1346 vs. 0.1372). This means the symbolic constraint helps within the jointly-trained system but the overhead of the constraint path slightly hurts the parts it does not improve. LOSO-CV stratified by dysarthric strata is required to determine whether the constraint provides a meaningful advantage for severe speakers.

### Component Ablations

`ablation_no_constraint_matrix_v6` (avg_per=0.1444) removes the `LearnableConstraintMatrix` while keeping `SeverityAdapter`. Its PER is worse than `baseline_v6` (0.1444 vs. 0.1372), indicating that the `SeverityAdapter` alone without the constraint matrix does not provide a net benefit. The full system (`baseline_v6`) outperforms this ablation, confirming the learnable constraint matrix adds value within the jointly-trained system.

### Error Profile

Error counts from `baseline_v5` (historical, pre-LOSO):
- Substitutions: 13,821
- Deletions: 4,338
- Insertions: 3,752

Insertion/Deletion ratio history demonstrating the progressive fix of insertion bias:

| Baseline | I/D ratio | Fix applied |
|---|---|---|
| `baseline_v1` | 56× | None (B3 attention mask bug active) |
| `baseline_v4` | ~4.6× (internal) → 0.87× (final) | `BlankPriorKLLoss`, staged warmup |
| `baseline_v5` | 0.9× | Same config, 30 epochs |

---

## Articulatory Accuracy

Evaluated utterance-level via global average pool (GAP) over time-axis features, with mode of phoneme label sequence as target (I5 fix). Measured on `baseline_v4`.

| Feature | Accuracy | Notes |
|---|---|---|
| Manner of articulation | **78.6%** | stop/fricative/nasal/liquid/glide/vowel/affricate/diphthong |
| Place of articulation | **79.1%** | bilabial/alveolar/velar/postalveolar/etc. |
| Voicing | **92.4%** | voiced/voiceless/vowel |

---

## Known Limitations

### Frame-CE Alignment (T-05)

`align_labels_to_logits()` in `src/utils/sequence_utils.py` uses proportional nearest-neighbor interpolation to map phoneme labels [B, L] to logit time dimension [B, T]. CTC does not provide forced alignment, so this assignment is approximate — frame 0..L-1 does not correspond to actual phoneme boundaries. For a 3-phoneme utterance decoded to 150 frames, the CE loss supervises using positionally-incorrect frame→phoneme assignments.

**Impact:** This is the most likely contributing cause of `per_constrained > per_neural` in early baselines (pre-v6). Frame-CE trains the phoneme classifier with near-random frame→phoneme associations, which may conflict with the symbolic layer's distribution-shifting. **Mitigation applied:** `lambda_ce` reduced from 0.35 to 0.10 (C-1). **Full fix:** CTC forced alignment via `torchaudio.functional.forced_align` (future work).

### Small-N Statistical Validity

Single-split test sets contain approximately 2 speakers. All single-split statistics — Pearson r (severity vs. PER), Welch t-test (dysarthric vs. control), per-speaker CI — are invalid at n≈2. Pearson r=-0.85 with p=0.353 at n=3 reflects this: the correlation is directionally consistent but not statistically significant. LOSO-CV (15 folds, `loso_v1`) resolves this for the overall PER aggregate. Stratified dysarthric-subset LOSO analysis remains pending.

### Confusion Matrix Validity

Substitution/deletion/insertion statistics and confusion matrices are computed from greedy CTC decoder outputs without frame-level phoneme boundaries. The identity of which phoneme was misarticulated at which frame is ambiguous without CTC forced alignment. Confusion statistics (e.g., T→AH dominance, vowel centralization patterns) are indicative but not fully defensible until CTCSegmentation or Montreal Forced Aligner is applied.

---

## Reproduce Results
```bash
# Reproduce baseline_v6 (full system, single split)
python run_pipeline.py --run-name baseline_v6

# Reproduce neural-only ablation
python run_pipeline.py --run-name ablation_neural_only_v7 --ablation neural_only

# Reproduce no-constraint-matrix ablation
python run_pipeline.py --run-name ablation_no_constraint_matrix_v6 --ablation no_constraint_matrix

# Reproduce LOSO (15/15 folds, ~32h)
python run_pipeline.py --run-name loso_v1 --loso

# Resume LOSO from last completed fold
python run_pipeline.py --run-name loso_v1 --loso --resume-loso

# Force-rerun specific LOSO folds
python run_pipeline.py --run-name loso_v1 --loso --resume-loso --loso-force-speakers M01,F01

# Eval-only with beam search and explainability on baseline_v6
python run_pipeline.py --run-name baseline_v6 --skip-train \
    --beam-search --beam-width 25 --explain --uncertainty
```