# DysarthriaNSR — Experiments, Results & Ablations

> Cross-reference: [docs/architecture.md](architecture.md) for component descriptions,
> [docs/reproducibility.md](reproducibility.md) for exact commands to reproduce these results.

---

## Dataset: TORGO Corpus

| Property | Value |
|----------|-------|
| Source | TORGO database — Rudzicz et al. (2012) |
| HuggingFace ID | `abnerh/TORGO-database` |
| Total speakers | 15 (8 dysarthric + 7 control) |
| Total utterances (post-manifest) | ~16,531 |
| Total duration | ~20h (approx.; exact in manifest `duration` column) |

### Speaker Severity

Severity scores derived from Rudzicz et al. (2012) mean percent-word-correct intelligibility ratings:
`severity = (1 - intelligibility/100) * 5.0`

| Speaker | Type | Approx. Intelligibility | Severity Score |
|---------|------|------------------------|----------------|
| F01 | Dysarthric | ~2% | 4.90 (severe) |
| F03 | Dysarthric | ~7% | 4.65 (severe) |
| F04 | Dysarthric | ~39% | 3.05 (moderate) |
| M01 | Dysarthric | ~2% | 4.90 (severe) |
| M02 | Dysarthric | ~4.5% | 4.78 (severe) |
| M03 | Dysarthric | ~8.2% | 4.59 (severe) |
| M04 | Dysarthric | ~43% | 2.85 (moderate) |
| M05 | Dysarthric | ~58% | 2.10 (mild) |
| FC01 | Control | ~100% | 0.00 |
| FC02 | Control | ~100% | 0.00 |
| FC03 | Control | ~100% | 0.00 |
| MC01 | Control | ~100% | 0.00 |
| MC02 | Control | ~100% | 0.00 |
| MC03 | Control | ~100% | 0.00 |
| MC04 | Control | ~100% | 0.00 |

---

## Evaluation Protocol

### Metric: Phoneme Error Rate (PER)

```
PER = (S + D + I) / N
```
where S = substitutions, D = deletions, I = insertions, N = reference phoneme count.
Computed using `editdistance.eval()` (Levenshtein distance at phoneme level).

**Aggregation:** **Macro-speaker PER** — compute per-utterance PER, group by speaker, average within-speaker, then average across speakers. Treats each speaker equally regardless of utterance count.

### Evaluation Protocol: Single Split

70%/15%/15% speaker-stratified split (`DataConfig.split_strategy="speaker_stratified"`):
- Train: ~10 speakers
- Val: ~2 speakers (used for early stopping; `val/per` is checkpointing metric)
- Test: ~2 speakers

**Known limitation:** With 15 speakers and a single 70/15/15 split, the test set contains approximately 2 speakers. All per-speaker statistics (PER by severity, dysarthric vs. control comparisons, Pearson r) are statistically invalid at n≈2. **LOSO-CV is mandatory for publication.** All single-split results reported here are interim development figures only.

### Evaluation Protocol: LOSO-CV (Required for Publication)

Leave-One-Speaker-Out cross-validation:
- 15 folds, one per speaker as test set
- Each fold: remaining 14 speakers split 12/2 for train/val
- Reported metric: macro-average PER across 15 fold test PER values with bootstrap 95% CI

---

## Baselines Compared

| System | Description | avg_per (single split) |
|--------|-------------|----------------------|
| `ablation_neural_only_v7` | HuBERT + PhonemeClassifier only; SymbolicConstraintLayer bypassed; no SeverityAdapter | **0.1346** |
| `ablation_no_constraint_matrix_v6` | HuBERT + SeverityAdapter + PhonemeClassifier; log-softmax in place of symbolic layer | 0.1444 |
| `baseline_v6` | Full system (HuBERT + SeverityAdapter + TemporalDownsampler + SymbolicConstraintLayer with learnable C) | 0.1372 |

### Symbolic Impact (baseline_v6)

The model forward pass outputs both `logits_neural` (pre-symbolic) and `log_probs_constrained` (post-symbolic). Both are greedily decoded during evaluation to assess the constraint layer's direct impact:

| Metric | baseline_v6 |
|--------|-------------|
| `per_neural` (neural classifier only, greedy) | 0.1451 |
| `per_constrained` (full system, greedy) | 0.1372 |
| Δ = constrained − neural | −0.0079 (symbolic helps vs internal neural path) |
| `p_value_neural_vs_constrained` | 0.0 (statistically significant) |
| Constraint helpful / neutral / harmful (per utterance) | 9.16% / 87.06% / 3.78% |

**Interpretation:** The constrained path improves over the model's internal neural sub-path in `baseline_v6`. However, the full neural-only ablation (`ablation_neural_only_v7`) still achieves the best global PER (0.135 < 0.137). This supports a "symbolic helps within-system but is not yet superior to pure HuBERT" conclusion. LOSO-CV on dysarthric-only strata is the decisive test.

---

## Ablations (B1–B5 Critical Fixes)

These five fixes were implemented between development sessions and are tracked in conversations `d9d04797` and `78e917b4`. They resolved systemic training failures.

| Fix | Component | Issue | Measured Impact |
|-----|-----------|-------|-----------------|
| B1 | `SymbolicKLLoss` | `reduction="batchmean"` divided by V=47; effective per-row weight was 0.001 (too weak). Changed to `reduction="sum"/V` with explicit normalisation; `lambda_symbolic_kl` raised 0.05→0.50 | Constraint matrix now tracks prior across training |
| B2 | `BlankPriorKLLoss` target | `target_prob` default/test fixture used 0.85 vs config 0.75; prior inconsistency. Aligned to 0.75 throughout | Blank KL loss now correctly-targeted |
| B3 | SpecAugment ordering | SpecAugment was applied after SeverityAdapter, masking adapter's severity signal. Moved to before SeverityAdapter | Severity conditioning no longer corrupted by masks |
| B4a | Log-prob underflow | `log_probs = log(P_final + 1e-12)` underflows to 0 in BF16. Changed to `clamp_min(1e-6)` | Stable log-probs in BF16 training |
| B4b | No-grad in rule tracking | `_track_activations` built a gradient graph even during inference, wasting memory. Wrapped in `torch.no_grad()` | Inference memory usage reduced |
| B5 | Val/test step missing keys | `logits_neural` and `hidden_states` absent from `compute_loss()` call in `validation_step`/`test_step`; CE loss silently fell back to constrained log-probs. Added both keys | Frame-CE computed on correct neural logits in validation |

### Additional Architecture Ablations (Documented in Codebase)

| Ablation mode | Description | Activation flag |
|--------------|-------------|--------|
| `no_spec_augment` | Remove SpecAugment from forward pass | `--ablation no_spec_augment` |
| `no_temporal_ds` | Remove TemporalDownsampler (direct 50Hz prediction) | `--ablation no_temporal_ds` |
| `no_art_heads` | Remove articulatory auxiliary supervision | `--ablation no_art_heads` |
| `symbolic_only` | CTC/CE disabled; test pure symbolic signal | `--ablation symbolic_only` |

Core ablations have now been run (`baseline_v6`, `ablation_neural_only_v7`, `ablation_no_constraint_matrix_v6`).
Recommended next pass is targeted dysarthric-fold optimization and additional component ablations (`no_spec_augment`, `no_temporal_ds`) under the LOSO protocol.

---

## Known Caveats & Limitations

### Statistical

1. **Test set contains ~2 speakers.** All reported single-split statistics (Pearson r, Welch t-test, Wilcoxon p-value, per-speaker PER) are computed on n≈2 and are not publication-valid. With n=2, 95% CI covers nearly the entire range [0,1].

2. **Severity stratification collapses to dysarthric/control.** All 7 control speakers have `severity=0.0`. Severity buckets mild/moderate/severe are therefore dysarthric-only; control speakers all appear in the "mild" bucket.

3. **Bootstrap CI computed on macro-speaker PER** (3–15 values). With n≈3 (single split), the CI is extremely wide and uninformative. With n=15 (LOSO), bootstrap CI over 15 fold PER values is appropriate.

4. **Inverted dysarthric/control PER ordering** observed in some single-split runs (control PER > dysarthric PER). Almost certainly an artefact of which 2 speakers land in the test set. LOSO-CV will correct this.

### Architectural

1. **Frame-CE alignment (T-05):** `align_labels_to_logits` pads/truncates phoneme labels to match logit time dimension without forced alignment. CE loss supervises using positionally-incorrect frame→phoneme assignments. Mitigated by reducing `lambda_ce` from 0.35 to 0.10. Full fix requires `torchaudio.functional.forced_align`.

2. **Confusion matrices without forced alignment:** Substitution/deletion/insertion statistics are computed from greedy CTC decoder outputs without frame-level phoneme boundaries. The identity of which phoneme was misarticulated is ambiguous without forced alignment.

3. **`OrdinalContrastiveLoss` zero-margin for control groups (partially mitigated):** All controls have `severity=0.0`, so all control–control pairs have `|sev_i − sev_j|=0` → `margin=0` → zero gradient contribution. Fixed by using continuous `TORGO_SEVERITY_MAP` severity scores in the forward path. However, the underlying binary design of control vs. dysarthric severity limits ordinal contrast.

---

## MLflow Tracking

All experiments are tracked with MLflow. Key logged values:

| Logged metric | Key |
|--------------|-----|
| Training loss | `train/loss` |
| CTC loss | `train/loss_ctc` |
| Mean blank probability | `train/blank_prob_mean` |
| Constraint β | `train/avg_beta` |
| Constraint matrix row entropy | `val/constraint_row_entropy` |
| Constraint matrix KL from prior | `val/constraint_kl_from_prior` |
| Validation PER (macro-speaker) | `val/per` |
| Dysarthric PER | `val/per_dysarthric` |
| Control PER | `val/per_control` |

**To access:**
```bash
mlflow ui --backend-store-uri file://$(pwd)/mlruns
# Open http://127.0.0.1:5000
```

**Key experiment names:** `DysarthriaNSR` (all runs share this experiment). Filter by `run_name` tag for specific baselines.

**Key runs documented in `docs/04_RESEARCH_AUDIT.md`:**
- `baseline_v4`, `baseline_v5` — historical pre-LOSO baselines
- `baseline_v6` — post-fix symbolic reference checkpoint
- `ablation_neural_only_v7` — best single-split performance
- `ablation_no_constraint_matrix_v6` — SeverityAdapter without learnable C
- `loso_v1` — 15/15 folds complete (publication aggregate)

---

## Evaluation Artifacts

Each evaluation run produces the following under `results/{run_name}/`:

| File | Description |
|------|-------------|
| `evaluation_results.json` | Full structured results (PER, CI, stratified, per-speaker, error counts, symbolic impact) |
| `config.yaml` | Exact run configuration for reproducibility |
| `confusion_matrix.png` | Top-30 phoneme confusion matrix, row-normalised |
| `per_by_length.png` | PER by utterance length × dysarthric/control status |
| `clinical_gap.png` | Dysarthric vs. control PER comparison bar |
| `rule_impact.png` | Symbolic constraint activation analysis / C matrix heatmap |
| `blank_probability_histogram.png` | Distribution of blank probabilities (target line at 0.75) |
| `per_phoneme_per.png` | Per-phoneme PER breakdown (top-30, colour-coded) |
| `articulatory_confusion.png` | Articulatory feature confusion heatmap |
| `severity_vs_per.png` | Scatter: severity score vs. per-speaker PER |
| `per_by_speaker.png` | Per-speaker PER bar chart, sorted by severity |
| `explanations.json` | Per-utterance explainability JSON (when `--explain`) |
