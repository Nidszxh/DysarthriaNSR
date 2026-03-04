# DysarthriaNSR — Comprehensive Next Steps & Gap Analysis
**Compiled from:** Senior Research Scientist Audit Report + ROADMAP.md cross-reference  
**Date:** February 2026  
**Status:** Post-baseline (`baseline_v1`, Test PER 0.567 ± 0.365)

---

## How to Read This Document

Findings are divided into three tiers:

- 🔴 **Critical** — Bugs or validity issues that must be fixed before any result is reported or shared
- 🟠 **High** — Gaps that significantly weaken the contribution; required for conference submission
- 🟡 **Medium / Research** — Enhancements that strengthen the paper or open new research threads

Each item includes a ROADMAP cross-reference showing whether it was already planned, partially planned, or entirely missing.

---

## Part 1 — Critical Bugs (Fix First)

### BUG-1 · Train/Eval Severity Scaling Mismatch
**Status:** Not in ROADMAP  
**Effort:** ~1 hour  
**Severity:** 🔴 Invalidates the adaptive-constraint claim

During training (`train.py`), severity is correctly scaled:
```python
severity = batch['status'].float() * 5.0   # → {0.0, 5.0}
```
But in `evaluate.py → evaluate_model()`, the raw status (0 or 1) is passed directly to the model without scaling. Inside `_compute_adaptive_beta`, this produces:
```
β_adaptive = β + 0.1 * (status / 5.0)   →   β + 0.02   (not β + 0.1)
```
The model at test time behaves as if severity adaptation is nearly disabled. All reported baseline numbers are affected.

**Fix:** In `evaluate.py`, replace
```python
severity = batch['status'].to(device)
```
with
```python
severity = batch['status'].float().to(device) * 5.0
```

---

### BUG-2 · Frame-Level CE Loss Applied to CTC-Aligned Labels
**Status:** Partially acknowledged in ROADMAP as "insertion bias"; root cause not identified  
**Effort:** 2–3 days to refactor  
**Severity:** 🔴 Causes the 21,290-insertion pathology; renders WER claims invalid

The frame-level `NLLLoss` in `compute_loss()` receives `log_probs_constrained` (shape `[B, T, V]`) paired with labels padded with `-100`. Because CTC does not provide forced alignment, the ground-truth phoneme labels cannot be meaningfully aligned to frame positions by simple truncation/padding. The result:

- Non-blank emissions on padding frames are neither penalized nor incentivized (the `-100` ignore silences them both ways)
- The CE loss does not teach the model when to emit blanks — only CTC does — but the CE loss's gradient still flows through β, creating a conflated signal
- This is the root cause of the 56× insertion/deletion ratio

**Options (pick one):**

**Option A — Remove CE loss entirely; rely on CTC only**  
Simplest fix. Revert `lambda_ce` to `0.0`. Validates whether CE was helping or hurting.

**Option B — Apply CE only to forced-alignment frames**  
Use CTC forced-alignment (e.g., via `torchaudio.functional.forced_align` or a Viterbi pass) to identify which frames correspond to which phonemes, then apply CE only there. This is correct but adds a preprocessing step.

**Option C — Replace frame CE with a blank-posterior KL regulariser**  
Add a soft constraint that the mean blank probability across all frames is ≥ 0.3:
```python
blank_probs = log_probs_constrained[:, :, blank_id].exp()   # [B, T]
target_blank = torch.full_like(blank_probs, 0.30)
loss_blank_kl = F.mse_loss(blank_probs.mean(dim=1), target_blank.mean(dim=1))
total_loss += 0.1 * loss_blank_kl
```
This directly addresses the insertion bias without requiring forced alignment.

**Recommended immediate action:** Implement Option A or C, retrain, and compare insertion counts before reporting any result.

---

### BUG-3 · Macro-Average PER Not Computed
**Status:** Acknowledged partially in ROADMAP ("per-speaker breakdown") but never actually implemented as a primary metric  
**Effort:** ~2 hours  
**Severity:** 🔴 Sample-mean PER is statistically misleading on TORGO

TORGO speakers have unequal utterance counts. Reporting `np.mean(per_scores)` over all samples allows high-volume speakers to dominate. Per conference standards, the primary reported metric must be macro-average PER (mean of per-speaker means):
```python
speaker_per_means = [np.mean(per_scores_for_speaker) for speaker in test_speakers]
macro_per = np.mean(speaker_per_means)
```

**Fix:** Update `evaluate_model()` to report `macro_per` as the primary headline metric alongside sample-mean for completeness.

---

## Part 2 — High Priority (Required for Conference Submission)

### HIGH-1 · Implement Leave-One-Speaker-Out (LOSO) Cross-Validation
**ROADMAP status:** Listed as "Cross-Validation Strategy" in architecture diagram; never implemented  
**Effort:** 3 days  
**Severity:** 🟠 Without this, no statistical claim about generalization is defensible

With 15 TORGO speakers, a single seed-42 train/val/test split yields a 95% CI of ±0.365 — spanning essentially the full range [0, 1]. Every conference reviewer will flag this.

**Implementation plan:**

1. Create `src/training/loso_cv.py` with a `LOSOCrossValidator` class
2. For each speaker `k` in 1..15: train on speakers ≠ k, evaluate on speaker `k`
3. Report macro-average PER per fold, then mean ± std across all 15 folds
4. Apply Holm–Bonferroni correction for the 15 pairwise significance tests

```python
def run_loso(config: Config, dataset: TorgoNeuroSymbolicDataset) -> Dict:
    speakers = dataset.df['speaker'].unique()
    fold_results = []
    
    for held_out_speaker in speakers:
        train_idx = dataset.df[dataset.df['speaker'] != held_out_speaker].index.tolist()
        test_idx  = dataset.df[dataset.df['speaker'] == held_out_speaker].index.tolist()
        # ... train model, evaluate, record macro_per
        fold_results.append({'speaker': held_out_speaker, 'macro_per': macro_per})
    
    return fold_results
```

---

### HIGH-2 · Validate and Correct Substitution Rules Against Empirical Confusion Matrix
**ROADMAP status:** Listed as "Symbolic rule discovery — auto-extract from confusion matrices" (Not Implemented)  
**Effort:** 1 day  
**Severity:** 🟠 Hard-coded rules not validated against actual model confusions

The 27 rules in `SymbolicConfig.substitution_rules` (e.g., B→P = 0.85) are from clinical phonology literature but have not been compared to the empirically observed confusion frequencies from `baseline_v1`.

**Implementation plan:**

1. Load `results/baseline_v1/evaluation_results.json` and extract the confusion matrix
2. For each hard-coded rule pair (source, target), compare rule weight to empirical confusion rate
3. Flag rules where the discrepancy is > 0.2 (likely reinforcing incorrect patterns)
4. Create `scripts/validate_symbolic_rules.py` that outputs a comparison table

This analysis is itself a publishable table showing whether the clinical prior aligns with acoustic reality on TORGO.

---

### HIGH-3 · Run Required Ablation Studies
**ROADMAP status:** Listed explicitly; never executed  
**Effort:** 3–5 days (primarily compute time)  
**Severity:** 🟠 No paper can be submitted without at minimum neural-only vs. neuro-symbolic

Minimum required ablations (in priority order):

| Variant | Config Change | Purpose |
|---|---|---|
| **Baseline HuBERT** | Freeze all layers, no fine-tuning | Upper bound on SSL baseline |
| **Neural-only** | `constraint_learnable=False`, `constraint_weight_init=1.0` (β=1.0) | Validates symbolic contribution |
| **Symbolic-heavy** | `constraint_weight_init=0.0` (β=0.0) | Tests rule-only performance |
| **No articulatory heads** | `lambda_articulatory=0.0` | Validates auxiliary head contribution |
| **β sweep** | Fixed β ∈ {0.0, 0.3, 0.5, 0.7, 1.0}, `constraint_learnable=False` | Sensitivity analysis |

Create `scripts/run_ablations.py` that iterates over the above configs and produces a unified results table.

---

### HIGH-4 · Resolve CTC Insertion Bias with Systematic Experiment
**ROADMAP status:** Listed as "in design phase"; mitigation plan exists but not executed  
**Effort:** 1 week  
**Severity:** 🟠 Cannot submit with a 56× insertion/deletion ratio without explanation and fix

Immediate diagnostic steps (add `scripts/diagnose_blank_probs.py`):

```python
# Step 1: Histogram of blank posterior probabilities
blank_probs = model_outputs[:, :, blank_id].exp()
plt.hist(blank_probs.flatten().cpu().numpy(), bins=50)

# Step 2: Compare blank probs dysarthric vs. control  
blank_dysarthric = blank_probs[status==1].mean()
blank_control    = blank_probs[status==0].mean()

# Step 3: Per-frame entropy
entropy = -(probs * (probs + 1e-12).log()).sum(-1)
```

Then test the following mitigations in order:
1. Increase `blank_priority_weight` from 1.5 → {2.0, 2.5, 3.0} and retrain
2. Add blank KL regulariser (see BUG-2, Option C)
3. If still >5K insertions, switch to Option B (forced alignment for CE)

Report I/D/S counts and PER for each mitigation configuration.

---

### HIGH-5 · Implement Speaker-Level Continuous Severity Scores
**ROADMAP status:** Not planned; binary status is a project-wide assumption  
**Effort:** 2–3 days  
**Severity:** 🟠 Binary severity defeats the purpose of "adaptive" β

TORGO provides intelligibility scores from Rudzicz et al. (2012). Map these to a continuous [0, 5] severity scale per speaker:

| Speaker | Intelligibility | Normalized Severity |
|---|---|---|
| M01 | 2.0% | 5.0 |
| M02 | 28.0% | 3.6 |
| M04 | 93.0% | 0.4 |
| F01 | 6.0% | 4.9 |
| FC01–FC03, MC01–MC04 | ~100% | 0.0 |

Add this mapping to `DataConfig` or `SymbolicConfig`, update `manifest.py` to include a `severity_score` column, and replace the binary `status * 5.0` scaling in both `train.py` and `evaluate.py`.

---

## Part 3 — Medium Priority (Strengthens the Paper)

### MED-1 · Acoustic Augmentation Pipeline
**ROADMAP status:** Not mentioned  
**Effort:** 1 week  
**Notes:** Peak normalization alone is insufficient for real-world clinical conditions

Add to `src/data/dataloader.py` or a new `src/data/augmentation.py`:

```python
import torchaudio.transforms as T

class DysarthricAugmentation:
    """Augmentations appropriate for dysarthric clinical data."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
        self.speed_factors = [0.85, 0.90, 0.95, 1.0, 1.05, 1.10]
        self.spec_aug = T.SpecAugment(
            n_time_masks=2, time_mask_param=50,
            n_freq_masks=2, freq_mask_param=27
        )
    
    def __call__(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if random.random() < self.p:
            factor = random.choice(self.speed_factors)
            waveform = T.Resample(sr, int(sr * factor))(waveform)
        return waveform
```

Apply augmentation only during training (not validation/test). Speed perturbation is especially important for dysarthric speech — the model should be invariant to the slower articulation rate of severe speakers.

---

### MED-2 · Learnable Constraint Matrix (Proposal 2 from Audit)
**ROADMAP status:** Not planned  
**Effort:** 2 weeks  
**Notes:** High publishability; the discovered C is a research result in itself

Replace the static `C` buffer in `SymbolicConstraintLayer` with a learnable parameter initialized from the articulatory prior:

```python
class LearnableConstraintMatrix(nn.Module):
    def __init__(self, num_phonemes: int, init_matrix: torch.Tensor):
        super().__init__()
        # Initialize from symbolic prior
        self.logit_C = nn.Parameter(torch.log(init_matrix.clamp(1e-8)))
    
    @property
    def C(self) -> torch.Tensor:
        return F.softmax(self.logit_C, dim=-1)
```

Add a KL anchor loss to prevent the matrix from diverging from the phonological prior:
```python
loss_symbolic_kl = F.kl_div(
    self.logit_C.log_softmax(-1),
    C_symbolic.log(),
    reduction='batchmean', log_target=True
)
total_loss += lambda_kl * loss_symbolic_kl  # lambda_kl ∈ {0.01, 0.05, 0.1}
```

---

### MED-3 · Cross-Attention Severity Adapter (Proposal 3 from Audit)
**ROADMAP status:** "Multi-speaker adaptation" listed as medium-term future work  
**Effort:** 1 week  
**Notes:** Replaces binary β scaling with representational severity conditioning

Insert a `SeverityAdapter` module between HuBERT encoder and PhonemeClassifier:

```python
class SeverityAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 768, severity_dim: int = 64):
        super().__init__()
        self.severity_proj = nn.Sequential(
            nn.Linear(1, severity_dim), nn.SiLU(), nn.Linear(severity_dim, hidden_dim)
        )
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, hidden_states, severity):  # severity: [B] continuous [0, 5]
        sev_ctx = self.severity_proj(severity.view(-1, 1, 1))   # [B, 1, 768]
        attn_out, _ = self.cross_attn(hidden_states, sev_ctx, sev_ctx)
        return self.layer_norm(hidden_states + attn_out)
```

Requires HIGH-5 (continuous severity scores) as a prerequisite.

---

### MED-4 · Attention Weight Extraction for Attribution
**ROADMAP status:** Listed as "not implemented"; required for explainability claims  
**Effort:** 2–3 days  
**Notes:** Required to substantiate the "clinically explainable" claim

Change the HuBERT forward call to expose attention weights:
```python
hubert_outputs = self.hubert(
    input_values,
    attention_mask=attention_mask,
    output_hidden_states=True,
    output_attentions=True,   # ← add this
    return_dict=True
)
attention_weights = hubert_outputs.attentions  # Tuple of [B, heads, T, T] per layer
```

Then add an `AttentionRollout` utility in `src/visualization/attention.py` for phoneme-level attribution visualizations.

---

### MED-5 · Per-Phoneme and Per-Speaker Error Rate Reporting
**ROADMAP status:** Partially implemented in `evaluate.py` but not surfaced in primary results  
**Effort:** 1 day  
**Notes:** Required for clinical interpretability; reviewers will ask for this

Add to `evaluate_model()`:
- A sorted table of per-phoneme PER highlighting the most-confused pairs
- Per-speaker PER with bootstrap CIs reported as the primary evaluation table
- Statistical significance tests (Welch's t-test) for dysarthric vs. control PER difference

---

## Part 4 — Research-Grade Enhancements (Future Publication Threads)

### RES-1 · Ordinal Contrastive Severity Loss (Proposal 1 from Audit)
**Prerequisite:** HIGH-5 (continuous severity scores)  
**Expected impact:** 3–5% relative PER reduction for severe speakers  
**Venue:** INTERSPEECH / TASLP

Adds a contrastive loss that pulls representations of similar-severity speakers together and pushes dissimilar ones apart, providing a data-driven basis for severity-adaptive β. See audit report Section III, Proposal 1 for full implementation.

---

### RES-2 · Uncertainty-Aware Decoding with Conformal Prediction (Proposal 4 from Audit)
**Prerequisite:** None (can add to existing model)  
**Expected impact:** Clinical utility; high SLP trust  
**Venue:** INTERSPEECH / ML4H workshop

Monte Carlo dropout at inference to estimate epistemic uncertainty per phoneme prediction. Output phoneme *sets* with guaranteed coverage (e.g., "the model predicts /b/ or /p/ with 95% confidence"). See audit report Section III, Proposal 4 for full implementation.

---

## Part 5 — Prioritized Execution Order

```
WEEK 1
  BUG-1 · Fix severity scaling in evaluate.py (1 hour)
  BUG-3 · Add macro-average PER metric (2 hours)
  HIGH-4 · Run blank-prob diagnostic → implement blank KL regulariser or remove CE loss
  
WEEK 2
  BUG-2 · Retrain with CE fix (Option A or C) and validate insertion count drops
  MED-5 · Add per-speaker + per-phoneme breakdown to evaluation report
  HIGH-5 · Load TORGO intelligibility scores; add severity_score to manifest

WEEK 3–4
  HIGH-1 · Implement LOSO cross-validation infrastructure and run 15 folds
  HIGH-3 · Run ablation study suite (neural-only, symbolic-only, no art heads, β sweep)

WEEK 5
  HIGH-2 · Validate substitution rules against empirical confusion matrix
  MED-1 · Add speed perturbation + SpecAugment augmentation and retrain

WEEK 6–7
  MED-2 · Implement learnable constraint matrix + KL anchor
  MED-3 · Implement SeverityAdapter (requires HIGH-5 continuous severity)

WEEK 8+
  RES-1 · Ordinal contrastive loss
  RES-2 · Uncertainty-aware conformal decoder
```

---

## Part 6 — Minimum Bar for INTERSPEECH 2026 Submission

The following is the minimum checklist before a draft paper can be submitted:

- [ ] **BUG-1** resolved — severity scaling consistent between train and eval
- [ ] **BUG-2** resolved — insertion count reduced to < 5,000 (from 21,290)
- [ ] **BUG-3** resolved — macro-average PER reported as primary metric
- [ ] **HIGH-1** complete — LOSO CV with macro-PER per fold and 95% bootstrap CI
- [ ] **HIGH-3** complete — at minimum: neural-only vs. neuro-symbolic ablation
- [ ] **HIGH-2** complete — symbolic rules validated against empirical confusion matrix
- [ ] **HIGH-5** complete — continuous severity scores used (not binary)
- [ ] Statistical significance test for dysarthric vs. control PER difference reported
- [ ] Per-speaker and per-phoneme error breakdowns included

---

## Appendix: ROADMAP.md Completion Status

| ROADMAP Item | Status | This Document |
|---|---|---|
| Data pipeline (download → manifest → dataloader) | ✅ Done | — |
| HuBERT + phoneme classifier + symbolic layer | ✅ Done | — |
| Multi-task training (CTC + CE + articulatory) | ✅ Done (but CE is pathological) | BUG-2 |
| Evaluation PER/WER + confusion matrices | ✅ Done (but sample-mean, not macro) | BUG-3 |
| Beam search decoding | ✅ Done | — |
| MLflow logging | ✅ Done | — |
| **Insertion bias mitigation** | ❌ Not done | HIGH-4, BUG-2 |
| **Ablation studies** | ❌ Not done | HIGH-3 |
| **LOSO cross-validation** | ❌ Not done | HIGH-1 |
| **Symbolic rule discovery from confusion matrix** | ❌ Not done | HIGH-2 |
| **Multi-speaker adaptation (speaker embedding)** | ❌ Not done | MED-3 |
| **Real-time streaming inference / ONNX** | ❌ Not done | RES-6 |
| **Clinician-facing dashboard** | ❌ Not done | Out of scope for paper |
| **Attention visualization** | ❌ Not done | MED-4 |
| **Continuous severity scoring** | ❌ Not done (was never planned) | HIGH-5 |
| **Acoustic augmentation** | ❌ Not done (was never planned) | MED-1 |
| **Cross-corpus evaluation** | ❌ Not done | RES-5 |
