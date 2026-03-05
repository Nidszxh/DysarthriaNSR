# DysarthriaNSR — TODO (March 2026)

Current baseline: **baseline_v5** (beam PER 0.4750, March 6 2026).  
Critical blocker: symbolic constraint layer degrades PER by 57% relative (`per_neural=0.305` vs `per_constrained=0.475`).  
LOSO-CV not yet run (n=3 test speakers; no statistically valid macro-PER or severity correlation).

---

## 🔴 IMMEDIATE (do first — blocks everything else)

- [ ] **Fix `ablation_mode='neural_only'`** in `train.py`  
  Currently does **not** disable `SeverityAdapter` / `LearnableConstraintMatrix` forward passes.  
  Fix: gate those submodules on `ablation_mode != 'neural_only'` in `NeuroSymbolicASR.forward()`.

- [ ] **Run neural-only ablation** (after fix above)  
  ```bash
  python run_pipeline.py --run-name ablation_neural_only --ablation neural_only \
      --beam-search --beam-width 25
  ```
  Expected: `per_constrained ≈ per_neural ≈ 0.305`; confirms neural floor on 3-speaker test split.

- [ ] **Disable learnable constraint and verify** (1-line config change)  
  `src/utils/config.py` → `use_learnable_constraint: bool = False`  
  Short smoke run to confirm `delta_per` drops near 0.

---

## 🟠 HIGH PRIORITY

- [ ] **Run LOSO-CV** (run *after* at least one of the constraint fixes above)  
  ```bash
  python run_pipeline.py --run-name loso_v1 --loso
  ```
  Runtime: ~15–22h overnight.  
  Output: `results/loso_v1_loso_summary.json` with macro-PER (n=15), 95% CI, severity correlation.  
  Success criterion: Pearson r severity↔PER with p < 0.05.

- [ ] **Regenerate publication figures for baseline_v5**  
  ```bash
  python scripts/generate_figures.py --run-name baseline_v5
  ```
  Outputs 6 diagnostic plots to `results/baseline_v5/figures/`.

---

## 🟡 MEDIUM PRIORITY (ablations)

- [ ] **`--ablation no_art_heads`** — disable articulatory auxiliary heads; measure PER impact  
  ```bash
  python run_pipeline.py --run-name ablation_no_art_heads --ablation no_art_heads \
      --beam-search --beam-width 25
  ```

- [ ] **`--ablation no_constraint_matrix`** — disable constraint blending entirely  
  ```bash
  python run_pipeline.py --run-name ablation_no_constraint --ablation no_constraint_matrix \
      --beam-search --beam-width 25
  ```

- [ ] **β fixed-value sweep** — set `constraint_learnable=False` and test β ∈ {0.0, 0.1, 0.3, 0.5}  
  Diagnose whether the learnable β or the constraint matrix itself is the problem.

- [ ] **Raise `lambda_symbolic_kl`** from 0.05 → 0.5  
  Current effective per-row penalty ≈ 0.001 (too weak to prevent matrix drift toward `<BLANK>`).

---

## 🟢 LOW PRIORITY / LONG-TERM

- [ ] **Inspect `LearnableConstraintMatrix` weight evolution**  
  Log `constraint_matrix.diag().mean()` to MLflow per epoch; compare to data-driven confusion matrix.

- [ ] **`SymbolicRuleTracker` fix**  
  `avg_confidence=0.131`; most activations are X→`<BLANK>`.  
  Audit `_track_activations()` call site; ensure per-frame rule IDs map to correct phoneme pairs.

- [ ] **CTC forced alignment**  
  Unblocks `PhonemeAttributor.attention_attribution` (currently disabled).  
  Approach: Montreal Forced Aligner or CTC segmentation library.

- [ ] **Compare LOSO results across ablation variants**  
  ```bash
  python scripts/generate_figures.py --run-name loso_v1 \
      --compare ablation_neural_only ablation_no_constraint
  ```

- [ ] **ONNX export + streaming inference**  
  Prerequisite for clinician dashboard demo.

- [ ] **Domain adaptation to UASpeech**  
  Transfer learning from TORGO checkpoint → minimal fine-tuning on UASpeech speakers.

---

## ✅ COMPLETED (March 2026)

- [x] All B1–B12 bugs fixed (CTC stride, KL direction, LOSO name mutation, speaker extraction, etc.)
- [x] `run_pipeline.py` orchestrator (single entry point)
- [x] baseline_v3 trained (first valid speaker splits; val/per 0.574)
- [x] baseline_v4 trained + evaluated (beam PER 0.4748; insertion bias resolved; I/D=0.87×)
- [x] baseline_v5 trained + evaluated (beam PER 0.4750; same config as v4 but 30 epochs)
- [x] Manifest regenerated (B12 fix; correct TORGO speaker IDs)
- [x] Smoke tests passing (7/7)
- [x] Explainability JSON + uncertainty MC-Dropout wired into evaluation pipeline
- [x] Publication-quality figure suite (`scripts/generate_figures.py`, 6 plots)
- [x] LOSO infrastructure implemented (`run_loso()` in `train.py`; `--loso` flag in `run_pipeline.py`)
