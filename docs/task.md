# DysarthriaNSR — Full Implementation Task

## Phase 0 — Critical Bug Fixes
- [x] Review all source files for bugs
- [/] Fix severity scaling mismatch: [evaluate.py](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py) passes raw 0/1, not 0/5 scale
- [/] Fix frame-CE label alignment pathology (NLL on mis-padded labels → insertion bias)
- [/] De-duplicate `input_lengths` clamp in [training_step](file:///home/nidszxh/Projects/DysarthriaNSR/train.py#330-424) (done twice, lines 358–367)
- [/] Fix metadata key: [dataloader.py](file:///home/nidszxh/Projects/DysarthriaNSR/src/data/dataloader.py) uses `metadata['transcript']` but manifest may have [label](file:///home/nidszxh/Projects/DysarthriaNSR/train.py#303-329) column, not `transcript`

## Phase 1 — Config & Infrastructure Upgrades
- [/] Add `SeverityConfig` dataclass with continuous severity mapping per TORGO speaker
- [/] Add LOSO cross-validation infrastructure to [train.py](file:///home/nidszxh/Projects/DysarthriaNSR/train.py) (`create_loso_splits()`)
- [/] Add `ablation_mode` flag to [TrainingConfig](file:///home/nidszxh/Projects/DysarthriaNSR/src/utils/config.py#63-103) (neural_only, symbolic_only, full)
- [/] Add `use_severity_adapter`, `use_learnable_constraint`, `use_ordinal_loss` flags to [ModelConfig](file:///home/nidszxh/Projects/DysarthriaNSR/src/utils/config.py#45-62)
- [/] Update [requirements.txt](file:///home/nidszxh/Projects/DysarthriaNSR/requirements.txt) with new dependencies (torch-geometric, scikit-learn, etc.)

## Phase 2 — Learnable Constraint Matrix (Proposal P2)
- [ ] Add `LearnableConstraintMatrix` class to [src/models/model.py](file:///home/nidszxh/Projects/DysarthriaNSR/src/models/model.py)
  - [ ] Parameterize `logit_C` (raw softmax-parameterized constraint matrix)
  - [ ] Add `symbolic_kl_loss()` method for KL divergence anchor
- [ ] Replace static `SymbolicConstraintLayer.constraint_matrix` buffer with `LearnableConstraintMatrix`
- [ ] Add `lambda_kl` weight to [TrainingConfig](file:///home/nidszxh/Projects/DysarthriaNSR/src/utils/config.py#63-103) and [compute_loss()](file:///home/nidszxh/Projects/DysarthriaNSR/train.py#201-254) in [train.py](file:///home/nidszxh/Projects/DysarthriaNSR/train.py)
- [ ] Log learned `C` snapshot to MLflow at end of each epoch

## Phase 3 — Cross-Attention Severity Adapter (Proposal P3)
- [ ] Add `SeverityAdapter` class to [src/models/model.py](file:///home/nidszxh/Projects/DysarthriaNSR/src/models/model.py)
  - [ ] Continuous severity projection: `nn.Linear(1, 64) → SiLU → nn.Linear(64, 768)`
  - [ ] `nn.MultiheadAttention` cross-attention (query=hidden_states, key/value=severity_ctx)
  - [ ] Residual + LayerNorm
- [ ] Insert `SeverityAdapter` between HuBERT output and [PhonemeClassifier](file:///home/nidszxh/Projects/DysarthriaNSR/src/models/model.py#439-480) in `NeuroSymbolicASR.forward()`
- [ ] Replace [_compute_adaptive_beta()](file:///home/nidszxh/Projects/DysarthriaNSR/src/models/model.py#321-345) binary scaling with new adapter (β remains for constraint blend)
- [ ] Add `severity_to_float()` utility mapping TORGO speaker IDs to continuous [0, 5] intelligibility scores

## Phase 4 — Ordinal Contrastive Severity Loss (Proposal P1)
- [ ] Add `OrdinalContrastiveLoss` class to new file `src/models/losses.py`
  - [ ] Pairwise cosine similarity matrix
  - [ ] Ordinal margin: `margin ∝ |sev_i - sev_j|`
  - [ ] Contrastive target: same severity → pull, different → push
- [ ] Add `lambda_ordinal` to [TrainingConfig](file:///home/nidszxh/Projects/DysarthriaNSR/src/utils/config.py#63-103)
- [ ] Add contrastive loss term to `DysarthriaASRLightning.compute_loss()`
- [ ] Log `loss_ordinal` to MLflow

## Phase 5 — Blank-Prior KL Regularisation (CTC insertion fix)
- [ ] Add `BlankPriorKLLoss` to `src/models/losses.py`
  - [ ] Target mean blank probability = 0.3 (tunable)
  - [ ] KL(P_blank_mean || target_distribution)
- [ ] Add `lambda_blank_kl` and `blank_target_prob` to [TrainingConfig](file:///home/nidszxh/Projects/DysarthriaNSR/src/utils/config.py#63-103)
- [ ] Integrate into [compute_loss()](file:///home/nidszxh/Projects/DysarthriaNSR/train.py#201-254) in [train.py](file:///home/nidszxh/Projects/DysarthriaNSR/train.py)
- [ ] Remove frame-CE loss applied to constrained log-probs; apply to `logits_neural` instead with proper temporal sampling
- [ ] Monitor `train/blank_prob_mean` metric to track insertion bias

## Phase 6 — ROADMAP Phase 6: Explainability Module
- [ ] Create `src/explainability/` package with [__init__.py](file:///home/nidszxh/Projects/DysarthriaNSR/src/__init__.py)
- [ ] Implement `PhonemeAttributor` class (`src/explainability/attribution.py`)
  - [ ] `alignment_attribution()`: Levenshtein alignment → (operation, predicted, expected, position, articulatory_features)
  - [ ] `attention_attribution()`: Extract HuBERT attention weights (`output_attentions=True`), aggregate to phoneme boundaries
  - [ ] Enable `output_attentions=True` in `NeuroSymbolicASR.forward()` and route weights
- [ ] Implement `SymbolicRuleTracker` class (`src/explainability/rule_tracker.py`)
  - [ ] `log_rule_activation()` with rule_id, context, confidence, timestamp
  - [ ] `generate_explanation()` with top-K rule frequency report
  - [ ] Integrate tracker into `SymbolicConstraintLayer.forward()` – log which substitution rules fired above `min_rule_confidence`
- [ ] Implement `ArticulatoryConfusionAnalyzer` (`src/explainability/articulator_analysis.py`)
  - [ ] Per-feature (manner/place/voice) confusion matrices from error alignments
  - [ ] `plot_feature_confusion()` visualization
- [ ] Implement `ExplainableOutputFormatter` (`src/explainability/output_format.py`)
  - [ ] Generate structured JSON explanation per utterance (see ROADMAP §6.4 schema)
  - [ ] Include: error type, position, articulatory features, probable cause, symbolic rule activated, neural confidence
- [ ] Wire explainability into [evaluate_model()](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py#676-931) in [evaluate.py](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py) (opt-in via `generate_explanations=True`)

## Phase 7 — ROADMAP Phase 7: Training Pipeline Enhancements
- [ ] Implement LOSO (`create_loso_splits()` in [train.py](file:///home/nidszxh/Projects/DysarthriaNSR/train.py))
  - [ ] Speaker-disjoint folds (N folds = N speakers)
  - [ ] Macro-average metrics across folds with per-fold bootstrap CI
- [ ] Add `--ablation [neural_only|symbolic_only|no_art_heads|full]` CLI flag to [train.py](file:///home/nidszxh/Projects/DysarthriaNSR/train.py)
- [ ] Add `layer_dropout` (stochastic depth) to [NeuroSymbolicASR](file:///home/nidszxh/Projects/DysarthriaNSR/src/models/model.py#482-704)
- [ ] Fix [configure_optimizers](file:///home/nidszxh/Projects/DysarthriaNSR/train.py#650-682) to use differential learning rates: encoder 1×, head 10×, symbolic 5×

## Phase 8 — ROADMAP Phase 8: Evaluation Metrics Enhancements
- [ ] Fix metric aggregation: macro-average PER over speakers, not samples
- [ ] Add severity-stratified WER to [evaluate_model()](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py#676-931)
  - [ ] Buckets: mild (0–2), moderate (2–4), severe (4–5)
- [ ] Add intelligibility correlation (Pearson r between predicted PER and severity score)
- [ ] Add statistical significance tests to [evaluate.py](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py)
  - [ ] Paired Wilcoxon test for model comparison (neural-only vs. neuro-symbolic)
  - [ ] Welch t-test for dysarthric vs. control PER gap
  - [ ] Holm–Bonferroni correction for multiple comparisons
- [ ] Add rule precision metric: correct rule applications / total rule applications
- [ ] Fix speaker key: `batch['speakers']` vs `batch['speaker']` mismatch (line 769, evaluate.py)

## Phase 9 — ROADMAP Phase 9: Uncertainty & Deployment
- [ ] Implement `UncertaintyAwareDecoder` (`src/models/uncertainty.py`)
  - [ ] Monte Carlo dropout forward passes (N=20)
  - [ ] Predictive entropy per frame → utterance-level uncertainty score
  - [ ] Report uncertainty alongside PER in evaluation results JSON
- [ ] Implement conformal phoneme sets (optional, post-MC dropout)

## Phase 10 — Verification & Ablation Studies
- [ ] Write smoke-test script `scripts/smoke_test.py` (1 epoch, 10 batches, checks all loss terms)
- [ ] Run 3 ablations and log results:
  - [ ] `neural_only` (β=1.0, no constraint matrix, no articulatory heads)
  - [ ] `symbolic_only` (β=0.0)
  - [ ] `full_neuro_symbolic` (current + all Phase 1–5 fixes)
- [ ] Validate learned constraint matrix C against empirical confusion matrix from test set
- [ ] Generate walkthrough.md with all results, charts, and ablation table
