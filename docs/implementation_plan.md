# Implementation Plan — DysarthriaNSR Research-Grade Upgrade

## Goal
Implement all audit findings and ROADMAP phases 6–9 in a structured, test-first manner. Changes are organized from lowest risk (bug fixes) to highest research novelty (new architectural modules). Every new module is designed to be additive and flag-controlled so existing runs remain reproducible.

---

## Phase 0 — Critical Bug Fixes

### Config & Train/Eval Consistency

#### [MODIFY] [train.py](file:///home/nidszxh/Projects/DysarthriaNSR/train.py)
- Remove duplicate `input_lengths = torch.clamp(...)` block (lines 364–367)
- In [training_step](file:///home/nidszxh/Projects/DysarthriaNSR/train.py#330-424), confirm `severity = batch['status'].float() * 5.0` is correct (✅ it is)

#### [MODIFY] [evaluate.py](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py)
- **Bug fix (S3):** In [evaluate_model()](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py#676-931) line 736, change `severity = batch['status'].to(device)` → `severity = batch['status'].float().to(device) * 5.0` to match train-time scaling
- **Bug fix (S4):** Change `avg_per = np.mean(per_scores)` (sample-mean) → speaker macro-average across `all_speakers`
- **Bug fix:** Fix key name: `batch['speakers']` (line 769) should be `batch['speakers']` — verify consistency with collator output key (`'speakers'` ✅ matches dataloader.py)

#### [MODIFY] [src/models/model.py](file:///home/nidszxh/Projects/DysarthriaNSR/src/models/model.py)
- In `NeuroSymbolicASR.forward()`, add `output_attentions=True` to `self.hubert(...)` call; route `hubert_outputs.attentions` into the output dict for Phase 6 explainability

---

## Phase 1 — Config Upgrades

#### [MODIFY] [src/utils/config.py](file:///home/nidszxh/Projects/DysarthriaNSR/src/utils/config.py)
- Add `TORGO_SEVERITY_MAP` dict: maps each speaker ID to a continuous severity float [0.0–5.0] from Rudzicz 2012 intelligibility ratings
- Add to [ModelConfig](file:///home/nidszxh/Projects/DysarthriaNSR/src/utils/config.py#45-62): `use_severity_adapter: bool = True`, `use_learnable_constraint: bool = True`
- Add to [TrainingConfig](file:///home/nidszxh/Projects/DysarthriaNSR/src/utils/config.py#63-103): `lambda_kl: float = 0.05`, `lambda_ordinal: float = 0.1`, `lambda_blank_kl: float = 0.05`, `blank_target_prob: float = 0.3`, `use_loso: bool = False`, `ablation_mode: str = "full"` (`"neural_only"`, `"symbolic_only"`, `"full"`)
- Add to [SymbolicConfig](file:///home/nidszxh/Projects/DysarthriaNSR/src/utils/config.py#131-158): `learnable_constraint: bool = True`

---

## Phase 2 — New Loss Functions

#### [NEW] [src/models/losses.py](file:///home/nidszxh/Projects/DysarthriaNSR/src/models/losses.py)
Three new loss classes, each independently controlled by a config flag:

1. **`OrdinalContrastiveLoss`** — Pulls embeddings of same-severity utterances together, pushes different-severity apart, with margins proportional to `|sev_i − sev_j|`. Uses mean-pooled HuBERT hidden states from `last_hidden_state`.

2. **`BlankPriorKLLoss`** — Computes KL divergence between the mean blank probability across valid frames and a target beta distribution (default: mean=0.3). Addresses the 56× insertion/deletion ratio.

3. **`SymbolicKLLoss`** — KL divergence between the learned `C` matrix rows and the static symbolic prior. Applied only when `use_learnable_constraint=True`.

---

## Phase 3 — Model Architecture Additions

#### [MODIFY] [src/models/model.py](file:///home/nidszxh/Projects/DysarthriaNSR/src/models/model.py)

**`SeverityAdapter(nn.Module)`** — New class:
- `severity_proj`: `Linear(1, 64) → SiLU → Linear(64, 768)` maps scalar severity → context vector
- `cross_attn`: `MultiheadAttention(768, heads=8)` with `hidden_states` as Q and severity context as K/V
- `layer_norm`: Residual + LayerNorm on adapter output
- Inserted in `NeuroSymbolicASR.forward()` between `hubert_outputs.last_hidden_state` and [PhonemeClassifier](file:///home/nidszxh/Projects/DysarthriaNSR/src/models/model.py#439-480)
- Only active when `ModelConfig.use_severity_adapter=True`

**`LearnableConstraintMatrix(nn.Module)`** — New class:
- `logit_C`: `nn.Parameter` of shape `[num_phonemes, num_phonemes]`, initialized from [log(C_static + ε)](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py#49-67)
- Property `C` returns `F.softmax(logit_C, dim=-1)` (row-stochastic)
- Replaces static `register_buffer('constraint_matrix', ...)` in [SymbolicConstraintLayer](file:///home/nidszxh/Projects/DysarthriaNSR/src/models/model.py#157-437)
- Only learnable when `ModelConfig.use_learnable_constraint=True`; falls back to static buffer otherwise

**`SymbolicConstraintLayer.forward()`** update:
- Returns both `P_neural` probabilities and the constrained output
- Attentions forwarded through if available

---

## Phase 4 — LOSO & Ablation Infrastructure

#### [MODIFY] [train.py](file:///home/nidszxh/Projects/DysarthriaNSR/train.py)

**`create_loso_splits(dataset, fold_speaker)`** — New function:
- Takes a speaker name as the held-out fold; returns train/val DataLoaders excluding that speaker
- `run_loso(config, dataset)` — Loops over all unique speakers, runs one training+eval per fold, aggregates macro-average PER with 95% CI

**`DysarthriaASRLightning.compute_loss()` additions:**
- Add `loss_ordinal` term (from `OrdinalContrastiveLoss`, using `hidden_states` output)
- Add `loss_blank_kl` term (from `BlankPriorKLLoss`)
- Add `loss_symbolic_kl` term (from `SymbolicKLLoss` when learnable constraint enabled)
- Log `train/blank_prob_mean` for insertion monitoring

**`--ablation` CLI flag:**
- `neural_only`: Sets `beta=1.0` (non-learnable), disables constraint matrix, disables articulatory heads
- `symbolic_only`: Sets `beta=0.0`
- `no_art_heads`: Disables manner/place/voice heads only
- `full`: Default, everything enabled

**Differential learning rates in [configure_optimizers()](file:///home/nidszxh/Projects/DysarthriaNSR/train.py#650-682):**
- HuBERT encoder: `lr * 0.1`
- PhonemeClassifier + SeverityAdapter: `lr * 1.0`
- SymbolicConstraintLayer (learnable C): `lr * 0.5`

---

## Phase 5 — Explainability Module (ROADMAP §6)

#### [NEW] [src/explainability/\_\_init\_\_.py](file:///home/nidszxh/Projects/DysarthriaNSR/src/explainability/__init__.py)
Package init, exports main classes.

#### [NEW] [src/explainability/attribution.py](file:///home/nidszxh/Projects/DysarthriaNSR/src/explainability/attribution.py)
**`PhonemeAttributor`**:
- `alignment_attribution(pred, ref)` → per-error dict: `{type, position, predicted, expected, articulatory_features, probable_cause}`
- `attention_attribution(attn_weights, phoneme_boundaries)` → frame-to-phoneme attention aggregation from HuBERT attention maps (12 layers, 12 heads → mean-head, last-4-layers average)

#### [NEW] [src/explainability/rule_tracker.py](file:///home/nidszxh/Projects/DysarthriaNSR/src/explainability/rule_tracker.py)
**`SymbolicRuleTracker`** (port from ROADMAP §6.2):
- `log_rule_activation(rule_id, input_context, output_correction, confidence)` 
- `generate_explanation()` → `{total_rules_fired, high_confidence_corrections, rule_frequency}`
- Integration: called from `SymbolicConstraintLayer._track_activations()` with rule IDs keyed to substitution rule pairs

#### [NEW] [src/explainability/articulator_analysis.py](file:///home/nidszxh/Projects/DysarthriaNSR/src/explainability/articulator_analysis.py)
**`ArticulatoryConfusionAnalyzer`**:
- Builds per-feature (manner/place/voice) confusion matrices from [analyze_phoneme_errors()](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py#439-493) output
- `plot_feature_confusion()` → 3-panel heatmap PNG

#### [NEW] [src/explainability/output_format.py](file:///home/nidszxh/Projects/DysarthriaNSR/src/explainability/output_format.py)
**`ExplainableOutputFormatter`**:
- `format_utterance(utt_id, ground_truth, prediction, errors, rule_stats)` → structured dict matching ROADMAP §6.4 JSON schema
- `save_explanations(results_dir)` → writes `explanations.json`

#### [MODIFY] [evaluate.py](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py)
- Add `generate_explanations=False` param to [evaluate_model()](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py#676-931)
- When `True`, instantiate `PhonemeAttributor` + `ExplainableOutputFormatter`; save `{results_dir}/explanations.json`
- Add articulatory confusion heatmaps alongside existing phoneme confusion matrix

---

## Phase 6 — Advanced Evaluation (ROADMAP §8)

#### [MODIFY] [evaluate.py](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py)
- **Macro-average PER:** Group `per_scores` by speaker, compute per-speaker mean, then macro-average across speakers
- **Severity-stratified WER:** Bucket speakers by continuous severity; report mean PER per bucket with CI
- **Intelligibility correlation:** `scipy.stats.pearsonr(per_by_speaker, severity_by_speaker)`
- **Statistical tests:** 
  - Paired Wilcoxon test comparing neuro-symbolic vs. neural-only PER across utterances
  - Welch t-test for dysarthric vs. control PER
  - Holm–Bonferroni correction when running multiple comparisons
- **Rule precision**: `correct_rule_uses / total_rule_uses` from `SymbolicRuleTracker` output
- Save all new metrics to `evaluation_results.json`

---

## Phase 7 — Uncertainty & Deployment (ROADMAP §9)

#### [NEW] [src/models/uncertainty.py](file:///home/nidszxh/Projects/DysarthriaNSR/src/models/uncertainty.py)
**`UncertaintyAwareDecoder`**:
- `predict_with_uncertainty(model, input_values, ..., n_samples=20)` → enables dropout during inference, stacks N forward passes
- Computes predictive entropy per-frame: `H = -Σ p̄ log p̄`
- Returns `{mean_log_probs, epistemic_uncertainty_per_frame, utterance_uncertainty}`
- Integrated as optional post-processing in [evaluate_model()](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py#676-931) when `compute_uncertainty=True`

---

## Verification Plan

> [!IMPORTANT]
> No existing unit tests were found in the repository. All verification will use the existing smoke-test pattern in [train.py](file:///home/nidszxh/Projects/DysarthriaNSR/train.py) and a new `scripts/smoke_test.py`.

### Automated Tests

**1. Smoke test (all losses, single batch):**
```bash
cd /home/nidszxh/Projects/DysarthriaNSR
python train.py --run-name smoke_all_phases --max_epochs 1 --limit_train_batches 5
```
Expected: No exceptions; `train/loss`, `train/loss_ctc`, `train/loss_ce`, `train/loss_ordinal`, `train/loss_blank_kl`, `train/blank_prob_mean` all logged.

**2. Severity scaling correctness:**
```bash
python -c "
from src.utils.config import TORGO_SEVERITY_MAP
assert all(0.0 <= v <= 5.0 for v in TORGO_SEVERITY_MAP.values()), 'Invalid severity range'
print('PASS: severity map values in [0, 5]')
"
```

**3. Learnable constraint matrix gradient check:**
```bash
python -c "
import torch
from src.models.model import SymbolicConstraintLayer
from src.utils.config import get_default_config
cfg = get_default_config()
# Instantiate with small vocab for speed
phn_to_id = {'<BLANK>':0,'<PAD>':1,'<UNK>':2,'P':3,'B':4}
id_to_phn = {v:k for k,v in phn_to_id.items()}
layer = SymbolicConstraintLayer(5, phn_to_id, id_to_phn, cfg.symbolic, learnable=True)
logits = torch.randn(2, 10, 5, requires_grad=False)
out = layer(logits)
loss = out['log_probs'].mean()
loss.backward()
assert layer.logit_C.grad is not None, 'Learnable C has no gradient'
print('PASS: learnable constraint matrix gradients flow')
"
```

**4. Severity scaling bug fix verification:**
```bash
python -c "
import torch
from evaluate import evaluate_model
# Verify that severity in evaluate_model is scaled by 5.0
import inspect, ast
src = inspect.getsource(evaluate_model)
assert '* 5.0' in src or '5.0' in src, 'FAIL: severity not scaled in evaluate_model'
print('PASS: severity scaling present in evaluate_model')
"
```

**5. Blank probability monitoring (insertion bias check):**
```bash
python -c "
import torch, torch.nn.functional as F
from src.models.losses import BlankPriorKLLoss
loss_fn = BlankPriorKLLoss(blank_id=0, target_prob=0.3)
logits = torch.zeros(2, 50, 5)  # All-zeros → uniform → blank prob = 0.2
logits[:,:,0] = 1.0             # Boost blank → prob ~0.44
log_probs = F.log_softmax(logits, dim=-1)
loss = loss_fn(log_probs, attention_mask=torch.ones(2, 50, dtype=torch.long))
assert loss.item() >= 0.0, 'KL loss must be non-negative'
print(f'PASS: BlankPriorKLLoss = {loss.item():.4f}')
"
```

**6. Explainability output format test:**
```bash
python -c "
from src.explainability.output_format import ExplainableOutputFormatter
fmt = ExplainableOutputFormatter()
result = fmt.format_utterance('test_001', 'hello world', 'hello', [], {})
assert 'utterance_id' in result
assert 'phoneme_analysis' in result
print('PASS: explainability formatter output structure valid')
"
```

**7. `OrdinalContrastiveLoss` sanity check:**
```bash
python -c "
import torch
from src.models.losses import OrdinalContrastiveLoss
loss_fn = OrdinalContrastiveLoss(margin_per_level=0.3)
embeddings = torch.randn(4, 10, 768)  # [B, T, 768]
severity = torch.tensor([0.0, 0.0, 5.0, 5.0])
loss = loss_fn(embeddings, severity)
assert loss.item() >= 0.0, 'Contrastive loss must be non-negative'
print(f'PASS: OrdinalContrastiveLoss = {loss.item():.4f}')
"
```

### Manual Verification
- After full training run, open `results/{run_name}/explanations.json` and verify the structure matches ROADMAP §6.4 schema (utterance_id, ground_truth, prediction, wer, phoneme_analysis, symbolic_rules_summary).
- Confirm `train/blank_prob_mean` in MLflow is trending toward 0.3 over epochs (not diverging to 0.0 or 1.0).
- Inspect the logged confusion matrix heatmap for the learned `C` matrix — diagonal should be dominant but with off-diagonal clusters corresponding to known dysarthric confusions (devoicing, liquid-gliding).
