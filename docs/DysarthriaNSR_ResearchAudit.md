# DysarthriaNSR — Senior Research Scientist Audit Report
**Reviewer Perspective:** Senior Research Scientist, Speech Processing & Deep Learning  
**Target Venues:** INTERSPEECH, ICASSP, IEEE/ACM TASLP  
**Date:** February 2026

---

## Part I — System Summary

### Design Philosophy
DysarthriaNSR is built on a **neuro-symbolic hypothesis**: that dysarthric misarticulations follow clinically documented articulatory confusion patterns (devoicing, fronting, liquid-gliding), and that injecting these as explicit symbolic priors into a neural ASR pipeline will improve robustness while preserving interpretability.

### Data Flow (End-to-End)
```
Raw WAV (16 kHz)
  │  Peak-normalize → truncate to 6 s
  ▼
HuBERT Feature Extractor (frozen CNN, stride=320)
  │  → [B, T≈300, 1024] raw features
  ▼
HuBERT Transformer Encoder (12 layers; layers 0–9 frozen during warmup)
  │  → [B, T, 768] contextualised representations
  ▼
PhonemeClassifier (768 → LayerNorm → GELU → Dropout(0.1) → 512 → 44)
  │  → logits_neural  [B, T, 44]
  ├─ Articulatory heads (768 → 512 → manner/place/voice)
  ▼
SymbolicConstraintLayer
  │  C = build_constraint_matrix()  [44×44, static, non-learned]
  │  β = learnable scalar (init 0.05, clamped [0, 0.8])
  │  β_adaptive = β + 0.1*(severity/5.0)          ← binary: 0 or 5
  │  P_final = β·(P_n @ C) + (1−β)·P_n
  ▼
log_probs_constrained  [B, T, 44]
  ▼
Loss = 0.8·CTC + 0.2·NLL(CE) + 0.1·(manner + place + voice CE)/3
  ▼
CTC Beam Search (width=10) → phoneme sequence → CMU lexicon → WER
```

### Baseline Results (January 2026)
| Metric | Value |
|---|---|
| Test PER | 0.567 ± 0.365 |
| Dysarthric PER | 0.541 |
| Control PER | 0.575 |
| Insertions / Deletions | 21,290 / 376 (56× ratio) |
| Learned β | ~0.50 |
| Trainable params | 416K / 94.8M |

---

## Part II — Gap Analysis

### Pillar 1 — Generalization

#### G1: Binary Severity is Clinically Uninformative
**Finding:** `speaker_severity = status * 5.0` maps every speaker to exactly one of two values (0.0 for control, 5.0 for dysarthric). The TORGO database actually spans mild to severe dysarthria with documented intelligibility scores.

**Impact:** The adaptive-β mechanism becomes a trivially learned threshold between two groups—defeats the purpose of "adaptive" weighting. The model cannot distinguish a mild F03 from a severe M01.

**SoA Gap:** The TORGO literature (Rudzicz et al., 2012; Liu et al.,  INTERSPEECH 2021) and UA-Speech benchmarks use 4-point severity scales with continuous intelligibility scores validated by trained listeners.

#### G2: Leave-One-Speaker-Out (LOSO) Not Executed
**Finding:** The ROADMAP mentions LOSO as the "cross-validation strategy" but the actual [create_dataloaders()](file:///home/nidszxh/Projects/DysarthriaNSR/train.py#684-801) does a *single* random speaker shuffle (seed=42), producing exactly one train/val/test partition. On a 15-speaker dataset, reported metrics are from 3 test speakers and carry massive variance (CI ± 0.365 spans the entire range [0, 1]).

**Impact:** A single partition cannot support claims of statistical significance or generalization. Conference reviewers will reject any PER result without LOSO or k-fold speaker-disjoint CV under appropriate multiple-comparison corrections.

#### G3: No Cross-Corpus Validation
**Finding:** The system is trained and evaluated exclusively on TORGO (English, head-mic + array, Canadian speakers, neurological etiology). There is no evaluation on UA-Speech, CUDYS, or Nemours databases.

**Impact:** Claims of general dysarthric speech modeling cannot be sustained.

---

### Pillar 2 — Robustness

#### R1: Absence of Acoustic Augmentation
**Finding:** The data pipeline applies only peak normalization. No speed perturbation, SpecAugment, room impulse response (RIR) convolution, background noise addition, or microphone response simulation is applied.

**Impact:** TORGO includes recordings from array and head microphones under controlled lab conditions. HuBERT features are not agnostic to SNR or channel characteristics despite their SSL pretraining, as shown in Robust HuBERT (Chen et al., ICASSP 2023). The model will degrade severely under real clinical conditions (quiet home recordings, tablet microphones).

**SoA Gap:** For dysarthric ASR, perturbation-based augmentation for rare speaker populations is considered essential (Xiong et al., TASLP 2020; Tu et al., INTERSPEECH 2022).

#### R2: Static Constraint Matrix vs. Dynamic Neural-Symbolic Fusion
**Finding:** [_build_constraint_matrix()](file:///home/nidszxh/Projects/DysarthriaNSR/src/models/model.py#203-260) constructs a fixed 44×44 matrix at model initialization from hard-coded substitution probabilities and binary articulatory features. The matrix is never updated during training.

**Impact:** 
- The articulatory similarity `C[i,j] = exp(−3 * sqrt(w_m*(m1≠m2) + w_p*(p1≠p2) + w_v*(v1≠v2)))` uses *categorical* binary distances. Manner=stop vs. manner=fricative receive the same penalty regardless of acoustic proximity.
- Hard-coded probabilities (e.g., B→P = 0.85) are not validated against the *empirical* confusion matrix of the trained model, creating a possibility of circular reinforcement of incorrect biases.
- The constraint matrix cannot adapt across speakers or severities.

**SoA Gap:** Differentiable neuro-symbolic integration (e.g., ATM-Net, IEEE TASLP 2023; LM-constrained CTC, Zeyer et al.) learns the symbolic weight jointly end-to-end.

#### R3: CTC Insertion Pathology is Unresolved
**Finding:** 21,290 insertions vs. 376 deletions represents a 56× ratio, a fundamental failure of the CTC training regime. The frame-level CE loss (λ=0.2) applied to constrained log-probabilities on temporally sparse phoneme labels—most frames map to `-100` padding—provides a perverse gradient: non-blank emissions on padding frames are not penalized by NLLLoss (which uses `ignore_index=-100`), but blank emissions are not incentivized either.

**Impact:** Any downstream WER claim is invalid because the decoder receives a log-probability distribution biased toward non-blank tokens across all frames. This is not an inference artifact—it is a training signal pathology.

**Root cause (missed in repo):** Frame-level CE loss with `-100` ignore assumes that labeled frames are *uniformly distributed* over time, but in CTC-aligned sequences the ground-truth phoneme boundaries are unknown. Aligning labels by simple truncation/padding creates a misaligned supervision signal.

---

### Pillar 3 — Interpretability

#### I1: β Convergence to ~0.5 is Not Interpretable
**Finding:** β converges to approximately 0.50, meaning the model settles at equal weighting of neural and symbolic pathways. Since β is a scalar (not per-phoneme, per-speaker, or per-layer), this value is uninterpretable: it reflects a training equilibrium, not a clinical insight.

**Impact:** There is no mechanism to determine *which* symbolic rules benefited recognition and *which* were harmful. The [rule_activations](file:///home/nidszxh/Projects/DysarthriaNSR/src/models/model.py#701-704) tracker only logs when the argmax changed—it does not expose the magnitude, direction, or phonological category of constraint influence.

#### I2: Attention Weights Not Exposed for Attribution
**Finding:** The model calls `output_hidden_states=True` but `output_attentions=False`. Transformer attention maps from HuBERT's 12 layers are discarded.

**Impact:** No GradCAM, Integrated Gradients, or attention-rollout analysis is feasible. The ROADMAP mentions attention visualization as "not implemented" but claims "clinical explainability."

#### I3: Symbolic Rules are Literature-Derived, Not Data-Validated
**Finding:** The 27 substitution rules in `SymbolicConfig.substitution_rules` cite clinical phonology literature but are *not validated against the empirical confusion matrix* of the trained model. The audit cannot confirm whether, for example, the hard-coded `B→P = 0.85` probability matches the observed B→P confusion rate in TORGO.

---

### Pillar 4 — Scientific Rigor

#### S1: Critical Data Leakage Vector — HuBERT Pretraining Overlap
**Finding:** The system uses `facebook/hubert-base-ls960`, pretrained on **LibriSpeech 960h**. While TORGO speakers are not in LibriSpeech, the g2p_en phoneme extractor used to generate labels for both datasets uses the same CMU phoneme dictionary and produces identical ARPABET sequences for the same English words. Because the frame-level CE loss projects HuBERT representations onto phoneme labels using the *same* phoneme set as the HuBERT pretraining objectives, this creates a **domain alignment artifact**: the PhonemeClassifier head learns to reproduce HuBERT's internal cluster-to-phoneme mapping for *typical* speech patterns, which may not reflect dysarthric phoneme confusions.

**Severity:** Moderate. Not a strict data leakage but a confounding pre-alignment that inflates performance on control speakers and explains the inverted dysarthric/control PER (control PER > dysarthric PER).

#### S2: Loss Imbalance — Total Loss > 1.0 Semantically
**Finding:** `total_loss = 0.8·CTC + 0.2·CE + 0.1·(art_mean)`. Since both CTC and NLL/CE are applied to the *same* `log_probs_constrained` (not `logits_neural`), the gradients from CE flow *through* the symbolic constraint layer (back-propagating into β), while CTC gradients also back-propagate through β. This creates a **conflated gradient signal**: β is simultaneously optimized for CTC alignment (sequence-level) and frame-level CE (frame-level), two objectives with contradictory assumptions about the temporal distribution of phonemes.

#### S3: Severity Scaling Creates Invalid Comparison
**Finding:** In [evaluate.py](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py), `all_status` = raw 0/1 values. In [train.py](file:///home/nidszxh/Projects/DysarthriaNSR/train.py), `severity = batch['status'].float() * 5.0`. During evaluation, `severity` is passed as `batch['status']` (0 or 1) directly to the model, which internally expects 0–5 scale values. If [evaluate_model()](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py#676-931) is called directly (i.e., [_compute_adaptive_beta](file:///home/nidszxh/Projects/DysarthriaNSR/src/models/model.py#321-345) receives 0 or 1 instead of 0 or 5), `β_adaptive` will be `β + 0.1*(status/5.0)` = `β + 0.02` instead of `β + 0.1`. This is a **train-eval scaling mismatch** that invalidates the severity-adaptive behavior at inference time.

> [!CAUTION]
> The test-time severity pipeline passes raw `batch['status']` (0 or 1) to the model in [evaluate_model()](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py#676-931) (line 736) rather than scaling by 5.0. This means the severity-adaptive β at test time is effectively the same as a non-adaptive β, silently invalidating the core adaptive-constraint claim.

#### S4: Metric Aggregation is Speaker-Unaware
**Finding:** `avg_per = np.mean(per_scores)` computes a simple mean over *samples*, not over *speakers*. Since TORGO speakers have vastly different sample counts (some have 400 utterances, others 200), high-volume speakers dominate the reported PER. A speaker with 400 utterances and PER=0.20 swamps three speakers contributing 100 utterances each with PER=0.70.

**SoA requirement:** PER should be reported as macro-average over speakers (mean of per-speaker PER means) with per-speaker bootstrap CIs.

#### S5: No Ablation Results
**Finding:** The ROADMAP lists 4 ablation variants (neural-only β=1.0, symbolic-only β=0.0, no auxiliary heads, β sweeps) but none are executed. The paper cannot be submitted to any conference without at minimum a neural-only vs. neuro-symbolic comparison to validate the contribution of the symbolic constraint layer.

---

## Part III — Research-Grade Enhancement Proposals

*Ordered by scientific impact and feasibility. Implementation complexity: L=Low (days), M=Medium (1–2 weeks), H=High (3–6 weeks).*

---

### Proposal 1 — Ordinal Contrastive Loss for Severity-Graded Representation *(High Impact)*
**Complexity:** M | **Venue fit:** INTERSPEECH / TASLP

**Motivation:** The system treats dysarthria as a binary label. In reality, it is an ordinal spectrum. A supervisor-level separation of representations by severity enables β to be truly data-adaptive instead of hard-coded binary.

**Technical Design:**
```python
class OrdinalContrastiveLoss(nn.Module):
    """
    Pulls representations of similar severity together,
    pushes dissimilar severity apart, using ordinal margin.
    """
    def __init__(self, margin_per_level: float = 0.3):
        super().__init__()
        self.margin = margin_per_level
    
    def forward(self, embeddings: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
        # Pool frame embeddings → utterance embedding: [B, 768]
        z = embeddings.mean(dim=1)  
        z = F.normalize(z, dim=-1)
        
        # Pairwise cosine similarity
        sim = torch.matmul(z, z.T)  # [B, B]
        
        # Ordinal severity margin: margin ∝ |sev_i - sev_j|
        sev_diff = (severity.unsqueeze(0) - severity.unsqueeze(1)).abs()
        margin = self.margin * sev_diff  # [B, B]
        
        # Contrastive: similar severity → sim > 0; different → sim < -margin
        target = (sev_diff == 0).float() * 2 - 1  # +1 same, -1 different
        loss = F.relu(margin - target * sim).mean()
        return loss
```

**Required:** Continuous severity scores from TORGO (use Rudzicz 2012 intelligibility ratings: 2.0–100%; normalized to [0, 5]).
**Expected Impact:** 3–5% relative PER reduction for severe dysarthric speakers; enables clinically meaningful β adaptation.

---

### Proposal 2 — Learnable Phonological Constraint Matrix via Differentiable Parsing *(High Impact, Novel)*
**Complexity:** H | **Venue fit:** TASLP / ACL (if framed as neuro-symbolic NLP)

**Motivation:** The static `C` matrix encodes *literature-assumed* substitution probabilities that may not match empirical TORGO confusions. Making C *differentiable and learnable* enables end-to-end discovery of dysarthric phonological rules from data.

**Technical Design:**
```python
class LearnableConstraintMatrix(nn.Module):
    """
    Constraint matrix initialized from articulatory phonology
    but updated end-to-end during training.
    """
    def __init__(self, num_phonemes: int, init_matrix: torch.Tensor):
        super().__init__()
        # Initialize from symbolic prior, then allow gradient update
        self.logit_C = nn.Parameter(torch.log(init_matrix.clamp(1e-8)))
    
    @property
    def C(self) -> torch.Tensor:
        # Row-normalized softmax ensures C remains a valid stochastic matrix
        return F.softmax(self.logit_C, dim=-1)
    
    def forward(self, P_neural: torch.Tensor) -> torch.Tensor:
        return torch.matmul(P_neural, self.C)
```

**Regularization:** Add a KL divergence term `KL(C_learned || C_symbolic)` to prevent C from diverging arbitrarily from the phonological prior:
```python
loss_symbolic_kl = F.kl_div(
    self.logit_C.log_softmax(-1),
    C_symbolic.log(),
    reduction='batchmean', log_target=True
)
total_loss += lambda_kl * loss_symbolic_kl
```

**Expected Impact:** The discovered `C` becomes a *publishable result* itself (dysarthric confusion topology from data), in addition to potential 5–8% relative PER reduction.

---

### Proposal 3 — Cross-Attention Severity Adapter *(Medium Impact, Engineering Sound)*
**Complexity:** M | **Venue fit:** INTERSPEECH

**Motivation:** The current β-scaling mechanism is a scalar multiplier on severity, with no learned mapping between severity and acoustic feature space. A cross-attention adapter enables the model to selectively amplify or suppress dysarthria-relevant features in a severity-conditioned manner.

**Technical Design:**
```python
class SeverityAdapter(nn.Module):
    """
    Injects speaker-level severity via cross-attention into HuBERT hidden states.
    Conditioned on a continuous severity embedding, not a binary flag.
    """
    def __init__(self, hidden_dim: int = 768, severity_dim: int = 64):
        super().__init__()
        # Project severity scalar → severity context vector
        self.severity_proj = nn.Sequential(
            nn.Linear(1, severity_dim),
            nn.SiLU(),
            nn.Linear(severity_dim, hidden_dim)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,       # [B, T, 768]
        severity: torch.Tensor             # [B] continuous [0, 5]
    ) -> torch.Tensor:
        # Severity context: [B, 1, 768]
        sev_ctx = self.severity_proj(severity.view(-1, 1, 1))
        
        # Cross-attend: query=hidden_states, key/value=severity_context
        attn_out, _ = self.cross_attn(hidden_states, sev_ctx, sev_ctx)
        return self.layer_norm(hidden_states + attn_out)
```

**Integration:** Insert `SeverityAdapter` between the HuBERT encoder output and the PhonemeClassifier. This replaces the post-hoc β scaling with a *representational* severity adaptation.

---

### Proposal 4 — Uncertainty-Aware CTC with Conformal Prediction *(Medium Impact, Clinical Value)*
**Complexity:** M | **Venue fit:** INTERSPEECH / ML4H workshop

**Motivation:** Clinical deployment requires not just accurate predictions but *calibrated uncertainty*. When a model is unsure, it should say so—critical for speech-language pathologist trust.

**Technical Design:**

**Step 1 — Monte Carlo Dropout for Epistemic Uncertainty:**
```python
class UncertaintyAwareDecoder:
    def __init__(self, model, n_samples: int = 20):
        self.model = model
        self.n_samples = n_samples
    
    def predict_with_uncertainty(
        self, 
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        severity: torch.Tensor
    ) -> Dict:
        # Enable dropout at inference
        self.model.train()  # activates dropout
        
        log_probs_samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                outputs = self.model(input_values, attention_mask, severity)
                log_probs_samples.append(outputs['log_probs_constrained'])
        
        self.model.eval()
        
        lp_stack = torch.stack(log_probs_samples, dim=0)  # [N, B, T, V]
        mean_lp = lp_stack.mean(0)                         # [B, T, V]
        
        # Predictive entropy as uncertainty
        probs = lp_stack.exp()
        mean_probs = probs.mean(0)                          # [B, T, V]
        entropy = -(mean_probs * (mean_probs + 1e-12).log()).sum(-1)  # [B, T]
        
        return {
            'mean_log_probs': mean_lp,
            'epistemic_uncertainty': entropy,
            'sample_variance': lp_stack.var(0)
        }
```

**Step 2 — Conformal Phoneme Sets:** Use conformal prediction to output phoneme *sets* with guaranteed coverage at test time. This generalizes naturally to producing "the model predicts /b/ or /p/ with 95% confidence"—a directly clinical output.

---


## Part IV — Prioritized Implementation Roadmap

| Priority | Proposal | Impact | Fix Type | Estimated Effort |
|---|---|---|---|---|
| 🔴 **Critical** | Fix severity scaling bug in [evaluate_model()](file:///home/nidszxh/Projects/DysarthriaNSR/evaluate.py#676-931) (S3) | Correctness | Bug fix | 1 hour |
| 🔴 **Critical** | Fix frame-CE / CTC label alignment pathology (R3, S2) | Correctness | Refactor | 2 days |
| 🔴 **Critical** | Implement LOSO cross-validation (G2) | Validity | Infrastructure | 3 days |
| 🟠 **High** | Ordinal contrastive severity loss (P1) | 3–5% PER | New module | 1 week |
| 🟠 **High** | Learnable constraint matrix with KL anchor (P2) | 5–8% PER + novel result | Architecture | 2 weeks |
| 🟡 **Medium** | Cross-attention severity adapter (P3) | Replaces binary β | Architecture | 1 week |
| 🟡 **Medium** | Uncertainty-aware conformal decoder (P4) | Clinical 
---

## Part V — Minimum Requirements for Conference Submission

To reach **INTERSPEECH 2026 submission quality**, the following must be completed before any paper draft:

1. **Fix the severity scaling bug** — the train/eval mismatch invalidates the core adaptive-constraint contribution claim.
2. **Execute LOSO cross-validation** — 15-fold CV with macro-averaged PER and 95% bootstrap CI per fold.
3. **Implement and report at minimum 3 ablations:**
   - Baseline HuBERT (no phoneme head fine-tuning)
   - Neural-only (β=1.0, no constraints)
   - Neuro-symbolic (current system, after bug fixes)
   - + any one novel proposal above
4. **Resolve the CTC insertion pathology** — Report separate I/D/S rates; test blank-prior KL regularization.
5. **Validate substitution rules against empirical confusion matrix** — Are the hard-coded rules (B→P=0.85, etc.) consistent with what the trained model actually confuses?
6. **Macro-average PER over speakers**, not samples.

---

*Report prepared as part of code review for DysarthriaNSR repository. All code citations reference the files at revision reviewed on 2026-02-28.*
