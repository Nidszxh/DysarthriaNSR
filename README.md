# DysarthriaNSR — Neuro-Symbolic ASR for Dysarthric Speech

**Target venue:** SPCOM 2026 | **Status:** LOSO-CV complete (15/15 folds)

> **LOSO macro PER · 0.2848** (95% CI: [0.1921, 0.3801])
> **LOSO weighted PER · 0.2299**
> **Folds complete · 15/15**

---

## What This Is

Dysarthria is a motor speech disorder — caused by conditions such as cerebral palsy and ALS — that causes severely reduced intelligibility. Commercial ASR systems fail for dysarthric speakers because they treat atypical phoneme realizations as noise. DysarthriaNSR addresses this by combining a pretrained HuBERT speech encoder with a learnable symbolic phoneme constraint layer grounded in articulatory phonetics.

Unlike a standard HuBERT fine-tune, DysarthriaNSR adds three key components: (1) a `LearnableConstraintMatrix` — a differentiable 47×47 phoneme confusion matrix initialized from articulatory priors (e.g., devoicing B→P, liquid gliding R→W) and trained end-to-end with a KL anchor to prevent drift; (2) a `SeverityAdapter` that injects a continuous severity score [0, 5] into HuBERT hidden states via cross-attention; and (3) a six-term multi-task loss that includes blank-prior KL regularization to suppress CTC insertion bias. The result is a system that provides clinically interpretable phoneme-level error analysis alongside recognition output.

---

## Key Results

| Model | Macro PER | Weighted PER | Notes |
|---|---|---|---|
| `loso_v1` (full system, LOSO) | **0.2848** (95% CI: [0.1921, 0.3801]) | **0.2299** | Publication result, 15/15 folds |
| `baseline_v6` (full system, single split) | 0.1372 | — | per_constrained; post-fix reference |
| `ablation_neural_only_v7` (single split) | **0.1346** | — | Best single-split PER to date |
| `ablation_no_constraint_matrix_v6` (single split) | 0.1444 | — | SeverityAdapter only, no learnable C |

The symbolic constraint story is nuanced. Within `baseline_v6`, the constrained path improves over the model's internal neural sub-path (per_constrained=0.1372 < per_neural=0.1451, p=0.0), and 9.16% of utterances benefit from constraint application versus only 3.78% that are harmed. However, a fully independent neural-only ablation (`ablation_neural_only_v7`, which bypasses SeverityAdapter and SymbolicConstraintLayer entirely) still achieves the best global single-split PER at 0.1346. This means the symbolic constraint helps within the jointly-trained system but has not yet demonstrated superiority over a pure HuBERT baseline in isolation. LOSO-CV stratified by dysarthric speaker groups — particularly the high-severity folds (M01, M02, M04, M05, F01) — is the decisive pending test for the SPCOM claim.

---

## Quick Start

### Installation
```bash
git clone <repo-url>
cd DysarthriaNSR
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

### Dataset Setup
```bash
# Step 1: Download TORGO audio from HuggingFace (abnerh/TORGO-database)
python src/data/download.py

# Step 2: Generate manifest with G2P + articulatory labels
python src/data/manifest.py
# Output: data/processed/torgo_neuro_symbolic_manifest.csv (16,531 rows)
```

### Run Commands
```bash
# Standard single-split training + evaluation
python run_pipeline.py --run-name baseline_v6

# Full LOSO cross-validation (publication result, ~32h on RTX 4060)
python run_pipeline.py --run-name loso_v1 --loso

# Resume LOSO from last completed fold
python run_pipeline.py --run-name loso_v1 --loso --resume-loso

# Smoke test (unit profile, 7/7 checks)
python scripts/smoke_test.py --profile unit
```

---

## Repository Layout
```
DysarthriaNSR/
├── run_pipeline.py          # Canonical entry point: train + eval orchestrator
├── train.py                 # DysarthriaASRLightning, run_loso()
├── evaluate.py              # evaluate_model(), BeamSearchDecoder, compute_per()
├── requirements.txt         # Pinned dependency stack
├── src/
│   ├── models/
│   │   ├── model.py         # NeuroSymbolicASR and all architectural components
│   │   ├── losses.py        # OrdinalContrastiveLoss, BlankPriorKLLoss, SymbolicKLLoss
│   │   └── uncertainty.py   # UncertaintyAwareDecoder (MC-Dropout)
│   ├── data/
│   │   ├── dataloader.py    # TorgoNeuroSymbolicDataset, NeuroSymbolicCollator
│   │   ├── manifest.py      # TORGO manifest generation (G2P → ARPABET)
│   │   └── download.py      # Audio download from HuggingFace
│   ├── utils/
│   │   ├── config.py        # All hyperparameters — single source of truth
│   │   └── sequence_utils.py
│   └── explainability/      # PhonemeAttributor, SymbolicRuleTracker, formatters
├── scripts/
│   ├── smoke_test.py        # Profiles: unit (7 checks), pipeline (CLI integration)
│   └── generate_figures.py  # Publication-quality figure CLI
├── docs/                    # User-facing documentation (see index below)
├── data/raw/audio/          # Downloaded TORGO .wav files
├── data/processed/          # Manifest CSV + feature_cache/
├── checkpoints/             # Per-run model checkpoints
└── results/                 # Per-run evaluation artifacts + figures
```

---

## Documentation Index

| File | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Complete model architecture reference: components, ablation modes, freeze schedule, multi-task loss |
| [docs/experiments.md](docs/experiments.md) | Reproducible record of all experiments, LOSO results, ablation analysis, known limitations |
| [docs/data.md](docs/data.md) | TORGO corpus, download/manifest pipeline, manifest schema, vocabulary, collator internals |
| [docs/training.md](docs/training.md) | Environment setup, configuration system, CLI reference, training dynamics, monitoring, troubleshooting |
| [docs/evaluation.md](docs/evaluation.md) | Metrics reference, decoding algorithms, symbolic impact analysis, explainability, uncertainty estimation |
| [docs/contributing.md](docs/contributing.md) | Code conventions, adding components/metrics, bug fix conventions, known codebase risks |

---

## Citation
```bibtex
@inproceedings{dysarthriaNSR2026,
  title     = {Neuro-Symbolic Phoneme Recognition for Dysarthric Speech with
               Articulatory Constraints and Severity-Adaptive Fusion},
  author    = {},
  booktitle = {Proceedings of SPCOM 2026},
  year      = {2026},
}

@article{rudzicz2012torgo,
  title     = {The TORGO database of acoustic and articulatory speech from speakers with dysarthria},
  author    = {Rudzicz, Frank and Namasivayam, Aravind Kumar and Bhasha, Tom},
  journal   = {Language Resources and Evaluation},
  volume    = {46},
  number    = {4},
  pages     = {523--541},
  year      = {2012},
  publisher = {Springer}
}
```

---

## License

This project is released under the [MIT License](LICENSE).