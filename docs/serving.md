# docs/serving.md — Inference Service, Docker & CI

> Cross-references: [evaluation.md](evaluation.md) for the decoders used by the API, [architecture.md](architecture.md) for the model internals.

---

## FastAPI Inference Service (`serve.py`)

A single-file FastAPI app that loads a trained `DysarthriaASRLightning` checkpoint once at startup and transcribes uploaded audio. It reuses the **same** audio-preprocessing path as the training dataset (mono → 16 kHz resample → HuBERT processor → `max_audio_length` truncation) and the production decoders from `evaluate.py` (`greedy_decode` / `BeamSearchDecoder` via `decode_predictions`).

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe: model loaded, checkpoint, device, uptime. Used by the Docker healthcheck. |
| `GET` | `/model` | Model metadata: run name, checkpoint, vocab size, ablation mode, HuBERT id/revision, sample rate. |
| `POST` | `/transcribe` | Upload an audio file → phoneme sequence. |

### `/transcribe` request fields (multipart form)

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | file | — | `.wav` / `.flac` / `.mp3` / `.ogg` / `.m4a`, ≤ 50 MB |
| `severity` | float | `null` | Dysarthria severity in [0, 5]; overrides speaker lookup |
| `speaker` | str | `null` | TORGO speaker ID; severity resolved from the severity map |
| `use_beam_search` | bool | `false` | Beam search vs. greedy CTC decoding |
| `beam_width` | int | `10` | Beam width (1–100) |
| `temperature` | float | `1.0` | CTC logit temperature (0–2) |

If neither `severity` nor `speaker` is provided, severity defaults to 0.0 (control). Explicit `severity` is honored by calling the underlying `NeuroSymbolicASR` directly (the Lightning wrapper would re-derive severity from a speaker ID).

### Response (example)

```json
{
  "run_name": "v4_final",
  "checkpoint": "epoch=28-val_per=0.505.ckpt",
  "device": "cuda",
  "severity": 4.9,
  "decoder": "greedy",
  "beam_width": null,
  "temperature": 1.0,
  "phonemes": ["P", "AH", "T"],
  "text": "P AH T",
  "num_phonemes": 3,
  "articulatory": {"manner": "stop", "place": "alveolar", "voice": "voiceless"},
  "latency_ms": 42.1
}
```

### Run locally

```bash
# With RUN_NAME + DEVICE env vars
RUN_NAME=v4_final DEVICE=cuda python serve.py --port 8000

# Explicit CLI args
python serve.py --run-name v4_final --host 0.0.0.0 --port 8000

# Or via uvicorn directly
uvicorn serve:app --host 0.0.0.0 --port 8000

# Request
curl -F "file=@data/raw/audio/train/M01_abc12345_s1_mic1_001.wav" \
     -F "severity=4.9" http://localhost:8000/transcribe
```

Requires a trained checkpoint under `checkpoints/{run_name}/` and the processed manifest under `data/processed/` (same layout as `run_pipeline.py`).

---

## Docker

`Dockerfile` is multi-stage:

| Stage | Contents |
|---|---|
| `base` | `nvidia/cuda:12.8.0-runtime-ubuntu22.04` + Python 3.10 venv + pinned `requirements.txt` (including `fastapi`/`uvicorn`) |
| `train` | Adds the repo; entrypoint `python run_pipeline.py` — canonical training/eval entry |
| `serve` | Adds the repo; `CMD uvicorn serve:app` on port 8000 |

`docker-compose.yml` defines:

- **`serve`** — the FastAPI service on `${SERVE_PORT:-8000}`, GPU-enabled (`--gpus all` via compose `deploy`), mounts `checkpoints/` and `data/` read-only, ships a `/health` healthcheck. Select the checkpoint run with `RUN_NAME=v4_final docker compose up --build serve`.
- **`train`** — a one-off training/eval container (profile `train`): `docker compose run train --run-name v4_final`.

The host needs a working NVIDIA driver + `nvidia-container-toolkit` for GPU access (the image itself bundles CUDA runtime libs via the `nvidia/cuda` base).

```bash
# Build + run the serving image (GPU)
docker compose up --build serve

# One-off training run
docker compose run train --run-name v4_final
```

---

## CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on push to `main`/`master` and on pull requests:

| Job | Steps |
|---|---|
| **lint** | `ruff check .` (repo-wide) + `mypy serve.py --follow-imports=silent` (the repo is not yet fully type-clean; the new service file is gated) |
| **test** | CPU-only torch install (avoids ~2.5 GB CUDA wheels), `pip install -r requirements.txt`, full `pytest` suite (135 tests), `scripts/smoke_test.py --profile unit` (8 checks), and a real `import serve` gate (validates that the app module and all its imports resolve without loading a model) |

Lint passes without modifying repo style: `ruff` is configured in `pyproject.toml` to match existing conventions (`N`, `UP`, `S324`, `E402` ignored — paper tensor notation, `Dict`/`Optional` py3.10 style, md5/sha1 cache fingerprints, and the project-root `sys.path` bootstrap).

---

## Honesty notes

- Docker, CI, and the serving API **exist in the repo** (`Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `serve.py`). They are exercised by `docker build --check`, `docker compose config`, the CI workflow, and the local test suite.
- The service has **no production uptime/requests-per-second numbers** — it is containerized and CI-tested but has not been load-tested in production.
- TORGO data is gitignored and not present in the development checkout, so end-to-end `/transcribe` requires a local checkpoint + manifest (see §Run locally).
