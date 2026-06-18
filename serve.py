"""
serve.py — DysarthriaNSR FastAPI Inference Service
===================================================

Lightweight HTTP inference endpoint around a trained ``DysarthriaASRLightning``
checkpoint. Reuses the same audio preprocessing path as the training dataset
(mono → 16 kHz resample → HuBERT processor) and the evaluation decoders.

Endpoints
---------
    GET  /health        Liveness/readiness probe (model loaded, device, checkpoint).
    GET  /model         Model metadata (run name, checkpoint, vocab size, device).
    POST /transcribe    Transcribe an uploaded audio file to phonemes.

Run (local)
-----------
    python serve.py --run-name v4_final --port 8000

Run (uvicorn)
-------------
    uvicorn serve:app --host 0.0.0.0 --port 8000
    # set RUN_NAME env var (default: v4_final), DEVICE=cuda|cpu

Run (Docker)
------------
    docker compose up --build serve

Example
-------
    curl -F "file=@sample.wav" -F "severity=4.9" \\
         http://localhost:8000/transcribe
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch

# Project-root on sys.path (supports running from any CWD)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Silence noisy third-party loggers (mirrors run_pipeline.py)
for _lib in (
    "huggingface_hub",
    "huggingface_hub.file_download",
    "huggingface_hub.hf_api",
    "httpx",
    "httpcore",
    "transformers",
    "pytorch_lightning",
    "urllib3",
    "requests",
):
    logging.getLogger(_lib).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402

from evaluate import decode_predictions  # noqa: E402
from src.data.dataloader import TorgoNeuroSymbolicDataset  # noqa: E402
from src.models.model import NeuroSymbolicASR  # noqa: E402
from src.utils.config import (  # noqa: E402
    Config,
    get_default_config,
    get_project_root,
    get_speaker_severity,
)
from train import DysarthriaASRLightning  # noqa: E402

_STARTED_AT = time.time()
MAX_AUDIO_BYTES = 50 * 1024 * 1024  # 50 MB upload cap
SUPPORTED_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


class _InferenceModel:
    """Loaded checkpoint + vocab maps + processor, ready for single-shot decode."""

    def __init__(self, run_name: str, device: str) -> None:
        self.run_name = run_name
        self.device = torch.device(
            device if device in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

        self.config: Config = get_default_config()
        self.config.experiment.run_name = run_name

        manifest = Path(self.config.data.manifest_path)
        if not manifest.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest}\n"
                "Run `python src/data/download.py && python src/data/manifest.py` first."
            )
        self.dataset = TorgoNeuroSymbolicDataset(
            manifest_path=str(manifest),
            processor_id=self.config.model.hubert_model_id,
            sampling_rate=self.config.data.sampling_rate,
            max_audio_length=self.config.data.max_audio_length,
        )
        self.processor = self.dataset.processor

        ckpt = self._resolve_checkpoint(run_name)
        model_arch = NeuroSymbolicASR(
            model_config=self.config.model,
            symbolic_config=self.config.symbolic,
            phn_to_id=self.dataset.phn_to_id,
            id_to_phn=self.dataset.id_to_phn,
            manner_to_id=self.dataset.manner_to_id,
            place_to_id=self.dataset.place_to_id,
            voice_to_id=self.dataset.voice_to_id,
        )
        self.lightning_model = DysarthriaASRLightning.load_from_checkpoint(
            str(ckpt),
            model=model_arch,
            config=self.config,
            phn_to_id=self.dataset.phn_to_id,
            id_to_phn=self.dataset.id_to_phn,
            strict=False,
        )
        self.lightning_model.to(self.device)
        self.lightning_model.eval()

        self.ckpt_name = ckpt.name
        self.phn_to_id = self.dataset.phn_to_id
        self.id_to_phn = self.dataset.id_to_phn
        self.ablation_mode = getattr(self.config.training, "ablation_mode", None) or "full"
        log.info(
            "Model ready: run=%s ckpt=%s device=%s vocab=%d ablation=%s",
            run_name,
            self.ckpt_name,
            self.device.type,
            len(self.phn_to_id),
            self.ablation_mode,
        )

    @staticmethod
    def _resolve_checkpoint(run_name: str) -> Path:
        """Best-scored checkpoint under checkpoints/{run_name} (lowest val_per wins)."""
        import re

        ckpt_dir = get_project_root() / "checkpoints" / run_name
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {ckpt_dir}\n"
                "Run training first (`python run_pipeline.py --run-name <name>`)."
            )
        all_ckpts = list(ckpt_dir.glob("*.ckpt"))
        if not all_ckpts:
            raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}.")
        scored = [
            (float(m.group(1)), p)
            for p in all_ckpts
            if (m := re.search(r"val_per=([0-9]+\.[0-9]+)", p.name)) is not None
        ]
        if scored:
            scored.sort(key=lambda t: t[0])
            return scored[0][1]
        last = ckpt_dir / "last.ckpt"
        if last.exists():
            return last
        return sorted(all_ckpts)[-1]

    @torch.no_grad()
    def transcribe(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        severity: float | None = None,
        speaker: str | None = None,
        use_beam_search: bool = False,
        beam_width: int = 10,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Decode a single mono waveform [T] into a phoneme sequence."""
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        if sample_rate != self.config.data.sampling_rate:
            import torchaudio.functional as taF

            waveform = taF.resample(waveform, sample_rate, self.config.data.sampling_rate)
        max_samples = int(self.config.data.max_audio_length * self.config.data.sampling_rate)
        if waveform.numel() > max_samples:
            waveform = waveform[:max_samples]

        input_values = self.processor(  # type: ignore[operator]
            waveform, sampling_rate=self.config.data.sampling_rate, return_tensors="pt"
        ).input_values
        if input_values.numel() > max_samples:
            input_values = input_values[:, :max_samples]
        input_values = input_values.to(self.device)

        B = 1
        attention_mask = torch.ones(B, input_values.size(1), dtype=torch.long, device=self.device)

        if severity is None:
            severity = get_speaker_severity(speaker) if speaker else 0.0
        severity_tensor = torch.tensor([severity], dtype=torch.float32, device=self.device)

        # Call the underlying NeuroSymbolicASR directly (not the Lightning
        # wrapper) so an explicit severity is honored instead of being
        # re-derived from a speaker ID.
        outputs = self.lightning_model.model(
            input_values=input_values,
            attention_mask=attention_mask,
            speaker_severity=severity_tensor,
            ablation_mode=self.ablation_mode,
        )
        log_probs = outputs["log_probs_constrained"]
        output_lengths = outputs["output_lengths"]

        predictions = decode_predictions(
            log_probs,
            self.phn_to_id,
            self.id_to_phn,
            use_beam_search=use_beam_search,
            beam_width=beam_width,
            output_lengths=output_lengths,
            temperature=temperature,
        )
        phonemes = predictions[0] if predictions else []

        result: dict[str, Any] = {
            "run_name": self.run_name,
            "checkpoint": self.ckpt_name,
            "device": self.device.type,
            "severity": round(float(severity), 3),
            "decoder": "beam" if use_beam_search else "greedy",
            "beam_width": beam_width if use_beam_search else None,
            "temperature": temperature,
            "phonemes": phonemes,
            "text": " ".join(phonemes),
            "num_phonemes": len(phonemes),
        }
        if outputs.get("logits_manner") is not None:
            result["articulatory"] = {
                "manner": self._argmax_class(outputs["logits_manner"], self.dataset.id_to_manner),
                "place": self._argmax_class(outputs["logits_place"], self.dataset.id_to_place),
                "voice": self._argmax_class(outputs["logits_voice"], self.dataset.id_to_voice),
            }
        return result

    @staticmethod
    def _argmax_class(logits: torch.Tensor, id_to_class: dict[int, str]) -> str:
        """Map utterance-level GAP logits [1, K] → class label string."""
        idx = int(logits.detach().cpu().argmax(dim=-1).item())
        return id_to_class.get(idx, str(idx))


_app_model: _InferenceModel | None = None


def get_model() -> _InferenceModel:
    if _app_model is None:
        raise RuntimeError("Model not initialized (startup failed).")
    return _app_model


@asynccontextmanager
async def lifespan(_: FastAPI) -> Any:
    """Load the model once at startup so requests never block on model loading."""
    global _app_model
    run_name = os.environ.get("RUN_NAME", "v4_final")
    device = os.environ.get("DEVICE", "")
    try:
        _app_model = _InferenceModel(run_name=run_name, device=device)
    except Exception as exc:
        log.error("Model load failed: %s", exc)
        _app_model = None
        raise
    yield
    _app_model = None


app = FastAPI(
    title="DysarthriaNSR Inference API",
    description="Neuro-symbolic phoneme recognition for dysarthric speech (TORGO).",
    version="2.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    model = get_model()
    return {
        "status": "ok",
        "model": "DysarthriaNSR",
        "run_name": model.run_name,
        "checkpoint": model.ckpt_name,
        "device": model.device.type,
        "uptime_seconds": round(time.time() - _STARTED_AT, 1),
    }


@app.get("/model")
def model_info() -> dict[str, Any]:
    model = get_model()
    return {
        "run_name": model.run_name,
        "checkpoint": model.ckpt_name,
        "device": model.device.type,
        "vocab_size": len(model.phn_to_id),
        "ablation_mode": model.ablation_mode,
        "sampling_rate": model.config.data.sampling_rate,
        "max_audio_length_seconds": model.config.data.max_audio_length,
        "hubert_model_id": model.config.model.hubert_model_id,
        "hubert_model_revision": model.config.model.hubert_model_revision,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Audio file (wav/flac/mp3/ogg/m4a)"),
    severity: float | None = Form(None, description="Dysarthria severity in [0, 5]"),
    speaker: str | None = Form(None, description="TORGO speaker ID (severity looked up)"),
    beam_width: int = Form(10, description="Beam width for beam search decoding"),
    use_beam_search: bool = Form(False, description="Use beam search instead of greedy decode"),
    temperature: float = Form(1.0, description="CTC logit temperature"),
) -> dict[str, Any]:
    """Transcribe an uploaded audio clip into a phoneme sequence."""
    model = get_model()

    if not file.filename or not file.filename.lower().endswith(SUPPORTED_SUFFIXES):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type. Allowed suffixes: {', '.join(SUPPORTED_SUFFIXES)}",
        )
    if severity is not None and not (0.0 <= severity <= 5.0):
        raise HTTPException(status_code=422, detail="severity must be in [0, 5]")
    if not (0.0 <= temperature <= 2.0):
        raise HTTPException(status_code=422, detail="temperature must be in [0, 2]")
    if beam_width < 1 or beam_width > 100:
        raise HTTPException(status_code=422, detail="beam_width must be in [1, 100]")

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=422, detail="Empty file")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_AUDIO_BYTES // (1024 * 1024)} MB limit")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to decode audio: {exc}") from exc
    finally:
        os.unlink(tmp_path)

    if waveform.numel() == 0:
        raise HTTPException(status_code=422, detail="Decoded audio is empty")

    t0 = time.perf_counter()
    result = model.transcribe(
        waveform,
        int(sample_rate),
        severity=severity,
        speaker=speaker,
        use_beam_search=use_beam_search,
        beam_width=beam_width,
        temperature=temperature,
    )
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Serve DysarthriaNSR via FastAPI.")
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME", "v4_final"))
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container serving binds all interfaces
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=os.environ.get("DEVICE", ""))
    args = parser.parse_args()

    os.environ["RUN_NAME"] = args.run_name
    os.environ["DEVICE"] = args.device

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
