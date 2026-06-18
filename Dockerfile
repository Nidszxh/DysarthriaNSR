# syntax=docker/dockerfile:1.7
#
# DysarthriaNSR multi-stage Dockerfile.
#
#   docker compose up --build serve     # run the FastAPI inference service
#   docker compose run train --run-name v4_final   # run training/eval (profile: train)
#
# The base image bundles NVIDIA CUDA runtime libs; the host still needs a
# working NVIDIA driver + nvidia-container-toolkit for GPU access.

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Ubuntu 22.04 ships Python 3.10 — the project's pyproject target (>= 3.10).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install pinned deps first for layer caching.
COPY requirements.txt pyproject.toml ./
RUN python3.10 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

# Training / evaluation image — entry point is the canonical orchestrator.
FROM base AS train
COPY . .
ENTRYPOINT ["python", "run_pipeline.py"]

# Inference service image — FastAPI app, HTTP on 8000.
FROM base AS serve
COPY . .
EXPOSE 8000
ENV RUN_NAME=v4_final
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
