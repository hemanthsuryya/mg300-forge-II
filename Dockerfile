# Tone Chaser
#
# Torch is installed from the CPU-only index on purpose: the default wheels drag
# in CUDA and take the image past 6 GB for no benefit, since this runs analysis
# on the CPU. Model weights are NOT baked in - they download on first use into
# the torch cache, which compose keeps in a named volume.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache \
    JOBS_DIR=/data/jobs \
    DATA_DIR=/data

# ffmpeg is required for decoding; libsndfile for soundfile; the rest is for
# building any wheel that has no arm64/amd64 binary.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch first, in its own layer, so app changes never re-download 200+ MB.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
# demucs depends on torch, but the CPU build above already satisfies that
# constraint, so pip leaves it alone instead of pulling the CUDA wheel.
RUN pip install -r requirements.txt

COPY app ./app
COPY web ./web
COPY tests ./tests
COPY docs ./docs

RUN mkdir -p /data/jobs /cache && \
    useradd -m -u 10001 tonechaser && \
    chown -R tonechaser:tonechaser /app /data /cache
USER tonechaser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
