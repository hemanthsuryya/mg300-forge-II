# Tone Sear - Hugging Face Spaces build.
#
# Differences from the compose build on the main branch, all forced by the
# Spaces runtime:
#   * listens on 7860, the only port Spaces routes to
#   * runs as uid 1000, the user Spaces executes the container as
#   * writes under /home/user, because /data only exists on paid persistent
#     storage - on the free tier it is not writable
#   * no Postgres: DATABASE_URL is left unset so app/db.py falls back to SQLite
#
# Torch is installed from the CPU-only index on purpose: the default wheels drag
# in CUDA and take the image past 6 GB for no benefit, since this runs analysis
# on the CPU. Model weights are NOT baked in - they download on first use.
FROM python:3.12-slim AS base

# Spaces runs the container as uid 1000. Create that user up front so every
# path below is owned by whoever actually executes the process.
RUN useradd -m -u 1000 user
ENV HOME=/home/user

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/home/user/.cache/torch \
    XDG_CACHE_HOME=/home/user/.cache \
    JOBS_DIR=/home/user/data/jobs \
    DATA_DIR=/home/user/data

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

RUN mkdir -p /home/user/data/jobs /home/user/.cache && \
    chown -R user:user /app /home/user
USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:7860/api/health || exit 1

# --proxy-headers so the app sees the real scheme behind the Spaces TLS proxy,
# which the Google OAuth callback URL depends on.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
