FROM python:3.11-slim

# Image processing system deps. (TTS deps — espeak-ng, cmake — were dropped
# when the in-process Kokoro fallback became an HTTP call to the kokoro-tts
# pod on teagolab-1. Keep build-essential for any pip-built C extensions in
# the panels stack.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev libwebp-dev zlib1g-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir ".[panels]"

COPY src/ src/
COPY models/ models/

# Collection data lives here (mount as volume)
RUN mkdir -p /data

CMD ["python", "-m", "src.bot"]
