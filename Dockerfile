FROM python:3.11-slim

# Image processing + TTS system deps + build tools (pyopenjtalk needs cmake/gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev libwebp-dev zlib1g-dev \
    espeak-ng \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir ".[panels,tts]"

COPY src/ src/
COPY models/ models/

# Pre-download Kokoro TTS model files during build (after COPY to not be overwritten)
RUN mkdir -p models/tts && python -c "\
import urllib.request; \
base = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0'; \
urllib.request.urlretrieve(f'{base}/kokoro-v1.0.int8.onnx', 'models/tts/kokoro-v1.0.int8.onnx'); \
urllib.request.urlretrieve(f'{base}/voices-v1.0.bin', 'models/tts/voices-v1.0.bin')"

# Download UniDic dictionary for Japanese phonemization (misaki JAG2P needs it)
RUN python -m unidic download

# Collection data lives here (mount as volume)
RUN mkdir -p /data

CMD ["python", "-m", "src.bot"]
