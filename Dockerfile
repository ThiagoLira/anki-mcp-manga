FROM python:3.11-slim

# Pillow + panels system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev libwebp-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .

# Install CPU-only torch first (avoids downloading 2GB CUDA libs),
# then install the rest of the project with panels extras
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[panels]"

COPY src/ src/

# Collection data lives here (mount as volume)
RUN mkdir -p /data

CMD ["python", "-m", "src.bot"]
