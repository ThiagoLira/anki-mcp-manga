FROM python:3.11-slim

# Pillow system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev libwebp-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/

# Collection data lives here (mount as volume)
RUN mkdir -p /data

CMD ["python", "-m", "src.bot"]
