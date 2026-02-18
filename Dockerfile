FROM python:3.11-slim

WORKDIR /app

# Install system deps for Pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends libjpeg62-turbo libwebp7 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/

# Collection data lives here (mount as volume)
RUN mkdir -p /data

EXPOSE 8000

CMD ["python", "-m", "src.server"]
