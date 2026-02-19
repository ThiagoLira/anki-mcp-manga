FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/

# Collection data lives here (mount as volume)
RUN mkdir -p /data

EXPOSE 8000

CMD ["python", "-m", "src.server"]
