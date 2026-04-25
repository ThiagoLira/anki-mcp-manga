#!/usr/bin/env python3
"""Smoke-test the janus kobold vision server with one manga image."""
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

PROMPT = """\
Read this manga page and:
1. Transcribe ALL dialogue panel by panel (reference panels as (1)(2)(3) in reading order, \
right-to-left top-to-bottom).
2. For each interesting vocabulary word, provide:
   - word: the vocabulary word
   - reading: hiragana reading
   - sentence: full Japanese sentence containing the word (target word in **bold**)
   - translation: English translation of the sentence (target word in **bold**)
Format your response as JSON with keys "panels" (list of {panel, dialogue}) and \
"vocabulary" (list of {word, reading, sentence, translation})."""

URL = "http://janus:5001/v1/chat/completions"
IMG = Path(sys.argv[1] if len(sys.argv) > 1 else "test_manga_images/manga.jpg")

data_url = f"data:image/jpeg;base64,{base64.b64encode(IMG.read_bytes()).decode()}"
payload = {
    "model": "local",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": PROMPT},
    ]}],
    "max_tokens": 600,
    "temperature": 0.2,
}

req = urllib.request.Request(
    URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
)
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    body = json.loads(r.read())
dt = time.time() - t0

usage = body.get("usage", {})
print(f"latency={dt:.1f}s prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")
print(f"tok/s={usage.get('completion_tokens', 0) / dt:.2f}")
print("---")
print(body["choices"][0]["message"]["content"])
