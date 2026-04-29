from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request

from .config import settings

logger = logging.getLogger(__name__)


def _generate_irodori(text: str, caption: str | None) -> bytes:
    """POST to the Irodori VoiceDesign HTTP server, return WAV bytes."""
    payload: dict[str, object] = {"text": text}
    if caption:
        payload["caption"] = caption
    body = json.dumps(payload).encode("utf-8")
    url = settings.irodori_tts_url.rstrip("/") + "/tts"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=settings.irodori_tts_timeout_s) as resp:
        if resp.status != 200:
            raise RuntimeError(f"irodori HTTP {resp.status}")
        return resp.read()


def _generate_kokoro(text: str) -> bytes:
    """POST to the Kokoro-FastAPI service (homelab models namespace), return WAV bytes."""
    payload = {
        "model": "kokoro",
        "input": text,
        "voice": settings.kokoro_tts_voice,
        "response_format": "wav",
    }
    body = json.dumps(payload).encode("utf-8")
    url = settings.kokoro_tts_url.rstrip("/") + "/v1/audio/speech"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=settings.kokoro_tts_timeout_s) as resp:
        if resp.status != 200:
            raise RuntimeError(f"kokoro HTTP {resp.status}")
        return resp.read()


def generate_tts(text: str, *, caption: str | None = None) -> bytes:
    """Synthesize Japanese TTS as WAV bytes ready for Anki media.

    Primary path: Irodori-TTS (VoiceDesign) over HTTP, with the speaker caption
    in `caption`. Falls back to the Kokoro-FastAPI service on any timeout /
    connection / HTTP error so card creation never blocks on Irodori.
    """
    irodori_url = (settings.irodori_tts_url or "").strip()
    if irodori_url:
        try:
            return _generate_irodori(text, caption)
        except (urllib.error.URLError, OSError, TimeoutError, RuntimeError) as exc:
            logger.warning(
                "Irodori TTS at %s failed (%s); falling back to Kokoro",
                irodori_url, exc,
            )
    return _generate_kokoro(text)


def tts_filename(wav_bytes: bytes) -> str:
    """Generate a deterministic filename for TTS audio based on content hash."""
    h = hashlib.sha256(wav_bytes).hexdigest()[:12]
    return f"tts_{h}.wav"
