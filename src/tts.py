from __future__ import annotations

import hashlib
import io
import json
import logging
import urllib.error
import urllib.request

from .config import settings

logger = logging.getLogger(__name__)

# Kokoro fallback model files — downloaded during Docker build to models/tts/
_KOKORO_MODEL_PATH = "models/tts/kokoro-v1.0.int8.onnx"
_KOKORO_VOICES_PATH = "models/tts/voices-v1.0.bin"
_KOKORO_VOICE = "jf_alpha"

_kokoro = None
_g2p = None


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        logger.info("Loading Kokoro TTS model...")
        _kokoro = Kokoro(_KOKORO_MODEL_PATH, _KOKORO_VOICES_PATH)
        logger.info("Kokoro TTS model loaded.")
    return _kokoro


def _get_g2p():
    global _g2p
    if _g2p is None:
        from misaki.ja import JAG2P
        logger.info("Loading Japanese G2P...")
        _g2p = JAG2P()
        logger.info("Japanese G2P loaded.")
    return _g2p


def _generate_kokoro(text: str) -> bytes:
    """Synthesize WAV bytes via in-process Kokoro ONNX (single fixed voice)."""
    import soundfile as sf
    kokoro = _get_kokoro()
    g2p = _get_g2p()
    phonemes, _ = g2p(text)
    samples, sample_rate = kokoro.create(
        phonemes, voice=_KOKORO_VOICE, speed=1.0, lang="ja", is_phonemes=True,
    )
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


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


def generate_tts(text: str, *, caption: str | None = None) -> bytes:
    """Synthesize Japanese TTS as WAV bytes ready for Anki media.

    Primary path: Irodori-TTS (VoiceDesign) over HTTP, with the speaker caption
    in `caption`. Falls back to in-process Kokoro on any timeout / connection /
    HTTP error so card creation never blocks on the remote server.
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
