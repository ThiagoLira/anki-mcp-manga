from __future__ import annotations

import hashlib
import io
import logging

import soundfile as sf

logger = logging.getLogger(__name__)

# Model files — downloaded during Docker build to models/tts/
_MODEL_PATH = "models/tts/kokoro-v1.0.int8.onnx"
_VOICES_PATH = "models/tts/voices-v1.0.bin"

_kokoro = None
_g2p = None


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        logger.info("Loading Kokoro TTS model...")
        _kokoro = Kokoro(_MODEL_PATH, _VOICES_PATH)
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


def generate_tts(text: str, voice: str = "jf_alpha", speed: float = 1.0) -> bytes:
    """Generate WAV audio bytes from Japanese text using Kokoro TTS.

    Returns WAV file bytes ready to be stored in Anki media.
    """
    kokoro = _get_kokoro()
    g2p = _get_g2p()

    # Use misaki's Japanese G2P for proper phonemization
    # (Kokoro's built-in phonemizer falls back to espeak which mangles Japanese)
    phonemes, _ = g2p(text)
    samples, sample_rate = kokoro.create(
        phonemes, voice=voice, speed=speed, lang="ja", is_phonemes=True,
    )

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


def tts_filename(wav_bytes: bytes) -> str:
    """Generate a deterministic filename for TTS audio based on content hash."""
    h = hashlib.sha256(wav_bytes).hexdigest()[:12]
    return f"tts_{h}.wav"
