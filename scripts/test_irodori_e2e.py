"""End-to-end test of styled-translation + Irodori TTS pipeline.

1. Loads a manga panel image.
2. Runs deterministic word extraction (panel detector + manga-ocr + fugashi).
3. Picks the first N candidates.
4. CardAgent.generate_cards: ONE batched LLM call producing
   translation + tts_text + voice_description_jp per card.
5. tts.generate_tts on each card: hits Irodori HTTP (Kokoro fallback on error).
6. Writes wavs + summary.json to outputs/e2e/.

Run from repo root:
    .venv/bin/python scripts/test_irodori_e2e.py [image-path] [N]

Requires the irodori-tts-server (sibling repo ~/repos/irodori-tts-server)
to be reachable at $IRODORI_TTS_URL (default http://janus:8200).
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import CardAgent
from src.tts import generate_tts, tts_filename


async def main() -> None:
    image_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1
        else ROOT / "test_manga_images" / "manga2.jpg"
    )
    n_select = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    out_dir = ROOT / "outputs" / "e2e"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] loading {image_path}")
    image_bytes = image_path.read_bytes()

    print(f"[2] CardAgent + word extraction (image={len(image_bytes)} bytes)")
    agent = CardAgent()
    t0 = time.time()
    extraction = await agent.extract_candidates(image_bytes)
    print(
        f"    extracted {len(extraction.candidates)} candidates "
        f"in {time.time() - t0:.1f}s"
    )
    for i, c in enumerate(extraction.candidates):
        print(
            f"    [{i:2}] word={c.word!r:>10} surface={c.surface!r:>10} "
            f"panel={c.panel_index} sent={c.sentence!r}"
        )

    if not extraction.candidates:
        print("no candidates; exiting")
        return

    selected = list(range(min(n_select, len(extraction.candidates))))
    print(f"\n[3] generate_cards: ONE batched LLM call for {len(selected)} cards")
    t0 = time.time()
    cards = await agent.generate_cards(extraction, selected)
    print(f"    LLM done in {time.time() - t0:.1f}s, produced {len(cards)} cards")

    summary: list[dict] = []
    for i, card in enumerate(cards):
        print(f"\n--- card {i}: {card.word} ---")
        print(f"  sentence:      {card.sentence}")
        print(f"  translation:   {card.translation}")
        print(f"  tts_text:      {card.tts_text}")
        print(f"  voice_desc_jp: {card.voice_description_jp}")
        t1 = time.time()
        try:
            wav = generate_tts(
                card.tts_text,
                caption=card.voice_description_jp or None,
            )
        except Exception as exc:
            print(f"  TTS failed: {exc}")
            summary.append({"i": i, "word": card.word, "error": str(exc)})
            continue
        gen_dt = time.time() - t1
        fn = tts_filename(wav)
        wav_path = out_dir / f"{i:02d}_{card.word}_{fn}"
        wav_path.write_bytes(wav)
        print(f"  -> {wav_path.name}  bytes={len(wav)}  gen={gen_dt:.1f}s")
        summary.append({
            "i": i,
            "word": card.word,
            "sentence": card.sentence,
            "translation": card.translation,
            "tts_text": card.tts_text,
            "voice_description_jp": card.voice_description_jp,
            "wav": wav_path.name,
            "wav_bytes": len(wav),
            "gen_seconds": round(gen_dt, 2),
        })

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(f"\n[done] {len(summary)} entries -> {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
