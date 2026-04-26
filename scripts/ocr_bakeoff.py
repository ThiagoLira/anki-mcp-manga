#!/usr/bin/env python3
"""Compare manga-ocr vs Qwen-MoE OCR quality on manga panels.

For each test image, run both OCR backends on every panel and dump results
side-by-side as JSON, plus the panel crop as PNG for visual cross-reference.
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.panel_detector import OnnxPanelDetector

load_dotenv()

IMAGE_DIR = ROOT / "test_manga_images"
OUTPUT_DIR = ROOT / "benchmark_results" / "ocr_bakeoff"

OCR_PROMPT = """Transcribe every Japanese text element in this manga panel, verbatim.
One string per balloon, narration box, sign, or SFX, in reading order (right-to-left, top-to-bottom).
- Do NOT translate.
- Do NOT add commentary.
- If a character is unclear, omit it rather than guess.
- If the panel has no Japanese text, return an empty list."""


class PanelOCR(BaseModel):
    lines: list[str]


def panel_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def panel_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


async def llm_ocr_panel(llm_structured, img: Image.Image) -> tuple[list[str], float]:
    from langchain_core.messages import HumanMessage

    data_url = panel_to_data_url(img)
    msg = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": OCR_PROMPT},
    ])
    t0 = time.time()
    try:
        result = await llm_structured.ainvoke([msg])
        return result.lines, time.time() - t0
    except Exception as e:
        return [f"<ERROR: {type(e).__name__}: {e}>"], time.time() - t0


def manga_ocr_panel(mocr, img: Image.Image) -> tuple[str, float]:
    t0 = time.time()
    try:
        text = mocr(img)
        return text, time.time() - t0
    except Exception as e:
        return f"<ERROR: {type(e).__name__}: {e}>", time.time() - t0


async def main_async(args: argparse.Namespace) -> None:
    api_key = os.environ.get(args.api_key_env, "not-needed")

    images = sorted(args.images_dir.glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not images:
        print(f"No images in {args.images_dir}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading manga-ocr (downloads ~400MB on first run)...")
    from manga_ocr import MangaOcr
    mocr = MangaOcr()
    print("manga-ocr ready.")

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        max_tokens=args.max_tokens,
    )
    llm_structured = llm.with_structured_output(PanelOCR)
    print(f"LLM ready: {args.model} @ {args.base_url}")

    detector = OnnxPanelDetector(model_path=args.panel_model_path)
    print("Panel detector ready.\n")

    for img_path in images:
        print(f"=== {img_path.name} ===")
        image_bytes = img_path.read_bytes()
        page = detector.detect(image_bytes)
        print(f"  panels: {len(page.panels)}")

        img_dir = OUTPUT_DIR / img_path.stem
        img_dir.mkdir(parents=True, exist_ok=True)

        panel_records = []
        for panel in page.panels:
            img = panel_to_pil(panel.image_bytes)
            crop_path = img_dir / f"panel_{panel.index:02d}.png"
            img.save(crop_path, format="PNG")

            mocr_text, mocr_s = manga_ocr_panel(mocr, img)
            llm_lines, llm_s = await llm_ocr_panel(llm_structured, img)

            record = {
                "panel": panel.index,
                "crop_file": crop_path.name,
                "manga_ocr": {"text": mocr_text, "elapsed_s": round(mocr_s, 2)},
                "llm_ocr": {"lines": llm_lines, "elapsed_s": round(llm_s, 2)},
            }
            panel_records.append(record)

            print(f"  panel {panel.index}: mocr={mocr_s:.1f}s llm={llm_s:.1f}s")
            print(f"    mocr: {mocr_text[:120]}")
            print(f"    llm:  {' | '.join(llm_lines)[:120]}")

        (img_dir / "summary.json").write_text(
            json.dumps({"image": img_path.name, "panels": panel_records},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()

    print(f"Results in {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="manga-ocr vs LLM OCR per panel")
    parser.add_argument("--model", default="local")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--api-key-env", default="DUMMY")
    parser.add_argument("--images-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--panel-model-path", default="models/panel_detector.onnx")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
