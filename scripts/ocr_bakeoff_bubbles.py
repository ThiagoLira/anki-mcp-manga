#!/usr/bin/env python3
"""Bakeoff round 2: manga-ocr fed PER TEXT BUBBLE vs LLM OCR per full panel.

Round 1 fed full panels to manga-ocr and it dreamed up garbled text — that's
expected because manga-ocr was trained on single-bubble crops. Here we use the
same ONNX MagiV2 model's text-class (label 1) detections to crop each bubble
and feed those individually to manga-ocr. The LLM still sees the full panel
for fair comparison against the v3 pipeline option.
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

import numpy as np
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.panel_detector import (
    OnnxPanelDetector,
    _preprocess_image,
    get_text_to_panel_mapping,
    sort_text_boxes_in_reading_order,
)

load_dotenv()

IMAGE_DIR = ROOT / "test_manga_images"
OUTPUT_DIR = ROOT / "benchmark_results" / "ocr_bakeoff_bubbles"

_TEXT_CLASS_IDX = 1
_SCORE_THRESHOLD = 0.2

OCR_PROMPT = """Transcribe every Japanese text element in this manga panel, verbatim.
One string per balloon, narration box, sign, or SFX, in reading order (right-to-left, top-to-bottom).
- Do NOT translate.
- Do NOT add commentary.
- If a character is unclear, omit it rather than guess.
- If the panel has no Japanese text, return an empty list."""


class PanelOCR(BaseModel):
    lines: list[str]


def postprocess_class(class_scores, boxes, orig_h, orig_w, class_idx):
    logits = class_scores[0]
    bboxes = boxes[0]
    labels = logits.argmax(axis=-1)
    max_logits = logits.max(axis=-1)
    confidences = 1.0 / (1.0 + np.exp(-max_logits))
    mask = labels == class_idx
    confidences = confidences[mask]
    bboxes = bboxes[mask]
    if len(confidences) == 0:
        return []
    cx, cy, bw, bh = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x1 = np.clip((cx - bw / 2) * orig_w, 0, orig_w)
    y1 = np.clip((cy - bh / 2) * orig_h, 0, orig_h)
    x2 = np.clip((cx + bw / 2) * orig_w, 0, orig_w)
    y2 = np.clip((cy + bh / 2) * orig_h, 0, orig_h)
    corner_boxes = np.stack([x1, y1, x2, y2], axis=1)
    score_mask = confidences > _SCORE_THRESHOLD
    return corner_boxes[score_mask].tolist()


def detect_text_bboxes(detector: OnnxPanelDetector, image_bytes: bytes):
    """Run the same ONNX session and return raw text-class bboxes."""
    detector._load_model()
    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    gray_np = np.array(original.convert("L").convert("RGB"))
    orig_h, orig_w = gray_np.shape[:2]
    pixel_values, pixel_mask = _preprocess_image(gray_np)
    class_scores, boxes = detector._session.run(
        None, {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
    )
    return original, postprocess_class(
        class_scores, boxes, orig_h, orig_w, _TEXT_CLASS_IDX
    )


def panel_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


async def llm_ocr_panel(llm_structured, img: Image.Image):
    from langchain_core.messages import HumanMessage

    msg = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": panel_to_data_url(img)}},
        {"type": "text", "text": OCR_PROMPT},
    ])
    t0 = time.time()
    try:
        result = await llm_structured.ainvoke([msg])
        return result.lines, time.time() - t0
    except Exception as e:
        return [f"<ERROR: {type(e).__name__}: {e}>"], time.time() - t0


def manga_ocr_bubble(mocr, img: Image.Image):
    t0 = time.time()
    try:
        return mocr(img), time.time() - t0
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

    print("Loading manga-ocr...")
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
        original, text_bboxes = detect_text_bboxes(detector, image_bytes)
        sorted_panel_bboxes = [p.bbox for p in page.panels]

        print(f"  panels: {len(page.panels)}  text bubbles: {len(text_bboxes)}")

        if text_bboxes:
            text_to_panel = get_text_to_panel_mapping(text_bboxes, sorted_panel_bboxes)
            text_order = sort_text_boxes_in_reading_order(text_bboxes, sorted_panel_bboxes)
            ordered_texts = [(text_bboxes[i], text_to_panel[i]) for i in text_order]
        else:
            ordered_texts = []

        bubbles_by_panel: dict[int, list] = {p.index: [] for p in page.panels}
        for bbox, panel_idx in ordered_texts:
            if 0 <= panel_idx < len(page.panels):
                bubbles_by_panel[panel_idx].append(bbox)

        img_dir = OUTPUT_DIR / img_path.stem
        img_dir.mkdir(parents=True, exist_ok=True)

        panel_records = []
        for panel in page.panels:
            panel_pil = Image.open(io.BytesIO(panel.image_bytes)).convert("RGB")
            panel_pil.save(img_dir / f"panel_{panel.index:02d}.png", format="PNG")

            llm_lines, llm_s = await llm_ocr_panel(llm_structured, panel_pil)

            mocr_bubbles = []
            mocr_total_s = 0.0
            for b_idx, bbox in enumerate(bubbles_by_panel[panel.index]):
                x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                bubble_pil = original.crop((x1, y1, x2, y2)).convert("RGB")
                bubble_path = img_dir / f"panel_{panel.index:02d}_bubble_{b_idx:02d}.png"
                bubble_pil.save(bubble_path, format="PNG")
                text, t = manga_ocr_bubble(mocr, bubble_pil)
                mocr_total_s += t
                mocr_bubbles.append({"bbox": bbox, "file": bubble_path.name, "text": text})

            record = {
                "panel": panel.index,
                "crop_file": f"panel_{panel.index:02d}.png",
                "manga_ocr_per_bubble": {
                    "bubbles": mocr_bubbles,
                    "elapsed_s": round(mocr_total_s, 2),
                    "count": len(mocr_bubbles),
                },
                "llm_ocr_full_panel": {
                    "lines": llm_lines,
                    "elapsed_s": round(llm_s, 2),
                },
            }
            panel_records.append(record)

            mocr_joined = " | ".join(b["text"] for b in mocr_bubbles)
            llm_joined = " | ".join(llm_lines)
            print(f"  panel {panel.index}: bubbles={len(mocr_bubbles)} mocr={mocr_total_s:.1f}s llm={llm_s:.1f}s")
            print(f"    mocr: {mocr_joined[:140]}")
            print(f"    llm:  {llm_joined[:140]}")

        (img_dir / "summary.json").write_text(
            json.dumps({"image": img_path.name, "panels": panel_records},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()

    print(f"Results in {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="manga-ocr per bubble vs LLM OCR per panel")
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
