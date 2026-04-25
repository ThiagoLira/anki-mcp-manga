#!/usr/bin/env python3
"""Benchmark the per-panel production pipeline end-to-end.

Runs the real CardAgent flow: detect panels -> summarise page -> extract
vocabulary per panel (with dedup). Results go to a per-model subfolder so
runs don't overwrite each other.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Project root is the parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import agent as agent_module
from src.agent import CardAgent
from src.panel_detector import PanelDetector

load_dotenv()

IMAGE_DIR = ROOT / "test_manga_images"
OUTPUT_DIR = ROOT / "benchmark_results"


def slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(".", "-")


async def run_one(agent: CardAgent, detector: PanelDetector, img_path: Path, out_dir: Path) -> dict:
    image_bytes = img_path.read_bytes()
    t_detect = time.time()
    page_analysis = detector.detect(image_bytes)
    detect_s = time.time() - t_detect
    n_panels = len(page_analysis.panels)

    t_agent = time.time()
    error = None
    try:
        result = await agent.process_image(
            "Extract vocabulary from this manga page and create cards.",
            image_bytes,
            page_analysis,
        )
        cards = [
            {"word": c.word, "reading": c.reading,
             "sentence": c.sentence, "translation": c.translation}
            for c in result.pending_cards
        ]
        summary_text = result.text
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        cards = []
        summary_text = ""
    agent_s = time.time() - t_agent

    record = {
        "image": img_path.name,
        "panels_detected": n_panels,
        "detect_s": round(detect_s, 2),
        "agent_s": round(agent_s, 2),
        "card_count": len(cards),
        "summary": summary_text,
        "cards": cards,
        "error": error,
    }

    img_dir = out_dir / img_path.stem
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / "result.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (img_dir / "summary.txt").write_text(result.text, encoding="utf-8")
    status = "FAIL" if error else "OK  "
    suffix = f"  error={error[:60]}" if error else ""
    print(
        f"  {status} {img_path.name}  panels={n_panels}  detect={detect_s:.1f}s  "
        f"agent={agent_s:.1f}s  cards={len(cards)}{suffix}"
    )
    return record


async def main_async(args: argparse.Namespace) -> None:
    api_key = os.environ.get(args.api_key_env, "not-needed")
    if api_key == "not-needed" and "openrouter" in args.base_url:
        print(f"ERROR: {args.api_key_env} not set")
        sys.exit(1)

    images = sorted(args.images_dir.glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not images:
        print(f"No images in {args.images_dir}")
        sys.exit(1)

    out_dir = OUTPUT_DIR / f"per_panel_{slug(args.model)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:    {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"Images:   {[p.name for p in images]}")
    print(f"Output:   {out_dir}\n")

    if args.force_per_panel:
        agent_module.MULTI_PANEL_THRESHOLD = 1  # always take per-panel path

    detector = PanelDetector(device=args.panel_device)
    agent = CardAgent(
        model=args.model, base_url=args.base_url, api_key=api_key, max_tokens=args.max_tokens
    )

    records = []
    for img in images:
        rec = await run_one(agent, detector, img, out_dir)
        records.append(rec)

    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "total_images": len(records),
        "total_cards": sum(r["card_count"] for r in records),
        "total_agent_s": round(sum(r["agent_s"] for r in records), 2),
        "images": records,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSummary saved to {out_dir / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the per-panel prod pipeline")
    parser.add_argument("--model", required=True, help="Model id to report under (folder name)")
    parser.add_argument(
        "--base-url", default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible base URL"
    )
    parser.add_argument(
        "--api-key-env", default="OPENROUTER_API_KEY",
        help="Env var holding the API key"
    )
    parser.add_argument("--images-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument(
        "--panel-device", default="cpu", choices=["cpu", "cuda"],
        help="Device for panel detector"
    )
    parser.add_argument(
        "--force-per-panel", action="store_true",
        help="Force per-panel path regardless of panel count (mirrors prod's multi-panel flow)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Max completion tokens per LLM call (default 2048)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
