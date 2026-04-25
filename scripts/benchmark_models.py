#!/usr/bin/env python3
"""Benchmark vision models for manga transcription + translation quality.

Runs all models in parallel using asyncio for fast execution.
"""

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

BENCHMARK_PROMPT = """\
Read this manga page and:
1. Transcribe ALL dialogue panel by panel (reference panels as ①②③ in reading order, \
right-to-left top-to-bottom).
2. For each interesting vocabulary word, provide:
   - word: the vocabulary word
   - reading: hiragana reading
   - sentence: full Japanese sentence containing the word (target word in **bold**)
   - translation: English translation of the sentence (target word in **bold**)
Format your response as JSON with keys "panels" (list of {panel, dialogue}) and \
"vocabulary" (list of {word, reading, sentence, translation})."""

# model_id -> (input_cost_per_M, output_cost_per_M)
MODEL_PRICING = {
    "qwen/qwen3.5-9b": (0.05, 0.15),
    "qwen/qwen3.5-flash-02-23": (0.065, 0.26),
    "qwen/qwen3.5-27b": (0.195, 1.56),
    "qwen/qwen3.5-35b-a3b": (0.1625, 1.30),
    "qwen/qwen3.5-122b-a10b": (0.26, 2.08),
    "qwen/qwen3.5-397b-a17b": (0.39, 2.34),
    "z-ai/glm-4.7-flash": (0.06, 0.40),
    "z-ai/glm-4.7": (0.39, 1.75),
    "z-ai/glm-4.6v": (0.30, 0.90),
    "mistralai/mistral-small-2603": (0.15, 0.60),
    "bytedance-seed/seed-1.6-flash": (0.075, 0.30),
    "bytedance-seed/seed-1.6": (0.25, 2.00),
    "bytedance-seed/seed-2.0-mini": (0.10, 0.40),
    "bytedance-seed/seed-2.0-lite": (0.25, 2.00),
    "xiaomi/mimo-v2-flash": (0.09, 0.29),
    "moonshotai/kimi-k2.5": (0.45, 2.20),
}

ALL_MODELS = list(MODEL_PRICING.keys())

IMAGE_DIR = Path("test_manga_images")
OUTPUT_DIR = Path("benchmark_results")


def load_image_as_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(".", "-")


async def benchmark_one(
    model_id: str, img_path: Path, data_url: str, api_key: str, base_url: str, timeout: int
) -> dict:
    """Benchmark a single model+image combination."""
    pricing = MODEL_PRICING.get(model_id, (0, 0))
    img_name = img_path.stem

    llm = ChatOpenAI(
        model=model_id,
        openai_api_key=api_key,
        openai_api_base=base_url,
        timeout=timeout,
        max_retries=0,
        temperature=0.3,
        max_tokens=1024,
        model_kwargs={"extra_body": {"repetition_penalty": 1.1, "min_p": 0.05}},
    )

    msg = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": BENCHMARK_PROMPT},
    ])

    # Retry up to 3 times on transient errors
    response = None
    last_err = None
    for attempt in range(3):
        try:
            t0 = time.time()
            response = await llm.ainvoke([msg])
            latency = time.time() - t0
            break
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "503" in err_str or "429" in err_str or "overloaded" in err_str.lower():
                await asyncio.sleep(3 * (attempt + 1))
            else:
                break

    if response is None:
        print(f"  FAIL  {model_id} / {img_path.name}: {last_err}")
        return {
            "model": model_id,
            "image": img_name,
            "status": "error",
            "error": str(last_err),
        }

    text = response.content
    usage = response.usage_metadata or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost = input_tokens * pricing[0] / 1_000_000 + output_tokens * pricing[1] / 1_000_000

    # Save raw output in a per-model subfolder
    model_dir = OUTPUT_DIR / slug(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"{img_name}.txt").write_text(text, encoding="utf-8")

    print(f"  OK    {model_id} / {img_path.name}  {latency:.1f}s  {input_tokens}+{output_tokens}tok  ${cost:.4f}")
    return {
        "model": model_id,
        "image": img_name,
        "status": "ok",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "latency_s": round(latency, 2),
    }


async def run_benchmark(
    models: list[str], images: list[Path], base_url: str, api_key_env: str, timeout: int
) -> dict:
    api_key = os.environ.get(api_key_env, "not-needed")
    if api_key == "not-needed" and "openrouter" in base_url:
        print(f"ERROR: {api_key_env} not set")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Pre-load all images as data URLs
    data_urls = {p: load_image_as_data_url(p) for p in images}

    # Launch all model×image tasks in parallel
    tasks = []
    for model_id in models:
        for img_path in images:
            tasks.append(
                benchmark_one(model_id, img_path, data_urls[img_path], api_key, base_url, timeout)
            )

    print(f"Launching {len(tasks)} parallel requests...\n")
    results = await asyncio.gather(*tasks)

    # Aggregate by model
    summary = {}
    for model_id in models:
        pricing = MODEL_PRICING.get(model_id, (0, 0))
        model_results = [r for r in results if r["model"] == model_id]
        ok = [r for r in model_results if r["status"] == "ok"]
        summary[model_id] = {
            "model": model_id,
            "pricing_input_per_M": pricing[0],
            "pricing_output_per_M": pricing[1],
            "total_cost_usd": round(sum(r["cost_usd"] for r in ok), 6),
            "total_input_tokens": sum(r["input_tokens"] for r in ok),
            "total_output_tokens": sum(r["output_tokens"] for r in ok),
            "avg_latency_s": round(sum(r["latency_s"] for r in ok) / len(ok), 2) if ok else 0,
            "success_count": len(ok),
            "fail_count": len(model_results) - len(ok),
            "images": {r["image"]: r for r in model_results},
        }

    # Per-model summary in each subfolder (one model per run is typical)
    for model_id, data in summary.items():
        model_dir = OUTPUT_DIR / slug(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "summary.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"\nPer-model summaries saved under {OUTPUT_DIR}/<model>/summary.json")
    return summary


def print_summary_table(summary: dict):
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<40} {'Cost':>8} {'Latency':>8} {'In Tok':>8} {'Out Tok':>8} {'OK':>3}")
    print("-" * 80)
    for model_id, data in sorted(summary.items(), key=lambda x: x[1]["total_cost_usd"]):
        print(
            f"{model_id:<40} "
            f"${data['total_cost_usd']:>7.4f} "
            f"{data['avg_latency_s']:>7.1f}s "
            f"{data['total_input_tokens']:>8} "
            f"{data['total_output_tokens']:>8} "
            f"{data['success_count']:>3}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark vision models for manga transcription")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, help="Model IDs to benchmark")
    parser.add_argument("--images-dir", type=Path, default=IMAGE_DIR, help="Directory with test images")
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Env var holding the API key (ignored for local servers)",
    )
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds")
    args = parser.parse_args()

    images = sorted(args.images_dir.glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not images:
        print(f"No images found in {args.images_dir}/")
        sys.exit(1)

    print(f"Images: {[p.name for p in images]}")
    print(f"Models: {len(args.models)}")

    summary = asyncio.run(
        run_benchmark(args.models, images, args.base_url, args.api_key_env, args.timeout)
    )
    print_summary_table(summary)


if __name__ == "__main__":
    main()
