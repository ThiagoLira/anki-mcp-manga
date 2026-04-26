#!/usr/bin/env python3
"""V3 pipeline: deterministic where possible, LLM only for translation.

Steps per page:
  1. ONNX detector: panels + text bboxes (single inference)
  2. NMS-dedupe text bboxes, map to panels in reading order
  3. manga-ocr each text bubble → sentence
  4. fugashi tokenize → wordfreq filter → candidate words (deterministic)
  5. ONE batched LLM call to translate all sentences with target word bolded
  6. Validate (word ⊂ sentence, reading is hiragana) → drop on fail
  7. Build cards
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
import time
import unicodedata
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
OUTPUT_DIR = ROOT / "benchmark_results"

_TEXT_CLASS_IDX = 1
_SCORE_THRESHOLD = 0.2
_NMS_IOU_THRESHOLD = 0.5

# fugashi POS classes to drop
DROP_POS = {"助詞", "助動詞", "記号", "補助記号", "感動詞", "代名詞",
            "接頭辞", "接尾辞", "数詞", "空白"}

# Hiragana Unicode block
_HIRA_LO, _HIRA_HI = "ぁ", "ゟ"


class TranslationItem(BaseModel):
    translation: str


class Translations(BaseModel):
    items: list[TranslationItem]


def slug(s: str) -> str:
    return s.replace("/", "_").replace(".", "-")


def postprocess_class(class_scores, boxes, orig_h, orig_w, class_idx, score_threshold):
    logits = class_scores[0]
    bboxes = boxes[0]
    labels = logits.argmax(axis=-1)
    max_logits = logits.max(axis=-1)
    confidences = 1.0 / (1.0 + np.exp(-max_logits))
    mask = labels == class_idx
    confidences = confidences[mask]
    bboxes = bboxes[mask]
    if len(confidences) == 0:
        return [], np.array([])
    cx, cy, bw, bh = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x1 = np.clip((cx - bw / 2) * orig_w, 0, orig_w)
    y1 = np.clip((cy - bh / 2) * orig_h, 0, orig_h)
    x2 = np.clip((cx + bw / 2) * orig_w, 0, orig_w)
    y2 = np.clip((cy + bh / 2) * orig_h, 0, orig_h)
    corners = np.stack([x1, y1, x2, y2], axis=1)
    score_mask = confidences > score_threshold
    return corners[score_mask].tolist(), confidences[score_mask]


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def nms(bboxes, scores, iou_thresh):
    order = list(np.argsort(scores)[::-1])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(bboxes[i], bboxes[j]) < iou_thresh]
    return keep


def detect_text_bubbles(detector: OnnxPanelDetector, image_bytes: bytes):
    detector._load_model()
    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    gray_np = np.array(original.convert("L").convert("RGB"))
    orig_h, orig_w = gray_np.shape[:2]
    pixel_values, pixel_mask = _preprocess_image(gray_np)
    class_scores, boxes = detector._session.run(
        None, {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
    )
    bboxes, scores = postprocess_class(
        class_scores, boxes, orig_h, orig_w, _TEXT_CLASS_IDX, _SCORE_THRESHOLD
    )
    if not bboxes:
        return original, []
    keep = nms(bboxes, scores, _NMS_IOU_THRESHOLD)
    return original, [bboxes[i] for i in keep]


_KATA_LO, _KATA_HI = "ァ", "ヿ"


def is_pure_katakana(s: str) -> bool:
    return bool(s) and all(_KATA_LO <= c <= _KATA_HI or c in "ー－ｰ" for c in s)


def candidates_from_lines(tagger, lines, freq_threshold, freq_fn):
    """Yield candidate dicts: {word, lemma, surface, reading, sentence}.

    `word` is the dictionary form (lemma) so Anki cards show e.g. 負う not 負わ.
    `surface` is the conjugated form actually appearing in the sentence — used
    for bolding the right span. The validation step still checks
    surface ⊂ sentence (lemma may not be in the sentence at all).
    """
    seen = set()
    for line in lines:
        for tok in tagger(line):
            feature = tok.feature
            pos = feature.pos1
            if pos in DROP_POS:
                continue
            if (feature.pos2 or "") == "固有名詞":
                continue
            surface = tok.surface
            lemma = feature.lemma or surface
            # Drop single-char surfaces (almost always stray particles / fragments)
            if len(surface) == 1:
                continue
            # Drop pure ASCII / punctuation
            if not any("　" <= c <= "鿿" or "＀" <= c <= "￯" for c in surface):
                continue
            # Drop short pure-katakana — usually OCR fragments or names
            if is_pure_katakana(surface) and len(surface) <= 3:
                continue
            # Common-word filter: skip if EITHER the surface or the lemma is
            # in the most-common bucket. Catches both 居る (low-freq kanji
            # lemma) and いる (high-freq actual surface).
            if max(freq_fn(surface), freq_fn(lemma)) >= freq_threshold:
                continue
            if lemma in seen:
                continue
            seen.add(lemma)
            reading_kata = feature.kana or ""
            yield {"word": lemma, "lemma": lemma, "surface": surface,
                   "reading_kata": reading_kata, "sentence": line}


def is_hiragana(s: str) -> bool:
    return bool(s) and all(_HIRA_LO <= c <= _HIRA_HI or c in "ー" for c in s)


def validate_card(card):
    """Validate that the conjugated `surface` form is in the sentence
    (the dictionary `word` may not be — that's expected for verbs/adjectives)."""
    sent_clean = re.sub(r"</?b>", "", card["sentence"])
    nfkc_surface = unicodedata.normalize("NFKC", card["surface"])
    nfkc_sent = unicodedata.normalize("NFKC", sent_clean)
    if nfkc_surface not in nfkc_sent:
        return "surface_not_in_sentence"
    if not is_hiragana(card["reading"]):
        return "reading_not_hiragana"
    if not card["translation"].strip() or not any(c.isascii() and c.isalpha() for c in card["translation"]):
        return "empty_or_non_english_translation"
    return None


def bold_word_in_sentence(sentence: str, surface: str) -> str:
    return re.sub(re.escape(surface), f"<b>{surface}</b>", sentence, count=1)


def page_transcript_block(ocr_per_panel: list[list[str]]) -> str:
    parts = []
    for i, lines in enumerate(ocr_per_panel):
        joined = " / ".join(lines) if lines else "(no text)"
        parts.append(f"Panel {i + 1}: {joined}")
    return "\n".join(parts)


TRANSLATION_PROMPT = """Translate Japanese sentences from a manga page to English.

## PAGE TRANSCRIPT (context — for resolving pronouns, subjects, tone)
{transcript}

## SENTENCES TO TRANSLATE
{numbered_pairs}

For each, return a natural English translation of the sentence, with the English equivalent of the marked word wrapped in <b>...</b>. Output JSON {{"items": [{{"translation": "..."}}, ...]}} — same length and order as the input list."""


async def translate_batch(llm_structured, transcript: str, candidates: list[dict]):
    if not candidates:
        return []
    pairs = "\n".join(
        f"{i + 1}. word=「{c['word']}」 (form in sentence: 「{c['surface']}」), sentence=「{c['sentence']}」"
        for i, c in enumerate(candidates)
    )
    prompt = TRANSLATION_PROMPT.format(transcript=transcript, numbered_pairs=pairs)
    from langchain_core.messages import HumanMessage
    msg = HumanMessage(content=prompt)
    result = await llm_structured.ainvoke([msg])
    return [item.translation for item in result.items]


async def process_image(
    img_path: Path, detector: OnnxPanelDetector, mocr, llm_structured,
    tagger, freq_fn, freq_threshold: float, out_dir: Path,
):
    image_bytes = img_path.read_bytes()

    t_detect = time.time()
    page = detector.detect(image_bytes)
    original, raw_text_bboxes = detect_text_bubbles(detector, image_bytes)
    t_detect = time.time() - t_detect
    sorted_panel_bboxes = [p.bbox for p in page.panels]

    if raw_text_bboxes:
        text_to_panel = get_text_to_panel_mapping(raw_text_bboxes, sorted_panel_bboxes)
        text_order = sort_text_boxes_in_reading_order(raw_text_bboxes, sorted_panel_bboxes)
        ordered = [(raw_text_bboxes[i], text_to_panel[i]) for i in text_order]
    else:
        ordered = []

    bubbles_by_panel: dict[int, list[list[float]]] = {p.index: [] for p in page.panels}
    for bbox, p_idx in ordered:
        if 0 <= p_idx < len(page.panels):
            bubbles_by_panel[p_idx].append(bbox)

    t_ocr = time.time()
    ocr_per_panel: list[list[str]] = []
    for panel in page.panels:
        lines = []
        for bbox in bubbles_by_panel[panel.index]:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            bubble = original.crop((x1, y1, x2, y2)).convert("RGB")
            lines.append(mocr(bubble))
        ocr_per_panel.append(lines)
    t_ocr = time.time() - t_ocr

    t_filter = time.time()
    all_candidates = []
    for lines in ocr_per_panel:
        for cand in candidates_from_lines(tagger, lines, freq_threshold, freq_fn):
            all_candidates.append(cand)
    # dedup across the page on lemma
    seen_lemmas = set()
    unique_candidates = []
    for c in all_candidates:
        if c["lemma"] in seen_lemmas:
            continue
        seen_lemmas.add(c["lemma"])
        unique_candidates.append(c)
    t_filter = time.time() - t_filter

    t_translate = time.time()
    transcript = page_transcript_block(ocr_per_panel)
    error = None
    try:
        translations = await translate_batch(llm_structured, transcript, unique_candidates)
        if len(translations) != len(unique_candidates):
            translations = (translations + [""] * len(unique_candidates))[:len(unique_candidates)]
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        translations = [""] * len(unique_candidates)
    t_translate = time.time() - t_translate

    import jaconv
    cards = []
    drop_counts = {"surface_not_in_sentence": 0, "reading_not_hiragana": 0,
                   "empty_or_non_english_translation": 0}
    drop_examples = []
    for cand, en in zip(unique_candidates, translations):
        card = {
            "word": cand["word"],
            "lemma": cand["lemma"],
            "surface": cand["surface"],
            "reading": jaconv.kata2hira(cand["reading_kata"]),
            "sentence": bold_word_in_sentence(cand["sentence"], cand["surface"]),
            "translation": en,
        }
        reason = validate_card(card)
        if reason:
            drop_counts[reason] += 1
            if len(drop_examples) < 5:
                drop_examples.append({"reason": reason, "card": card})
        else:
            cards.append(card)

    record = {
        "image": img_path.name,
        "panels_detected": len(page.panels),
        "text_bubbles_detected": len(ordered),
        "ocr_per_panel": ocr_per_panel,
        "candidates_after_filter": len(unique_candidates),
        "card_count": len(cards),
        "cards": cards,
        "dropped": drop_counts,
        "dropped_examples": drop_examples,
        "timing_s": {
            "detect": round(t_detect, 2),
            "ocr": round(t_ocr, 2),
            "filter": round(t_filter, 2),
            "translate": round(t_translate, 2),
        },
        "error": error,
    }

    img_dir = out_dir / img_path.stem
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / "result.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    status = "FAIL" if error else "OK  "
    suffix = f"  error={error[:60]}" if error else ""
    total = t_detect + t_ocr + t_filter + t_translate
    print(
        f"  {status} {img_path.name}  panels={len(page.panels)} bubbles={len(ordered)}  "
        f"detect={t_detect:.1f}s ocr={t_ocr:.1f}s translate={t_translate:.1f}s "
        f"total={total:.1f}s  cands={len(unique_candidates)} cards={len(cards)} "
        f"dropped={sum(drop_counts.values())}{suffix}"
    )
    return record


async def main_async(args: argparse.Namespace) -> None:
    api_key = os.environ.get(args.api_key_env, "not-needed")

    images = sorted(args.images_dir.glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not images:
        print(f"No images in {args.images_dir}")
        sys.exit(1)

    out_dir = OUTPUT_DIR / f"v3_{slug(args.model)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading manga-ocr...")
    from manga_ocr import MangaOcr
    mocr = MangaOcr()
    print("manga-ocr ready.")

    print("Loading fugashi + wordfreq...")
    import fugashi
    from wordfreq import zipf_frequency
    tagger = fugashi.Tagger()
    freq_fn = lambda w: zipf_frequency(w, "ja")
    print("Tokenizer ready.")

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        max_tokens=args.max_tokens,
    )
    llm_structured = llm.with_structured_output(Translations)
    print(f"LLM ready: {args.model} @ {args.base_url}")

    detector = OnnxPanelDetector(model_path=args.panel_model_path)
    print(f"Output: {out_dir}\n")

    records = []
    for img in images:
        rec = await process_image(
            img, detector, mocr, llm_structured, tagger, freq_fn,
            args.freq_threshold, out_dir,
        )
        records.append(rec)

    summary = {
        "model": args.model,
        "freq_threshold": args.freq_threshold,
        "total_images": len(records),
        "total_cards": sum(r["card_count"] for r in records),
        "total_translate_s": round(sum(r["timing_s"]["translate"] for r in records), 2),
        "total_ocr_s": round(sum(r["timing_s"]["ocr"] for r in records), 2),
        "images": records,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSummary: {out_dir / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 pipeline: manga-ocr + fugashi + LLM translate")
    parser.add_argument("--model", default="local")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--api-key-env", default="DUMMY")
    parser.add_argument("--images-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--panel-model-path", default="models/panel_detector.onnx")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--freq-threshold", type=float, default=5.0,
                        help="Drop words whose Zipf frequency >= this (default 5.0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
