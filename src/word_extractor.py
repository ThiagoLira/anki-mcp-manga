"""Deterministic word-extraction pipeline.

Detects panels + text bubbles, OCRs each bubble with manga-ocr, tokenizes
with fugashi, filters with wordfreq. No LLM calls — produces a candidate
list the user can review before any translation work happens.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .panel_detector import (
    OnnxPanelDetector,
    _preprocess_image,
    get_text_to_panel_mapping,
    sort_text_boxes_in_reading_order,
)

logger = logging.getLogger(__name__)

_TEXT_CLASS_IDX = 1
# MagiV2 author's default for text is 0.3; we sit slightly below it for recall.
# Going lower (0.1) flooded manga-ocr with false positives — and manga-ocr is
# documented to hallucinate Japanese on any image, even empty ones, so each
# false-positive bubble produced garbage text and duplicate fragments.
_SCORE_THRESHOLD = 0.2
_NMS_IOU_THRESHOLD = 0.5
# Drop bubble detections smaller than this fraction of the page area —
# tiny boxes are almost always spurious and manga-ocr will hallucinate on them.
_MIN_BUBBLE_AREA_FRAC = 0.003

_HIRA_LO, _HIRA_HI = "ぁ", "ゟ"
_KATA_LO, _KATA_HI = "ァ", "ヿ"

# fugashi POS tags to drop entirely
DROP_POS = {"助詞", "助動詞", "記号", "補助記号", "感動詞", "代名詞",
            "接頭辞", "接尾辞", "数詞", "空白"}


@dataclass
class WordCandidate:
    word: str           # lemma (dictionary form) — what the user learns
    surface: str        # form actually appearing in the sentence — what gets bolded
    reading: str        # hiragana reading
    sentence: str       # raw OCR'd Japanese sentence containing the surface
    panel_index: int    # 0-based panel this came from
    panel_image: bytes  # cropped panel image, attached to the eventual card
    translation: str = ""  # optional pre-baked English gloss (Explain Page path)


@dataclass
class CandidateExtraction:
    candidates: list[WordCandidate]
    ocr_per_panel: list[list[str]]  # full transcript, used for translation context
    panel_images: list[bytes] = field(default_factory=list)  # WebP bytes, aligned with ocr_per_panel


# ---------------------------------------------------------------------------
# ONNX text-bbox extraction (reuses the panel detector's session)
# ---------------------------------------------------------------------------

def _postprocess_class(class_scores, boxes, orig_h, orig_w, class_idx, score_threshold):
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


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms(bboxes, scores, iou_thresh):
    order = list(np.argsort(scores)[::-1])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _iou(bboxes[i], bboxes[j]) < iou_thresh]
    return keep


def _detect_text_bubbles(detector: OnnxPanelDetector, image_bytes: bytes):
    """Run ONNX inference and return the original PIL image plus deduped text bboxes."""
    detector._load_model()
    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    gray_np = np.array(original.convert("L").convert("RGB"))
    orig_h, orig_w = gray_np.shape[:2]
    pixel_values, pixel_mask = _preprocess_image(gray_np)
    class_scores, boxes = detector._session.run(
        None, {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
    )
    bboxes, scores = _postprocess_class(
        class_scores, boxes, orig_h, orig_w, _TEXT_CLASS_IDX, _SCORE_THRESHOLD
    )
    if not bboxes:
        logger.info("_detect_text_bubbles: 0 bubbles above threshold %.2f", _SCORE_THRESHOLD)
        return original, []
    page_area = float(orig_h) * float(orig_w)
    min_area = page_area * _MIN_BUBBLE_AREA_FRAC
    area_keep = [
        i for i, b in enumerate(bboxes)
        if (b[2] - b[0]) * (b[3] - b[1]) >= min_area
    ]
    dropped_tiny = len(bboxes) - len(area_keep)
    bboxes = [bboxes[i] for i in area_keep]
    scores = scores[area_keep] if len(scores) else scores
    if not bboxes:
        logger.info(
            "_detect_text_bubbles: %d bubbles dropped as too small (<%.1f%% of page)",
            dropped_tiny, _MIN_BUBBLE_AREA_FRAC * 100,
        )
        return original, []
    keep = _nms(bboxes, scores, _NMS_IOU_THRESHOLD)
    kept_scores = sorted((float(scores[i]) for i in keep), reverse=True)
    logger.info(
        "_detect_text_bubbles: %d above threshold %.2f, %d dropped tiny, %d after NMS, scores=[%s]",
        len(bboxes) + dropped_tiny, _SCORE_THRESHOLD, dropped_tiny, len(keep),
        ", ".join(f"{s:.2f}" for s in kept_scores),
    )
    return original, [bboxes[i] for i in keep]


# ---------------------------------------------------------------------------
# Tokenization filters
# ---------------------------------------------------------------------------

def _is_pure_katakana(s: str) -> bool:
    return bool(s) and all(_KATA_LO <= c <= _KATA_HI or c in "ー－ｰ" for c in s)


def _has_japanese_chars(s: str) -> bool:
    return any("　" <= c <= "鿿" or "＀" <= c <= "￯" for c in s)


def _dedup_substring_lines(lines: list[str]) -> list[str]:
    """Drop OCR lines that are substrings of another line in the same panel.

    manga-ocr hallucinates on bad crops and the model can also produce
    multiple partial reads of the same bubble when our detector emits
    overlapping boxes. Both cases show up as one line being a prefix or
    substring of another — keep the longest, drop the fragments.
    """
    cleaned = [s.strip() for s in lines if s and s.strip()]
    if len(cleaned) <= 1:
        return cleaned
    # Sort longest-first so we keep maximal lines and drop their fragments
    ordered = sorted(cleaned, key=len, reverse=True)
    kept: list[str] = []
    for line in ordered:
        if any(line in k for k in kept):
            continue
        kept.append(line)
    # Preserve original (reading) order
    seen: set[str] = set()
    out: list[str] = []
    for line in cleaned:
        if line in kept and line not in seen:
            out.append(line)
            seen.add(line)
    return out


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class WordExtractor:
    """Run the deterministic pipeline. All heavy state is lazy-loaded."""

    def __init__(
        self,
        panel_detector: OnnxPanelDetector | None = None,
        freq_threshold: float = 5.0,
    ):
        self._detector = panel_detector
        self._freq_threshold = freq_threshold
        self._mocr = None
        self._tagger = None
        self._kata2hira = None
        self._zipf = None

    def _ensure_loaded(self):
        if self._detector is None:
            from .config import settings
            onnx_path = Path(settings.panel_model_path)
            if not onnx_path.exists():
                raise RuntimeError(
                    f"ONNX panel detector model not found at {onnx_path}; "
                    "WordExtractor currently requires the ONNX path."
                )
            self._detector = OnnxPanelDetector(model_path=str(onnx_path))
        if self._mocr is None:
            from manga_ocr import MangaOcr
            logger.info("Loading manga-ocr (first call may take a while)...")
            self._mocr = MangaOcr()
            logger.info("manga-ocr ready.")
        if self._tagger is None:
            import fugashi
            self._tagger = fugashi.Tagger()
        if self._kata2hira is None:
            import jaconv
            self._kata2hira = jaconv.kata2hira
        if self._zipf is None:
            from wordfreq import zipf_frequency
            self._zipf = zipf_frequency

    def extract(self, image_bytes: bytes) -> CandidateExtraction:
        self._ensure_loaded()

        page = self._detector.detect(image_bytes)
        original, raw_text_bboxes = _detect_text_bubbles(self._detector, image_bytes)
        sorted_panel_bboxes = [p.bbox for p in page.panels]

        if raw_text_bboxes:
            text_to_panel = get_text_to_panel_mapping(raw_text_bboxes, sorted_panel_bboxes)
            text_order = sort_text_boxes_in_reading_order(raw_text_bboxes, sorted_panel_bboxes)
            ordered = [(raw_text_bboxes[i], text_to_panel[i]) for i in text_order]
        else:
            ordered = []

        bubbles_by_panel: dict[int, list] = {p.index: [] for p in page.panels}
        for bbox, p_idx in ordered:
            if 0 <= p_idx < len(page.panels):
                bubbles_by_panel[p_idx].append(bbox)
        logger.info(
            "bubbles per panel: %s",
            {idx: len(bubs) for idx, bubs in bubbles_by_panel.items()},
        )

        # OCR each bubble, in panel reading order
        ocr_per_panel: list[list[str]] = []
        panel_images: list[bytes] = []
        sentence_panels: list[tuple[int, bytes, str]] = []  # (panel_idx, panel_image, sentence)
        for panel in page.panels:
            raw_lines: list[str] = []
            for bbox in bubbles_by_panel[panel.index]:
                x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                bubble = original.crop((x1, y1, x2, y2)).convert("RGB")
                raw_lines.append(self._mocr(bubble))
            lines = _dedup_substring_lines(raw_lines)
            if len(raw_lines) != len(lines):
                logger.info(
                    "panel %d: %d OCR lines → %d after substring dedup",
                    panel.index, len(raw_lines), len(lines),
                )
            for line in lines:
                sentence_panels.append((panel.index, panel.image_bytes, line))
            ocr_per_panel.append(lines)
            panel_images.append(panel.image_bytes)

        candidates = self._tokenize_and_filter(sentence_panels)
        logger.info(
            "WordExtractor: %d panels, %d bubbles, %d candidate words",
            len(page.panels), len(ordered), len(candidates),
        )
        return CandidateExtraction(
            candidates=candidates,
            ocr_per_panel=ocr_per_panel,
            panel_images=panel_images,
        )

    def _tokenize_and_filter(self, sentence_panels) -> list[WordCandidate]:
        seen_lemmas: set[str] = set()
        out: list[WordCandidate] = []
        for panel_idx, panel_image, sentence in sentence_panels:
            for tok in self._tagger(sentence):
                feature = tok.feature
                if feature.pos1 in DROP_POS:
                    continue
                if (feature.pos2 or "") == "固有名詞":
                    continue
                surface = tok.surface
                lemma = feature.lemma or surface
                if len(surface) == 1:
                    continue
                if not _has_japanese_chars(surface):
                    continue
                if _is_pure_katakana(surface) and len(surface) <= 3:
                    continue
                if max(self._zipf(surface, "ja"), self._zipf(lemma, "ja")) >= self._freq_threshold:
                    continue
                # Strip fugashi's quirky proper-noun lemma suffix like "ティーチ-teacch"
                lemma_clean = re.sub(r"-[a-zA-Z]+$", "", lemma)
                if lemma_clean in seen_lemmas:
                    continue
                seen_lemmas.add(lemma_clean)
                reading = self._kata2hira(feature.kana or "")
                out.append(WordCandidate(
                    word=lemma_clean,
                    surface=surface,
                    reading=reading,
                    sentence=sentence,
                    panel_index=panel_idx,
                    panel_image=panel_image,
                ))
        return out
