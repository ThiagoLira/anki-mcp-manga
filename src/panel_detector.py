"""Panel detection and reading-order sorting for manga pages using MagiV2.

Ported from MangaWhisperer (image_processor.py + utils.py).
"""

from __future__ import annotations

import io
import logging
from copy import deepcopy
from dataclasses import dataclass
from itertools import groupby

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Panel:
    index: int  # 0-based, in reading order
    bbox: list[float]  # [x1, y1, x2, y2]
    image_bytes: bytes  # cropped panel as WebP


@dataclass
class PageAnalysis:
    panels: list[Panel]
    annotated_image: bytes  # full page with panel numbers drawn


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _convert_to_list_of_lists(rects):
    """Normalise rects from tensors / ndarrays to plain lists."""
    try:
        import torch
        if isinstance(rects, torch.Tensor):
            return rects.tolist()
    except ImportError:
        pass
    if isinstance(rects, np.ndarray):
        return rects.tolist()
    return [[a, b, c, d] for a, b, c, d in rects]


def _box(x1, y1, x2, y2):
    """Create a Shapely box (lazy import)."""
    from shapely.geometry import box
    return box(x1, y1, x2, y2)


def _point(x, y):
    from shapely.geometry import Point
    return Point(x, y)


def erode_rectangle(bbox: list[float], erosion_factor: float) -> list[float]:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    if w < h:
        aspect_ratio = w / h
        ef_w = erosion_factor * aspect_ratio
        ef_h = erosion_factor
    else:
        aspect_ratio = h / w
        ef_w = erosion_factor
        ef_h = erosion_factor * aspect_ratio
    w = w - w * ef_w
    h = h - h * ef_h
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def intersects(rectA: list[float], rectB: list[float]) -> bool:
    return _box(*rectA).intersects(_box(*rectB))


def _get_distance(rectA: list[float], rectB: list[float]) -> float:
    return _box(*rectA).distance(_box(*rectB))


def merge_overlapping_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda x: x[0])
    merged: list[tuple[float, float]] = []
    prev_x1, prev_x2 = ranges[0]
    for x1, x2 in ranges[1:]:
        if x1 > prev_x2:
            merged.append((prev_x1, prev_x2))
            prev_x1, prev_x2 = x1, x2
        else:
            prev_x2 = max(prev_x2, x2)
    merged.append((prev_x1, prev_x2))
    return merged


# ---------------------------------------------------------------------------
# Directional checks
# ---------------------------------------------------------------------------

def _is_strictly_above(rectA, rectB):
    return rectA[3] < rectB[1]


def _is_strictly_below(rectA, rectB):
    return rectB[3] < rectA[1]


def _is_strictly_left_of(rectA, rectB):
    return rectA[2] < rectB[0]


def _is_strictly_right_of(rectA, rectB):
    return rectB[2] < rectA[0]


# ---------------------------------------------------------------------------
# Edge determination for reading-order graph
# ---------------------------------------------------------------------------

def _use_cuts_to_determine_edge(a: int, b: int, rects: list[list[float]]):
    rects = deepcopy(rects)
    while True:
        xmin = min(rects[a][0], rects[b][0])
        ymin = min(rects[a][1], rects[b][1])
        xmax = max(rects[a][2], rects[b][2])
        ymax = max(rects[a][3], rects[b][3])
        rect_index = [
            i for i in range(len(rects))
            if intersects(rects[i], [xmin, ymin, xmax, ymax])
        ]
        rects_copy = [
            rects[i] for i in range(len(rects))
            if intersects(rects[i], [xmin, ymin, xmax, ymax])
        ]

        # Try horizontal split
        overlapping_y = merge_overlapping_ranges(
            [(y1, y2) for _, y1, _, y2 in rects_copy]
        )
        panel_to_split: dict[int, int] = {}
        for split_idx, (y1, y2) in enumerate(overlapping_y):
            for i, idx in enumerate(rect_index):
                if y1 <= rects_copy[i][1] <= rects_copy[i][3] <= y2:
                    panel_to_split[idx] = split_idx
        if panel_to_split.get(a) is not None and panel_to_split.get(b) is not None:
            if panel_to_split[a] != panel_to_split[b]:
                return panel_to_split[a] < panel_to_split[b]

        # Try vertical split
        overlapping_x = merge_overlapping_ranges(
            [(x1, x2) for x1, _, x2, _ in rects_copy]
        )
        panel_to_split = {}
        for split_idx, (x1, x2) in enumerate(overlapping_x[::-1]):
            for i, idx in enumerate(rect_index):
                if x1 <= rects_copy[i][0] <= rects_copy[i][2] <= x2:
                    panel_to_split[idx] = split_idx
        if panel_to_split.get(a) is not None and panel_to_split.get(b) is not None:
            if panel_to_split[a] != panel_to_split[b]:
                return panel_to_split[a] < panel_to_split[b]

        # Erode and retry
        rects = [erode_rectangle(r, 0.05) for r in rects]


def is_there_a_directed_edge(a: int, b: int, rects: list[list[float]]):
    """Return truthy if panel a should come before panel b in reading order."""
    rectA, rectB = rects[a], rects[b]
    centre_a = [rectA[0] + (rectA[2] - rectA[0]) / 2, rectA[1] + (rectA[3] - rectA[1]) / 2]
    centre_b = [rectB[0] + (rectB[2] - rectB[0]) / 2, rectB[1] + (rectB[3] - rectB[1]) / 2]
    if np.allclose(np.array(centre_a), np.array(centre_b)):
        return _box(*rectA).area > _box(*rectB).area

    copy_a = list(rectA)
    copy_b = list(rectB)
    while True:
        if _is_strictly_above(copy_a, copy_b) and not _is_strictly_left_of(copy_a, copy_b):
            return 1
        if _is_strictly_above(copy_b, copy_a) and not _is_strictly_left_of(copy_b, copy_a):
            return 0
        if _is_strictly_right_of(copy_a, copy_b) and not _is_strictly_below(copy_a, copy_b):
            return 1
        if _is_strictly_right_of(copy_b, copy_a) and not _is_strictly_below(copy_b, copy_a):
            return 0
        if _is_strictly_below(copy_a, copy_b) and _is_strictly_right_of(copy_a, copy_b):
            return _use_cuts_to_determine_edge(a, b, rects)
        if _is_strictly_below(copy_b, copy_a) and _is_strictly_right_of(copy_b, copy_a):
            return _use_cuts_to_determine_edge(a, b, rects)
        # Overlapping — erode and retry
        copy_a = erode_rectangle(copy_a, 0.05)
        copy_b = erode_rectangle(copy_b, 0.05)


# ---------------------------------------------------------------------------
# Panel sorting (graph-based topological sort)
# ---------------------------------------------------------------------------

def sort_panels(rects) -> list[int]:
    """Return panel indices sorted in manga reading order (RTL, top-to-bottom)."""
    import networkx as nx

    before_rects = _convert_to_list_of_lists(rects)
    eroded = [erode_rectangle(r, 0.05) for r in before_rects]

    G = nx.DiGraph()
    G.add_nodes_from(range(len(eroded)))
    for i in range(len(eroded)):
        for j in range(len(eroded)):
            if i == j:
                continue
            if is_there_a_directed_edge(i, j, eroded):
                G.add_edge(i, j, weight=_get_distance(eroded[i], eroded[j]))
            else:
                G.add_edge(j, i, weight=_get_distance(eroded[i], eroded[j]))

    # Remove cycles by breaking highest-weight edge
    while True:
        cycles = sorted(nx.simple_cycles(G))
        cycles = [c for c in cycles if len(c) > 1]
        if not cycles:
            break
        cycle = cycles[0]
        edges = list(zip(cycle, cycle[1:] + cycle[:1]))
        worst = max(edges, key=lambda e: G.edges[e]["weight"])
        G.remove_edge(*worst)

    return list(nx.topological_sort(G))


# ---------------------------------------------------------------------------
# Text-box ordering
# ---------------------------------------------------------------------------

def get_text_to_panel_mapping(
    text_bboxes: list[list[float]], sorted_panel_bboxes: list[list[float]]
) -> list[int]:
    mapping: list[int] = []
    for tb in text_bboxes:
        text_poly = _box(*tb)
        intersections: list[tuple[float, int]] = []
        distances: list[tuple[float, int]] = []
        if not sorted_panel_bboxes:
            mapping.append(-1)
            continue
        for j, pb in enumerate(sorted_panel_bboxes):
            panel_poly = _box(*pb)
            if text_poly.intersects(panel_poly):
                intersections.append((text_poly.intersection(panel_poly).area, j))
            distances.append((text_poly.distance(panel_poly), j))
        if not intersections:
            mapping.append(min(distances, key=lambda x: x[0])[1])
        else:
            mapping.append(max(intersections, key=lambda x: x[0])[1])
    return mapping


def sort_texts_within_panel(rects: list[list[float]]) -> list[int]:
    """Sort text boxes by distance to top-right reference point."""
    smallest_y = float("inf")
    greatest_x = float("-inf")
    for x1, y1, x2, y2 in rects:
        smallest_y = min(smallest_y, y1)
        greatest_x = max(greatest_x, x2)
    ref = _point(greatest_x, smallest_y)
    indexed = [(_box(*r), i) for i, r in enumerate(rects)]
    indexed.sort(key=lambda x: ref.distance(x[0]))
    return [i for _, i in indexed]


def sort_text_boxes_in_reading_order(
    text_bboxes, sorted_panel_bboxes
) -> list[int]:
    text_bboxes = _convert_to_list_of_lists(text_bboxes)
    sorted_panel_bboxes = _convert_to_list_of_lists(sorted_panel_bboxes)
    if not text_bboxes:
        return []

    panel_ids = get_text_to_panel_mapping(text_bboxes, sorted_panel_bboxes)
    indices = list(range(len(text_bboxes)))
    # Sort by panel id
    pairs = sorted(zip(indices, panel_ids), key=lambda x: x[1])
    indices = [p[0] for p in pairs]
    sorted_panel_ids = [p[1] for p in pairs]

    # Group by panel and sort within each panel
    groups = [list(g) for _, g in groupby(range(len(indices)), key=lambda i: sorted_panel_ids[i])]
    for group in groups:
        subset_indices = [indices[i] for i in group]
        subset_bboxes = [text_bboxes[i] for i in subset_indices]
        within_order = sort_texts_within_panel(subset_bboxes)
        for gi, wi in zip(group, within_order):
            indices[gi] = subset_indices[wi]

    return indices


# ---------------------------------------------------------------------------
# Annotation drawing
# ---------------------------------------------------------------------------

_CIRCLED_NUMBERS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def _annotate_page(image: Image.Image, panel_bboxes: list[list[float]]) -> Image.Image:
    """Draw circled panel numbers on the image."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    # Scale font to ~4% of image height
    font_size = max(24, int(image.height * 0.04))
    try:
        font = ImageFont.truetype("/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    for i, (x1, y1, x2, y2) in enumerate(panel_bboxes):
        label = _CIRCLED_NUMBERS[i] if i < len(_CIRCLED_NUMBERS) else str(i + 1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Draw background circle
        r = font_size * 0.7
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0, 200))
        # Draw text centered
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw / 2, cy - th / 2), label, fill="white", font=font)

    return annotated


# ---------------------------------------------------------------------------
# PanelDetector — main entry point
# ---------------------------------------------------------------------------

def _build_page_analysis(
    image_np: np.ndarray, raw_panels: list[list[float]]
) -> PageAnalysis:
    """Sort panels, crop, annotate — shared by both PyTorch and ONNX paths."""
    if not raw_panels:
        logger.warning("No panels detected, returning full page as single panel.")
        h, w = image_np.shape[:2]
        raw_panels = [[0, 0, w, h]]

    if len(raw_panels) == 1:
        order = [0]
    else:
        order = sort_panels(raw_panels)

    sorted_bboxes = [raw_panels[i] for i in order]

    panels: list[Panel] = []
    pil_image = Image.fromarray(image_np)
    for idx, bbox in enumerate(sorted_bboxes):
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cropped = pil_image.crop((x1, y1, x2, y2)).convert("RGB")
        buf = io.BytesIO()
        cropped.save(buf, format="WebP", quality=85)
        panels.append(Panel(index=idx, bbox=bbox, image_bytes=buf.getvalue()))

    annotated = _annotate_page(pil_image, sorted_bboxes)
    ann_buf = io.BytesIO()
    annotated.save(ann_buf, format="WebP", quality=85)

    return PageAnalysis(panels=panels, annotated_image=ann_buf.getvalue())


class PanelDetector:
    """Lazy-loading MagiV2 wrapper for manga panel detection (PyTorch)."""

    def __init__(self, device: str = "cuda"):
        self._device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel

        logger.info("Loading MagiV2 model on %s...", self._device)
        self._model = (
            AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True)
            .to(self._device)
            .eval()
        )
        logger.info("MagiV2 model loaded.")

    def detect(self, image_bytes: bytes) -> PageAnalysis:
        """Detect panels, sort in reading order, crop, and annotate."""
        import torch

        self._load_model()

        # Load image as numpy array (RGB)
        image = Image.open(io.BytesIO(image_bytes)).convert("L").convert("RGB")
        image_np = np.array(image)

        # Run model
        with torch.no_grad():
            results = self._model.predict_detections_and_associations([image_np])

        page = results[0]
        raw_panels = _convert_to_list_of_lists(page["panels"])

        return _build_page_analysis(image_np, raw_panels)


# ---------------------------------------------------------------------------
# ONNX inference path — no PyTorch / transformers required
# ---------------------------------------------------------------------------

# Preprocessing constants (ConditionalDETR / ImageNet defaults)
_RESIZE_SHORTEST = 800
_RESIZE_LONGEST = 1333
_IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Post-processing constants
_PANEL_CLASS_IDX = 2  # MagiV2 class ordering: char=0, text=1, panel=2, tail=3
_SCORE_THRESHOLD = 0.2
_NMS_IOU_THRESHOLD = 0.5


def _preprocess_image(
    image_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize, normalise, and format an image for the ONNX panel detector.

    Args:
        image_np: RGB uint8 array (H, W, 3).

    Returns:
        pixel_values: (1, 3, H', W') float32
        pixel_mask:   (1, H', W') float32
    """
    h, w = image_np.shape[:2]

    # Scale so shortest edge = 800, longest capped at 1333
    short, long = (w, h) if w <= h else (h, w)
    size = _RESIZE_SHORTEST
    if long / short * size > _RESIZE_LONGEST:
        size = int(round(_RESIZE_LONGEST * short / long))

    if w <= h:
        new_w = size
        new_h = int(size * h / w)
    else:
        new_h = size
        new_w = int(size * w / h)

    pil_img = Image.fromarray(image_np).resize((new_w, new_h), Image.BILINEAR)
    img = np.array(pil_img, dtype=np.float32) / 255.0

    # Normalise with ImageNet statistics
    img = (img - _IMAGE_MEAN) / _IMAGE_STD

    # HWC → CHW, add batch dimension
    pixel_values = img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H', W')
    pixel_mask = np.ones((1, new_h, new_w), dtype=np.float32)

    return pixel_values, pixel_mask


def _suppress_overlapping_panels(
    boxes: np.ndarray, scores: np.ndarray, overlap_threshold: float = 0.5
) -> list[int]:
    """Greedy overlap-based panel suppression matching MagiV2 logic.

    Keeps panels whose area is NOT >50% covered by the union of
    already-accepted panels (sorted by score descending).
    """
    order = scores.argsort()[::-1]
    keep: list[int] = []
    # Track union of kept panels as a list of boxes (use shapely for intersection)
    union_x1, union_y1, union_x2, union_y2 = 0.0, 0.0, 0.0, 0.0
    union_poly = _box(0, 0, 0, 0)

    for idx in order:
        bx1, by1, bx2, by2 = boxes[idx]
        panel_poly = _box(float(bx1), float(by1), float(bx2), float(by2))
        panel_area = panel_poly.area
        if panel_area == 0:
            continue
        if union_poly.intersection(panel_poly).area / panel_area > overlap_threshold:
            continue
        keep.append(int(idx))
        union_poly = union_poly.union(panel_poly)

    return keep


def _postprocess_detections(
    class_scores: np.ndarray,
    boxes: np.ndarray,
    orig_h: int,
    orig_w: int,
) -> list[list[float]]:
    """Convert raw model outputs to panel bounding boxes in pixel coords.

    Replicates MagiV2's ``predict_detections_and_associations`` logic:
    argmax class → sigmoid confidence → filter panels → overlap suppression.

    Args:
        class_scores: (1, N, C) raw logits
        boxes:        (1, N, 4) sigmoid values in centre format (cx, cy, w, h)
        orig_h, orig_w: original image dimensions

    Returns:
        List of [x1, y1, x2, y2] bounding boxes.
    """
    logits = class_scores[0]  # (N, C)
    bboxes = boxes[0]  # (N, 4)

    # MagiV2 scoring: argmax across classes, sigmoid on the winning logit
    labels = logits.argmax(axis=-1)  # (N,)
    max_logits = logits.max(axis=-1)  # (N,)
    confidences = 1.0 / (1.0 + np.exp(-max_logits))

    # Keep only detections classified as panel (label == 2)
    panel_mask = labels == _PANEL_CLASS_IDX
    confidences = confidences[panel_mask]
    bboxes = bboxes[panel_mask]

    if len(confidences) == 0:
        return []

    # Centre (cx, cy, w, h) → corner (x1, y1, x2, y2), scaled to pixels
    cx, cy, bw, bh = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x1 = np.clip((cx - bw / 2) * orig_w, 0, orig_w)
    y1 = np.clip((cy - bh / 2) * orig_h, 0, orig_h)
    x2 = np.clip((cx + bw / 2) * orig_w, 0, orig_w)
    y2 = np.clip((cy + bh / 2) * orig_h, 0, orig_h)
    corner_boxes = np.stack([x1, y1, x2, y2], axis=1)

    # Score threshold + overlap-based suppression (matches MagiV2 logic)
    score_mask = confidences > _SCORE_THRESHOLD
    corner_boxes = corner_boxes[score_mask]
    confidences = confidences[score_mask]

    if len(confidences) == 0:
        return []

    keep = _suppress_overlapping_panels(corner_boxes, confidences)
    return corner_boxes[keep].tolist()


class OnnxPanelDetector:
    """ONNX-based panel detector — no PyTorch or transformers required."""

    def __init__(self, model_path: str = "models/panel_detector.onnx"):
        self._model_path = model_path
        self._session = None

    def _load_model(self):
        if self._session is not None:
            return
        import onnxruntime as ort

        logger.info("Loading ONNX panel detector from %s ...", self._model_path)
        self._session = ort.InferenceSession(
            self._model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        logger.info(
            "ONNX panel detector ready (providers: %s)",
            self._session.get_providers(),
        )

    def detect(self, image_bytes: bytes) -> PageAnalysis:
        """Detect panels, sort in reading order, crop, and annotate."""
        self._load_model()

        image = Image.open(io.BytesIO(image_bytes)).convert("L").convert("RGB")
        image_np = np.array(image)
        orig_h, orig_w = image_np.shape[:2]

        pixel_values, pixel_mask = _preprocess_image(image_np)
        class_scores, boxes = self._session.run(
            None,
            {"pixel_values": pixel_values, "pixel_mask": pixel_mask},
        )

        raw_panels = _postprocess_detections(class_scores, boxes, orig_h, orig_w)

        return _build_page_analysis(image_np, raw_panels)
