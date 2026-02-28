#!/usr/bin/env python3
"""Compare panel detection outputs: full MagiV2 (PyTorch) vs ONNX export.

Runs both backends on every image in test_manga_images/ and verifies that
the ONNX path produces identical panel bounding boxes.

Usage:
    python scripts/test_onnx_vs_pytorch.py [--model models/panel_detector.onnx]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.panel_detector import (
    PanelDetector,
    OnnxPanelDetector,
    _convert_to_list_of_lists,
)


def _bbox_lists_close(
    a: list[list[float]], b: list[list[float]], atol: float = 2.0
) -> tuple[bool, str]:
    """Check if two lists of bounding boxes match within tolerance.

    Allows small coordinate differences (atol pixels) and reorders by
    centre-y then centre-x to handle different ordering.
    """
    if len(a) != len(b):
        return False, f"count mismatch: {len(a)} vs {len(b)}"

    if len(a) == 0:
        return True, "both empty"

    def sort_key(box):
        cy = (box[1] + box[3]) / 2
        cx = (box[0] + box[2]) / 2
        return (round(cy, -1), round(cx, -1))

    a_sorted = sorted(a, key=sort_key)
    b_sorted = sorted(b, key=sort_key)

    max_diff = 0.0
    for ba, bb in zip(a_sorted, b_sorted):
        diff = max(abs(x - y) for x, y in zip(ba, bb))
        max_diff = max(max_diff, diff)

    if max_diff <= atol:
        return True, f"max diff {max_diff:.1f}px (tol {atol}px)"
    return False, f"max diff {max_diff:.1f}px EXCEEDS tolerance {atol}px"


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX vs PyTorch panel comparison")
    parser.add_argument(
        "--model",
        default="models/panel_detector.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--images",
        default="test_manga_images",
        help="Directory with test images",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=3.0,
        help="Absolute tolerance in pixels (default: 3.0)",
    )
    args = parser.parse_args()

    image_dir = Path(args.images)
    images = sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.webp"))
    )
    if not images:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    print(f"Found {len(images)} test image(s)")
    print()

    # --- Load both detectors ---
    print("Loading PyTorch (full MagiV2)...")
    pt_detector = PanelDetector(device="cpu")
    print("Loading ONNX model...")
    onnx_detector = OnnxPanelDetector(model_path=args.model)

    all_passed = True
    for img_path in images:
        print(f"\n{'='*60}")
        print(f"Image: {img_path.name}")
        print(f"{'='*60}")

        image_bytes = img_path.read_bytes()

        # Run PyTorch
        pt_result = pt_detector.detect(image_bytes)
        pt_bboxes = [p.bbox for p in pt_result.panels]
        print(f"  PyTorch: {len(pt_bboxes)} panels")
        for i, b in enumerate(pt_bboxes):
            print(f"    [{i}] {[round(x, 1) for x in b]}")

        # Run ONNX
        onnx_result = onnx_detector.detect(image_bytes)
        onnx_bboxes = [p.bbox for p in onnx_result.panels]
        print(f"  ONNX:    {len(onnx_bboxes)} panels")
        for i, b in enumerate(onnx_bboxes):
            print(f"    [{i}] {[round(x, 1) for x in b]}")

        # Compare
        ok, msg = _bbox_lists_close(pt_bboxes, onnx_bboxes, atol=args.atol)
        status = "PASS" if ok else "FAIL"
        print(f"  Result:  {status} — {msg}")
        if not ok:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
