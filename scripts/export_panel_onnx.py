#!/usr/bin/env python3
"""Export MagiV2 panel detection components to ONNX format.

Downloads the full MagiV2 model (~2GB), extracts only the panel detection
sub-models (~190MB), and exports them to a single ONNX file.

Usage:
    python scripts/export_panel_onnx.py [-o models/panel_detector.onnx]

Requirements:
    pip install torch transformers>=4.44 onnx onnxruntime
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# ---------------------------------------------------------------------------
# Slim wrapper — only the three components needed for panel detection
# ---------------------------------------------------------------------------

class PanelDetectorWrapper(nn.Module):
    """Wraps detection_transformer + class_labels_classifier + bbox_predictor."""

    def __init__(self, full_model: nn.Module):
        super().__init__()
        self.detection_transformer = full_model.detection_transformer
        self.class_labels_classifier = full_model.class_labels_classifier
        self.bbox_predictor = full_model.bbox_predictor
        self.num_non_obj_tokens = full_model.num_non_obj_tokens

    def forward(
        self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.detection_transformer(
            pixel_values=pixel_values, pixel_mask=pixel_mask
        )

        obj_tokens = output.last_hidden_state[:, : -self.num_non_obj_tokens]
        class_scores = self.class_labels_classifier(obj_tokens)

        reference = output.reference_points[: -self.num_non_obj_tokens]
        reference_before_sigmoid = inverse_sigmoid(reference).transpose(0, 1)

        boxes = self.bbox_predictor(obj_tokens)
        boxes[..., :2] += reference_before_sigmoid
        boxes = boxes.sigmoid()

        return class_scores, boxes


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_model_stats(model: nn.Module) -> None:
    panel_names = {"detection_transformer", "class_labels_classifier", "bbox_predictor"}
    total = 0
    panel_total = 0

    print("\n=== Model Architecture ===")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        size_mb = params * 4 / (1024 * 1024)
        total += params
        needed = "KEEP" if name in panel_names else "drop"
        if name in panel_names:
            panel_total += params
        print(f"  {name:40s} {params:>12,} params  ({size_mb:>7.1f} MB)  [{needed}]")

    total_mb = total * 4 / (1024**2)
    panel_mb = panel_total * 4 / (1024**2)
    print(f"  {'TOTAL':40s} {total:>12,} params  ({total_mb:>7.1f} MB)")
    print(f"\n  Panel-only: {panel_total:,} params ({panel_mb:.1f} MB)")
    print(f"  Reduction:  {(1 - panel_total / total) * 100:.0f}% fewer parameters")

    # Print num_non_obj_tokens
    if hasattr(model, "num_non_obj_tokens"):
        print(f"  num_non_obj_tokens: {model.num_non_obj_tokens}")


def test_wrapper(model: nn.Module) -> PanelDetectorWrapper:
    """Verify the wrapper forward pass works before exporting."""
    wrapper = PanelDetectorWrapper(model)
    wrapper.eval()

    print("\nTesting wrapper forward pass...")
    test_pv = torch.randn(1, 3, 800, 800)
    test_pm = torch.ones(1, 800, 800)

    with torch.no_grad():
        try:
            scores, boxes = wrapper(test_pv, test_pm)
            print(f"  class_scores: {scores.shape}")
            print(f"  boxes:        {boxes.shape}")
        except AttributeError as e:
            print(f"  ERROR: {e}")
            print("\n  Inspecting detection_transformer output...")
            output = model.detection_transformer(
                pixel_values=test_pv, pixel_mask=test_pm
            )
            attrs = [a for a in dir(output) if not a.startswith("_")]
            print(f"  Output type: {type(output).__name__}")
            print(f"  Fields: {attrs}")
            sys.exit(1)

    return wrapper


# ---------------------------------------------------------------------------
# Export & validation
# ---------------------------------------------------------------------------

def export_onnx(
    wrapper: PanelDetectorWrapper, output_path: Path, opset_version: int = 17
) -> None:
    dummy_pv = torch.randn(1, 3, 800, 800)
    dummy_pm = torch.ones(1, 800, 800)

    print(f"\nExporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        wrapper,
        (dummy_pv, dummy_pm),
        str(output_path),
        input_names=["pixel_values", "pixel_mask"],
        output_names=["class_scores", "boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "pixel_mask": {0: "batch", 1: "height", 2: "width"},
            "class_scores": {0: "batch"},
            "boxes": {0: "batch"},
        },
        opset_version=opset_version,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")


def validate_onnx(wrapper: PanelDetectorWrapper, onnx_path: Path) -> bool:
    import onnxruntime as ort

    print("\n=== Validation (PyTorch vs ONNX) ===")

    rng = np.random.RandomState(42)
    test_pv = torch.from_numpy(rng.randn(1, 3, 800, 800).astype(np.float32))
    test_pm = torch.ones(1, 800, 800)

    # PyTorch
    with torch.no_grad():
        pt_scores, pt_boxes = wrapper(test_pv, test_pm)

    # ONNX
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_scores, ort_boxes = session.run(
        None,
        {"pixel_values": test_pv.numpy(), "pixel_mask": test_pm.numpy()},
    )

    score_diff = float(np.abs(pt_scores.numpy() - ort_scores).max())
    box_diff = float(np.abs(pt_boxes.numpy() - ort_boxes).max())
    print(f"  class_scores max diff: {score_diff:.6e}")
    print(f"  boxes max diff:        {box_diff:.6e}")

    # Also test with different input size to verify dynamic axes
    test_pv2 = torch.from_numpy(rng.randn(1, 3, 600, 1000).astype(np.float32))
    test_pm2 = torch.ones(1, 600, 1000)

    with torch.no_grad():
        pt_scores2, pt_boxes2 = wrapper(test_pv2, test_pm2)
    ort_scores2, ort_boxes2 = session.run(
        None,
        {"pixel_values": test_pv2.numpy(), "pixel_mask": test_pm2.numpy()},
    )
    score_diff2 = float(np.abs(pt_scores2.numpy() - ort_scores2).max())
    box_diff2 = float(np.abs(pt_boxes2.numpy() - ort_boxes2).max())
    print(f"  (alt size)  scores:   {score_diff2:.6e}  boxes: {box_diff2:.6e}")

    tol = 1e-4
    passed = max(score_diff, box_diff, score_diff2, box_diff2) < tol
    print(f"  {'PASSED' if passed else 'FAILED'} (tolerance: {tol})")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MagiV2 panel detector to ONNX"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="models/panel_detector.onnx",
        help="Output path (default: models/panel_detector.onnx)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load full model ---
    print("Downloading MagiV2 model from HuggingFace...")
    print("  (first run downloads ~2 GB)")
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        "ragavsachdeva/magiv2", trust_remote_code=True
    )
    model.eval()

    print_model_stats(model)

    # --- Check image processor config ---
    try:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained("ragavsachdeva/magiv2")
        print(f"\n=== Image Processor Config ===")
        print(f"  size: {getattr(processor, 'size', 'N/A')}")
        print(f"  mean: {getattr(processor, 'image_mean', 'N/A')}")
        print(f"  std:  {getattr(processor, 'image_std', 'N/A')}")
    except Exception:
        print("\n  No image processor on HF — using ConditionalDETR defaults")
        print("  (shortest=800, longest=1333, ImageNet mean/std)")

    # --- Test wrapper ---
    wrapper = test_wrapper(model)

    # --- Export ---
    export_onnx(wrapper, output_path, opset_version=args.opset)

    # --- Validate ---
    if not args.skip_validation:
        try:
            ok = validate_onnx(wrapper, output_path)
            if not ok:
                print("\nWARNING: validation failed — outputs may differ")
                sys.exit(1)
        except Exception as e:
            print(f"\nValidation error: {e}")
            print("Run with --skip-validation to bypass")
            sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
