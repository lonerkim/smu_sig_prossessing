#!/usr/bin/env python3
"""
Qualitative Comparison Grid Generator (Iteration 9)

Processes an input video with the top 6 presets and creates a 2x3 grid PNG
showing before/after for each preset.

Usage:
    python scripts/iter9_comparison_grid.py -i input/analog_whoop_footage.mp4
    python scripts/iter9_comparison_grid.py -i input/analog_whoop_footage.mp4 --frame 5
    python scripts/iter9_comparison_grid.py -i input/test_small.jpg

Output:
    output/comparison_grid/<video_name>_<preset>_grid.png  (individual)
    output/comparison_grid/<video_name>_master_grid.png     (2x3 combined)
"""
from __future__ import annotations

import argparse
import os
import sys
import math

import cv2
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.filters import reset_temporal_state

# ─── Top 6 presets ──────────────────────────────────────────────────

TOP_PRESETS = {
    "adaptive": PipelineConfig.adaptive(),
    "analog-clean": PipelineConfig.analog_clean(),
    "fast-premium": PipelineConfig.fast_premium(),
    "bm3d-denoise": PipelineConfig.bm3d_denoise(),
    "temporal-premium": PipelineConfig.temporal_premium(),
    "chroma-focus": PipelineConfig.chroma_focus(),
}

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

OUT_BASE = os.path.join(PROJECT_ROOT, "output", "comparison_grid")


def load_frame(input_path: str, frame_idx: int = 0) -> np.ndarray | None:
    """Load a single frame from an image or video file."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext in IMAGE_EXTS:
        img = cv2.imread(input_path)
        return img
    elif ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"  ⚠ Cannot open: {input_path}")
            return None
        for _ in range(frame_idx + 1):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                print(f"  ⚠ Cannot read frame {frame_idx}")
                return None
        cap.release()
        return frame
    else:
        print(f"  ⚠ Unsupported format: {ext}")
        return None


def process_with_preset(frame: np.ndarray, preset_name: str,
                        cfg: PipelineConfig) -> np.ndarray:
    """Process a frame with the given preset config."""
    reset_temporal_state()
    # For adaptive, we'd need AdaptivePipeline, but fall back to the default config
    return pl.apply_pipeline(frame, cfg)


def create_before_after_pair(origin: np.ndarray, processed: np.ndarray,
                             preset_name: str, label: str,
                             cell_w: int = 480) -> np.ndarray:
    """Create a before/after side-by-side image with labels."""
    h, w = origin.shape[:2]
    # Resize each half to cell_w/2 width
    half_w = cell_w // 2
    scale = half_w / w
    new_h = int(h * scale)

    origin_rs = cv2.resize(origin, (half_w, new_h))
    processed_rs = cv2.resize(processed, (half_w, new_h))

    # Canvas: two images side by side + label area at top and bottom
    label_h = 35
    canvas = np.zeros((new_h + label_h * 2, cell_w, 3), dtype=np.uint8)

    # Top label
    cv2.putText(canvas, f"{label} — {preset_name}",
                (10, label_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)

    # Images
    canvas[label_h:label_h + new_h, :half_w] = origin_rs
    canvas[label_h:label_h + new_h, half_w:] = processed_rs

    # Bottom labels for each half
    cv2.putText(canvas, "Before", (half_w // 2 - 30, label_h + new_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(canvas, "After", (half_w + half_w // 2 - 25, label_h + new_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Divider line
    cv2.line(canvas, (half_w, label_h), (half_w, label_h + new_h),
             (100, 100, 100), 1)

    return canvas


def create_master_grid(pairs: list[np.ndarray],
                       preset_names: list[str]) -> np.ndarray:
    """Arrange pairs into a 2x3 grid."""
    cols, rows = 3, 2
    cell_h = pairs[0].shape[0]
    cell_w = pairs[0].shape[1]

    # Small gap between cells
    gap = 4
    grid_w = cols * cell_w + (cols - 1) * gap
    grid_h = rows * cell_h + (rows - 1) * gap

    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx, pair in enumerate(pairs):
        r, c = divmod(idx, cols)
        y0 = r * (cell_h + gap)
        x0 = c * (cell_w + gap)
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = pair

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Comparison grid generator — top 6 presets before/after",
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input video or image path")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index to extract from video (default: 0)")
    parser.add_argument("--cell-width", type=int, default=640,
                        help="Width of each before/after cell (default: 640)")

    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        # Try relative to project root
        input_path = os.path.join(PROJECT_ROOT, input_path)
    if not os.path.isfile(input_path):
        print(f"  ⚠ File not found: {args.input}")
        sys.exit(1)

    # Load frame
    print(f"\n📥 Loading frame {args.frame} from {os.path.basename(input_path)}")
    origin = load_frame(input_path, args.frame)
    if origin is None:
        sys.exit(1)

    h, w = origin.shape[:2]
    print(f"   Resolution: {w}x{h}")

    name = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(OUT_BASE, exist_ok=True)

    # Process with each preset
    pairs: list[np.ndarray] = []
    preset_names: list[str] = []

    for preset_name, cfg in TOP_PRESETS.items():
        print(f"\n  🔧 Processing: {preset_name}...")
        processed = process_with_preset(origin, preset_name, cfg)

        # Save individual grid
        ind_path = os.path.join(OUT_BASE, f"{name}_{preset_name}_grid.png")
        pair = create_before_after_pair(origin, processed, preset_name,
                                        label=os.path.basename(args.input),
                                        cell_w=args.cell_width)
        cv2.imwrite(ind_path, pair)
        print(f"    → {ind_path}")

        pairs.append(pair)
        preset_names.append(preset_name)

    # Create 2x3 master grid
    master = create_master_grid(pairs, preset_names)
    master_path = os.path.join(OUT_BASE, f"{name}_master_grid.png")
    cv2.imwrite(master_path, master)
    print(f"\n🎨 Master grid → {master_path}")

    print(f"\n✅ Done — {len(pairs)} presets processed")
    print(f"   Individual grids: output/comparison_grid/{name}_<preset>_grid.png")
    print(f"   Master grid:      output/comparison_grid/{name}_master_grid.png")


if __name__ == "__main__":
    main()
