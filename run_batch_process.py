#!/usr/bin/env python3
"""
Batch process all input videos with top-5 presets, saving output frames
and comparison grids for visual evaluation.

Usage:
    .venv/bin/python3 run_batch_process.py
    .venv/bin/python3 run_batch_process.py --sample 3  # fewer frames
    .venv/bin/python3 run_batch_process.py --presets optimal-balanced,analog-clean,wavelet-denoise
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import PRESETS
from smu_sig_prossessing.filters import reset_temporal_state

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE, "input")
OUTPUT_DIR = os.path.join(BASE, "output", "batch_process")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")
GRID_DIR = os.path.join(OUTPUT_DIR, "grids")

# Top-5 presets by perceptual quality (NIQE) from ablation sweep
TOP_PRESETS_BY_NIQE = [
    "optimal-balanced",   # NIQE=7.28
    "analog-clean",       # NIQE=7.33
    "wavelet-denoise",    # NIQE=7.35
    "ntsc-plus",          # NIQE=7.35
    "fast-denoise",       # NIQE=7.57, fastest
]


def make_degraded(frame: np.ndarray, mode: str, strength: float) -> np.ndarray:
    """Apply degradation to a frame."""
    if mode == "none":
        return frame.copy()
    from smu_sig_prossessing.degradation import degrade_image
    if mode.startswith("ntsc"):
        intensity = mode.split("-")[1]
        result = degrade_image(frame, use_ntsc=True, ntsc_intensity=intensity)
        if result.shape != frame.shape:
            result = cv2.resize(result, (frame.shape[1], frame.shape[0]))
        return result
    return degrade_image(frame, use_ntsc=False, strength=strength)


def save_comparison_row(
    origin: np.ndarray,
    restored: np.ndarray,
    path: str,
    label: str = "",
) -> str:
    """Save origin | restored side-by-side with label."""
    h, w = origin.shape[:2]
    canvas = np.zeros((h + 30, w * 2, 3), dtype=np.uint8)
    canvas[:h, :w] = cv2.resize(origin, (w, h))
    canvas[:h, w:] = cv2.resize(restored, (w, h))

    for i, lbl in enumerate(["Input", label or "Output"]):
        cv2.putText(canvas, lbl, (i * w + 10, h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, canvas)
    return path


def save_master_grid(
    input_frame: np.ndarray,
    preset_outputs: dict[str, np.ndarray],
    path: str,
    video_name: str,
) -> str:
    """Create a grid showing input frame + output from each preset."""
    n_presets = len(preset_outputs)
    cols = min(3, n_presets + 1)  # +1 for input
    rows = (n_presets + 1 + cols - 1) // cols

    h, w = input_frame.shape[:2]
    label_h = 24
    cell_w = w + 10
    cell_h = h + label_h + 10

    canvas = np.zeros((rows * cell_h + 10, cols * cell_w + 10, 3), dtype=np.uint8)

    # First cell: Input
    y0, x0 = 10, 10
    canvas[y0:y0 + h, x0:x0 + w] = cv2.resize(input_frame, (w, h))
    cv2.putText(canvas, f"{video_name} — Input",
                (x0 + 5, y0 + h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    items = [("Input", input_frame)] + list(preset_outputs.items())
    for idx, (name, img) in enumerate(items):
        r = idx // cols
        c = idx % cols
        if idx == 0:
            continue  # already placed input
        y0 = r * cell_h + 10
        x0 = c * cell_w + 10
        img_resized = cv2.resize(img, (w, h))
        canvas[y0:y0 + h, x0:x0 + w] = img_resized
        cv2.putText(canvas, name, (x0 + 5, y0 + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, canvas)
    return path


def process_video(
    video_path: str,
    presets: list[str],
    degrade_mode: str,
    strength: float,
    sample_frames: int,
    out_dir: str,
) -> dict:
    """Process a single video with all presets and save comparison frames."""
    name = os.path.splitext(os.path.basename(video_path))[0]
    result = {
        "video": video_path,
        "name": name,
        "presets": {},
        "status": "ok",
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"video": video_path, "name": name, "status": "error",
                "error": "Cannot open"}

    frames = []
    for _ in range(sample_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return {"video": video_path, "name": name, "status": "error",
                "error": "No frames read"}

    # Process each preset
    preset_outputs = {}
    for preset_name in presets:
        cfg = PRESETS.get(preset_name)
        if cfg is None:
            continue

        reset_temporal_state()
        t0 = time.perf_counter()
        processed_frames = []
        for frame in frames:
            from smu_sig_prossessing import pipeline as pl
            restored = pl.apply_pipeline(frame.copy(), cfg)
            processed_frames.append(restored)
        elapsed = time.perf_counter() - t0

        preset_outputs[preset_name] = processed_frames
        result["presets"][preset_name] = {
            "time_sec": round(elapsed, 3),
            "fps": round(sample_frames / elapsed, 1) if elapsed > 0 else 0,
        }

        # Save first-frame comparison
        cmp_path = os.path.join(out_dir, "samples", f"{name}_{preset_name}.png")
        save_comparison_row(frames[0], processed_frames[0], cmp_path,
                            label=preset_name)
        result["presets"][preset_name]["comparison"] = cmp_path

    # Save master grid
    if preset_outputs:
        first_frame_outputs = {k: v[0] for k, v in preset_outputs.items()}
        grid_path = os.path.join(out_dir, "grids", f"{name}_master_grid.png")
        save_master_grid(frames[0], first_frame_outputs, grid_path, name)
        result["master_grid"] = grid_path

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch process all input videos with top presets"
    )
    parser.add_argument("--presets", type=str, default=None,
                        help="Comma-separated preset list (default: top-5 by NIQE)")
    parser.add_argument("--sample", type=int, default=3,
                        help="Number of sample frames per video (default: 3)")
    parser.add_argument("--degrade", type=str, default="none",
                        help="Degrade mode (default: none)")
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: output/batch_process/)")
    args = parser.parse_args()

    # Resolve presets
    if args.presets:
        preset_list = [p.strip() for p in args.presets.split(",")]
    else:
        preset_list = TOP_PRESETS_BY_NIQE

    out_dir = args.output or OUTPUT_DIR
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "grids"), exist_ok=True)

    # Find all input videos
    video_exts = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
    video_files = []
    for ext in video_exts:
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    video_files = sorted(set(video_files))

    if not video_files:
        print("No video files found in input/")
        sys.exit(1)

    print(f"🎬 Batch processing {len(video_files)} videos with {len(preset_list)} presets")
    print(f"   Presets: {', '.join(preset_list)}")
    print(f"   Sample frames: {args.sample}")
    print(f"   Degrade: {args.degrade}")
    print()

    all_results = []
    for vf in video_files:
        print(f"📹 Processing: {os.path.basename(vf)}...")
        r = process_video(vf, preset_list, args.degrade, args.strength,
                          args.sample, out_dir)
        all_results.append(r)
        if r["status"] == "ok":
            print(f"   ✅ {len(r['presets'])} presets processed")
            for pname, pinfo in r["presets"].items():
                print(f"      {pname:25s} {pinfo['time_sec']:.2f}s ({pinfo['fps']:.0f}fps)")
        else:
            print(f"   ⚠ ERROR: {r.get('error', 'unknown')}")

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 SUMMARY")
    print(f"{'=' * 60}")
    success = [r for r in all_results if r["status"] == "ok"]
    failed = [r for r in all_results if r["status"] != "ok"]
    print(f"   Total videos: {len(all_results)}")
    print(f"   Success: {len(success)}")
    print(f"   Failed: {len(failed)}")
    print(f"   Output: {out_dir}")
    print(f"   Samples: {os.path.join(out_dir, 'samples')}/")
    print(f"   Grids: {os.path.join(out_dir, 'grids')}/")

    # Save manifest
    manifest = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "presets": preset_list,
        "sample_frames": args.sample,
        "degrade": args.degrade,
        "results": all_results,
    }
    manifest_path = os.path.join(out_dir, "batch_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"   Manifest: {manifest_path}")

    print(f"\n✅ Batch processing complete")


if __name__ == "__main__":
    main()
