#!/usr/bin/env python3
"""
아날로그 영상 잡음 및 색 왜곡 완화 — 범용 영상처리 파이프라인

Usage:
    python main.py -p "input/*.jpg" --preset wiener-only
    python main.py -p "input/analog_whoop_footage.mp4" --preset edge-preserving
    python main.py -p "input/*.{jpg,png,mp4}" --preset research-best
    python main.py --list-filters
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.filters import list_filters

# ─── Preset map ─────────────────────────────────────────────────────

PRESETS: dict[str, PipelineConfig] = {
    "wiener-only": PipelineConfig.wiener_only(),
    "edge-preserving": PipelineConfig.edge_preserving(),
    "aggressive": PipelineConfig.aggressive(),
    "research-best": PipelineConfig.research_best(),
}

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_RAW = os.path.join(BASE, "output", "raw")
OUT_PROC = os.path.join(BASE, "output", "processed")


# ─── File discovery ─────────────────────────────────────────────────

def resolve_files(pattern: str) -> list[str]:
    """Expand a glob pattern into absolute file paths."""
    if pattern.startswith('"') and pattern.endswith('"'):
        pattern = pattern[1:-1]
    if pattern.startswith("'") and pattern.endswith("'"):
        pattern = pattern[1:-1]
    files = sorted(glob.glob(os.path.expanduser(pattern), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.expanduser(os.path.join(BASE, pattern)), recursive=True))
    if not files:
        print(f"  ⚠ No files matched: {pattern}")
    return [f for f in files if os.path.isfile(f)]


def classify_file(path: str) -> str | None:
    """Return 'image', 'video', or None."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return None


# ─── Processing ─────────────────────────────────────────────────────

def process_image(path: str, cfg: PipelineConfig) -> None:
    img = cv2.imread(path)
    if img is None:
        print(f"  ⚠ Cannot read: {path}")
        return
    name = os.path.splitext(os.path.basename(path))[0]
    out_path_raw = os.path.join(OUT_RAW, f"{name}.png")
    out_path_proc = os.path.join(OUT_PROC, f"{name}.png")

    cv2.imwrite(out_path_raw, img)
    result = pl.apply_pipeline(img, cfg)
    cv2.imwrite(out_path_proc, result)

    h, w = img.shape[:2]
    print(f"  ✅ {name:30s}  {w}x{h}  →  {out_path_proc}")


def process_video(path: str, cfg: PipelineConfig) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  ⚠ Cannot open: {path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    name = os.path.splitext(os.path.basename(path))[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    raw_writer = cv2.VideoWriter(os.path.join(OUT_RAW, f"{name}.mp4"), fourcc, fps, (w, h))
    proc_writer = cv2.VideoWriter(os.path.join(OUT_PROC, f"{name}.mp4"), fourcc, fps, (w, h))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_writer.write(frame)
        processed = pl.apply_pipeline(frame, cfg)
        proc_writer.write(processed)
        count += 1
        if count % 100 == 0 and n_frames > 0:
            print(f"    {count}/{n_frames} frames", end="\r")

    raw_writer.release()
    proc_writer.release()
    cap.release()
    print(f"  ✅ {name:30s}  {w}x{h}  {count}frames  →  {os.path.join(OUT_PROC, f'{name}.mp4')}")


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="아날로그 영상 잡음 완화 파이프라인 — 실제 영상/이미지 처리",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예제:\n"
            "  python main.py -p \"input/*.jpg\" --preset wiener-only\n"
            "  python main.py -p \"input/analog_whoop_footage.mp4\" --preset edge-preserving\n"
            "  python main.py -p \"input/*.{jpg,png,mp4}\" --preset research-best\n"
            "  python main.py -p \"input/*\" --preset aggressive\n"
            "  python main.py --list-filters\n"
        )
    )
    parser.add_argument("-p", "--path", type=str, default=None,
                        help="Input file glob pattern (e.g. 'input/*.jpg' or 'input/*.{mp4,jpg}')")
    parser.add_argument("--preset", type=str, default="wiener-only",
                        choices=list(PRESETS.keys()),
                        help="Pipeline preset to apply (default: wiener-only)")
    parser.add_argument("--list-filters", action="store_true",
                        help="Show all available filters and exit")

    args = parser.parse_args()

    if args.list_filters:
        print("📋 Available Filters:")
        print(pl.list_available_filters())
        return

    if not args.path:
        parser.print_help()
        print("\n  ⚠  Use -p to specify input files or --list-filters to see available filters.")
        return

    # Resolve files
    files = resolve_files(args.path)
    if not files:
        sys.exit(1)

    # Separate by type
    images = [f for f in files if classify_file(f) == "image"]
    videos = [f for f in files if classify_file(f) == "video"]
    unknown = [f for f in files if classify_file(f) is None]

    if unknown:
        print(f"  ⚠ Skipping {len(unknown)} file(s) with unsupported extension:")
        for f in unknown:
            print(f"     {f}")

    if not images and not videos:
        print("  ⚠ No image or video files found.")
        sys.exit(1)

    # Prepare output dirs
    os.makedirs(OUT_RAW, exist_ok=True)
    os.makedirs(OUT_PROC, exist_ok=True)

    cfg = PRESETS[args.preset]
    print(f"\n🔧 Preset: {cfg.label}")
    print(f"   Filters: {' → '.join(s.name for s in cfg.stages if s.enabled)}")
    print(f"   Input:   {len(files)} file(s) ({len(images)} image, {len(videos)} video)")
    print(f"   Output:  {OUT_RAW}/  (raw input copy)")
    print(f"            {OUT_PROC}/  (after pipeline)")
    print()

    for f in images:
        process_image(f, cfg)

    for f in videos:
        process_video(f, cfg)

    print(f"\n✅ Done — {len(images) + len(videos)} file(s) processed.")


if __name__ == "__main__":
    main()
