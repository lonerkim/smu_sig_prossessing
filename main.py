#!/usr/bin/env python3
"""
아날로그 영상 잡음 및 색 왜곡 완화 — 범용 영상처리 파이프라인

Flow:
  input (origin) → degrade → output/raw/ (degraded) → pipeline → output/processed/ (restored)

Usage:
    python main.py -p "input/*.jpg" --preset wiener-only
    python main.py -p "input/digital_whoop_footage.mp4" --preset edge-preserving --degrade ntsc-heavy
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
from smu_sig_prossessing.degradation import degrade_image

# ─── Preset map ─────────────────────────────────────────────────────

PRESETS: dict[str, PipelineConfig] = {
    "wiener-only": PipelineConfig.wiener_only(),
    "edge-preserving": PipelineConfig.edge_preserving(),
    "aggressive": PipelineConfig.aggressive(),
    "research-best": PipelineConfig.research_best(),
}

# ─── Degrade modes ──────────────────────────────────────────────────

DEGRADE_MODES: dict[str, dict] = {
    "none":       {"use_ntsc": False, "label": "no degradation"},
    "basic":      {"use_ntsc": False, "label": "basic synthetic noise"},
    "ntsc-light": {"use_ntsc": True,  "ntsc_intensity": "light",  "label": "NTSC light"},
    "ntsc-medium":{"use_ntsc": True,  "ntsc_intensity": "medium", "label": "NTSC medium"},
    "ntsc-heavy": {"use_ntsc": True,  "ntsc_intensity": "heavy",  "label": "NTSC heavy"},
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

def make_degraded(origin: np.ndarray, mode: str) -> np.ndarray:
    """Apply degradation to an original image."""
    params = {k: v for k, v in DEGRADE_MODES[mode].items() if k != "label"}
    if mode == "none":
        return origin.copy()
    return degrade_image(origin, **params)


def process_image(path: str, cfg: PipelineConfig, degrade_mode: str) -> None:
    origin = cv2.imread(path)
    if origin is None:
        print(f"  ⚠ Cannot read: {path}")
        return
    name = os.path.splitext(os.path.basename(path))[0]

    # 1) Degrade → save to raw/
    degraded = make_degraded(origin, degrade_mode)
    out_raw = os.path.join(OUT_RAW, f"{name}.png")
    cv2.imwrite(out_raw, degraded)

    # 2) Pipeline → save to processed/
    restored = pl.apply_pipeline(degraded, cfg)
    out_proc = os.path.join(OUT_PROC, f"{name}.png")
    cv2.imwrite(out_proc, restored)

    h, w = origin.shape[:2]
    print(f"  ✅ {name:30s}  {w}x{h}  degraded → {out_raw}")
    print(f"  {'':33s} restored → {out_proc}")


def process_video(path: str, cfg: PipelineConfig, degrade_mode: str) -> None:
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
    raw_path = os.path.join(OUT_RAW, f"{name}.mp4")
    proc_path = os.path.join(OUT_PROC, f"{name}.mp4")
    raw_writer = cv2.VideoWriter(raw_path, fourcc, fps, (w, h))
    proc_writer = cv2.VideoWriter(proc_path, fourcc, fps, (w, h))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Degrade → write to raw
        degraded = make_degraded(frame, degrade_mode)
        raw_writer.write(degraded)

        # Pipeline → write to processed
        restored = pl.apply_pipeline(degraded, cfg)
        proc_writer.write(restored)

        count += 1
        if n_frames > 0 and count % 30 == 0:
            pct = count / n_frames * 100
            print(f"    {count}/{n_frames} frames ({pct:.0f}%)", end="\r")

    raw_writer.release()
    proc_writer.release()
    cap.release()
    print(f"  ✅ {name:30s}  {w}x{h}  {count}frames  degraded → {raw_path}")
    print(f"  {'':33s} restored → {proc_path}")


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="아날로그 영상 잡음 완화 파이프라인 — 실제 영상/이미지 처리",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예제:\n"
            "  python main.py -p \"input/*.jpg\" --preset wiener-only\n"
            "  python main.py -p \"input/digital_whoop_footage.mp4\" --preset edge-preserving --degrade ntsc-heavy\n"
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
    parser.add_argument("--degrade", type=str, default="ntsc-medium",
                        choices=list(DEGRADE_MODES.keys()),
                        help="Degradation mode applied to origin before pipeline (default: ntsc-medium)")
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
    degrade_label = DEGRADE_MODES[args.degrade]["label"]

    print(f"\n🔧 Preset:     {cfg.label}")
    print(f"   Filters:    {' → '.join(s.name for s in cfg.stages if s.enabled)}")
    print(f"   Degrade:    {degrade_label} ({args.degrade})")
    print(f"   Input:      {len(files)} file(s) ({len(images)} image, {len(videos)} video)")
    print(f"   Output raw: {OUT_RAW}/  (degraded — after noise injection)")
    print(f"   Output proc:{OUT_PROC}/  (restored — after pipeline)")
    print()

    for f in images:
        process_image(f, cfg, args.degrade)

    for f in videos:
        process_video(f, cfg, args.degrade)

    print(f"\n✅ Done — {len(images) + len(videos)} file(s) processed.")


if __name__ == "__main__":
    main()
