#!/usr/bin/env python3
"""
아날로그 영상 잡음 및 색 왜곡 완화 — 범용 영상처리 파이프라인

Flow:
  input (origin) → degrade → output/raw/ (degraded) → pipeline → output/processed/ (restored)
                                     ↕
              output/*_comparison.png (origin | degraded | restored)

Usage:
    python main.py -p "input/*.jpg" --preset wiener-only
    python main.py -p "input/digital_whoop_footage.mp4" --preset edge-preserving --degrade ntsc-heavy
    python main.py -p "input/*.{jpg,png,mp4}" --preset research-best --strength 0.7
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
    "edge-preserve": PipelineConfig.edge_preserve(),
    "fast-denoise": PipelineConfig.fast_denoise(),
    "nlm-denoise": PipelineConfig.nlm_denoise(),
    "wiener-denoise": PipelineConfig.wiener_denoise(),
    "wavelet-denoise": PipelineConfig.wavelet_denoise(),
    "guided-denoise": PipelineConfig.guided_denoise(),
    "tv-denoise": PipelineConfig.tv_denoise_preset(),
    "aniso-denoise": PipelineConfig.aniso_denoise(),
    "dct-denoise": PipelineConfig.dct_denoise(),
    "st-video": PipelineConfig.spatial_temporal_video(),
    "optimized-fast": PipelineConfig.optimized_fast(),
    "optimized-quality": PipelineConfig.optimized_quality(),
    "max-quality": PipelineConfig.max_quality(),
    "video-enhanced": PipelineConfig.video_enhanced(),
    "video-ultra": PipelineConfig.video_ultra(),
    "ntsc-plus": PipelineConfig.ntc_plus(),
    "fast-premium": PipelineConfig.fast_premium(),
    "aggressive": PipelineConfig.aggressive(),
    "research-best": PipelineConfig.research_best(),
    "analog-clean": PipelineConfig.analog_clean(),
    "analog-heavy": PipelineConfig.analog_heavy(),
    "adaptive": PipelineConfig.adaptive(),
    # ── New BM3D / Retinex presets ──────────────────────────────
    "bm3d-denoise": PipelineConfig.bm3d_denoise(),
    "retinex-enhance": PipelineConfig.retinex_enhance(),
    "retinex-bm3d": PipelineConfig.retinex_bm3d(),
    "bm3d-fast": PipelineConfig.bm3d_fast(),
}

# ─── Degrade modes ──────────────────────────────────────────────────

DEGRADE_MODES: dict[str, str] = {
    "none": "none",
    "basic": "basic",
    "ntsc-light": "ntsc-light",
    "ntsc-medium": "ntsc-medium",
    "ntsc-heavy": "ntsc-heavy",
}

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_RAW = os.path.join(BASE, "output", "raw")
OUT_PROC = os.path.join(BASE, "output", "processed")
OUT_CMP = os.path.join(BASE, "output")


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
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return None


# ─── Degrade helper ─────────────────────────────────────────────────

def make_degraded(origin: np.ndarray, mode: str, strength: float) -> np.ndarray:
    """Apply degradation to an original. strength (0-1) scales 'basic' noise."""
    if mode == "none":
        return origin.copy()
    if mode.startswith("ntsc"):
        intensity = mode.split("-")[1]  # light, medium, heavy
        result = degrade_image(origin, use_ntsc=True, ntsc_intensity=intensity)
        # NTSC may change dimensions (even height requirement)
        if result.shape != origin.shape:
            result = cv2.resize(result, (origin.shape[1], origin.shape[0]))
        return result
    # basic
    return degrade_image(origin, use_ntsc=False, strength=strength)


def save_comparison(origin: np.ndarray, degraded: np.ndarray,
                    restored: np.ndarray, path: str) -> str:
    """Side-by-side: origin | degraded | restored with labels."""
    h, w = origin.shape[:2]
    # Downscale if too wide for comparison
    total_w = w * 3
    if total_w > 3000:
        scale = 3000 / total_w
        h, w = int(h * scale), int(w * scale)
        origin = cv2.resize(origin, (w, h))
        degraded = cv2.resize(degraded, (w, h))
        restored = cv2.resize(restored, (w, h))

    canvas = np.zeros((h + 40, w * 3, 3), dtype=np.uint8)  # +40 for label area
    canvas[:h, :w] = cv2.resize(origin, (w, h))
    canvas[:h, w:2*w] = cv2.resize(degraded, (w, h))
    canvas[:h, 2*w:3*w] = cv2.resize(restored, (w, h))

    for i, lbl in enumerate(["Original", "Degraded", "Restored"]):
        cv2.putText(canvas, lbl, (i * w + 10, h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, canvas)
    return path


# ─── Processing ─────────────────────────────────────────────────────

def process_image(path: str, cfg: PipelineConfig,
                  degrade_mode: str, strength: float,
                  adaptive_pipeline: 'AdaptivePipeline | None' = None) -> str | None:
    origin = cv2.imread(path)
    if origin is None:
        print(f"  ⚠ Cannot read: {path}")
        return
    name = os.path.splitext(os.path.basename(path))[0]

    # 1) Degrade → raw/
    degraded = make_degraded(origin, degrade_mode, strength)
    cv2.imwrite(os.path.join(OUT_RAW, f"{name}.png"), degraded)

    # 2) Pipeline → processed/
    if adaptive_pipeline is not None:
        restored = adaptive_pipeline.process(degraded)
    else:
        restored = pl.apply_pipeline(degraded, cfg)
    cv2.imwrite(os.path.join(OUT_PROC, f"{name}.png"), restored)

    # 3) Side-by-side comparison
    cmp_path = os.path.join(OUT_CMP, f"{name}_comparison.png")
    save_comparison(origin, degraded, restored, cmp_path)

    h, w = origin.shape[:2]
    print(f"  ✅ {name:30s}  {w}x{h}")
    print(f"      degraded → {os.path.join(OUT_RAW, f'{name}.png')}")
    print(f"      restored → {os.path.join(OUT_PROC, f'{name}.png')}")
    print(f"      compare  → {cmp_path}")

    # Return paths for potential delivery
    return cmp_path


def process_video(path: str, cfg: PipelineConfig,
                  degrade_mode: str, strength: float,
                  sample_frames: int = 0,
                  adaptive_pipeline: 'AdaptivePipeline | None' = None) -> str | None:
    from smu_sig_prossessing.filters import reset_temporal_state
    reset_temporal_state()  # reset temporal filters before each video
    if adaptive_pipeline is not None:
        adaptive_pipeline.reset()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  ⚠ Cannot open: {path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    name = os.path.splitext(os.path.basename(path))[0]

    if sample_frames > 0:
        # ── Heuristic sample mode ──────────────────────────────────
        print(f"\n  📋 SAMPLE MODE: processing first {sample_frames} frame(s)")
        sample_comparisons = []
        for f_idx in range(sample_frames):
            ret, frame = cap.read()
            if not ret:
                break
            degraded = make_degraded(frame, degrade_mode, strength)
            if adaptive_pipeline is not None:
                restored = adaptive_pipeline.process(degraded)
            else:
                restored = pl.apply_pipeline(degraded, cfg)
            # Save individual sample comparison
            samp_cmp = os.path.join(OUT_CMP, f"{name}_sample_{f_idx:03d}.png")
            save_comparison(frame, degraded, restored, samp_cmp)
            sample_comparisons.append(samp_cmp)
            print(f"    Frame {f_idx}: {samp_cmp}")

        cap.release()

        # Also save a multi-frame grid
        if len(sample_comparisons) > 1:
            grid_path = os.path.join(OUT_CMP, f"{name}_sample_grid.png")
            _save_sample_grid(sample_comparisons, grid_path, (w, h))
            print(f"    Multi-frame grid → {grid_path}")

        print(f"  ✅ Sample complete — {len(sample_comparisons)} frame(s)")
        return sample_comparisons[0] if sample_comparisons else None

    # ── Full video mode ────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    raw_path = os.path.join(OUT_RAW, f"{name}.mp4")
    proc_path = os.path.join(OUT_PROC, f"{name}.mp4")
    raw_writer = cv2.VideoWriter(raw_path, fourcc, fps, (w, h))
    proc_writer = cv2.VideoWriter(proc_path, fourcc, fps, (w, h))

    # Save first-frame comparison
    first_origin = None

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if first_origin is None:
            first_origin = frame.copy()

        degraded = make_degraded(frame, degrade_mode, strength)
        raw_writer.write(degraded)

        if adaptive_pipeline is not None:
            restored = adaptive_pipeline.process(degraded)
        else:
            restored = pl.apply_pipeline(degraded, cfg)
        proc_writer.write(restored)

        count += 1
        if n_frames > 0 and count % 30 == 0:
            pct = count / n_frames * 100
            print(f"    {count}/{n_frames} frames ({pct:.0f}%)", end="\r")

    raw_writer.release()
    proc_writer.release()
    cap.release()

    # Comparison from first frame
    cmp_path = None
    if first_origin is not None:
        d0 = make_degraded(first_origin, degrade_mode, strength)
        if adaptive_pipeline is not None:
            r0 = adaptive_pipeline.process(d0)
        else:
            r0 = pl.apply_pipeline(d0, cfg)
        cmp_path = os.path.join(OUT_CMP, f"{name}_comparison.png")
        save_comparison(first_origin, d0, r0, cmp_path)
        print(f"      compare  → {cmp_path}")

    print(f"  ✅ {name:30s}  {w}x{h}  {count}frames")
    print(f"      degraded → {raw_path}")
    print(f"      restored → {proc_path}")
    return cmp_path


def _save_sample_grid(sample_paths: list[str], out_path: str,
                      frame_size: tuple[int, int]) -> str:
    """Arrange sample comparison images into a grid."""
    import math
    n = len(sample_paths)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fw, fh = frame_size
    fw3 = fw * 3
    # Read first image to get actual comparison image height
    first_img = cv2.imread(sample_paths[0])
    if first_img is None:
        return out_path
    cmp_h = first_img.shape[0]
    cell_h = cmp_h + 10
    canvas = np.zeros((rows * cell_h + 10,
                       cols * fw3 + 10, 3), dtype=np.uint8)

    for idx, sp in enumerate(sample_paths):
        r, c = divmod(idx, cols)
        img = cv2.imread(sp)
        if img is None:
            continue
        h_im = img.shape[0]
        y0 = r * cell_h + 5
        x0 = c * fw3 + 5
        canvas[y0:y0 + h_im, x0:x0 + fw3] = img[:h_im, :fw3]
        cv2.putText(canvas, f"Frame {idx}",
                    (x0 + 5, y0 + h_im + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(out_path, canvas)
    return out_path


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="아날로그 영상 잡음 완화 파이프라인 — 실제 영상/이미지 처리",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예제:\n"
            "  python main.py -p \"input/*.jpg\" --preset wiener-only --strength 0.5\n"
            "  python main.py -p \"input/digital_whoop_footage.mp4\" --preset edge-preserving --degrade ntsc-heavy\n"
            "  python main.py -p \"input/*.{jpg,png,mp4}\" --preset research-best --strength 0.7\n"
            "  python main.py -p \"input/*\" --preset aggressive\n"
            "  python main.py --list-filters\n"
        )
    )
    parser.add_argument("-p", "--path", type=str, default=None,
                        help="Input file glob pattern (e.g. 'input/*.jpg' or 'input/*.{mp4,jpg}')")
    parser.add_argument("--preset", type=str, default="edge-preserve",
                        choices=list(PRESETS.keys()),
                        help="Pipeline preset to apply (default: edge-preserve)")
    parser.add_argument("--degrade", type=str, default="basic",
                        choices=list(DEGRADE_MODES.keys()),
                        help="Degradation type (default: basic)")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="Degradation strength 0.0–1.0 for 'basic' mode (default: 0.5). "
                             "Higher = more noise, darker, harder to restore.")
    parser.add_argument("--list-filters", action="store_true",
                        help="Show all available filters and exit")
    parser.add_argument("--sample", type=int, default=0, metavar="N",
                        help="Process only N frames (video) for heuristic preview (default: full)")

    args = parser.parse_args()

    if args.list_filters:
        print("📋 Available Filters:")
        print(pl.list_available_filters())
        return

    if not args.path:
        parser.print_help()
        print("\n  ⚠  Use -p to specify input files or --list-filters to see available filters.")
        return

    args.strength = np.clip(args.strength, 0.0, 1.0)

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

    # If adaptive preset selected, create the AdaptivePipeline instance
    adaptive_pipe = None
    if args.preset == "adaptive":
        from smu_sig_prossessing.adaptive import AdaptivePipeline
        adaptive_pipe = AdaptivePipeline(verbose=True)
        print(f"\n🔧 Preset:     Adaptive (auto-tuned per frame)")
    else:
        print(f"\n🔧 Preset:     {cfg.label}")
        print(f"   Filters:    {' → '.join(s.name for s in cfg.stages if s.enabled)}")
    print(f"   Degrade:    {args.degrade}  (strength={args.strength:.1f})")
    print(f"   Input:      {len(files)} file(s) ({len(images)} image, {len(videos)} video)")
    print(f"   Output:")
    print(f"      raw/       = degraded (after noise injection)")
    print(f"      processed/ = restored (after pipeline)")
    print(f"      *_comparison.png = origin | degraded | restored")
    print()

    for f in images:
        process_image(f, cfg, args.degrade, args.strength,
                      adaptive_pipeline=adaptive_pipe)

    for f in videos:
        process_video(f, cfg, args.degrade, args.strength,
                      sample_frames=args.sample,
                      adaptive_pipeline=adaptive_pipe)

    print(f"\n✅ Done — {len(images) + len(videos)} file(s) processed.")


if __name__ == "__main__":
    main()
