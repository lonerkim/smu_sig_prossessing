#!/usr/bin/env python3
"""
Ablation study runner — systematically test preset/filter combinations
and record PSNR, SSIM, Edge Retention, and Timing.

Usage:
    python run_ablation.py                          # test all presets
    python run_ablation.py --presets edge-preserve,guided-denoise,tv-denoise
    python run_ablation.py --strength 0.5 --degrade ntsc-light
    python run_ablation.py --ablation median,nlm,bilateral  # test filter on/off
    python run_ablation.py --param-tune wiener noise_var 100,200,400,800
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.degradation import degrade_image
from smu_sig_prossessing.evaluation import (
    calculate_psnr, calculate_ssim,
)
from smu_sig_prossessing.filters import FILTER_REGISTRY

# ─── Config ──────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(BASE, "input", "REAL_WORLD_PICTURE.jpg")
OUT_DIR = os.path.join(BASE, "output", "ablation")

ALL_PRESETS: dict[str, PipelineConfig] = {
    "edge-preserve": PipelineConfig.edge_preserve(),
    "fast-denoise": PipelineConfig.fast_denoise(),
    "nlm-denoise": PipelineConfig.nlm_denoise(),
    "wiener-denoise": PipelineConfig.wiener_denoise(),
    "wavelet-denoise": PipelineConfig.wavelet_denoise(),
    "guided-denoise": PipelineConfig.guided_denoise(),
    "tv-denoise": PipelineConfig.tv_denoise_preset(),
    "aniso-denoise": PipelineConfig.aniso_denoise(),
    "dct-denoise": PipelineConfig.dct_denoise(),
    "optimized-fast": PipelineConfig.optimized_fast(),
    "optimized-quality": PipelineConfig.optimized_quality(),
    "max-quality": PipelineConfig.max_quality(),
    "video-enhanced": PipelineConfig.video_enhanced(),
    "aggressive": PipelineConfig.aggressive(),
    "research-best": PipelineConfig.research_best(),
}


# ─── Edge Retention ──────────────────────────────────────────────────

def edge_retention(original: np.ndarray, processed: np.ndarray) -> float:
    """Ratio of edge magnitude preserved (Sobel-based). 1.0 = perfect."""
    def _edge_mag(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx**2 + gy**2)
    e_orig = _edge_mag(original)
    e_proc = _edge_mag(processed)
    return float(np.sum(e_proc) / max(np.sum(e_orig), 1e-10))


# ─── Ablation Test ──────────────────────────────────────────────────

def run_single_test(
    degraded: np.ndarray, cfg: PipelineConfig, label: str,
    original: np.ndarray | None = None
) -> dict:
    """Run a single pipeline config and return metrics."""
    t0 = time.time()
    restored = pl.apply_pipeline(degraded, cfg)
    elapsed = time.time() - t0

    result = {
        "label": label,
        "time_s": round(elapsed, 4),
        "filters": " → ".join(s.name for s in cfg.stages if s.enabled),
    }

    if original is not None:
        # Match shapes
        o, r = original.copy(), restored.copy()
        if o.shape != r.shape:
            r = cv2.resize(r, (o.shape[1], o.shape[0]))
        result["psnr"] = round(calculate_psnr(o, r), 2)
        result["ssim"] = round(calculate_ssim(o, r), 4)
        result["edge"] = round(edge_retention(o, r), 3)

    return result


def run_preset_suite(
    original: np.ndarray,
    degraded: np.ndarray,
    presets: list[str],
) -> list[dict]:
    results = []
    for name in presets:
        cfg = ALL_PRESETS[name]
        r = run_single_test(degraded, cfg, name, original)
        results.append(r)
        _print_result(r)
    return results


def run_ablation_suite(
    degraded: np.ndarray,
    base_cfg: PipelineConfig,
    filter_names: list[str],
    original: np.ndarray | None = None,
) -> list[dict]:
    """
    For each filter in filter_names, run the pipeline with it ON and OFF
    to measure its individual contribution.
    """
    results = []

    # Baseline: all filters in base_cfg as-is
    r_base = run_single_test(degraded, base_cfg, "full", original)
    results.append(r_base)
    _print_result(r_base)

    for fname in filter_names:
        # Check if the filter exists in the config
        fc = base_cfg.get(fname)
        if fc is None:
            continue

        # OFF variant
        cfg_off = base_cfg.copy()
        cfg_off.disable(fname)
        r_off = run_single_test(degraded, cfg_off, f"-{fname}", original)
        results.append(r_off)
        _print_result(r_off)

        # ON with altered params (stronger)
        cfg_on = base_cfg.copy()
        cfg_on.disable(fname)
        adjusted = {}
        for k, v in fc.params.items():
            if isinstance(v, bool):
                adjusted[k] = v
            elif isinstance(v, int):
                adjusted[k] = max(1, int(v * 1.5)) | 1  # ensure odd for kernel sizes
            elif isinstance(v, float):
                adjusted[k] = v * 1.5
            else:
                adjusted[k] = v
        cfg_on.add(fname, **adjusted)
        r_on = run_single_test(degraded, cfg_on, f"+{fname}(1.5x)", original)
        results.append(r_on)
        _print_result(r_on)

    return results


def run_param_tune(
    degraded: np.ndarray,
    base_cfg: PipelineConfig,
    filter_name: str,
    param_name: str,
    values: list[float],
    original: np.ndarray | None = None,
) -> list[dict]:
    """Vary a single parameter across values."""
    results = []
    for val in values:
        cfg = base_cfg.copy()
        fc = cfg.get(filter_name)
        if fc is None:
            cfg.add(filter_name, **{param_name: val})
        else:
            # Remove and re-add to keep order... just update param
            fc.params[param_name] = val
        r = run_single_test(degraded, cfg,
                            f"{filter_name}.{param_name}={val}",
                            original)
        results.append(r)
        _print_result(r)
    return results


# ─── Output ──────────────────────────────────────────────────────────

def _print_result(r: dict) -> None:
    parts = [f"  {r['label']:35s}"]
    if "psnr" in r:
        parts.append(f"PSNR={r['psnr']:6.2f}  SSIM={r['ssim']:.4f}  Edge={r['edge']:.2f}")
    parts.append(f"  {r['time_s']:.4f}s")
    print("  ".join(parts))


def write_csv(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        if not results:
            return
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  📄 Results saved → {path}")


def write_report(results: list[dict], path: str, header: str = "") -> None:
    """Write a human-readable markdown report."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# Ablation Study Report\n\n")
        f.write(f"**{header}**\n\n" if header else "")
        f.write("| Label | PSNR | SSIM | Edge | Time (s) | Filters |\n")
        f.write("|-------|------|------|------|----------|--------|\n")
        for r in results:
            psnr = f"{r['psnr']:.2f}" if "psnr" in r else "—"
            ssim = f"{r['ssim']:.4f}" if "ssim" in r else "—"
            edge = f"{r['edge']:.2f}" if "edge" in r else "—"
            f.write(f"| {r['label']} | {psnr} | {ssim} | {edge} | "
                    f"{r['time_s']:.4f} | {r.get('filters', '')} |\n")
    print(f"  📄 Report → {path}")


def save_grid_comparison(
    original: np.ndarray,
    degraded: np.ndarray,
    results: list[dict],
    path: str,
) -> str:
    """
    Grid comparison: top row = original | degraded, rows below = each result.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = len(results) + 1  # +1 for original+degraded row
    h, w = original.shape[:2]
    # Scale down if too large
    scale = min(1.0, 1200 / (w * 2))
    if scale < 1.0:
        disp_w, disp_h = int(w * scale), int(h * scale)
    else:
        disp_w, disp_h = w, h

    cell_w = disp_w
    cell_h = disp_h + 30  # label area
    canvas_h = cell_h * n + 10
    canvas_w = cell_w * 2 + 10
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    def _place(img, row, col, label=""):
        """Place a (disp_w x disp_h) image at grid position."""
        y0 = row * cell_h + 5
        x0 = col * cell_w + 5
        resized = cv2.resize(img, (disp_w, disp_h))
        canvas[y0:y0 + disp_h, x0:x0 + disp_w] = resized
        if label:
            cv2.putText(canvas, label, (x0 + 5, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Row 0: original | degraded
    _place(original, 0, 0, "Original")
    _place(degraded, 0, 1, "Degraded")

    for i, r in enumerate(results):
        label = r["label"]
        # Re-process to get output image
        cfg = ALL_PRESETS.get(label, None)
        if cfg is None:
            continue
        restored = pl.apply_pipeline(degraded, cfg)
        meta = f"PSNR={r.get('psnr', '—')} SSIM={r.get('ssim', '—')}"
        _place(restored, i + 1, 0, label)
        cv2.putText(canvas, meta,
                    (5 + cell_w, (i + 1) * cell_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imwrite(path, canvas)
    return path


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study runner — systematic pipeline evaluation",
    )
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE,
                        help="Input image path (default: REAL_WORLD_PICTURE)")
    parser.add_argument("--presets", type=str, default="",
                        help="Comma-separated preset names (default: all)")
    parser.add_argument("--degrade", type=str, default="basic",
                        choices=["none", "basic", "ntsc-light", "ntsc-medium", "ntsc-heavy"],
                        help="Degradation type")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="Degradation strength 0–1")
    parser.add_argument("--ablation", type=str, default="",
                        help="Comma-separated filter names for ON/OFF ablation")
    parser.add_argument("--param-tune", type=str, default="",
                        help="filter.param,val1,val2,... e.g. 'wiener.noise_var,100,200,400'")
    parser.add_argument("--grid", action="store_true",
                        help="Save visual grid comparison")
    parser.add_argument("--export", type=str, default="auto",
                        help="Output dir (default: output/ablation/)")

    args = parser.parse_args()

    # Load original
    original = cv2.imread(args.image)
    if original is None:
        print(f"  ⚠ Cannot read: {args.image}")
        sys.exit(1)

    img_name = os.path.splitext(os.path.basename(args.image))[0]

    # Degrade
    if args.degrade.startswith("ntsc"):
        degraded = degrade_image(original, use_ntsc=True,
                                 ntsc_intensity=args.degrade.split("-")[1])
    elif args.degrade == "none":
        degraded = original.copy()
    else:
        degraded = degrade_image(original, use_ntsc=False,
                                 strength=args.strength)

    print(f"\n🔧 Image:       {args.image} ({original.shape[1]}x{original.shape[0]})")
    print(f"   Degrade:     {args.degrade}  strength={args.strength}")
    print(f"   Deg PSNR:    {calculate_psnr(original, degraded):.2f} dB")
    print(f"   Deg SSIM:    {calculate_ssim(original, degraded):.4f}")
    print()

    # Determine test mode
    if args.param_tune:
        # Parse: filter.param,val1,val2,...
        parts = args.param_tune.split(",")
        fp = parts[0]  # e.g. "wiener.noise_var"
        filter_name, param_name = fp.split(".")
        values = [float(v) for v in parts[1:]]
        base_cfg = ALL_PRESETS.get("edge-preserve", PipelineConfig.edge_preserve())
        results = run_param_tune(degraded, base_cfg, filter_name, param_name, values, original)

    elif args.ablation:
        filter_names = [f.strip() for f in args.ablation.split(",")]
        base_cfg = ALL_PRESETS.get("edge-preserve", PipelineConfig.edge_preserve())
        results = run_ablation_suite(degraded, base_cfg, filter_names, original)

    else:
        # Preset comparison
        if args.presets:
            preset_names = [p.strip() for p in args.presets.split(",")]
        else:
            preset_names = list(ALL_PRESETS.keys())
        results = []
        for name in preset_names:
            if name not in ALL_PRESETS:
                print(f"  ⚠ Unknown preset: {name}")
                continue
            cfg = ALL_PRESETS[name]
            r = run_single_test(degraded, cfg, name, original)
            results.append(r)
            _print_result(r)

    print(f"\n  ✅ {len(results)} test(s) completed.")

    # Save
    out_dir = args.export if args.export != "auto" else os.path.join(OUT_DIR, img_name)
    os.makedirs(out_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = os.path.join(out_dir, f"ablation_{ts}.csv")
    write_csv(results, csv_path)

    # Markdown report
    md_path = os.path.join(out_dir, f"ablation_{ts}.md")
    write_report(results, md_path, header=f"{args.degrade} strength={args.strength}")

    # Optional grid
    if args.grid:
        grid_path = os.path.join(out_dir, f"grid_{ts}.png")
        try:
            save_grid_comparison(original, degraded, results, grid_path)
            print(f"  🖼️  Grid → {grid_path}")
        except Exception as e:
            print(f"  ⚠ Grid failed: {e}")

    # Print summary
    if results and "psnr" in results[0]:
        psnrs = [r["psnr"] for r in results if "psnr" in r]
        ssims = [r["ssim"] for r in results if "ssim" in r]
        if psnrs:
            best_idx = int(np.argmax(psnrs))
            print(f"\n  🏆 Best PSNR: {results[best_idx]['label']} = {psnrs[best_idx]:.2f} dB")
        if ssims:
            best_idx = int(np.argmax(ssims))
            print(f"  🏆 Best SSIM: {results[best_idx]['label']} = {ssims[best_idx]:.4f}")


if __name__ == "__main__":
    main()
