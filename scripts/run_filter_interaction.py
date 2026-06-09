#!/usr/bin/env python3
"""
Filter Interaction Analysis — Test 2-filter and 3-filter combinations
systematically to find novel high-performing combos beyond existing presets.

Strategy:
  1. Test individual filters to establish baselines
  2. Test 2-filter combos (denoise + enhance, color + denoise, etc.)
  3. Test 3-filter combos (best denoise + best color + best enhance)
  4. Find combos that beat current best preset

Usage:
    .venv/bin/python3 run_filter_interaction.py
    .venv/bin/python3 run_filter_interaction.py --quick  # fewer combos
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from itertools import combinations

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.auto_evaluation import AutoEvaluator
from smu_sig_prossessing.filters import reset_temporal_state, FILTER_REGISTRY
from main import PRESETS as ALL_PRESETS

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO = os.path.join(BASE, "input", "analog_whoop_footage.mp4")
OUTPUT_DIR = os.path.join(BASE, "output", "filter_interaction")
os.makedirs(OUTPUT_DIR, exist_ok=True)

evaluator = AutoEvaluator()

# Define filter families and their default parameters
DENOISE_FILTERS = {
    "bilateral": {"d": 7, "sigma_color": 50, "sigma_space": 50},
    "cross_bilateral": {"guide_sigma": 1.0, "d": 5, "sigma_color": 30, "sigma_space": 30},
    "guided_filter": {"radius": 3, "eps": 100.0},
    "nlm": {"h": 5, "template_window": 7, "search_window": 21},
    "median": {"ksize": 3},
    "wavelet": {"wavelet": "db4", "level": 2, "threshold_mode": "soft"},
    "dct": {"patch_size": 8, "h_dct": 25.0},  # patch_collaborative
}

ENHANCE_FILTERS = {
    "unsharp_mask": {"strength": 0.3, "radius": 0.5, "threshold": 10},
    "histogram_eq_clahe": {"clip_limit": 1.5, "tile_size": 8},
    "adaptive_equalize": {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4},
    "channel_correction": {"clamp_min": 0.85, "clamp_max": 1.25},
    "detail_boost": {"strength": 0.3, "sigma_s": 3.0, "sigma_r": 0.15, "threshold": 0.02},
}

COLOR_FILTERS = {
    "chroma_denoise": {"strength": 0.3},
    "grey_edge": {"strength": 0.25, "sigma_smooth": 1.0},
}

TEMPORAL_FILTERS = {
    "temporal_motion": {"strength": 0.3},
    "temporal_nlm_multi": {"h": 8, "h_color": 8, "temporal_window": 3, "max_frames": 5},
}

ANALOG_FILTERS = {
    "scanline_remove": {"mode": "detect", "blend": 0.5},
    "flicker_stabilize": {"strength": 0.6, "window": 10},
}

# Best known presets for comparison
BENCHMARK_PRESETS = {
    "fast-denoise": ALL_PRESETS["fast-denoise"],
    "analog-clean": ALL_PRESETS["analog-clean"],
    "wavelet-denoise": ALL_PRESETS["wavelet-denoise"],
    "ultralight": ALL_PRESETS["ultralight"],
    "ntsc-plus": ALL_PRESETS["ntsc-plus"],
    "grey-ultralight": ALL_PRESETS["grey-ultralight"],
    "temporal-premium": ALL_PRESETS["temporal-premium"],
}

# Filter name -> registry mapping (some names differ)
FILTER_NAME_MAP = {
    "dct": "patch_collaborative",
}


def make_cfg(filters: list[tuple[str, dict]]) -> PipelineConfig:
    """Build a PipelineConfig from a list of (name, params) tuples."""
    cfg = PipelineConfig(label=" + ".join(f[0] for f in filters))
    for name, params in filters:
        reg_name = FILTER_NAME_MAP.get(name, name)
        cfg.add(reg_name, **params)
    return cfg


def run_single_test(
    cfg: PipelineConfig,
    label: str,
    frame: np.ndarray,
) -> dict:
    """Run a pipeline config on a frame and return metrics."""
    try:
        reset_temporal_state()
        t0 = time.perf_counter()
        restored = pl.apply_pipeline(frame.copy(), cfg)
        elapsed = time.perf_counter() - t0

        result = evaluator.evaluate(frame, restored, label=label,
                                     degraded=frame, verbose=False)
        m = {mt.name: mt.value for mt in result.metrics}

        return {
            "label": label,
            "status": "ok",
            "composite_score": round(result.composite_score, 2),
            "psnr": round(m.get("psnr", 0), 2),
            "ssim": round(m.get("ssim", 0), 4),
            "color_fidelity_dE": round(m.get("color_fidelity", 0), 2),
            "edge_retention": round(m.get("edge_retention", 0), 3),
            "noise_level": round(m.get("noise_level", 0), 1),
            "detail_recovery": round(m.get("detail_recovery", 0), 3),
            "vif": round(m.get("vif", 0), 4),
            "niqe": round(m.get("niqe", 0), 2),
            "time_ms": round(elapsed * 1000, 1),
        }
    except Exception as e:
        return {"label": label, "status": "error", "reason": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Filter interaction analysis")
    parser.add_argument("--quick", action="store_true",
                        help="Run fewer combinations (skip 3-filter combos)")
    args = parser.parse_args()

    # Load a single frame from the input video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"ERROR: Cannot read {INPUT_VIDEO}")
        sys.exit(1)

    print(f"📹 Testing on first frame of {os.path.basename(INPUT_VIDEO)}")
    print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print()

    all_results = []

    # ── Phase 1: Run benchmark presets ──────────────────────────────
    print("=" * 80)
    print("PHASE 1: Benchmark Presets")
    print("=" * 80)
    for name, cfg in BENCHMARK_PRESETS.items():
        r = run_single_test(cfg, f"[BENCH] {name}", frame)
        all_results.append(r)
        if r["status"] == "ok":
            print(f"  {r['label']:40s} Score={r['composite_score']:6.2f}  "
                  f"PSNR={r['psnr']:5.2f}  NIQE={r['niqe']:.2f}  {r['time_ms']:7.1f}ms")
        else:
            print(f"  {name:40s} ERROR: {r.get('reason', '')}")

    # ── Phase 2: Individual filters ─────────────────────────────────
    print(f"\n{'=' * 80}")
    print("PHASE 2: Individual Filter Baselines")
    print("=" * 80)

    all_filter_configs = {}
    all_filter_configs.update(DENOISE_FILTERS)
    all_filter_configs.update(ENHANCE_FILTERS)
    all_filter_configs.update(COLOR_FILTERS)

    for name, params in all_filter_configs.items():
        cfg = make_cfg([(name, params)])
        r = run_single_test(cfg, f"[SINGLE] {name}", frame)
        all_results.append(r)
        if r["status"] == "ok":
            print(f"  {r['label']:40s} Score={r['composite_score']:6.2f}  "
                  f"PSNR={r['psnr']:5.2f}  NIQE={r['niqe']:.2f}  {r['time_ms']:7.1f}ms")

    # ── Phase 3: 2-filter combinations ──────────────────────────────
    print(f"\n{'=' * 80}")
    print("PHASE 3: 2-Filter Combinations (Denoise + Enhance)")
    print("=" * 80)

    # Denoise + Enhance pairs
    for den_name, den_params in DENOISE_FILTERS.items():
        for enh_name, enh_params in ENHANCE_FILTERS.items():
            label = f"[2F] {den_name}+{enh_name}"
            cfg = make_cfg([(den_name, den_params), (enh_name, enh_params)])
            r = run_single_test(cfg, label, frame)
            all_results.append(r)
            if r["status"] == "ok":
                print(f"  {r['label']:40s} Score={r['composite_score']:6.2f}  "
                      f"PSNR={r['psnr']:5.2f}  NIQE={r['niqe']:.2f}  {r['time_ms']:7.1f}ms")

    # Denoise + Color pairs
    for den_name, den_params in DENOISE_FILTERS.items():
        for col_name, col_params in COLOR_FILTERS.items():
            label = f"[2F] {den_name}+{col_name}"
            cfg = make_cfg([(col_name, col_params), (den_name, den_params)])
            r = run_single_test(cfg, label, frame)
            all_results.append(r)
            if r["status"] == "ok":
                print(f"  {r['label']:40s} Score={r['composite_score']:6.2f}  "
                      f"PSNR={r['psnr']:5.2f}  NIQE={r['niqe']:.2f}  {r['time_ms']:7.1f}ms")

    # ── Phase 4: 3-filter combinations (best combos) ────────────────
    if not args.quick:
        print(f"\n{'=' * 80}")
        print("PHASE 4: 3-Filter Combinations")
        print("=" * 80)

        # Top denoisers × top enhancers × color filters
        top_denoisers = ["bilateral", "cross_bilateral", "guided_filter", "wavelet", "nlm"]
        top_enhancers = ["channel_correction", "unsharp_mask", "adaptive_equalize"]
        top_colors = ["chroma_denoise", "grey_edge"]

        for den in top_denoisers:
            for enh in top_enhancers:
                for col in top_colors:
                    label = f"[3F] {col}+{den}+{enh}"
                    cfg = make_cfg([
                        (col, COLOR_FILTERS[col]),
                        (den, DENOISE_FILTERS[den]),
                        (enh, ENHANCE_FILTERS[enh]),
                    ])
                    r = run_single_test(cfg, label, frame)
                    all_results.append(r)
                    if r["status"] == "ok":
                        print(f"  {r['label']:40s} Score={r['composite_score']:6.2f}  "
                              f"PSNR={r['psnr']:5.2f}  NIQE={r['niqe']:.2f}  {r['time_ms']:7.1f}ms")

    # ── Phase 5: Novel combos inspired by best presets ──────────────
    print(f"\n{'=' * 80}")
    print("PHASE 5: Novel Combination Candidates")
    print("=" * 80)

    novel_combos = [
        # Inspired by fast-denoise (best composite), try variants
        ([("bilateral", {"d": 5, "sigma_color": 30, "sigma_space": 30}),
          ("unsharp_mask", {"strength": 0.2, "radius": 0.5, "threshold": 10}),
          ("chroma_denoise", {"strength": 0.2})],
         "bilateral+unsharp+chroma"),
        # Cross_bilateral variants
        ([("cross_bilateral", {"guide_sigma": 1.0, "d": 5, "sigma_color": 30, "sigma_space": 30}),
          ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
          ("detail_boost", {"strength": 0.3, "sigma_s": 3.0, "sigma_r": 0.15, "threshold": 0.02})],
         "crossbilat+chan+detailboost"),
        # Wavelet + grey_edge (chroma-free variant)
        ([("grey_edge", {"strength": 0.22, "sigma_smooth": 1.0}),
          ("wavelet", {"wavelet": "db4", "level": 2, "threshold_mode": "soft"}),
          ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
          ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4})],
         "grey+wavelet+adapteq"),
        # Bilateral + CLAHE + detail_boost (fast denoising variant)
        ([("bilateral", {"d": 5, "sigma_color": 30, "sigma_space": 30}),
          ("histogram_eq_clahe", {"clip_limit": 1.5, "tile_size": 8}),
          ("detail_boost", {"strength": 0.25, "sigma_s": 3.0, "sigma_r": 0.15, "threshold": 0.02})],
         "bilateral+clahe+detailboost"),
        # Temporal_motion + guided + channel (ultra-fast temporal)
        ([("temporal_motion", {"strength": 0.3}),
          ("guided_filter", {"radius": 3, "eps": 100.0}),
          ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25})],
         "temporal+guided+chan"),
        # Rolling guidance + chroma + unsharp
        ([("chroma_denoise", {"strength": 0.3}),
          ("rolling_guidance", {"sigma_s": 3.0, "sigma_r": 0.08, "n_iter": 3}),
          ("unsharp_mask", {"strength": 0.2, "radius": 0.5, "threshold": 10}),
          ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25})],
         "chroma+rolling+unsharp+chan"),
        # Wavelet + nlm (dual denoising approach)
        ([("nlm", {"h": 3, "template_window": 7, "search_window": 21}),
          ("wavelet", {"wavelet": "db4", "level": 2, "threshold_mode": "soft"}),
          ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25})],
         "nlm+wavelet+chan"),
        # Ultralight + grey_edge variant
        ([("grey_edge", {"strength": 0.3, "sigma_smooth": 0.5}),
          ("cross_bilateral", {"guide_sigma": 1.0, "d": 5, "sigma_color": 30, "sigma_space": 30}),
          ("unsharp_mask", {"strength": 0.2, "radius": 0.5, "threshold": 10})],
         "grey+crossbilat+unsharp"),
        # Try bilateral with double unsharp
        ([("bilateral", {"d": 5, "sigma_color": 25, "sigma_space": 25}),
          ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
          ("unsharp_mask", {"strength": 0.15, "radius": 0.3, "threshold": 3})],
         "bilateral+unsharp2x"),
    ]

    for stages, label in novel_combos:
        cfg = make_cfg(stages)
        r = run_single_test(cfg, f"[NOVEL] {label}", frame)
        all_results.append(r)
        if r["status"] == "ok":
            print(f"  {r['label']:40s} Score={r['composite_score']:6.2f}  "
                  f"PSNR={r['psnr']:5.2f}  NIQE={r['niqe']:.2f}  {r['time_ms']:7.1f}ms")

    # ── Rank all results ────────────────────────────────────────────
    valid = [r for r in all_results if r.get("status") == "ok"]
    valid.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

    print(f"\n{'=' * 100}")
    print("🏆 FINAL RANKING (all combinations by Composite Score)")
    print(f"{'=' * 100}")
    print(f"{'Rank':<6} {'Label':<42} {'Score':<8} {'PSNR':<7} {'NIQE':<7} "
          f"{'ΔE':<7} {'VIF':<7} {'Time(ms)':<9}")
    print("-" * 100)
    for i, r in enumerate(valid[:50]):
        print(
            f"  #{i+1:<2}  {r['label']:<42} "
            f"{r['composite_score']:<8.2f} "
            f"{r['psnr']:<7.2f} "
            f"{r['niqe']:<7.2f} "
            f"{r['color_fidelity_dE']:<7.2f} "
            f"{r['vif']:<7.4f} "
            f"{r['time_ms']:<9.1f}"
        )

    # ── Save results ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "input": os.path.basename(INPUT_VIDEO),
        "stats": {
            "total_tests": len(all_results),
            "succeeded": len(valid),
            "failed": len(all_results) - len(valid),
        },
        "results": valid,
    }

    json_path = os.path.join(OUTPUT_DIR, f"filter_interaction_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n📄 JSON results → {json_path}")

    print(f"\n✅ Filter interaction analysis complete — {len(valid)} successful tests")


if __name__ == "__main__":
    main()
