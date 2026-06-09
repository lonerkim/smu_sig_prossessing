#!/usr/bin/env python3
"""
Filter Interaction Analysis v2 — Optimize for NIQE/BRISQUE no-reference metrics.

Based on comprehensive benchmark findings (optimal-balanced NIQE=7.28, chroma-focus BRISQUE=55.00),
this script tests carefully chosen new filter combinations to improve on the current best.

Usage:
    .venv/bin/python3 run_filter_interaction_v2.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.auto_evaluation import AutoEvaluator
from smu_sig_prossessing.filters import reset_temporal_state

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO = os.path.join(BASE, "input", "analog_whoop_footage.mp4")
OUTPUT_DIR = os.path.join(BASE, "output", "filter_interaction_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

evaluator = AutoEvaluator()

# ── Define new filter combinations to test ──────────────────────────

def make_combo(name: str, stages: list[tuple]) -> PipelineConfig:
    """Create a PipelineConfig from a list of (filter_name, params_dict) tuples."""
    cfg = PipelineConfig(label=name)
    for filter_name, params in stages:
        cfg.add(filter_name, **params)
    return cfg

COMBOS: dict[str, PipelineConfig] = {
    # ── Variations on chroma-focus (best BRISQUE+NIQE balance) ──────
    "cf-wavelet": make_combo("Chroma+Wavelet", [
        ("chroma_denoise", {"strength": 0.5}),
        ("wavelet", {"wavelet": "db4", "level": 2, "threshold_mode": "soft"}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
    ]),

    "cf-rolling-strong": make_combo("Chroma+Rolling Strong", [
        ("chroma_denoise", {"strength": 0.8}),
        ("rolling_guidance", {"sigma_s": 4.0, "sigma_r": 0.12, "n_iter": 4}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
    ]),

    "cf-bilateral": make_combo("Chroma+Bilateral", [
        ("chroma_denoise", {"strength": 0.5}),
        ("bilateral", {"d": 7, "sigma_color": 50, "sigma_space": 50}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
    ]),

    # ── Optimal-balanced variants ───────────────────────────────────
    "ob-strong-detail": make_combo("OB Strong Detail", [
        ("wavelet", {"wavelet": "db4", "level": 3, "threshold_mode": "soft"}),
        ("detail_boost", {"strength": 0.5, "sigma_s": 3.0, "sigma_r": 0.15, "threshold": 0.02}),
        ("chroma_denoise", {"strength": 0.3}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 5}),
    ]),

    "ob-bior4": make_combo("OB Bior4.4 Wavelet", [
        ("wavelet", {"wavelet": "bior4.4", "level": 2, "threshold_mode": "soft", "n_shifts": 3}),
        ("detail_boost", {"strength": 0.3, "sigma_s": 3.0, "sigma_r": 0.15, "threshold": 0.02}),
        ("chroma_denoise", {"strength": 0.2}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 5}),
    ]),

    "ob-guided": make_combo("OB+Guided", [
        ("guided_filter", {"radius": 3, "eps": 100.0}),
        ("wavelet", {"wavelet": "db4", "level": 2, "threshold_mode": "soft"}),
        ("detail_boost", {"strength": 0.3, "sigma_s": 3.0, "sigma_r": 0.15, "threshold": 0.02}),
        ("chroma_denoise", {"strength": 0.2}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 5}),
    ]),

    # ── Novel hybrid combos ─────────────────────────────────────────
    "hybrid-rolling-wavelet": make_combo("Rolling+Wavelet", [
        ("rolling_guidance", {"sigma_s": 3.0, "sigma_r": 0.08, "n_iter": 3}),
        ("wavelet", {"wavelet": "db4", "level": 2, "threshold_mode": "soft"}),
        ("chroma_denoise", {"strength": 0.3}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
    ]),

    "hybrid-cross-wavelet": make_combo("CrossBilat+Wavelet", [
        ("cross_bilateral", {"guide_sigma": 1.0, "d": 5, "sigma_color": 30, "sigma_space": 30}),
        ("wavelet", {"wavelet": "db4", "level": 2, "threshold_mode": "soft"}),
        ("chroma_denoise", {"strength": 0.2}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
    ]),

    "hybrid-cross-chroma": make_combo("CrossBilat+Chroma+Detail", [
        ("cross_bilateral", {"guide_sigma": 1.0, "d": 5, "sigma_color": 30, "sigma_space": 30}),
        ("detail_boost", {"strength": 0.3, "sigma_s": 3.0, "sigma_r": 0.15, "threshold": 0.02}),
        ("chroma_denoise", {"strength": 0.3}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
    ]),

    # ── BRISQUE-focused combos ──────────────────────────────────────
    "bq-temporal-chroma": make_combo("Temp+Chroma BRISQUE", [
        ("temporal_nlm_multi", {"h": 10, "h_color": 8, "temporal_window": 2, "max_frames": 5}),
        ("chroma_denoise", {"strength": 0.4}),
        ("guided_filter", {"radius": 3, "eps": 50.0}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 10}),
    ]),

    "bq-bilateral-strong": make_combo("Bilateral Strong BRISQUE", [
        ("median", {"ksize": 3}),
        ("bilateral", {"d": 15, "sigma_color": 120, "sigma_space": 120}),
        ("wavelet", {"wavelet": "db4", "level": 2, "threshold_mode": "soft"}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ]),

    # ── Ultra-fast quality combos ───────────────────────────────────
    "fast-chroma-detail": make_combo("Fast Chroma+Detail", [
        ("chroma_denoise", {"strength": 0.4}),
        ("detail_boost", {"strength": 0.25, "sigma_s": 2.0, "sigma_r": 0.12, "threshold": 0.015}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
    ]),

    "fast-guided-chroma": make_combo("Fast Guided+Chroma", [
        ("guided_filter", {"radius": 3, "eps": 100.0}),
        ("chroma_denoise", {"strength": 0.3}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
    ]),

    # ── BM4D + chroma (bm4d is fast at 36ms!) ───────────────────────
    "bm4d-chroma": make_combo("BM4D+Chroma", [
        ("bm4d_volume", {"sigma_psd": 12.0, "temporal_window": 2, "max_frames": 8}),
        ("chroma_denoise", {"strength": 0.3}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
    ]),

    "bm4d-wavelet": make_combo("BM4D+Wavelet", [
        ("bm4d_volume", {"sigma_psd": 12.0, "temporal_window": 2, "max_frames": 8}),
        ("wavelet", {"wavelet": "db4", "level": 2, "threshold_mode": "soft"}),
        ("chroma_denoise", {"strength": 0.2}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 5}),
    ]),

    # ── NLM-based quality combos ────────────────────────────────────
    "nlm-chroma": make_combo("NLM+Chroma", [
        ("nlm", {"h": 6, "template_window": 7, "search_window": 21}),
        ("chroma_denoise", {"strength": 0.3}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
    ]),

    "nlm-wavelet-chroma": make_combo("NLM+Wavelet+Chroma", [
        ("nlm", {"h": 5, "template_window": 7, "search_window": 21}),
        ("wavelet", {"wavelet": "db4", "level": 2, "threshold_mode": "soft"}),
        ("chroma_denoise", {"strength": 0.2}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
        ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 5}),
    ]),
}


def get_first_frames(video_path: str, n: int = 3) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def evaluate_combo(name: str, cfg: PipelineConfig, frames: list[np.ndarray]) -> dict:
    try:
        reset_temporal_state()
        t0 = time.perf_counter()
        restored = pl.apply_pipeline(frames[0].copy(), cfg)
        elapsed = time.perf_counter() - t0

        result = evaluator.evaluate(
            frames[0], restored, label=name, degraded=frames[0], verbose=False
        )
        m = {mt.name: mt.value for mt in result.metrics}

        return {
            "name": name,
            "status": "ok",
            "composite_score": round(result.composite_score, 2),
            "niqe": round(m.get("niqe", 0), 2),
            "brisque": round(m.get("brisque", 0), 2),
            "psnr": round(m.get("psnr", 0), 2),
            "ssim": round(m.get("ssim", 0), 4),
            "color_fidelity_dE": round(m.get("color_fidelity", 0), 2),
            "edge_retention": round(m.get("edge_retention", 0), 3),
            "noise_level": round(m.get("noise_level", 0), 1),
            "detail_recovery": round(m.get("detail_recovery", 0), 3),
            "artifact_score": round(m.get("artifact_score", 0), 2),
            "vif": round(m.get("vif", 0), 4),
            "time_ms": round(elapsed * 1000, 1),
            "filters": " → ".join(s.name for s in cfg.stages if s.enabled),
        }
    except Exception as e:
        return {"name": name, "status": "error", "reason": str(e)}


def main():
    print(f"🧪 Filter Interaction Analysis v2 — {len(COMBOS)} new combinations")
    print(f"{'=' * 90}")

    frames = get_first_frames(INPUT_VIDEO, 3)
    if not frames:
        print("ERROR: Cannot load video")
        sys.exit(1)

    print(f"📹 Video: {os.path.basename(INPUT_VIDEO)} ({frames[0].shape[1]}x{frames[0].shape[0]})")
    print()

    results = []
    for name, cfg in sorted(COMBOS.items()):
        entry = evaluate_combo(name, cfg, frames)
        results.append(entry)

        if entry["status"] == "ok":
            print(
                f"  {name:30s}  "
                f"NIQE={entry['niqe']:5.2f}  "
                f"BRISQUE={entry['brisque']:5.2f}  "
                f"Score={entry['composite_score']:5.2f}  "
                f"PSNR={entry['psnr']:5.2f}  "
                f"SSIM={entry['ssim']:.4f}  "
                f"{entry['time_ms']:7.1f}ms"
            )
        else:
            print(f"  {name:30s}  ❌ {entry.get('reason', 'error')}")

    # ── Rankings ────────────────────────────────────────────────────
    valid = [r for r in results if r["status"] == "ok"]

    for metric, label, reverse in [
        ("niqe", "NIQE (lower=better)", False),
        ("brisque", "BRISQUE (lower=better)", False),
        ("composite_score", "Composite Score (higher=better)", True),
    ]:
        sorted_r = sorted(valid, key=lambda x: x.get(metric, 999), reverse=reverse)
        print(f"\n{'─' * 90}")
        print(f"🏆 RANKING BY {label}")
        print(f"{'─' * 90}")
        print(f"{'Rank':<6} {'Name':<30} {metric:<12} {'NIQE':<7} {'BRISQUE':<8} {'Score':<7} {'Time(ms)':<9} {'SSIM':<8}")
        print(f"{'─' * 90}")
        for i, r in enumerate(sorted_r):
            print(
                f"  #{i+1:<2}  {r['name']:<30} "
                f"{r.get(metric, 0):<12.2f} "
                f"{r.get('niqe', 0):<7.2f} "
                f"{r.get('brisque', 0):<8.2f} "
                f"{r['composite_score']:<7.2f} "
                f"{r['time_ms']:<9.1f} "
                f"{r.get('ssim', 0):<8.4f}"
            )

    # ── Comparison with current best presets ────────────────────────
    print(f"\n{'=' * 90}")
    print("📊 COMPARISON WITH CURRENT BEST PRESETS")
    print(f"{'=' * 90}")
    print(f"{'Preset':<30} {'NIQE':<7} {'BRISQUE':<8} {'Score':<7} {'Time(ms)':<9}")
    print(f"{'─' * 90}")
    best_known = [
        ("optimal-balanced", 7.28, 74.93, 63.39, 915.8),
        ("analog-clean", 7.33, 67.77, 70.23, 833.2),
        ("chroma-focus", 7.57, 55.00, 66.61, 201.6),
        ("bm4d-temporal", 7.41, 63.55, 66.35, 36.7),
    ]
    for name, niqe, brisque, score, time_ms in best_known:
        print(f"  {name:<30} {niqe:<7.2f} {brisque:<8.2f} {score:<7.2f} {time_ms:<9.1f}")

    print(f"{'─' * 90}")
    print("  NEW COMBOS:")
    for r in sorted(valid, key=lambda x: x.get("niqe", 999))[:5]:
        print(
            f"  {r['name']:<30} {r['niqe']:<7.2f} {r['brisque']:<8.2f} "
            f"{r['composite_score']:<7.2f} {r['time_ms']:<9.1f}"
        )

    # ── Save ────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"filter_interaction_v2_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "input": os.path.basename(INPUT_VIDEO),
            "num_combos": len(results),
            "results": results,
        }, f, indent=2)
    print(f"\n📄 Results saved → {out_path}")


if __name__ == "__main__":
    main()
