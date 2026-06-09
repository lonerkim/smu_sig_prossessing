#!/usr/bin/env python3
"""
Filter Interaction Analysis v3 — Targets NIQE <7.2 + BRISQUE <55 simultaneously.

Based on comprehensive benchmark results:
  Current best NIQE: optimal-bior4=7.26 (BRISQUE=75.63, 858ms)
  Current best BRISQUE: max-quality=42.05 (NIQE=8.79, 229ms)
  Best balance: chroma-focus NIQE=7.57 BRISQUE=55.00 (201ms)
  Fast: fast-guided-chroma NIQE=7.63 BRISQUE=55.42 (55ms)

Strategy:
  1. Combine bior4.4 wavelet (best NIQE) with chroma-focused post-processing
  2. Try different wavelet families (bior6.8, sym5)
  3. Vary wavelet levels and shift parameters
  4. Test grey_edge + wavelet combinations for color correction
  5. Try cascade: chroma_denoise → wavelet → detail_boost → chroma_denoise → unsharp
  6. Test bm4d_volume with gentle parameters (non-wavelet path)

Usage:
    .venv/bin/python3 run_filter_interaction_v3.py
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
OUTPUT_DIR = os.path.join(BASE, "output", "filter_interaction_v3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

evaluator = AutoEvaluator()


def make_combo(name: str, stages: list[tuple]) -> PipelineConfig:
    cfg = PipelineConfig(label=name)
    for filter_name, params in stages:
        cfg.add(filter_name, **params)
    return cfg


# ── New filter combinations ───────────────────────────────────────────

COMBOS: dict[str, PipelineConfig] = {}

# --- Strategy 1: Bior4.4 wavelet variants with chroma post-processing ---

COMBOS["bior4-chroma-strong"] = make_combo("Bior4+StrongChroma", [
    ("wavelet", {"wavelet": "bior4.4", "level": 2, "threshold_mode": "soft", "n_shifts": 3}),
    ("chroma_denoise", {"strength": 0.6}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
])

COMBOS["bior4-guided-chroma"] = make_combo("Bior4+Guided+Chroma", [
    ("wavelet", {"wavelet": "bior4.4", "level": 2, "threshold_mode": "soft", "n_shifts": 3}),
    ("guided_filter", {"radius": 3, "eps": 100.0}),
    ("chroma_denoise", {"strength": 0.4}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 10}),
])

# --- Strategy 2: Different wavelet families ---

COMBOS["bior6.8-wavelet"] = make_combo("Bior6.8+Detail+Chroma", [
    ("wavelet", {"wavelet": "bior6.8", "level": 2, "threshold_mode": "soft", "n_shifts": 3}),
    ("detail_boost", {"strength": 0.3, "sigma_s": 3.0, "sigma_r": 0.15, "threshold": 0.02}),
    ("chroma_denoise", {"strength": 0.3}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 5}),
])

COMBOS["bior6.8-light"] = make_combo("Bior6.8+LightPost", [
    ("wavelet", {"wavelet": "bior6.8", "level": 2, "threshold_mode": "soft", "n_shifts": 2}),
    ("chroma_denoise", {"strength": 0.3}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
])

COMBOS["sym5-wavelet"] = make_combo("Sym5+Detail+Chroma", [
    ("wavelet", {"wavelet": "sym5", "level": 2, "threshold_mode": "soft", "n_shifts": 3}),
    ("detail_boost", {"strength": 0.3, "sigma_s": 3.0, "sigma_r": 0.15, "threshold": 0.02}),
    ("chroma_denoise", {"strength": 0.3}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 5}),
])

COMBOS["db8-wavelet"] = make_combo("DB8+Detail+Chroma", [
    ("wavelet", {"wavelet": "db8", "level": 2, "threshold_mode": "soft", "n_shifts": 3}),
    ("detail_boost", {"strength": 0.3, "sigma_s": 3.0, "sigma_r": 0.15, "threshold": 0.02}),
    ("chroma_denoise", {"strength": 0.3}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 5}),
])

# --- Strategy 3: Wavelet level/parameter variations ---

COMBOS["bior4-level3"] = make_combo("Bior4.4 Level3", [
    ("wavelet", {"wavelet": "bior4.4", "level": 3, "threshold_mode": "soft", "n_shifts": 3}),
    ("detail_boost", {"strength": 0.25, "sigma_s": 3.0, "sigma_r": 0.15, "threshold": 0.02}),
    ("chroma_denoise", {"strength": 0.3}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 5}),
])

COMBOS["bior4-noshift"] = make_combo("Bior4.4 NoShift", [
    ("wavelet", {"wavelet": "bior4.4", "level": 2, "threshold_mode": "soft", "n_shifts": 1}),
    ("detail_boost", {"strength": 0.3, "sigma_s": 3.0, "sigma_r": 0.15, "threshold": 0.02}),
    ("chroma_denoise", {"strength": 0.3}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
])

# --- Strategy 4: Chroma-first pipeline ---

COMBOS["chroma-bior4-detail"] = make_combo("Chroma→Bior4→Detail", [
    ("chroma_denoise", {"strength": 0.5}),
    ("wavelet", {"wavelet": "bior4.4", "level": 2, "threshold_mode": "soft", "n_shifts": 3}),
    ("detail_boost", {"strength": 0.25, "sigma_s": 3.0, "sigma_r": 0.15, "threshold": 0.02}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 5}),
])

COMBOS["chroma-guided-bior4"] = make_combo("Chroma→Guided→Bior4", [
    ("chroma_denoise", {"strength": 0.5}),
    ("guided_filter", {"radius": 3, "eps": 100.0}),
    ("wavelet", {"wavelet": "bior4.4", "level": 2, "threshold_mode": "soft", "n_shifts": 3}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 10}),
])

# --- Strategy 5: Grey-edge + wavelet (color correction + denoise) ---

COMBOS["grey-bior4-chroma"] = make_combo("GreyEdge+Bior4+Chroma", [
    ("grey_edge", {"strength": 0.25, "sigma_smooth": 1.0}),
    ("wavelet", {"wavelet": "bior4.4", "level": 2, "threshold_mode": "soft", "n_shifts": 3}),
    ("chroma_denoise", {"strength": 0.3}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 5}),
])

COMBOS["grey-guided-chroma"] = make_combo("Grey+Guided+Chroma", [
    ("grey_edge", {"strength": 0.25, "sigma_smooth": 1.0}),
    ("guided_filter", {"radius": 3, "eps": 100.0}),
    ("chroma_denoise", {"strength": 0.4}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
])

# --- Strategy 6: BM4D variants with gentle post-processing ---

COMBOS["bm4d-gentle"] = make_combo("BM4D Gentle", [
    ("bm4d_volume", {"sigma_psd": 10.0, "temporal_window": 2, "max_frames": 8}),
    ("chroma_denoise", {"strength": 0.2}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 10}),
])

COMBOS["bm4d-guided"] = make_combo("BM4D+Guided", [
    ("bm4d_volume", {"sigma_psd": 12.0, "temporal_window": 2, "max_frames": 8}),
    ("guided_filter", {"radius": 3, "eps": 50.0}),
    ("chroma_denoise", {"strength": 0.3}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
])

# --- Strategy 7: Fast quality combos ---

COMBOS["guided-chroma-unsharp"] = make_combo("Guided+Chroma+Unsharp", [
    ("guided_filter", {"radius": 3, "eps": 100.0}),
    ("chroma_denoise", {"strength": 0.5}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.2, "radius": 0.5, "threshold": 5}),
])

COMBOS["crossbilat-detail-bior4"] = make_combo("CrossBilat→Detail→Bior4", [
    ("cross_bilateral", {"guide_sigma": 1.0, "d": 5, "sigma_color": 30, "sigma_space": 30}),
    ("detail_boost", {"strength": 0.25, "sigma_s": 3.0, "sigma_r": 0.15, "threshold": 0.02}),
    ("wavelet", {"wavelet": "bior4.4", "level": 2, "threshold_mode": "soft", "n_shifts": 2}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 10}),
])

# --- Strategy 8: Temporal + Bior4 ---

COMBOS["temporal-bior4"] = make_combo("TempNLM+Bior4", [
    ("temporal_nlm_multi", {"h": 8, "h_color": 8, "temporal_window": 2, "max_frames": 5}),
    ("wavelet", {"wavelet": "bior4.4", "level": 2, "threshold_mode": "soft", "n_shifts": 3}),
    ("chroma_denoise", {"strength": 0.2}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 10}),
])

# --- Strategy 9: Aggressive detail preservation ---

COMBOS["aniso-bior4"] = make_combo("AnisoDiff+Bior4+Chroma", [
    ("anisotropic_diffusion", {"kappa": 30, "n_iter": 5, "gamma": 0.15}),
    ("wavelet", {"wavelet": "bior4.4", "level": 2, "threshold_mode": "soft", "n_shifts": 3}),
    ("chroma_denoise", {"strength": 0.3}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.15, "radius": 0.5, "threshold": 5}),
])

# --- Strategy 10: NLM + Bior4 ---

COMBOS["nlm-bior4-chroma"] = make_combo("NLM+Bior4+Chroma", [
    ("nlm", {"h": 4, "template_window": 7, "search_window": 21}),
    ("wavelet", {"wavelet": "bior4.4", "level": 2, "threshold_mode": "soft", "n_shifts": 3}),
    ("chroma_denoise", {"strength": 0.2}),
    ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ("adaptive_equalize", {"clip_limit": 1.5, "tile_size": 8, "brightness_preserve": 0.4}),
    ("unsharp_mask", {"strength": 0.1, "radius": 0.5, "threshold": 10}),
])


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
    print(f"🧪 Filter Interaction Analysis v3 — {len(COMBOS)} new combinations")
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
                f"  {name:35s}  "
                f"NIQE={entry['niqe']:5.2f}  "
                f"BRISQUE={entry['brisque']:5.2f}  "
                f"Score={entry['composite_score']:5.2f}  "
                f"PSNR={entry['psnr']:5.2f}  "
                f"SSIM={entry['ssim']:.4f}  "
                f"{entry['time_ms']:7.1f}ms"
            )
        else:
            print(f"  {name:35s}  ❌ {entry.get('reason', 'error')}")

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
        print(f"{'Rank':<6} {'Name':<35} {metric:<12} {'NIQE':<7} {'BRISQUE':<8} {'Score':<7} {'Time(ms)':<9}")
        print(f"{'─' * 90}")
        for i, r in enumerate(sorted_r):
            print(
                f"  #{i+1:<2}  {r['name']:<35} "
                f"{r.get(metric, 0):<12.2f} "
                f"{r.get('niqe', 0):<7.2f} "
                f"{r.get('brisque', 0):<8.2f} "
                f"{r['composite_score']:<7.2f} "
                f"{r['time_ms']:<9.1f}"
            )

    # ── Comparison with current best presets ────────────────────────
    print(f"\n{'=' * 90}")
    print("📊 COMPARISON WITH CURRENT BEST PRESETS (from comprehensive benchmark)")
    print(f"{'=' * 90}")
    print(f"{'Preset':<35} {'NIQE':<7} {'BRISQUE':<8} {'Score':<7} {'Time(ms)':<9}")
    print(f"{'─' * 90}")
    best_known = [
        ("optimal-bior4", 7.26, 75.63, 63.29, 858.6),
        ("optimal-balanced", 7.28, 74.93, 63.39, 915.8),
        ("analog-clean", 7.33, 67.77, 70.23, 833.2),
        ("wavelet-denoise", 7.35, 67.62, 69.87, 776.0),
        ("chroma-focus", 7.57, 55.00, 66.61, 201.6),
        ("fast-guided-chroma", 7.63, 55.42, 68.02, 55.7),
        ("research-best", 7.86, 45.24, 65.29, 892.5),
        ("max-quality", 8.79, 42.05, 56.28, 229.0),
    ]
    for name, niqe, brisque, score, time_ms in best_known:
        print(f"  {name:<35} {niqe:<7.2f} {brisque:<8.2f} {score:<7.2f} {time_ms:<9.1f}")

    print(f"{'─' * 90}")
    print("  NEW COMBOS (top 5 by NIQE):")
    for r in sorted(valid, key=lambda x: x.get("niqe", 999))[:5]:
        print(
            f"  {r['name']:<35} {r['niqe']:<7.2f} {r['brisque']:<8.2f} "
            f"{r['composite_score']:<7.2f} {r['time_ms']:<9.1f}"
        )

    print(f"{'─' * 90}")
    print("  NEW COMBOS (top 5 by BRISQUE):")
    for r in sorted(valid, key=lambda x: x.get("brisque", 999))[:5]:
        print(
            f"  {r['name']:<35} {r['niqe']:<7.2f} {r['brisque']:<8.2f} "
            f"{r['composite_score']:<7.2f} {r['time_ms']:<9.1f}"
        )

    # ── Save ────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"filter_interaction_v3_{ts}.json")
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
