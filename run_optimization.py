#!/usr/bin/env python3
"""
Comprehensive Optimization & Benchmarking Script
=================================================
Tests ALL presets, sweeps BM3D/Retinex parameters, validates NTSC robustness,
and measures scaling behaviour — all results saved to output/optimization/.

Usage:
    python run_optimization.py
    python run_optimization.py --quick      (skip BM3D/Retinex sweeps)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.degradation import degrade_image
from smu_sig_prossessing.auto_evaluation import AutoEvaluator, EvalResult
from main import PRESETS  # all 24 presets

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_IMG = os.path.join(BASE, "input", "test_small.jpg")
OUT_DIR = os.path.join(BASE, "output", "optimization")
os.makedirs(OUT_DIR, exist_ok=True)

evaluator = AutoEvaluator()


# ─── Helpers ─────────────────────────────────────────────────────────

def load_origin() -> np.ndarray:
    """Load the reference (clean) image."""
    img = cv2.imread(INPUT_IMG)
    if img is None:
        raise FileNotFoundError(f"Cannot read {INPUT_IMG}")
    print(f"  Input: {os.path.basename(INPUT_IMG)}  {img.shape[1]}x{img.shape[0]}")
    return img


def benchmark_preset(origin: np.ndarray, degraded: np.ndarray,
                     preset_name: str, cfg: PipelineConfig,
                     label: str | None = None) -> tuple[EvalResult, float]:
    """Run one preset, return (EvalResult, elapsed_seconds)."""
    t0 = time.perf_counter()
    restored = pl.apply_pipeline(degraded, cfg)
    elapsed = time.perf_counter() - t0
    r = evaluator.evaluate(origin, restored, label=label or preset_name,
                           degraded=degraded, verbose=False)
    return r, elapsed


def make_degraded(origin: np.ndarray, use_ntsc: bool = False,
                  ntsc_intensity: str = "medium",
                  strength: float = 0.5) -> np.ndarray:
    """Create degraded image with requested parameters."""
    if use_ntsc:
        return degrade_image(origin, use_ntsc=True, ntsc_intensity=ntsc_intensity)
    return degrade_image(origin, use_ntsc=False, strength=strength)


def extract_metrics(r: EvalResult) -> dict:
    """Pull named metric values from an EvalResult into a flat dict."""
    m = {met.name: met.value for met in r.metrics}
    m["composite_score"] = r.composite_score
    m["label"] = r.label
    return m


def write_csv(rows: list[dict], path: str):
    """Write list of dicts to CSV, preserving key order of first row."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  📄 Saved → {path}")


def fmt_time(sec: float) -> str:
    if sec < 1:
        return f"{sec*1000:.1f}ms"
    return f"{sec:.3f}s"


# ═══════════════════════════════════════════════════════════════════════
#  TEST 1: Full Preset Benchmark
# ═══════════════════════════════════════════════════════════════════════

def test_full_preset_benchmark(origin: np.ndarray,
                                degraded: np.ndarray) -> list[dict]:
    """Benchmark EVERY preset, measure all 8 metrics + time."""
    print(f"\n{'='*70}")
    print("📊 TEST 1: Full Preset Benchmark")
    print(f"{'='*70}")
    print(f"  Degrade: strength=0.5 (basic)")
    print(f"  Presets: {len(PRESETS)} total")
    print(f"{'─'*70}")

    results: list[dict] = []
    for idx, (pname, cfg) in enumerate(PRESETS.items(), 1):
        try:
            r, elapsed = benchmark_preset(origin, degraded, pname, cfg)
            m = extract_metrics(r)
            m["time_sec"] = round(elapsed, 4)
            m["time_str"] = fmt_time(elapsed)
            m["preset"] = pname
            results.append(m)
            print(f"  [{idx:2d}/{len(PRESETS)}] {pname:22s}  "
                  f"Score={r.composite_score:5.1f}  "
                  f"PSNR={m.get('psnr',0):5.2f}  "
                  f"SSIM={m.get('ssim',0):.4f}  "
                  f"ΔE={m.get('color_fidelity',0):.2f}  "
                  f"Edge={m.get('edge_retention',0):.3f}  "
                  f"VIF={m.get('vif',0):.4f}  "
                  f"⏱ {m['time_str']}")
        except Exception as e:
            print(f"  [{idx:2d}/{len(PRESETS)}] {pname:22s}  ❌ FAILED: {e}")
            results.append({
                "preset": pname, "label": pname,
                "composite_score": -1, "time_sec": -1, "time_str": "FAILED",
                "psnr": -1, "ssim": -1, "color_fidelity": -1,
                "edge_retention": -1, "noise_level": -1, "detail_recovery": -1,
                "artifact_score": -1, "vif": -1,
            })

    # Sort by composite descending
    results.sort(key=lambda x: x["composite_score"], reverse=True)

    # Save ranking CSV
    csv_path = os.path.join(OUT_DIR, "ranking_by_composite.csv")
    write_csv(results, csv_path)

    # Print ranking table
    print(f"\n{'─'*70}")
    print("🏆 RANKING (by Composite Score)")
    print(f"{'─'*70}")
    print(f"  {'#':>3s}  {'Preset':22s}  {'Score':>6s}  {'PSNR':>5s}  {'SSIM':>6s}  "
          f"{'ΔE':>5s}  {'Edge':>5s}  {'Noise':>6s}  {'Detail':>6s}  "
          f"{'Artif':>6s}  {'VIF':>5s}  {'Time':>8s}")
    print(f"  {'─'*3}  {'─'*22}  {'─'*6}  {'─'*5}  {'─'*6}  "
          f"{'─'*5}  {'─'*5}  {'─'*6}  {'─'*6}  "
          f"{'─'*6}  {'─'*5}  {'─'*8}")
    for i, row in enumerate(results, 1):
        print(f"  {i:3d}  {row['preset']:22s}  "
              f"{row['composite_score']:6.1f}  "
              f"{row.get('psnr',0):5.2f}  "
              f"{row.get('ssim',0):6.4f}  "
              f"{row.get('color_fidelity',0):5.2f}  "
              f"{row.get('edge_retention',0):5.3f}  "
              f"{row.get('noise_level',0):6.1f}  "
              f"{row.get('detail_recovery',0):6.3f}  "
              f"{row.get('artifact_score',0):6.2f}  "
              f"{row.get('vif',0):5.4f}  "
              f"{row.get('time_str','?'):>8s}")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  TEST 2: BM3D Parameter Sweep
# ═══════════════════════════════════════════════════════════════════════

def test_bm3d_sweep(origin: np.ndarray, degraded: np.ndarray) -> list[dict]:
    """Sweep BM3D sigma_psd values."""
    print(f"\n{'='*70}")
    print("🔬 TEST 2: BM3D Parameter Sweep")
    print(f"{'='*70}")
    print(f"  Pipeline: median(3) → bm3d(sigma_psd=X) → channel_correction → unsharp")
    print(f"  Sweep: sigma_psd = [5, 10, 15, 20, 25, 30, 35, 40]")

    sigma_values = [5, 10, 15, 20, 25, 30, 35, 40]
    results: list[dict] = []

    for sigma in sigma_values:
        try:
            cfg = PipelineConfig(label=f"BM3D sweep σ={sigma}")
            cfg.add("median", ksize=3)
            cfg.add("bm3d", sigma_psd=sigma, stage_arg=3)
            cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
            cfg.add("unsharp_mask", strength=0.3, radius=0.5, threshold=5)

            r, elapsed = benchmark_preset(origin, degraded, f"bm3d_sigma_{sigma}",
                                           cfg)
            m = extract_metrics(r)
            m["sigma_psd"] = sigma
            m["time_sec"] = round(elapsed, 4)
            m["time_str"] = fmt_time(elapsed)
            results.append(m)
            print(f"  σ={sigma:2d}  Score={r.composite_score:5.1f}  "
                  f"PSNR={m['psnr']:5.2f}  SSIM={m['ssim']:.4f}  "
                  f"ΔE={m['color_fidelity']:.2f}  "
                  f"Edge={m['edge_retention']:.3f}  "
                  f"Noise={m['noise_level']:.1f}  "
                  f"Detail={m['detail_recovery']:.3f}  "
                  f"VIF={m['vif']:.4f}  ⏱{m['time_str']}")
        except Exception as e:
            print(f"  σ={sigma:2d}  ❌ FAILED: {e}")
            results.append({
                "sigma_psd": sigma, "label": f"bm3d_sigma_{sigma}",
                "composite_score": -1, "time_sec": -1, "time_str": "FAILED",
                "psnr": -1, "ssim": -1, "color_fidelity": -1,
                "edge_retention": -1, "noise_level": -1, "detail_recovery": -1,
                "artifact_score": -1, "vif": -1,
            })

    csv_path = os.path.join(OUT_DIR, "bm3d_sweep.csv")
    write_csv(results, csv_path)

    # Find best
    valid = [r for r in results if r["composite_score"] > 0]
    if valid:
        best = max(valid, key=lambda x: x["composite_score"])
        print(f"\n  🏆 Best BM3D sigma_psd = {best['sigma_psd']}  "
              f"(Composite: {best['composite_score']:.2f})")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  TEST 3: Retinex Parameter Sweep
# ═══════════════════════════════════════════════════════════════════════

def test_retinex_sweep(origin: np.ndarray, degraded: np.ndarray) -> list[dict]:
    """Sweep retinex scale combinations and gain/offset."""
    print(f"\n{'='*70}")
    print("🔬 TEST 3: Retinex Parameter Sweep")
    print(f"{'='*70}")

    # Scale combos to test
    scale_configs = [
        # (sigma_list, weights, label_suffix)
        ([15, 80, 250], [1/3, 1/3, 1/3], "MSRCP_std"),
        ([10, 50, 200], [1/3, 1/3, 1/3], "MSRCP_wide"),
        ([20, 100, 300], [1/3, 1/3, 1/3], "MSRCP_narrow"),
        ([15, 80], [0.5, 0.5], "SSR_15-80"),
        ([50, 200], [0.5, 0.5], "SSR_50-200"),
        ([15, 80, 250], [0.3, 0.4, 0.3], "MSRCP_weighted"),
    ]

    gain_offset_configs = [
        # (gain, offset, label_suffix)
        (5.0, 0.0, "g5.0_o0"),
        (3.0, 0.0, "g3.0_o0"),
        (8.0, 0.0, "g8.0_o0"),
        (5.0, 0.5, "g5.0_o0.5"),
        (5.0, -0.5, "g5.0_o-0.5"),
        (10.0, 0.0, "g10_o0"),
    ]

    results: list[dict] = []

    # Part A: Scale combinations (fixed gain=5, offset=0)
    print(f"\n  Part A — Scale combinations (gain=5, offset=0):")
    for sigma_list, weights, suffix in scale_configs:
        try:
            cfg = PipelineConfig(label=f"Retinex {suffix}")
            cfg.add("retinex", sigma_list=sigma_list, weights=weights,
                    gain=5.0, offset=0.0)
            cfg.add("bilateral", d=7, sigma_color=50, sigma_space=50)
            cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
            cfg.add("histogram_eq_clahe", clip_limit=2.0, tile_size=8)

            r, elapsed = benchmark_preset(origin, degraded,
                                           f"retinex_{suffix}", cfg)
            m = extract_metrics(r)
            m["test_type"] = "scale"
            m["sigma_list"] = str(sigma_list)
            m["weights"] = str(weights)
            m["gain"] = 5.0
            m["offset"] = 0.0
            m["config_label"] = suffix
            m["time_sec"] = round(elapsed, 4)
            m["time_str"] = fmt_time(elapsed)
            results.append(m)
            print(f"  {suffix:20s}  Score={r.composite_score:5.1f}  "
                  f"PSNR={m['psnr']:5.2f}  SSIM={m['ssim']:.4f}  "
                  f"ΔE={m['color_fidelity']:.2f}  Edge={m['edge_retention']:.3f}  "
                  f"Noise={m['noise_level']:.1f}  ⏱{m['time_str']}")
        except Exception as e:
            print(f"  {suffix:20s}  ❌ FAILED: {e}")

    # Part B: Gain/offset sweep (fixed scales [15,80,250])
    print(f"\n  Part B — Gain/offset sweep (scales=[15,80,250]):")
    for gain, offset, suffix in gain_offset_configs:
        try:
            cfg = PipelineConfig(label=f"Retinex {suffix}")
            cfg.add("retinex", sigma_list=[15, 80, 250],
                    weights=[1/3, 1/3, 1/3],
                    gain=gain, offset=offset)
            cfg.add("bilateral", d=7, sigma_color=50, sigma_space=50)
            cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
            cfg.add("histogram_eq_clahe", clip_limit=2.0, tile_size=8)

            r, elapsed = benchmark_preset(origin, degraded,
                                           f"retinex_{suffix}", cfg)
            m = extract_metrics(r)
            m["test_type"] = "gain_offset"
            m["sigma_list"] = "[15, 80, 250]"
            m["weights"] = "[1/3, 1/3, 1/3]"
            m["gain"] = gain
            m["offset"] = offset
            m["config_label"] = suffix
            m["time_sec"] = round(elapsed, 4)
            m["time_str"] = fmt_time(elapsed)
            results.append(m)
            print(f"  {suffix:20s}  Score={r.composite_score:5.1f}  "
                  f"PSNR={m['psnr']:5.2f}  SSIM={m['ssim']:.4f}  "
                  f"ΔE={m['color_fidelity']:.2f}  Edge={m['edge_retention']:.3f}  "
                  f"Noise={m['noise_level']:.1f}  ⏱{m['time_str']}")
        except Exception as e:
            print(f"  {suffix:20s}  ❌ FAILED: {e}")

    csv_path = os.path.join(OUT_DIR, "retinex_sweep.csv")
    write_csv(results, csv_path)

    valid = [r for r in results if r.get("composite_score", -1) > 0]
    if valid:
        best = max(valid, key=lambda x: x["composite_score"])
        print(f"\n  🏆 Best retinex config: {best.get('config_label','?')}  "
              f"(Composite: {best['composite_score']:.2f})")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  TEST 4: NTSC Robustness
# ═══════════════════════════════════════════════════════════════════════

def test_ntsc_robustness(origin: np.ndarray,
                          full_ranking: list[dict]) -> list[dict]:
    """Test top-5 presets against NTSC-heavy degradation."""
    print(f"\n{'='*70}")
    print("📺 TEST 4: NTSC Robustness Test")
    print(f"{'='*70}")

    # Get top-5 preset names from full ranking
    top5 = [r["preset"] for r in full_ranking[:5] if r["composite_score"] > 0]
    print(f"  Top-5 presets: {', '.join(top5)}")

    # NTSC-heavy degraded image
    degraded_ntsc = make_degraded(origin, use_ntsc=True,
                                   ntsc_intensity="heavy")
    print(f"  NTSC: heavy mode")

    results: list[dict] = []
    for idx, pname in enumerate(top5, 1):
        try:
            cfg = PRESETS[pname]
            r, elapsed = benchmark_preset(origin, degraded_ntsc, pname, cfg,
                                           label=f"{pname}_ntsc-heavy")
            m = extract_metrics(r)
            m["time_sec"] = round(elapsed, 4)
            m["time_str"] = fmt_time(elapsed)
            m["preset"] = pname
            results.append(m)
            print(f"  [{idx}/5] {pname:22s}  "
                  f"Score={r.composite_score:5.1f}  "
                  f"PSNR={m.get('psnr',0):5.2f}  "
                  f"SSIM={m.get('ssim',0):.4f}  "
                  f"ΔE={m.get('color_fidelity',0):.2f}  "
                  f"Edge={m.get('edge_retention',0):.3f}  "
                  f"Noise={m.get('noise_level',0):.1f}  "
                  f"VIF={m.get('vif',0):.4f}  "
                  f"⏱{m['time_str']}")
        except Exception as e:
            print(f"  [{idx}/5] {pname:22s}  ❌ FAILED: {e}")
            results.append({
                "preset": pname, "composite_score": -1,
                "time_sec": -1, "time_str": "FAILED",
                "psnr": -1, "ssim": -1, "color_fidelity": -1,
                "edge_retention": -1, "noise_level": -1,
                "detail_recovery": -1, "artifact_score": -1, "vif": -1,
            })

    csv_path = os.path.join(OUT_DIR, "ntsc_results.csv")
    write_csv(results, csv_path)

    valid = [r for r in results if r.get("composite_score", -1) > 0]
    if valid:
        best = max(valid, key=lambda x: x["composite_score"])
        print(f"\n  🏆 Best for NTSC-heavy: {best['preset']}  "
              f"(Composite: {best['composite_score']:.2f})")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  TEST 5: Strength Scaling
# ═══════════════════════════════════════════════════════════════════════

def test_strength_scaling(origin: np.ndarray,
                           full_ranking: list[dict]) -> list[dict]:
    """Test top-5 presets across different degradation strengths."""
    print(f"\n{'='*70}")
    print("📈 TEST 5: Strength Scaling Test")
    print(f"{'='*70}")

    # Get top-5 preset names
    top5 = [r["preset"] for r in full_ranking[:5] if r["composite_score"] > 0]
    strengths = [0.3, 0.5, 0.7, 1.0]

    results: list[dict] = []

    for pname in top5:
        cfg = PRESETS[pname]
        print(f"\n  Preset: {pname}")
        for s in strengths:
            try:
                degraded_s = make_degraded(origin, use_ntsc=False, strength=s)
                r, elapsed = benchmark_preset(origin, degraded_s, pname, cfg,
                                               label=f"{pname}_s{s}")
                m = extract_metrics(r)
                m["time_sec"] = round(elapsed, 4)
                m["time_str"] = fmt_time(elapsed)
                m["preset"] = pname
                m["strength"] = s
                results.append(m)
                print(f"    strength={s:.1f}  Score={r.composite_score:5.1f}  "
                      f"PSNR={m['psnr']:5.2f}  SSIM={m['ssim']:.4f}  "
                      f"ΔE={m['color_fidelity']:.2f}  Edge={m['edge_retention']:.3f}  "
                      f"Noise={m['noise_level']:.1f}  VIF={m['vif']:.4f}  "
                      f"⏱{m['time_str']}")
            except Exception as e:
                print(f"    strength={s:.1f}  ❌ FAILED: {e}")
                results.append({
                    "preset": pname, "strength": s,
                    "composite_score": -1, "time_sec": -1, "time_str": "FAILED",
                    "psnr": -1, "ssim": -1, "color_fidelity": -1,
                    "edge_retention": -1, "noise_level": -1,
                    "detail_recovery": -1, "artifact_score": -1, "vif": -1,
                })

    csv_path = os.path.join(OUT_DIR, "strength_scaling.csv")
    write_csv(results, csv_path)

    return results


# ═══════════════════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════

def write_recommendations(full_ranking: list[dict],
                           bm3d_results: list[dict],
                           retinex_results: list[dict],
                           ntsc_results: list[dict],
                           strength_results: list[dict]):
    """Write a summary recommendations file."""
    lines = []
    lines.append("# Optimization & Benchmark Results")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Input image: {INPUT_IMG}")
    lines.append(f"")
    lines.append(f"## 1. Full Preset Ranking (strength=0.5, basic)")
    lines.append(f"")
    lines.append(f"| Rank | Preset | Composite | PSNR | SSIM | ΔE | Edge | Time |")
    lines.append(f"|------|--------|-----------|------|------|----|------|------|")
    for i, row in enumerate(full_ranking[:10], 1):
        lines.append(
            f"| {i} | {row['preset']} | {row['composite_score']:.1f} | "
            f"{row.get('psnr',0):.2f} | {row.get('ssim',0):.4f} | "
            f"{row.get('color_fidelity',0):.2f} | "
            f"{row.get('edge_retention',0):.3f} | "
            f"{row.get('time_str','?')} |"
        )
    lines.append(f"")

    # Best BM3D
    valid_bm3d = [r for r in bm3d_results if r.get("composite_score", -1) > 0]
    if valid_bm3d:
        best_b = max(valid_bm3d, key=lambda x: x["composite_score"])
        lines.append(f"## 2. BM3D Parameter Sweep")
        lines.append(f"")
        lines.append(f"- **Optimal sigma_psd**: {best_b['sigma_psd']} "
                     f"(Composite: {best_b['composite_score']:.2f})")
        lines.append(f"- Full data: `bm3d_sweep.csv`")
        lines.append(f"")

    # Best Retinex
    valid_ret = [r for r in retinex_results if r.get("composite_score", -1) > 0]
    if valid_ret:
        best_rt = max(valid_ret, key=lambda x: x["composite_score"])
        lines.append(f"## 3. Retinex Parameter Sweep")
        lines.append(f"")
        lines.append(f"- **Optimal config**: {best_rt.get('config_label','?')} "
                     f"(Composite: {best_rt['composite_score']:.2f})")
        lines.append(f"- Full data: `retinex_sweep.csv`")
        lines.append(f"")

    # Best NTSC
    valid_ntsc = [r for r in ntsc_results if r.get("composite_score", -1) > 0]
    if valid_ntsc:
        best_n = max(valid_ntsc, key=lambda x: x["composite_score"])
        lines.append(f"## 4. NTSC Robustness (heavy)")
        lines.append(f"")
        lines.append(f"- **Best preset for NTSC-heavy**: {best_n['preset']} "
                     f"(Composite: {best_n['composite_score']:.2f})")
        lines.append(f"- Full data: `ntsc_results.csv`")
        lines.append(f"")

    # Strength scaling summary
    if strength_results:
        lines.append(f"## 5. Strength Scaling")
        lines.append(f"")
        # For each preset, show how score varies with strength
        presets_seen = set()
        for row in strength_results:
            if row["preset"] not in presets_seen:
                presets_seen.add(row["preset"])
                lines.append(f"- **{row['preset']}**: ",)
                for s in [0.3, 0.5, 0.7, 1.0]:
                    match = [r for r in strength_results
                             if r["preset"] == row["preset"] and r.get("strength") == s]
                    if match:
                        lines.append(f"    s={s:.1f} → Score={match[0].get('composite_score',0):.1f}  "
                                     f"PSNR={match[0].get('psnr',0):.2f}  "
                                     f"SSIM={match[0].get('ssim',0):.4f}")
        lines.append(f"")
        lines.append(f"- Full data: `strength_scaling.csv`")
        lines.append(f"")

    # Best overall recommendations
    lines.append(f"## Recommendations")
    lines.append(f"")

    if full_ranking and full_ranking[0]["composite_score"] > 0:
        best_overall = full_ranking[0]
        lines.append(f"### 🥇 Best Overall Preset")
        lines.append(f"- **{best_overall['preset']}** — "
                     f"Composite: {best_overall['composite_score']:.1f}, "
                     f"PSNR: {best_overall.get('psnr',0):.2f}, "
                     f"SSIM: {best_overall.get('ssim',0):.4f}, "
                     f"Time: {best_overall.get('time_str','?')}")

    # Find fastest among top-5
    top5_valid = [r for r in full_ranking[:5] if r.get("time_sec", -1) > 0]
    if top5_valid:
        fastest = min(top5_valid, key=lambda x: x["time_sec"])
        lines.append(f"### ⚡ Fastest in Top-5")
        lines.append(f"- **{fastest['preset']}** — "
                     f"{fastest.get('time_str','?')}, "
                     f"Score: {fastest['composite_score']:.1f}")

    if valid_bm3d:
        lines.append(f"### 🔬 Optimal BM3D Sigma")
        lines.append(f"- **sigma_psd = {best_b['sigma_psd']}** "
                     f"(Score: {best_b['composite_score']:.1f})")

    if valid_ntsc:
        lines.append(f"### 📺 Best for NTSC")
        lines.append(f"- **{best_n['preset']}** "
                     f"(Score: {best_n['composite_score']:.1f})")

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"All result files are in `output/optimization/`")

    path = os.path.join(OUT_DIR, "recommendations.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  📄 Recommendations → {path}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive optimization & benchmarking for SMU signal processing pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="Skip BM3D/Retinex parameter sweeps")
    args = parser.parse_args()

    print(f"{'═'*70}")
    print(f"  SMU Signal Processing — Optimization & Benchmarking")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output: {OUT_DIR}")
    print(f"{'═'*70}")

    # Load image
    try:
        origin = load_origin()
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    # Create standard degraded image
    degraded = make_degraded(origin, use_ntsc=False, strength=0.5)

    # ── TEST 1: Full Preset Benchmark ──────────────────────────────
    full_ranking = test_full_preset_benchmark(origin, degraded)

    # ── TEST 2: BM3D Sweep ────────────────────────────────────────
    bm3d_results = []
    if not args.quick:
        bm3d_results = test_bm3d_sweep(origin, degraded)
    else:
        print(f"\n  ⏭ Skipping BM3D sweep (--quick)")

    # ── TEST 3: Retinex Sweep ──────────────────────────────────────
    retinex_results = []
    if not args.quick:
        retinex_results = test_retinex_sweep(origin, degraded)
    else:
        print(f"\n  ⏭ Skipping Retinex sweep (--quick)")

    # ── TEST 4: NTSC Robustness ────────────────────────────────────
    ntsc_results = test_ntsc_robustness(origin, full_ranking)

    # ── TEST 5: Strength Scaling ───────────────────────────────────
    strength_results = test_strength_scaling(origin, full_ranking)

    # ── Write recommendations ──────────────────────────────────────
    write_recommendations(full_ranking, bm3d_results, retinex_results,
                           ntsc_results, strength_results)

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  ✅ ALL TESTS COMPLETE")
    print(f"  Results saved to: {OUT_DIR}")
    print(f"    - ranking_by_composite.csv")
    if not args.quick:
        print(f"    - bm3d_sweep.csv")
        print(f"    - retinex_sweep.csv")
    print(f"    - ntsc_results.csv")
    print(f"    - strength_scaling.csv")
    print(f"    - recommendations.md")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
