#!/usr/bin/env python3
"""
Comprehensive optimization & benchmarking script.
Tests all presets, sweeps BM3D/Retinex params, evaluates NTSC robustness.
"""
from __future__ import annotations
import sys, os, csv, json, time, math
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.degradation import degrade_image
from smu_sig_prossessing.auto_evaluation import AutoEvaluator
from main import PRESETS

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output", "optimization")
os.makedirs(OUT_DIR, exist_ok=True)

evaluator = AutoEvaluator()

def load_test_img(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load {path}")
    return img

def time_pipeline(img: np.ndarray, cfg: PipelineConfig, n_warmup: int = 2, n_timed: int = 10) -> float:
    """Time a pipeline. Returns seconds per call."""
    for _ in range(n_warmup):
        pl.apply_pipeline(img, cfg)
    t0 = time.perf_counter()
    for _ in range(n_timed):
        pl.apply_pipeline(img, cfg)
    t1 = time.perf_counter()
    return (t1 - t0) / n_timed

def run_full_benchmark():
    print("=" * 70)
    print("  1. FULL PRESET BENCHMARK")
    print("=" * 70)
    img = load_test_img(os.path.join(BASE, "input", "test_small.jpg"))
    h, w = img.shape[:2]
    degraded = degrade_image(img, use_ntsc=False, strength=0.5)
    print(f"   Image: {w}x{h}, degrade=basic(0.5)")

    results = []
    preset_names = list(PRESETS.keys())
    # Skip adaptive for benchmark (it's dynamic)
    skip = {"adaptive"}
    for pname in preset_names:
        if pname in skip:
            continue
        try:
            cfg = PRESETS[pname]
            t_sec = time_pipeline(degraded, cfg)
            restored = pl.apply_pipeline(degraded, cfg)
            res = evaluator.evaluate(img, restored, label=pname, degraded=degraded, verbose=False)
            res.notes = f"{t_sec:.4f}s"
            results.append(res)
            print(f"   ✅ {pname:25s}  score={res.composite_score:6.2f}  time={t_sec:.4f}s")
        except Exception as e:
            print(f"   ❌ {pname:25s}  ERROR: {e}")

    # Sort by composite descending
    results.sort(key=lambda r: r.composite_score, reverse=True)
    
    # Save CSV
    csv_path = os.path.join(OUT_DIR, "ranking_by_composite.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["rank", "preset", "composite", "psnr", "ssim", "color_ciede2000",
                       "edge_retention", "noise_level", "detail_recovery", 
                       "artifact_score", "vif", "time_s"])
        for i, r in enumerate(results):
            m = {m.name: m.value for m in r.metrics}
            wtr.writerow([i+1, r.label, round(r.composite_score, 2),
                          round(m.get("psnr", 0), 3), round(m.get("ssim", 0), 4),
                          round(m.get("color_fidelity", 0), 4),
                          round(m.get("edge_retention", 0), 4),
                          round(m.get("noise_level", 0), 2),
                          round(m.get("detail_recovery", 0), 4),
                          round(m.get("artifact_score", 0), 4),
                          round(m.get("vif", 0), 4),
                          r.notes.replace("s", "")])
    print(f"\n   📁 Saved: {csv_path}")

    # Print top 10
    print(f"\n   🏆 TOP 10 PRESETS (by Composite Score):")
    print(f"   {'Rank':<6} {'Preset':<25} {'Score':<8} {'PSNR':<8} {'SSIM':<8} {'ΔE':<8} {'Edge':<8} {'Time':<8}")
    print(f"   " + "-"*75)
    for i, r in enumerate(results[:10]):
        m = {m.name: m.value for m in r.metrics}
        print(f"   #{i+1:<4} {r.label:<25} {r.composite_score:<8.2f} "
              f"{m.get('psnr',0):<8.2f} {m.get('ssim',0):<8.4f} "
              f"{m.get('color_fidelity',0):<8.2f} {m.get('edge_retention',0):<8.3f} "
              f"{r.notes}")
    return results

def run_bm3d_sweep():
    print("\n" + "=" * 70)
    print("  2. BM3D PARAMETER SWEEP")
    print("=" * 70)
    img = load_test_img(os.path.join(BASE, "input", "test_small.jpg"))
    degraded = degrade_image(img, use_ntsc=False, strength=0.5)

    sigmas = [5, 10, 15, 20, 25, 30, 35, 40]
    results = []
    for s in sigmas:
        try:
            cfg = PipelineConfig(label=f"BM3D σ={s}")
            cfg.add("median", ksize=3)
            cfg.add("bm3d", sigma_psd=s)
            cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
            cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)
            restored = pl.apply_pipeline(degraded, cfg)
            res = evaluator.evaluate(img, restored, label=f"bm3d_sigma_{s}", degraded=degraded, verbose=False)
            results.append((s, res))
            print(f"   σ={s:3d}  score={res.composite_score:6.2f}  psnr={res.get('psnr').value:.2f}")
        except Exception as e:
            print(f"   σ={s:3d}  ERROR: {e}")

    csv_path = os.path.join(OUT_DIR, "bm3d_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["sigma", "composite", "psnr", "ssim", "noise_level"])
        for s, r in results:
            m = {m.name: m.value for m in r.metrics}
            wtr.writerow([s, round(r.composite_score, 2), round(m.get("psnr",0), 3),
                         round(m.get("ssim",0), 4), round(m.get("noise_level",0), 2)])
    print(f"   📁 Saved: {csv_path}")

    if results:
        best_s, best_r = max(results, key=lambda x: x[1].composite_score)
        print(f"\n   🏆 Optimal BM3D sigma = {best_s} (score={best_r.composite_score:.2f})")
    return results

def run_retinex_sweep():
    print("\n" + "=" * 70)
    print("  3. RETINEX PARAMETER SWEEP")
    print("=" * 70)
    img = load_test_img(os.path.join(BASE, "input", "test_small.jpg"))
    degraded = degrade_image(img, use_ntsc=False, strength=0.5)

    configs = [
        ("retinex_small", [15, 50, 120]),
        ("retinex_medium", [15, 80, 250]),
        ("retinex_large", [30, 150, 400]),
        ("retinex_xlarge", [50, 200, 500]),
        ("retinex_single_50", [50]),
        ("retinex_single_250", [250]),
    ]
    results = []
    for label, scales in configs:
        try:
            cfg = PipelineConfig(label=label)
            cfg.add("retinex", sigma_list=scales)
            cfg.add("bilateral", d=5, sigma_color=30, sigma_space=30)
            cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
            cfg.add("histogram_eq_clahe", clip_limit=1.5, tile_size=8)
            restored = pl.apply_pipeline(degraded, cfg)
            res = evaluator.evaluate(img, restored, label=label, degraded=degraded, verbose=False)
            results.append((label, scales, res))
            print(f"   {label:20s}  scales={str(scales):20s}  score={res.composite_score:.2f}")
        except Exception as e:
            print(f"   {label:20s}  ERROR: {e}")

    csv_path = os.path.join(OUT_DIR, "retinex_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["label", "scales", "composite", "psnr", "ssim", "color_fidelity"])
        for label, scales, r in results:
            m = {m.name: m.value for m in r.metrics}
            wtr.writerow([label, str(scales), round(r.composite_score, 2),
                         round(m.get("psnr",0), 3), round(m.get("ssim",0), 4),
                         round(m.get("color_fidelity",0), 4)])
    print(f"   📁 Saved: {csv_path}")

    if results:
        best_label, best_scales, best_r = max(results, key=lambda x: x[2].composite_score)
        print(f"\n   🏆 Best Retinex: {best_label} scales={best_scales} (score={best_r.composite_score:.2f})")
    return results

def run_ntsc_test():
    print("\n" + "=" * 70)
    print("  4. NTSC ROBUSTNESS TEST")
    print("=" * 70)
    img = load_test_img(os.path.join(BASE, "input", "test_small.jpg"))
    degraded = degrade_image(img, use_ntsc=True, ntsc_intensity="heavy")
    print(f"   NTSC-heavy degrade applied")

    # Test top 5 candidates for NTSC
    candidates = ["wavelet-denoise", "bm3d-denoise", "retinex-bm3d", 
                  "video-enhanced", "optimized-fast", "bm3d-fast", "optimized-quality"]
    results = []
    for pname in candidates:
        try:
            cfg = PRESETS[pname]
            restored = pl.apply_pipeline(degraded, cfg)
            res = evaluator.evaluate(img, restored, label=pname, degraded=degraded, verbose=False)
            results.append((pname, res))
            print(f"   ✅ {pname:25s}  score={res.composite_score:6.2f}")
        except Exception as e:
            print(f"   ❌ {pname:25s}  ERROR: {e}")

    csv_path = os.path.join(OUT_DIR, "ntsc_results.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["preset", "composite", "psnr", "ssim", "color_fidelity", "edge_retention"])
        for pname, r in sorted(results, key=lambda x: x[1].composite_score, reverse=True):
            m = {m.name: m.value for m in r.metrics}
            wtr.writerow([pname, round(r.composite_score, 2), round(m.get("psnr",0), 3),
                         round(m.get("ssim",0), 4), round(m.get("color_fidelity",0), 4),
                         round(m.get("edge_retention",0), 4)])
    print(f"   📁 Saved: {csv_path}")

    if results:
        best_p, best_r = max(results, key=lambda x: x[1].composite_score)
        print(f"\n   🏆 Best for NTSC: {best_p} (score={best_r.composite_score:.2f})")
    return results

def run_strength_ablation():
    print("\n" + "=" * 70)
    print("  5. STRENGTH SCALING TEST")
    print("=" * 70)
    img = load_test_img(os.path.join(BASE, "input", "test_small.jpg"))
    strengths = [0.3, 0.5, 0.7, 1.0]
    
    # Test the top 5 presets across strengths
    top_presets = ["optimized-fast", "bm3d-denoise", "video-enhanced", 
                   "wavelet-denoise", "max-quality", "retinex-bm3d"]
    
    rows = []
    for s in strengths:
        degraded = degrade_image(img, use_ntsc=False, strength=s)
        for pname in top_presets:
            try:
                cfg = PRESETS[pname]
                restored = pl.apply_pipeline(degraded, cfg)
                res = evaluator.evaluate(img, restored, label=f"{pname} s={s}", degraded=degraded, verbose=False)
                rows.append((s, pname, res.composite_score, 
                           res.get("psnr").value, res.get("ssim").value))
                print(f"   s={s:.1f} {pname:25s}  score={res.composite_score:.2f}")
            except Exception as e:
                print(f"   s={s:.1f} {pname:25s}  ERROR: {e}")

    csv_path = os.path.join(OUT_DIR, "strength_results.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["strength", "preset", "composite", "psnr", "ssim"])
        for s, pname, comp, ps, ss in rows:
            wtr.writerow([s, pname, round(comp, 2), round(ps, 3), round(ss, 4)])
    print(f"   📁 Saved: {csv_path}")
    return rows

def generate_report(full_results, bm3d_results, retinex_results, ntsc_results, strength_rows):
    print("\n" + "=" * 70)
    print("  GENERATING REPORT")
    print("=" * 70)

    lines = []
    lines.append("# Optimization & Benchmark Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Test image: input/test_small.jpg (1600×740)")
    lines.append(f"Degrade: basic(0.5) unless noted")
    lines.append("")

    # Overall ranking
    lines.append("## Overall Preset Ranking (Composite Score)")
    lines.append("")
    lines.append("| Rank | Preset | Composite | PSNR | SSIM | ΔE²⁰⁰⁰ | Edge | VIF | Time(s) |")
    lines.append("|------|--------|-----------|------|------|--------|------|-----|---------|")
    for i, r in enumerate(full_results[:20]):
        m = {m.name: m.value for m in r.metrics}
        lines.append(f"| {i+1} | {r.label} | **{r.composite_score:.1f}** | "
                     f"{m.get('psnr',0):.2f} | {m.get('ssim',0):.4f} | "
                     f"{m.get('color_fidelity',0):.2f} | {m.get('edge_retention',0):.3f} | "
                     f"{m.get('vif',0):.4f} | {r.notes} |")
    lines.append("")

    # BM3D sweep
    if bm3d_results:
        lines.append("## BM3D Parameter Sweep")
        lines.append("")
        lines.append("| σ_psd | Composite | PSNR | SSIM | Noise Level |")
        lines.append("|-------|-----------|------|------|-------------|")
        best_s, best_r = max(bm3d_results, key=lambda x: x[1].composite_score)
        for s, r in bm3d_results:
            m = {m.name: m.value for m in r.metrics}
            marker = "🏆" if s == best_s else ""
            lines.append(f"| {s} {marker} | {r.composite_score:.1f} | "
                         f"{m.get('psnr',0):.2f} | {m.get('ssim',0):.4f} | {m.get('noise_level',0):.1f} |")
        lines.append("")

    # Retinex sweep
    if retinex_results:
        lines.append("## Retinex Scale Sweep")
        lines.append("")
        lines.append("| Config | Scales | Composite | PSNR | SSIM | ΔE²⁰⁰⁰ |")
        lines.append("|--------|--------|-----------|------|------|--------|")
        for label, scales, r in retinex_results:
            m = {m.name: m.value for m in r.metrics}
            lines.append(f"| {label} | {scales} | {r.composite_score:.1f} | "
                         f"{m.get('psnr',0):.2f} | {m.get('ssim',0):.4f} | "
                         f"{m.get('color_fidelity',0):.2f} |")
        lines.append("")

    # NTSC results
    if ntsc_results:
        lines.append("## NTSC Heavy Robustness")
        lines.append("")
        lines.append("| Preset | Composite | PSNR | SSIM | Edge |")
        lines.append("|--------|-----------|------|------|------|")
        for pname, r in sorted(ntsc_results, key=lambda x: x[1].composite_score, reverse=True):
            m = {m.name: m.value for m in r.metrics}
            lines.append(f"| {pname} | {r.composite_score:.1f} | "
                         f"{m.get('psnr',0):.2f} | {m.get('ssim',0):.4f} | "
                         f"{m.get('edge_retention',0):.3f} |")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    if full_results:
        best_all = full_results[0]
        lines.append(f"- **Best overall**: {best_all.label} ({best_all.composite_score:.1f})")
    if bm3d_results:
        lines.append(f"- **Best BM3D sigma**: σ={best_s} for basic noise (strength=0.5)")
    if retinex_results:
        best_label, best_scales, best_r = max(retinex_results, key=lambda x: x[2].composite_score)
        lines.append(f"- **Best Retinex config**: {best_label} scales={best_scales}")
    if ntsc_results:
        best_p, best_r = max(ntsc_results, key=lambda x: x[1].composite_score)
        lines.append(f"- **Best for NTSC heavy**: {best_p} ({best_r.composite_score:.1f})")
    
    lines.append("")
    lines.append("### Suggested New Top Presets (v3.2)")
    lines.append("")
    lines.append("Based on the benchmark, configure the following optimized presets:")
    lines.append("")
    lines.append("1. **bm3d-optimized**: median(3) → bm3d(σ_psd=15) → channel_correction → unsharp")
    lines.append("2. **retinex-bm3d**: median(3) → bm3d(σ_psd=15) → retinex(MSR) → channel_correction → unsharp")
    lines.append("3. **ntsc-bm3d**: bm3d(σ_psd=10) → wavelet → channel_correction → unsharp")

    report_path = os.path.join(OUT_DIR, "optimization_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"   📁 Saved: {report_path}")
    
    # Also save JSON summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_presets_tested": len(full_results),
        "top_preset": full_results[0].label if full_results else None,
        "top_score": full_results[0].composite_score if full_results else None,
        "bm3d_optimal_sigma": best_s if bm3d_results else None,
        "best_ntsc_preset": best_p if ntsc_results else None
    }
    json_path = os.path.join(OUT_DIR, "optimization_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   📁 Saved: {json_path}")


if __name__ == "__main__":
    print(f"🔬 SMU Signal Processing Optimization Suite")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Output: {OUT_DIR}")
    
    r1 = run_full_benchmark()
    r2 = run_bm3d_sweep()
    r3 = run_retinex_sweep()
    r4 = run_ntsc_test()
    r5 = run_strength_ablation()
    generate_report(r1, r2, r3, r4, r5)
    
    print(f"\n{'=' * 70}")
    print(f"✅ OPTIMIZATION COMPLETE")
    print(f"   Reports saved to: {OUT_DIR}")
    print(f"{'=' * 70}")
