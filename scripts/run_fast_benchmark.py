#!/usr/bin/env python3
"""Quick benchmark with small image."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.degradation import degrade_image
from smu_sig_prossessing.evaluation import evaluate
from main import PRESETS

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output", "optimization")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    img = cv2.imread(os.path.join(BASE, "input", "test_small.jpg"))
    if img is None:
        print("ERROR: cannot load image")
        return
    # Resize to 400x185 for fast benchmark
    small = cv2.resize(img, (400, 185))
    h, w = small.shape[:2]
    degraded = degrade_image(small, use_ntsc=False, strength=0.5)
    print(f"Benchmark image: {w}x{h} (resized from 1600x740 for speed)")
    
    skip = {"adaptive"}
    results = []
    for pname in sorted(PRESETS.keys()):
        if pname in skip:
            continue
        try:
            cfg = PRESETS[pname]
            t0 = time.perf_counter()
            restored = pl.apply_pipeline(degraded, cfg)
            t = time.perf_counter() - t0
            r = evaluate(small, restored, verbose=False)
            r["time"] = round(t, 4)
            results.append(r)
            print(f"  {pname:25s}  PSNR={r['psnr']:6.2f}  SSIM={r['ssim']:.4f}  {t*1000:6.1f}ms")
        except Exception as e:
            print(f"  {pname:25s}  ERROR: {e}")

    results.sort(key=lambda r: r["psnr"], reverse=True)
    print(f"\n{'='*60}")
    print("TOP 10 by PSNR:")
    print(f"{'Rank':<5} {'Preset':<25} {'PSNR':<7} {'SSIM':<7} {'Time':<7}")
    print("-"*51)
    for i, r in enumerate(results[:10]):
        print(f"  #{i+1:<2} {r['label']:<25} {r['psnr']:<7.2f} {r['ssim']:<7.4f} {r['time']*1000:<7.1f}ms")
    
    csv_out = os.path.join(OUT_DIR, "benchmark_results.csv")
    with open(csv_out, "w") as f:
        f.write("rank,preset,psnr,ssim,time_s\n")
        for i, r in enumerate(results):
            f.write(f"{i+1},{r['label']},{r['psnr']:.2f},{r['ssim']:.4f},{r['time']:.4f}\n")
    print(f"\nSaved: {csv_out}")

    # BM3D sweep
    print(f"\n{'='*60}")
    print("BM3D PARAM SWEEP:")
    for s in [5, 10, 15, 20, 25]:
        try:
            cfg = PipelineConfig(label=f"bm3d_{s}")
            cfg.add("bm3d", sigma_psd=s)
            t0 = time.perf_counter()
            restored = pl.apply_pipeline(degraded, cfg)
            t = time.perf_counter() - t0
            r = evaluate(small, restored, verbose=False)
            print(f"  σ={s:3d}  PSNR={r['psnr']:6.2f}  SSIM={r['ssim']:.4f}  {t*1000:6.1f}ms")
        except Exception as e:
            print(f"  σ={s:3d}  ERROR: {e}")

    # NTSC test
    print(f"\n{'='*60}")
    print("NTSC HEAVY:")
    ntsc_deg = degrade_image(small, use_ntsc=True, ntsc_intensity="heavy")
    for pname in ["wavelet-denoise", "bm3d-denoise", "video-enhanced", "optimized-fast", "retinex-bm3d"]:
        try:
            cfg = PRESETS[pname]
            restored = pl.apply_pipeline(ntsc_deg, cfg)
            r = evaluate(small, restored, verbose=False)
            print(f"  {pname:25s}  PSNR={r['psnr']:6.2f}  SSIM={r['ssim']:.4f}")
        except Exception as e:
            print(f"  {pname:25s}  ERROR: {e}")
    
    print(f"\n✅ Done")

if __name__ == "__main__":
    main()
