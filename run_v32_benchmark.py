#!/usr/bin/env python3
"""Benchmark new v3.2 presets + retinex optimization."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing.degradation import degrade_image
from smu_sig_prossessing.auto_evaluation import AutoEvaluator
from main import PRESETS

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output", "optimization")
os.makedirs(OUT_DIR, exist_ok=True)

evaluator = AutoEvaluator()

def main():
    img = cv2.imread(os.path.join(BASE, "input", "test_small.jpg"))
    if img is None: return
    small = cv2.resize(img, (400, 185))
    degraded = degrade_image(small, use_ntsc=False, strength=0.5)

    # Test new presets
    print("=== NEW v3.2 PRESETS ===")
    new_presets = ["video-ultra", "ntsc-plus", "fast-premium"]
    for pname in new_presets:
        try:
            cfg = PRESETS[pname]
            t0 = time.perf_counter()
            restored = pl.apply_pipeline(degraded, cfg)
            t = time.perf_counter() - t0
            res = evaluator.evaluate(small, restored, label=pname, degraded=degraded, verbose=False)
            print(f"  {pname:20s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
                  f"SSIM={res.get('ssim').value:.4f}  ΔE={res.get('color_fidelity').value:.2f}  "
                  f"VIF={res.get('vif').value:.4f}  {t*1000:7.1f}ms")
        except Exception as e:
            print(f"  {pname:20s}  ERROR: {e}")
    
    # Retinex optimization (gentler gain to preserve quality)
    print("\n=== RETINEX OPTIMIZATION ===")
    retinex_configs = [
        ("retinex-gentle", [15, 50, 120], 2.0, 0.0),
        ("retinex-soft", [15, 80, 250], 1.0, 0.0),
        ("retinex-mid", [15, 80, 250], 3.0, 0.0),
        ("retinex-aggressive", [15, 80, 250], 5.0, 0.0),
        ("retinex-single", [80], 2.0, 0.0),
        ("retinex-dual", [30, 150], 2.0, 0.0),
    ]
    for label, scales, gain, offset in retinex_configs:
        try:
            cfg = PipelineConfig(label=label)
            cfg.add("retinex", sigma_list=scales, gain=gain, offset=offset)
            cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
            t0 = time.perf_counter()
            restored = pl.apply_pipeline(degraded, cfg)
            t = time.perf_counter() - t0
            res = evaluator.evaluate(small, restored, label=label, degraded=degraded, verbose=False)
            print(f"  {label:20s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
                  f"ΔE={res.get('color_fidelity').value:.2f}  {t*1000:7.1f}ms")
        except Exception as e:
            print(f"  {label:20s}  ERROR: {e}")

    # NTSC comparison with new presets
    print("\n=== NTSC HEAVY (new presets) ===")
    ntsc_deg = degrade_image(small, use_ntsc=True, ntsc_intensity="heavy")
    for pname in ["ntsc-plus", "video-ultra", "wavelet-denoise", "video-enhanced", "edge-preserve"]:
        try:
            cfg = PRESETS[pname]
            restored = pl.apply_pipeline(ntsc_deg, cfg)
            res = evaluator.evaluate(small, restored, label=pname, degraded=ntsc_deg, verbose=False)
            print(f"  {pname:20s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
                  f"SSIM={res.get('ssim').value:.4f}  ΔE={res.get('color_fidelity').value:.2f}")
        except Exception as e:
            print(f"  {pname:20s}  ERROR: {e}")

    # Speed challenge: can we break 50fps with quality?
    print("\n=== SPEED CHALLENGE (quality vs speed frontier) ===")
    speed_presets = ["fast-denoise", "guided-denoise", "st-video", "optimized-fast", "fast-premium"]
    for pname in speed_presets:
        try:
            cfg = PRESETS[pname]
            # Time 50 iterations
            t0 = time.perf_counter()
            for _ in range(20):
                pl.apply_pipeline(degraded, cfg)
            t = (time.perf_counter() - t0) / 20
            restored = pl.apply_pipeline(degraded, cfg)
            res = evaluator.evaluate(small, restored, label=pname, degraded=degraded, verbose=False)
            fps = 1.0 / t if t > 0 else float('inf')
            print(f"  {pname:20s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
                  f"Time={t*1000:7.2f}ms  FPS={fps:6.1f}")
        except Exception as e:
            print(f"  {pname:20s}  ERROR: {e}")

    print(f"\n✅ v3.2 benchmark complete")

if __name__ == "__main__":
    main()
