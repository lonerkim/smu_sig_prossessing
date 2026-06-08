#!/usr/bin/env python3
"""
Iter9 — Ablation on temporal-premium: disable one filter at a time.
Properly applies pipeline to degraded image, restores and compares to original.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing.auto_evaluation import AutoEvaluator
from smu_sig_prossessing.degradation import degrade_image

img = cv2.imread(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input", "test_small.jpg"))
small = cv2.resize(img, (400, 185))
degraded = degrade_image(small, use_ntsc=False, strength=0.5)

evaluator = AutoEvaluator()

base_cfg = PipelineConfig.temporal_premium()
filters = [s.name for s in base_cfg.stages if s.enabled]

print("=" * 70)
print("TEMPORAL-PREMIUM ABLATION (apply to degraded → compare vs original)")
print("=" * 70)

# Full pipeline on DEGRADED
t0 = time.perf_counter()
restored = pl.apply_pipeline(degraded, base_cfg)
t_full = time.perf_counter() - t0
res_full = evaluator.evaluate(small, restored, label='full', degraded=degraded, verbose=False)
print(f"\n  FULL:    Score={res_full.composite_score:6.2f}  PSNR={res_full.get('psnr').value:5.2f}  "
      f"SSIM={res_full.get('ssim').value:.4f}  DE={res_full.get('color_fidelity').value:.2f}  "
      f"VIF={res_full.get('vif').value:.4f}  {t_full*1000:.1f}ms")
print(f"  Filters: {' → '.join(filters)}")

# Ablation: remove one filter at a time
results = [('full', res_full.composite_score, 0.0)]
for fn in filters:
    cfg = base_cfg.copy()
    cfg.disable(fn)
    t0 = time.perf_counter()
    restored = pl.apply_pipeline(degraded, cfg)
    t = time.perf_counter() - t0
    res = evaluator.evaluate(small, restored, label=f'-{fn}', degraded=degraded, verbose=False)
    diff = res_full.composite_score - res.composite_score
    marker = "⚠ LOST" if diff > 0 else "✅ GAIN"
    results.append((f'-{fn}', res.composite_score, diff))
    print(f"  -{fn:25s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
          f"SSIM={res.get('ssim').value:.4f}  DE={res.get('color_fidelity').value:.2f}  "
          f"VIF={res.get('vif').value:.4f}  {marker} {diff:+.2f}  {t*1000:.1f}ms")

print(f"\n{'='*70}")
print("SUMMARY:")
print(f"{'Variant':<30s} {'Score':<8s} {'Diff':<8s}")
print("-" * 46)
for r in results:
    print(f"{r[0]:<30s} {r[1]:<8.2f} {r[2]:+8.2f}")
