#!/usr/bin/env python3
"""
Iter9 — Parameter sweep on temporal-premium key params.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing.auto_evaluation import AutoEvaluator
from smu_sig_prossessing.degradation import degrade_image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
evaluator = AutoEvaluator()

def make_cfg(label, *stages):
    cfg = PipelineConfig(label=label)
    for name, enabled, params in stages:
        cfg.add(name, enabled=enabled, **params)
    return cfg

def test_preset(name, cfg, origin, degraded_img):
    t0 = time.perf_counter()
    restored = pl.apply_pipeline(degraded_img, cfg)
    t = time.perf_counter() - t0
    res = evaluator.evaluate(origin, restored, label=name, degraded=degraded_img, verbose=False)
    return res, t

img = cv2.imread(os.path.join(BASE, "input", "test_small.jpg"))
small = cv2.resize(img, (400, 185))
degraded = degrade_image(small, use_ntsc=False, strength=0.5)
ntsc_deg = degrade_image(small, use_ntsc=True, ntsc_intensity="heavy")

# Base config template
def tp_cfg(**overrides):
    params = dict(h=8, h_color=8, temporal_window=3, max_frames=5,
                  gf_eps=50.0, cd_strength=0.2, ae_clip=1.5, us_strength=0.1)
    params.update(overrides)
    return make_cfg("tp",
        ("temporal_nlm_multi", True, dict(h=params['h'], h_color=params['h_color'], 
                                           temporal_window=params['temporal_window'], max_frames=params['max_frames'])),
        ("guided_filter", True, dict(radius=3, eps=params['gf_eps'])),
        ("chroma_denoise", True, dict(strength=params['cd_strength'])),
        ("channel_correction", True, dict(clamp_min=0.85, clamp_max=1.25)),
        ("adaptive_equalize", True, dict(clip_limit=params['ae_clip'], tile_size=8, brightness_preserve=0.4)),
        ("unsharp_mask", True, dict(strength=params['us_strength'], radius=0.5, threshold=10)),
    )

print("=" * 70)
print("PARAMETER SWEEP: temporal_nlm_multi h")
print("=" * 70)
results = []
for h_val in [4, 5, 6, 7, 8, 10, 12, 15]:
    cfg = tp_cfg(h=h_val, h_color=h_val)
    res, t = test_preset(f"TP(h={h_val})", cfg, small, degraded)
    results.append((h_val, res, t))
    print(f"  h={h_val:3d}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
          f"SSIM={res.get('ssim').value:.4f}  DE={res.get('color_fidelity').value:.2f}  "
          f"VIF={res.get('vif').value:.4f}  {t*1000:.1f}ms")

if results:
    best = max(results, key=lambda x: x[1].composite_score)
    print(f"\n  🏆 Best h = {best[0]} (Score={best[1].composite_score:.2f})")

# ── Guided filter eps sweep ──
print(f"\n{'='*70}")
print("PARAMETER SWEEP: guided_filter eps")
print("=" * 70)
results2 = []
for eps_val in [10, 25, 50, 75, 100, 150, 200]:
    cfg = tp_cfg(gf_eps=eps_val)
    res, t = test_preset(f"TP(gf_eps={eps_val})", cfg, small, degraded)
    results2.append((eps_val, res, t))
    print(f"  eps={eps_val:4d}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
          f"SSIM={res.get('ssim').value:.4f}  DE={res.get('color_fidelity').value:.2f}  "
          f"VIF={res.get('vif').value:.4f}  {t*1000:.1f}ms")

if results2:
    best2 = max(results2, key=lambda x: x[1].composite_score)
    print(f"\n  🏆 Best eps = {best2[0]} (Score={best2[1].composite_score:.2f})")

# ── Chroma_denoise strength sweep ──
print(f"\n{'='*70}")
print("PARAMETER SWEEP: chroma_denoise strength")
print("=" * 70)
results3 = []
for cd_val in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
    cfg = tp_cfg(cd_strength=cd_val)
    res, t = test_preset(f"TP(cd={cd_val})", cfg, small, degraded)
    results3.append((cd_val, res, t))
    print(f"  cd={cd_val:4.1f}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
          f"SSIM={res.get('ssim').value:.4f}  DE={res.get('color_fidelity').value:.2f}  "
          f"VIF={res.get('vif').value:.4f}  {t*1000:.1f}ms")

if results3:
    best3 = max(results3, key=lambda x: x[1].composite_score)
    print(f"\n  🏆 Best cd_strength = {best3[0]} (Score={best3[1].composite_score:.2f})")

# ── Unsharp mask strength sweep ──
print(f"\n{'='*70}")
print("PARAMETER SWEEP: unsharp_mask strength")
print("=" * 70)
results4 = []
for us_val in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
    cfg = tp_cfg(us_strength=us_val)
    res, t = test_preset(f"TP(us={us_val})", cfg, small, degraded)
    results4.append((us_val, res, t))
    print(f"  us={us_val:4.1f}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
          f"SSIM={res.get('ssim').value:.4f}  DE={res.get('color_fidelity').value:.2f}  "
          f"VIF={res.get('vif').value:.4f}  {t*1000:.1f}ms")

if results4:
    best4 = max(results4, key=lambda x: x[1].composite_score)
    print(f"\n  🏆 Best us_strength = {best4[0]} (Score={best4[1].composite_score:.2f})")

# ── Combined optimal ──
print(f"\n{'='*70}")
print("COMBINED OPTIMAL (best params from sweeps)")
print("=" * 70)
best_h = max(results, key=lambda x: x[1].composite_score)[0]
best_eps = max(results2, key=lambda x: x[1].composite_score)[0]
best_cd = max(results3, key=lambda x: x[1].composite_score)[0]
best_us = max(results4, key=lambda x: x[1].composite_score)[0]

print(f"  Best params: h={best_h}, gf_eps={best_eps}, cd={best_cd}, us={best_us}")
opt_cfg = tp_cfg(h=best_h, h_color=best_h, gf_eps=best_eps, cd_strength=best_cd, us_strength=best_us)
res, t = test_preset("TP-optimal", opt_cfg, small, degraded)
print(f"  TP-optimal:  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
      f"SSIM={res.get('ssim').value:.4f}  DE={res.get('color_fidelity').value:.2f}  "
      f"VIF={res.get('vif').value:.4f}  {t*1000:.1f}ms")

# Compare with original
orig_cfg = tp_cfg()
res_orig, t_orig = test_preset("TP-orig", orig_cfg, small, degraded)
print(f"  TP-orig:     Score={res_orig.composite_score:6.2f}  PSNR={res_orig.get('psnr').value:5.2f}  "
      f"SSIM={res_orig.get('ssim').value:.4f}  DE={res_orig.get('color_fidelity').value:.2f}  "
      f"VIF={res_orig.get('vif').value:.4f}  {t_orig*1000:.1f}ms")
improvement = res.composite_score - res_orig.composite_score
print(f"  {'✅' if improvement > 0 else '➡️'} Improvement: {improvement:+.2f} points")

# Also test on NTSC
print(f"\n{'='*70}")
print("PARAMETER SWEEP (NTSC-HEAVY): temporal_nlm_multi h")
print("=" * 70)
results_ntsc = []
for h_val in [4, 5, 6, 7, 8, 10, 12, 15]:
    cfg = tp_cfg(h=h_val, h_color=h_val)
    res, t = test_preset(f"TP(h={h_val})", cfg, small, ntsc_deg)
    results_ntsc.append((h_val, res, t))
    print(f"  h={h_val:3d}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
          f"SSIM={res.get('ssim').value:.4f}  DE={res.get('color_fidelity').value:.2f}  {t*1000:.1f}ms")

if results_ntsc:
    best_ntsc = max(results_ntsc, key=lambda x: x[1].composite_score)
    print(f"\n  🏆 Best NTSC h = {best_ntsc[0]} (Score={best_ntsc[1].composite_score:.2f})")
