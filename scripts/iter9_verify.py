#!/usr/bin/env python3
"""
Iter9 — Verify improvements and test on real-world image + NTSC.
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
    """Create a PipelineConfig with ordered stages. Each stage: (name, enabled, params_dict)"""
    cfg = PipelineConfig(label=label)
    for name, enabled, params in stages:
        cfg.add(name, enabled=enabled, **params)
    return cfg

def test_preset(name, cfg, origin, degraded_img):
    t0 = time.perf_counter()
    restored = pl.apply_pipeline(degraded_img, cfg)
    t = time.perf_counter() - t0
    res = evaluator.evaluate(origin, restored, label=name, degraded=degraded_img, verbose=False)
    return res, t, restored

# ── Presets to test ──
presets = {
    "temporal-premium(orig)": PipelineConfig.temporal_premium(),
    "TP+US(0.3)": make_cfg("TP+US 0.3",
        ("temporal_nlm_multi", True, dict(h=8, h_color=8, temporal_window=3, max_frames=5)),
        ("guided_filter", True, dict(radius=3, eps=50.0)),
        ("chroma_denoise", True, dict(strength=0.2)),
        ("channel_correction", True, dict(clamp_min=0.85, clamp_max=1.25)),
        ("adaptive_equalize", True, dict(clip_limit=1.5, tile_size=8, brightness_preserve=0.4)),
        ("unsharp_mask", True, dict(strength=0.3, radius=0.5, threshold=10)),
    ),
    "TP+US(0.5)": make_cfg("TP+US 0.5",
        ("temporal_nlm_multi", True, dict(h=8, h_color=8, temporal_window=3, max_frames=5)),
        ("guided_filter", True, dict(radius=3, eps=50.0)),
        ("chroma_denoise", True, dict(strength=0.2)),
        ("channel_correction", True, dict(clamp_min=0.85, clamp_max=1.25)),
        ("adaptive_equalize", True, dict(clip_limit=1.5, tile_size=8, brightness_preserve=0.4)),
        ("unsharp_mask", True, dict(strength=0.5, radius=0.5, threshold=10)),
    ),
    "TP+DetailBoost": make_cfg("TP+DB",
        ("temporal_nlm_multi", True, dict(h=8, h_color=8, temporal_window=3, max_frames=5)),
        ("guided_filter", True, dict(radius=3, eps=50.0)),
        ("detail_boost", True, dict(strength=0.2, sigma_s=3.0, sigma_r=0.15, threshold=0.02)),
        ("chroma_denoise", True, dict(strength=0.2)),
        ("channel_correction", True, dict(clamp_min=0.85, clamp_max=1.25)),
        ("adaptive_equalize", True, dict(clip_limit=1.5, tile_size=8, brightness_preserve=0.4)),
        ("unsharp_mask", True, dict(strength=0.15, radius=0.5, threshold=10)),
    ),
    "TP+US0.3+DB": make_cfg("TP+US0.3+DB",
        ("temporal_nlm_multi", True, dict(h=8, h_color=8, temporal_window=3, max_frames=5)),
        ("guided_filter", True, dict(radius=3, eps=50.0)),
        ("detail_boost", True, dict(strength=0.2, sigma_s=3.0, sigma_r=0.15, threshold=0.02)),
        ("chroma_denoise", True, dict(strength=0.2)),
        ("channel_correction", True, dict(clamp_min=0.85, clamp_max=1.25)),
        ("adaptive_equalize", True, dict(clip_limit=1.5, tile_size=8, brightness_preserve=0.4)),
        ("unsharp_mask", True, dict(strength=0.3, radius=0.5, threshold=10)),
    ),
}

# ── Test on test_small.jpg (400px) ──
print("=" * 70)
print("TEST: test_small.jpg (400x185) — basic degrade strength=0.5")
print("=" * 70)
img = cv2.imread(os.path.join(BASE, "input", "test_small.jpg"))
small = cv2.resize(img, (400, 185))
degraded = degrade_image(small, use_ntsc=False, strength=0.5)

results_small = []
for name, cfg in presets.items():
    res, t, _ = test_preset(name, cfg, small, degraded)
    results_small.append((name, res, t))
    print(f"  {name:25s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
          f"SSIM={res.get('ssim').value:.4f}  DE={res.get('color_fidelity').value:.2f}  "
          f"VIF={res.get('vif').value:.4f}  {t*1000:.1f}ms")

# ── Test on REAL_WORLD_PICTURE.jpg (full size) ──
print(f"\n{'='*70}")
print("TEST: REAL_WORLD_PICTURE.jpg (4000x1848) — degrade strength=0.5")
print("=" * 70)
real_img = cv2.imread(os.path.join(BASE, "input", "REAL_WORLD_PICTURE.jpg"))
if real_img is not None:
    real_degraded = degrade_image(real_img, use_ntsc=False, strength=0.5)
    results_real = []
    for name, cfg in presets.items():
        try:
            res, t, _ = test_preset(name, cfg, real_img, real_degraded)
            results_real.append((name, res, t))
            print(f"  {name:25s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
                  f"SSIM={res.get('ssim').value:.4f}  DE={res.get('color_fidelity').value:.2f}  "
                  f"VIF={res.get('vif').value:.4f}  {t*1000:.1f}ms")
        except Exception as e:
            print(f"  {name:25s}  ERROR: {e}")
else:
    print("  Cannot load REAL_WORLD_PICTURE.jpg")

# ── NTSC test ──
print(f"\n{'='*70}")
print("NTSC-HEAVY TEST: test_small.jpg (400px)")
print("=" * 70)
ntsc_deg = degrade_image(small, use_ntsc=True, ntsc_intensity="heavy")
results_ntsc = []
for name, cfg in presets.items():
    try:
        res, t, _ = test_preset(name, cfg, small, ntsc_deg)
        results_ntsc.append((name, res, t))
        print(f"  {name:25s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
              f"SSIM={res.get('ssim').value:.4f}  DE={res.get('color_fidelity').value:.2f}  "
              f"{t*1000:.1f}ms")
    except Exception as e:
        print(f"  {name:25s}  ERROR: {e}")

# ── Rankings ──
for test_name, results in [("basic(0.5) small", results_small), 
                            ("REAL_WORLD", results_real if 'results_real' in dir() and results_real else []),
                            ("NTSC-heavy", results_ntsc)]:
    if not results:
        continue
    results.sort(key=lambda x: x[1].composite_score, reverse=True)
    print(f"\n{'='*70}")
    print(f"RANKING: {test_name}")
    print(f"{'Rank':<5} {'Preset':<28s} {'Score':<8s} {'PSNR':<8s} {'SSIM':<8s} {'DE':<8s} {'Time':<8s}")
    print("-" * 65)
    for i, (name, res, t) in enumerate(results):
        print(f"  #{i+1:<2} {name:<28s} {res.composite_score:<8.2f} "
              f"{res.get('psnr').value:<8.2f} {res.get('ssim').value:<8.4f} "
              f"{res.get('color_fidelity').value:<8.2f} {t*1000:<8.1f}ms")

# ── Combined overall ranking ──
print(f"\n{'='*70}")
print("OVERALL AVERAGE RANKING (all 3 tests)")
print("=" * 70)
combined = {}
for name in presets:
    scores_sum = 0
    count = 0
    for rlist in [results_small, results_ntsc]:
        for n, res, t in rlist:
            if n == name:
                scores_sum += res.composite_score
                count += 1
    if count > 0:
        combined[name] = scores_sum / count

for n, res, t in results_small:
    for rlist in [results_small, results_ntsc]:
        pass

# Simple ranking by small image test
results_small.sort(key=lambda x: x[1].composite_score, reverse=True)
print(f"{'Rank':<5} {'Preset':<28s} {'Score':<8s}")
print("-" * 41)
for i, (name, res, t) in enumerate(results_small):
    # Find NTSC score for same preset
    ntsc_score = None
    for n, nr, nt in results_ntsc:
        if n == name:
            ntsc_score = nr.composite_score
            break
    ntsc_str = f"  NTSC={ntsc_score:.2f}" if ntsc_score else ""
    print(f"  #{i+1:<2} {name:<28s} {res.composite_score:<8.2f}{ntsc_str}")
