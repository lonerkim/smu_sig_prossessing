#!/usr/bin/env python3
"""
Iter9 — Creative combo experiments.
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

experiments = []

# 1. Baseline
experiments.append(("temporal-premium (orig)", PipelineConfig.temporal_premium()))

# 2. Temporal with median prefilter
experiments.append(("TP+Median", make_cfg("TP+Median",
    ("median", True, dict(ksize=3)),
    ("temporal_nlm_multi", True, dict(h=8, h_color=8, temporal_window=3, max_frames=5)),
    ("guided_filter", True, dict(radius=3, eps=50.0)),
    ("chroma_denoise", True, dict(strength=0.2)),
    ("channel_correction", True, dict(clamp_min=0.85, clamp_max=1.25)),
    ("adaptive_equalize", True, dict(clip_limit=1.5, tile_size=8, brightness_preserve=0.4)),
    ("unsharp_mask", True, dict(strength=0.1, radius=0.5, threshold=10)),
)))

# 3. Temporal with stronger chroma_denoise
experiments.append(("TP+Chroma(0.5)", make_cfg("TP+Chroma0.5",
    ("temporal_nlm_multi", True, dict(h=8, h_color=8, temporal_window=3, max_frames=5)),
    ("guided_filter", True, dict(radius=3, eps=50.0)),
    ("chroma_denoise", True, dict(strength=0.5)),
    ("channel_correction", True, dict(clamp_min=0.85, clamp_max=1.25)),
    ("adaptive_equalize", True, dict(clip_limit=1.5, tile_size=8, brightness_preserve=0.4)),
    ("unsharp_mask", True, dict(strength=0.1, radius=0.5, threshold=10)),
)))

# 4. Temporal with cross_bilateral instead of guided_filter (faster alternative)
experiments.append(("TP+CrossBilat", make_cfg("TP+CrossBilat",
    ("temporal_nlm_multi", True, dict(h=8, h_color=8, temporal_window=3, max_frames=5)),
    ("cross_bilateral", True, dict(guide_sigma=1.0, d=5, sigma_color=30, sigma_space=30)),
    ("chroma_denoise", True, dict(strength=0.2)),
    ("channel_correction", True, dict(clamp_min=0.85, clamp_max=1.25)),
    ("adaptive_equalize", True, dict(clip_limit=1.5, tile_size=8, brightness_preserve=0.4)),
    ("unsharp_mask", True, dict(strength=0.1, radius=0.5, threshold=10)),
)))

# 5. Temporal_nlm_multi with guided_filter eps=100 (weaker denoise, more detail)
experiments.append(("TP+GF(eps100)", make_cfg("TP+GF(eps100)",
    ("temporal_nlm_multi", True, dict(h=8, h_color=8, temporal_window=3, max_frames=5)),
    ("guided_filter", True, dict(radius=3, eps=100.0)),
    ("chroma_denoise", True, dict(strength=0.2)),
    ("channel_correction", True, dict(clamp_min=0.85, clamp_max=1.25)),
    ("adaptive_equalize", True, dict(clip_limit=1.5, tile_size=8, brightness_preserve=0.4)),
    ("unsharp_mask", True, dict(strength=0.1, radius=0.5, threshold=10)),
)))

# 6. Rolling guidance with temporal (replace guided with rolling)
experiments.append(("TP+RollingGuide", make_cfg("TP+RollingGuide",
    ("temporal_nlm_multi", True, dict(h=8, h_color=8, temporal_window=3, max_frames=5)),
    ("rolling_guidance", True, dict(sigma_s=3.0, sigma_r=0.08, n_iter=3)),
    ("chroma_denoise", True, dict(strength=0.2)),
    ("channel_correction", True, dict(clamp_min=0.85, clamp_max=1.25)),
    ("adaptive_equalize", True, dict(clip_limit=1.5, tile_size=8, brightness_preserve=0.4)),
    ("unsharp_mask", True, dict(strength=0.1, radius=0.5, threshold=10)),
)))

# 7. Temporal_nlm_multi with lower h (weaker denoise, faster)
experiments.append(("TP+h5", make_cfg("TP+h5",
    ("temporal_nlm_multi", True, dict(h=5, h_color=5, temporal_window=3, max_frames=5)),
    ("guided_filter", True, dict(radius=3, eps=50.0)),
    ("chroma_denoise", True, dict(strength=0.2)),
    ("channel_correction", True, dict(clamp_min=0.85, clamp_max=1.25)),
    ("adaptive_equalize", True, dict(clip_limit=1.5, tile_size=8, brightness_preserve=0.4)),
    ("unsharp_mask", True, dict(strength=0.1, radius=0.5, threshold=10)),
)))

# 8. Cross_bilateral + detail_boost + chroma (fast high-quality alternative)
experiments.append(("CB+DB+Chroma", make_cfg("CB+DB+Chroma",
    ("cross_bilateral", True, dict(guide_sigma=1.0, d=7, sigma_color=50, sigma_space=50)),
    ("detail_boost", True, dict(strength=0.3, sigma_s=3.0, sigma_r=0.15, threshold=0.02)),
    ("chroma_denoise", True, dict(strength=0.3)),
    ("channel_correction", True, dict(clamp_min=0.85, clamp_max=1.25)),
    ("adaptive_equalize", True, dict(clip_limit=1.5, tile_size=8, brightness_preserve=0.4)),
    ("unsharp_mask", True, dict(strength=0.2, radius=0.5, threshold=10)),
)))

# 9. Wavelet + guided + chroma (combine wavelet's strength with guided)
experiments.append(("Wavelet+GF+Chroma", make_cfg("Wavelet+GF+Chroma",
    ("median", True, dict(ksize=3)),
    ("wavelet", True, dict(wavelet="db4", level=2, threshold_mode="soft")),
    ("guided_filter", True, dict(radius=3, eps=50.0)),
    ("chroma_denoise", True, dict(strength=0.3)),
    ("channel_correction", True, dict(clamp_min=0.85, clamp_max=1.25)),
    ("adaptive_equalize", True, dict(clip_limit=1.5, tile_size=8, brightness_preserve=0.4)),
    ("unsharp_mask", True, dict(strength=0.15, radius=0.5, threshold=5)),
)))

# 10. NLM + guided (non-temporal NLM as alternative)
experiments.append(("NLM+GF+Chroma", make_cfg("NLM+GF+Chroma",
    ("nlm", True, dict(h=8, template_window=7, search_window=21)),
    ("guided_filter", True, dict(radius=3, eps=50.0)),
    ("chroma_denoise", True, dict(strength=0.3)),
    ("channel_correction", True, dict(clamp_min=0.85, clamp_max=1.25)),
    ("adaptive_equalize", True, dict(clip_limit=1.5, tile_size=8, brightness_preserve=0.4)),
    ("unsharp_mask", True, dict(strength=0.15, radius=0.5, threshold=5)),
)))

# Run
print("=" * 70)
print("CREATIVE COMBOS — test_small.jpg 400px, basic(0.5)")
print("=" * 70)
results = []
for name, cfg in experiments:
    try:
        res, t = test_preset(name, cfg, small, degraded)
        results.append((name, res, t))
        print(f"  {name:25s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
              f"SSIM={res.get('ssim').value:.4f}  DE={res.get('color_fidelity').value:.2f}  "
              f"VIF={res.get('vif').value:.4f}  {t*1000:.1f}ms")
    except Exception as e:
        print(f"  ❌ {name:25s}  ERROR: {e}")
        import traceback; traceback.print_exc()

# Ranking
print(f"\n{'='*70}")
print("RANKING:")
print(f"{'Rank':<5} {'Preset':<28s} {'Score':<8s} {'PSNR':<8s} {'SSIM':<8s} {'DE':<8s} {'Time':<8s}")
print("-" * 65)
results.sort(key=lambda x: x[1].composite_score, reverse=True)
for i, (name, res, t) in enumerate(results):
    print(f"  #{i+1:<2} {name:<28s} {res.composite_score:<8.2f} "
          f"{res.get('psnr').value:<8.2f} {res.get('ssim').value:<8.4f} "
          f"{res.get('color_fidelity').value:<8.2f} {t*1000:<8.1f}ms")
