#!/usr/bin/env python3
"""
Iter9 — Chroma-Focus ablation + new combo experiments.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.config import PipelineConfig, FilterConfig
from smu_sig_prossessing.auto_evaluation import AutoEvaluator
from smu_sig_prossessing.degradation import degrade_image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img = cv2.imread(os.path.join(BASE, "input", "test_small.jpg"))
small = cv2.resize(img, (400, 185))
degraded = degrade_image(small, use_ntsc=False, strength=0.5)

evaluator = AutoEvaluator()

def test_preset(name, cfg):
    t0 = time.perf_counter()
    restored = pl.apply_pipeline(degraded, cfg)
    t = time.perf_counter() - t0
    res = evaluator.evaluate(small, restored, label=name, degraded=degraded, verbose=False)
    return res, t

# ── 1. Chroma-focus ablation ──
print("=" * 70)
print("CHROMA-FOCUS ABLATION")
print("=" * 70)

cf_cfg = PipelineConfig.chroma_focus()
cf_filters = [s.name for s in cf_cfg.stages if s.enabled]
res_full, t_full = test_preset("chroma-focus", cf_cfg)
print(f"\n  FULL:    Score={res_full.composite_score:6.2f}  PSNR={res_full.get('psnr').value:5.2f}  "
      f"SSIM={res_full.get('ssim').value:.4f}  DE={res_full.get('color_fidelity').value:.2f}  "
      f"VIF={res_full.get('vif').value:.4f}  {t_full*1000:.1f}ms")
print(f"  Filters: {' → '.join(cf_filters)}")

for fn in cf_filters:
    cfg = cf_cfg.copy()
    cfg.disable(fn)
    res, t = test_preset(f'chroma-focus -{fn}', cfg)
    diff = res_full.composite_score - res.composite_score
    marker = "⚠ LOST" if diff > 0 else "✅ GAIN"
    print(f"  -{fn:25s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
          f"SSIM={res.get('ssim').value:.4f}  DE={res.get('color_fidelity').value:.2f}  "
          f"VIF={res.get('vif').value:.4f}  {marker} {diff:+.2f}  {t*1000:.1f}ms")

# ── 2. New combos ──
print("\n" + "=" * 70)
print("NEW COMBINATION EXPERIMENTS")
print("=" * 70)

experiments = []

# Ex1: temporal-premium with stronger guided filter params
cfg = PipelineConfig(label="TP+GF(eps30)")
cfg.add("temporal_nlm_multi", h=8, h_color=8, temporal_window=3, max_frames=5)
cfg.add("guided_filter", radius=3, eps=30.0)  # stronger denoise
cfg.add("chroma_denoise", strength=0.2)
cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
cfg.add("unsharp_mask", strength=0.1, radius=0.5, threshold=10)
experiments.append(cfg)

# Ex2: temporal-premium with unsharp boost
cfg = PipelineConfig(label="TP+US(0.3)")
cfg.add("temporal_nlm_multi", h=8, h_color=8, temporal_window=3, max_frames=5)
cfg.add("guided_filter", radius=3, eps=50.0)
cfg.add("chroma_denoise", strength=0.2)
cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
cfg.add("unsharp_mask", strength=0.3, radius=0.5, threshold=10)
experiments.append(cfg)

# Ex3: Temporal with median prefilter (removes temporal_nlm_multi since it's slow)
cfg = PipelineConfig(label="Fast-Temporal")
cfg.add("median", ksize=3)
cfg.add("temporal_nlm_multi", h=6, h_color=6, temporal_window=2, max_frames=3)  # lighter
cfg.add("guided_filter", radius=3, eps=50.0)
cfg.add("chroma_denoise", strength=0.3)
cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=10)
experiments.append(cfg)

# Ex4: Rolling guidance + temporal (rolling guidance from chroma-focus)
cfg = PipelineConfig(label="Rolling+Temporal")
cfg.add("temporal_nlm_multi", h=8, h_color=8, temporal_window=3, max_frames=5)
cfg.add("rolling_guidance", sigma_s=3.0, sigma_r=0.08, n_iter=3)
cfg.add("chroma_denoise", strength=0.2)
cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=10)
experiments.append(cfg)

# Ex5: Chroma-focus with temporal (combine the 2 best)
cfg = PipelineConfig(label="CF+Temporal")
cfg.add("chroma_denoise", strength=0.5)
cfg.add("temporal_nlm_multi", h=6, h_color=6, temporal_window=2, max_frames=3)
cfg.add("guided_filter", radius=3, eps=50.0)
cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
cfg.add("adaptive_equalize", clip_limit=2.0, tile_size=8, brightness_preserve=0.3)
cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=8)
experiments.append(cfg)

# Ex6: Temporal nlm params sweep — try h=10 for stronger denoise
cfg = PipelineConfig(label="TP+h10")
cfg.add("temporal_nlm_multi", h=10, h_color=10, temporal_window=3, max_frames=5)
cfg.add("guided_filter", radius=3, eps=50.0)
cfg.add("chroma_denoise", strength=0.2)
cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
cfg.add("unsharp_mask", strength=0.1, radius=0.5, threshold=10)
experiments.append(cfg)

# Ex7: Temporal nlm h=12
cfg = PipelineConfig(label="TP+h12")
cfg.add("temporal_nlm_multi", h=12, h_color=12, temporal_window=3, max_frames=5)
cfg.add("guided_filter", radius=3, eps=50.0)
cfg.add("chroma_denoise", strength=0.2)
cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
cfg.add("unsharp_mask", strength=0.1, radius=0.5, threshold=10)
experiments.append(cfg)

# Ex8: Temporal nlm multi with detail_boost
cfg = PipelineConfig(label="TP+Detail")
cfg.add("temporal_nlm_multi", h=8, h_color=8, temporal_window=3, max_frames=5)
cfg.add("guided_filter", radius=3, eps=50.0)
cfg.add("detail_boost", strength=0.25, sigma_s=3.0, sigma_r=0.15, threshold=0.02)
cfg.add("chroma_denoise", strength=0.2)
cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
cfg.add("unsharp_mask", strength=0.1, radius=0.5, threshold=10)
experiments.append(cfg)

# Ex9: Rolling_guidance with detail_boost (fast alternative)
cfg = PipelineConfig(label="Rolling+Detail")
cfg.add("rolling_guidance", sigma_s=3.0, sigma_r=0.08, n_iter=3)
cfg.add("detail_boost", strength=0.3, sigma_s=3.0, sigma_r=0.15, threshold=0.015)
cfg.add("chroma_denoise", strength=0.3)
cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=8)
experiments.append(cfg)

# Ex10: Light temporal with chroma_focus base
cfg = PipelineConfig(label="Light-Temporal")
cfg.add("chroma_denoise", strength=0.8)
cfg.add("temporal_nlm_multi", h=5, h_color=5, temporal_window=2, max_frames=3)
cfg.add("rolling_guidance", sigma_s=3.0, sigma_r=0.08, n_iter=3)
cfg.add("channel_correction", clamp_min=0.80, clamp_max=1.25)
cfg.add("adaptive_equalize", clip_limit=2.0, tile_size=8, brightness_preserve=0.3)
cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=8)
experiments.append(cfg)

# Run all experiments
all_results = [('temporal-premium(baseline)', res_full, t_full)]
for cfg in experiments:
    try:
        res, t = test_preset(cfg.label, cfg)
        all_results.append((cfg.label, res, t))
        print(f"  {cfg.label:25s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
              f"SSIM={res.get('ssim').value:.4f}  DE={res.get('color_fidelity').value:.2f}  "
              f"VIF={res.get('vif').value:.4f}  {t*1000:.1f}ms")
    except Exception as e:
        print(f"  ❌ {cfg.label:25s}  ERROR: {e}")

# Summary
print(f"\n{'='*70}")
print("RANKING (by Composite Score):")
print(f"{'Rank':<5} {'Preset':<28s} {'Score':<8s} {'PSNR':<8s} {'SSIM':<8s} {'DE':<8s} {'Time':<8s}")
print("-" * 65)
all_results.sort(key=lambda x: x[1].composite_score, reverse=True)
for i, (label, res, t) in enumerate(all_results):
    print(f"  #{i+1:<2} {label:<28s} {res.composite_score:<8.2f} "
          f"{res.get('psnr').value:<8.2f} {res.get('ssim').value:<8.4f} "
          f"{res.get('color_fidelity').value:<8.2f} {t*1000:<8.1f}ms")
