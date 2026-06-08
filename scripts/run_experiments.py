#!/usr/bin/env python3
"""Run improvement experiments — called by iter_loop.py every 2h."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing.degradation import degrade_image
from smu_sig_prossessing.auto_evaluation import AutoEvaluator
from smu_sig_prossessing.filters import reset_temporal_state, FILTER_REGISTRY

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
img = cv2.imread(os.path.join(BASE, "input", "test_small.jpg"))
if img is None:
    print("NO_IMAGE"); sys.exit(1)
small = cv2.resize(img, (400, 185))
degraded = degrade_image(small, use_ntsc=False, strength=0.5)
eval = AutoEvaluator()

print(f"Filters available: {len(FILTER_REGISTRY)}")

# Experiment 1: Grey-Edge strength sweep (fast)
results = []
for s in [0.12, 0.18, 0.22, 0.28, 0.35]:
    reset_temporal_state()
    cfg = PipelineConfig(label=f"exp_ge_{s}")
    cfg.add("grey_edge", strength=s, sigma_smooth=1.0)
    cfg.add("temporal_nlm_multi", h=8, h_color=8, temporal_window=3, max_frames=5)
    cfg.add("guided_filter", radius=3, eps=50.0)
    cfg.add("chroma_denoise", strength=0.2)
    cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
    cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
    cfg.add("unsharp_mask", strength=0.1, radius=0.5, threshold=10)
    reset_temporal_state()
    try:
        restored = pl.apply_pipeline(degraded, cfg)
        res = eval.evaluate(small, restored, label=f"ge_{s}", degraded=degraded, verbose=False)
        print(f"  ge_s={s:.2f}: Score={res.composite_score:.2f} PSNR={res.get('psnr').value:.2f} DE={res.get('color_fidelity').value:.2f}")
        results.append((res.composite_score, s))
    except Exception as e:
        print(f"  ge_s={s:.2f}: ERROR {e}")

# Experiment 2: Without channel_correction when grey_edge is present (redundant)
reset_temporal_state()
cfg = PipelineConfig(label="exp_nocc")
cfg.add("grey_edge", strength=0.25, sigma_smooth=1.0)
cfg.add("temporal_nlm_multi", h=8, h_color=8, temporal_window=3, max_frames=5)
cfg.add("guided_filter", radius=3, eps=50.0)
cfg.add("chroma_denoise", strength=0.2)
cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
cfg.add("unsharp_mask", strength=0.1, radius=0.5, threshold=10)
reset_temporal_state()
try:
    restored = pl.apply_pipeline(degraded, cfg)
    res = eval.evaluate(small, restored, label="exp_nocc", degraded=degraded, verbose=False)
    print(f"  no-channel-corr: Score={res.composite_score:.2f} PSNR={res.get('psnr').value:.2f} DE={res.get('color_fidelity').value:.2f}")
    results.append((res.composite_score, "no-cc"))
except Exception as e:
    print(f"  no-cc: ERROR {e}")

# Experiment 3: Grey-Edge with rolling guidance (fast combo)
reset_temporal_state()
cfg = PipelineConfig(label="exp_ge_rolling")
cfg.add("grey_edge", strength=0.25, sigma_smooth=1.0)
cfg.add("rolling_guidance", sigma_s=3.0, sigma_r=0.08, n_iter=3)
cfg.add("guided_filter", radius=3, eps=50.0)
cfg.add("chroma_denoise", strength=0.3)
cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
cfg.add("unsharp_mask", strength=0.15, radius=0.5, threshold=8)
reset_temporal_state()
try:
    restored = pl.apply_pipeline(degraded, cfg)
    res = eval.evaluate(small, restored, label="exp_ge_rolling", degraded=degraded, verbose=False)
    print(f"  ge-rolling: Score={res.composite_score:.2f} PSNR={res.get('psnr').value:.2f} DE={res.get('color_fidelity').value:.2f}")
    results.append((res.composite_score, "ge-rolling"))
except Exception as e:
    print(f"  ge-rolling: ERROR {e}")

# Experiment 4: Grey-Edge + NLM only (bare minimum quality)
reset_temporal_state()
cfg = PipelineConfig(label="exp_ge_nlm")
cfg.add("grey_edge", strength=0.25, sigma_smooth=1.0)
cfg.add("nlm", h=8, template_window=7, search_window=21)
cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=10)
reset_temporal_state()
try:
    restored = pl.apply_pipeline(degraded, cfg)
    res = eval.evaluate(small, restored, label="exp_ge_nlm", degraded=degraded, verbose=False)
    print(f"  ge-nlm: Score={res.composite_score:.2f} PSNR={res.get('psnr').value:.2f} DE={res.get('color_fidelity').value:.2f}")
    results.append((res.composite_score, "ge-nlm"))
except Exception as e:
    print(f"  ge-nlm: ERROR {e}")

# Print best experiment
if results:
    results.sort(reverse=True)
    print(f"\n  BEST: {results[0][1]} = {results[0][0]:.2f}")
