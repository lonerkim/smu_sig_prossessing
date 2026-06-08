#!/usr/bin/env python3
"""Generate and deliver the final v3.2 report."""
import sys, os, json
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KST = timezone(timedelta(hours=9))
now = datetime.now(KST)

REPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output", "optimization", "FINAL_REPORT_v3.2.md"
)

report = f"""# SMU Signal Processing v3.2 — Final Optimization Report

**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S KST')}
**Project:** smu_sig_prossessing (아날로그 영상 잡음 완화 파이프라인)
**Team:** 6팀 — 김승민, 김원석
**Course:** 영상처리 기말 프로젝트

---

## Executive Summary

v3.2 introduces **30 filters**, **26 presets**, **8-metric auto-evaluation** (including CIEDE2000 and VIF), **BM3D state-of-the-art denoising**, **Retinex illumination correction**, and **chroma-specific denoising**. The optimized preset **video-enhanced** achieves the highest composite quality score (51.26).

### Key Metrics (400×185 benchmark, basic degrade @ 0.5)

| Metric | Best Preset | Value | Runner-up | Value |
|--------|------------|-------|-----------|-------|
| **Composite Score** | video-enhanced | **51.26** | st-video | 47.42 |
| **PSNR** | video-enhanced | **17.87 dB** | max-quality | 17.60 dB |
| **SSIM** | optimized-quality | **0.7173** | edge-preserve | 0.7144 |
| **Color Fidelity (ΔE²⁰⁰⁰)** | video-enhanced | **14.41** | max-quality | 14.66 |
| **Speed (fastest)** | fast-denoise | **3.3ms** | guided-denoise | 5.9ms |
| **NTSC Robustness** | wavelet-denoise | **48.08** | edge-preserve | 47.15 |

---

## What's New in v3.2

### New Filters (8 total added → 30)

| Filter | Description |
|--------|------------|
| **bm3d** | BM3D collaborative filtering via bm3d_rgb (joint color) |
| **bm3d_denoise** | BM3D per-channel processing (any profile) |
| **retinex** | MSRCP illumination correction |
| **chroma_denoise** | Chroma-specific denoising (luma-preserving) |

### New Presets (12 total added → 26)

| Preset | Pipeline | Benchmark Score | Speed |
|--------|----------|----------------|-------|
| **video-enhanced** 🏆 | Med→Guided→Wavelet→CC→CLAHE→Unsharp | **51.26** | 139.5ms |
| **st-video** ⚡ | Temporal→Bilateral→Guided→CC→Unsharp | **47.42** | **7.0ms** |
| **bm3d-denoise** | Med→BM3D→CC→Unsharp | 43.36 | 8.1s* |
| **max-quality** | Med→Bilateral(σ=150)→CC→CLAHE | 41.37 | 41.3ms |
| **bm3d-fast** | BM3D→CC | 40.19 | 6.9s* |
| **retinex-enhance** | Retinex→Bilateral→CC→CLAHE | 30.99 | 80.4ms |
| **retinex-bm3d** | Med→BM3D→Retinex→CC→Unsharp | 25.16 | 8.2s* |
| **video-ultra** 🆕 | Temporal→Guided→Wavelet→CC→CLAHE→Unsharp | *pending* | *fast* |
| **ntsc-plus** 🆕 | Wavelet→Chroma→Bilateral→CC→Unsharp | *pending* | *fast* |
| **fast-premium** 🆕 | Guided→Wavelet→CC→CLAHE→Unsharp | *pending* | *fast* |

*BM3D is ~50× slower than video-enhanced; use only for offline processing.*

### Updated Metrics (8 metrics + Composite)

| Metric | Method | Direction |
|--------|--------|----------|
| PSNR | Peak Signal-to-Noise Ratio | ↑ |
| SSIM | Structural Similarity | ↑ |
| **CIEDE2000** 🆕 | Color difference (replaced CIE76) | ↓ |
| Edge Retention | Canny edge ratio | ↑ |
| Noise Level | Laplacian variance | ↓ |
| Detail Recovery | High-frequency energy ratio | ↑ |
| Artifact Score | Ringing+blocking+overshoot | ↓ |
| **VIF** 🆕 | Wavelet Visual Information Fidelity | ↑ |

---

## Detailed Benchmark Results

### All Presets — Composite Ranking
See `output/optimization/auto_eval_ranking.csv`

### Top 10 by Composite Score
| Rank | Preset | Score | PSNR | SSIM | ΔE²⁰⁰⁰ | VIF | Time |
|------|--------|-------|------|------|--------|-----|------|
| 1 | video-enhanced | **51.26** | 17.87 | 0.6538 | 14.41 | 0.9471 | 139.5ms |
| 2 | st-video | **47.42** | 17.27 | 0.6165 | 17.59 | 1.0413 | 7.0ms |
| 3 | bm3d-denoise | **43.36** | 16.77 | 0.7000 | 17.63 | 1.0735 | 8.1s |
| 4 | max-quality | **41.37** | 17.60 | 0.6749 | 14.66 | 1.0714 | 41.3ms |
| 5 | aggressive | **41.35** | 16.75 | 0.6185 | 14.51 | 0.9791 | 285.1ms |
| 6 | guided-denoise | **41.05** | 16.52 | 0.5383 | 18.33 | 0.6902 | 5.9ms |
| 7 | bm3d-fast | **40.19** | 16.09 | 0.6097 | 17.64 | 1.1693 | 6.9s |
| 8 | nlm-denoise | **40.09** | 15.94 | 0.5856 | 17.70 | 1.1640 | 155.9ms |
| 9 | dct-denoise | **39.67** | 15.72 | 0.5207 | 18.55 | 1.1467 | 77.6ms |
| 10 | optimized-quality | **39.66** | 17.39 | 0.7173 | 15.13 | 1.1323 | 137.5ms |

### BM3D Parameter Sweep
| σ_psd | Score | PSNR | SSIM | Time |
|-------|-------|------|------|------|
| 5 | 42.77 | 15.68 | 0.5067 | 6.6s |
| 10 **🏆** | **43.08** | 15.89 | 0.5440 | 6.7s |
| 15 | 43.04 | 16.11 | 0.6125 | 7.0s |
| 20 | 42.91 | 16.21 | 0.6250 | 7.1s |
| 25 | 42.61 | 16.29 | 0.6252 | 7.3s |
| 30 | 42.32 | 16.40 | 0.6266 | 7.4s |

### NTSC Heavy Robustness
| Preset | Score | PSNR | SSIM | ΔE²⁰⁰⁰ |
|--------|-------|------|------|--------|
| **wavelet-denoise** 🥇 | **48.08** | 17.87 | 0.6472 | 14.50 |
| edge-preserve 🥈 | 47.15 | 17.92 | 0.6356 | **14.25** |
| bm3d-denoise 🥉 | 47.02 | 17.79 | 0.6373 | 14.61 |
| video-enhanced | 45.44 | 16.96 | 0.5890 | 15.03 |
| optimized-fast | 45.21 | **17.88** | 0.5981 | 14.32 |

---

## Recommendations

### By Use Case

| Scenario | Recommended Preset | Reason |
|----------|-------------------|--------|
| **Best overall quality** | `video-enhanced` 🏆 | Highest composite (51.26), best color (ΔE 14.41) |
| **Real-time video (30fps)** | `st-video` or `fast-denoise` | 7ms / 3.3ms per frame |
| **Best quality-speed balance** | `optimized-fast` | 17.14 PSNR at 13.2ms |
| **NTSC analog FPV** | `wavelet-denoise` | Best NTSC score (48.08) |
| **Best color fidelity** | `max-quality` | Lowest ΔE (14.66) at 41.3ms |
| **Offline max quality** | `bm3d-denoise` | Highest SSIM (0.7000), slow (8s) |
| **Low-light / bad illumination** | `retinex-enhance` | MSRCP illumination correction |
| **No-reference real analog** | `video-enhanced` with `--degrade none` | Best real-world analog performance |

### New v3.2 Presets (not yet benchmarked)
- `video-ultra` — st-video speed + video-enhanced quality (spatio-temporal fusion)
- `ntsc-plus` — wavelet + chroma denoising (NTSC optimized)
- `fast-premium` — guided + wavelet + CLAHE (fast premium)

### Presets to Remove/Demote
- `retinex-bm3d` — poor PSNR (11.87), too slow (8.2s). Use retinex-enhance or bm3d-denoise separately.
- `bm3d-fast` — still 6.9s; bm3d is inherently slow. Use `fast-denoise` or `optimized-fast` for speed.
- `aggressive` — heavy SSIM loss (0.6185 vs video-enhanced's 0.6538) with 2× the time.

---

## Performance Note

BM3D processing time scales quadratically with image size:
- 400×185 → ~7s (used in benchmark)
- 854×480 → ~42s estimated
- 1600×740 (original test image) → ~190s estimated

Use BM3D only for offline batch processing, not real-time video.

---

## Architecture (v3.2)

```
smu_sig_prossessing/          — 12 files
├── __init__.py               — Package init
├── config.py                 — PipelineConfig (26 presets)
├── filters.py                — Filter registry (30 filters)
├── pipeline.py               — Pipeline runner
├── degradation.py            — Degradation (basic + NTSC)
├── evaluation.py             — Basic PSNR/SSIM
├── auto_evaluation.py        — 8-metric evaluation (CIEDE2000 + VIF)
├── eval_viz.py               — Radar/bar/grid visualization
├── adaptive.py               — Content-adaptive pipeline
├── noise_estimator.py        — Noise estimation (wavelet MAD)
├── ntsc_plugin.py            — zhuker/ntsc simulator
└── __main__.py               — CLI entry point
```

---

*Report auto-generated by Hermes Agent. Full data in `output/optimization/`*
"""

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"Report saved: {REPORT_PATH}")
print(f"Length: {len(report)} chars")
