# Iteration 9 — Autonomous Improvement Results

**Date:** 2026-06-09
**Scope:** Filter ablation sweep + 27-frame batch evaluation + qualitative comparison
**Baseline:** v3.3 with 37 filters, 35 presets (now 36)

---

## 1. Ablation Sweep (Previous Run)

Ablation on `temporal-premium` and `chroma-focus` presets completed with detailed results:

| Filter Removed | Score Loss | Impact |
|---|---|---|
| guided_filter | -7.30 pts | **Critical** |
| channel_correction | -5.09 pts | **Critical** |
| adaptive_equalize | -3.46 pts | **Important** |
| temporal_nlm_multi | -0.43 pts | Minor |
| chroma_denoise | -0.11 pts | Minor |
| unsharp_mask | ±0.00 pts | Negligible |

**Key insight:** `guided_filter` + `channel_correction` + `adaptive_equalize` are the critical components.

## 2. Parameter Optimization

Based on sweep across 6 parameter ranges:

| Parameter | Old Value | New Value | Gain |
|---|---|---|---|
| temporal_nlm_multi h | 8 | **10** | +0.01 |
| guided_filter eps | 50 | **10** | +0.06 |
| chroma_denoise strength | 0.2 | **0.1** | +0.03 |
| NTSC h | 8 | **4** | +0.23 |

## 3. New NIQE Metric

Added **Natural Image Quality Evaluator (NIQE)** metric — no-reference quality assessment based on MSCN coefficient analysis (simplified Mittal et al. 2012 approach). Lower score = more natural appearance. Integrated into composite scoring with 11.5% weight.

## 4. 27-Frame Batch Evaluation

Extracted frames from analog_whoop_footage.mp4 (13 frames) + digital_whoop_footage.mp4 (12 frames) + 2 still images.

### Results (pending — evaluation running)

## 5. All-Preset Comparison HTML

Generated interactive comparison page at `output/iter9_comparison.html` comparing all 35 presets on test_small.jpg with embedded images and score overlays.

**Top 3 on test_small.jpg (400x185, basic degrade 0.5):**
1. 🥇 temporal-premium — Score=54.12, PSNR=19.63, SSIM=0.6395, NIQE=11.92
2. 🥈 chroma-focus — (will update)
3. 🥉 super-premium-fast — (will update)

## 6. Config Changes

- `temporal-premium`: Updated h=8→h=10, eps=50→eps=10, chroma_denoise=0.2→0.1
- **New preset: temporal-ntsc**: h=4 optimized for NTSC-heavy content
- **New metric: NIQE** added to AutoEvaluator with 11.5% weight
- `run_auto_eval.py`: Updated from 16→36 presets
- Marco export includes NIQE column

## 7. Automation

- 2-hour cron loop: `smu-sig-iter9-autonomous-improve` (job_id: 4d940f63fdc3)

## 8. Next Steps

1. Investigate video-specific temporal window sweep for temporal_nlm_multi
2. Explore fusion of temporal-premium + wavelet for mixed noise types
3. Run full visual comparison across all video frame types
4. Investigate why some frames score higher than others (frame content dependency)
