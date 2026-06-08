# Iteration 9 — Ablation Sweep + Parameter Optimization

**Date:** 2026-06-09
**Scope:** Autonomous improvement run — filter ablation, parameter sweep, new preset exploration
**Baseline:** v3.3 with 37 filters, 35 presets, best score 55.34 (temporal-premium)

---

## 1. Ablation on temporal-premium

`temporal-premium` 구성: `temporal_nlm_multi → guided_filter → chroma_denoise → channel_correction → adaptive_equalize → unsharp_mask`

| Filter removed | Score Loss | Impact |
|---|---|---|
| guided_filter | -7.30 pts | **Critical** — edge preservation backbone |
| channel_correction | -5.09 pts | **Critical** — color balance essential |
| adaptive_equalize | -3.46 pts | **Important** — contrast recovery matters |
| temporal_nlm_multi | -0.43 pts | Minor on single images (essential for video) |
| chroma_denoise | -0.11 pts | Minor — slight color improvement |
| unsharp_mask | ±0.00 pts | Negligible on composite metric |

**Key insight:** `guided_filter` + `channel_correction` + `adaptive_equalize` are the most critical components. Temporal NLM adds value mainly for video sequences.

## 2. Ablation on chroma-focus

`chroma-focus` 구성: `chroma_denoise → rolling_guidance → channel_correction → adaptive_equalize → unsharp_mask`

| Filter removed | Score Loss | Impact |
|---|---|---|
| rolling_guidance | -7.34 pts | **Critical** — primary denoiser |
| channel_correction | -5.42 pts | **Critical** |
| adaptive_equalize | -3.45 pts | **Important** |
| chroma_denoise | -0.26 pts | Minor on this test |
| unsharp_mask | ±0.00 pts | Negligible |

## 3. Parameter Sweep Results

### temporal_nlm_multi h (400px basic 0.5)
| h | Score | PSNR | SSIM | ΔE |
|---|---|---|---|---|
| 4 | 53.92 | 18.73 | 0.604 | 13.63 |
| 6 | 54.11 | 18.77 | 0.613 | 13.58 |
| **8 (old)** | **54.19** | 18.79 | 0.618 | 13.49 |
| **10 (best)** | **54.20** | 18.81 | 0.622 | 13.50 |
| 12 | 54.12 | 18.79 | 0.628 | 13.58 |
| 15 | 53.64 | 18.73 | 0.640 | 13.73 |

🏆 **Optimal h=10** (marginal +0.01 over h=8)

### guided_filter eps
| eps | Score | PSNR |
|---|---|---|
| **10 (best)** | **54.25** | 18.80 |
| 50 (old) | 54.19 | 18.79 |
| 100 | 54.19 | 18.79 |
| 200 | 54.18 | 18.79 |

🏆 **Optimal eps=10** (+0.06 over eps=50)

### chroma_denoise strength
| cd | Score | ΔE |
|---|---|---|
| 0.0 | 54.21 | 13.49 |
| **0.1 (best)** | **54.22** | 13.48 |
| 0.2 (old) | 54.19 | 13.49 |
| 0.5 | 54.21 | 13.48 |
| 0.8 | 53.89 | 13.96 |

🏆 **Optimal cd=0.1** (marginal +0.03 over 0.2)

### unsharp_mask strength
All values (0.0–0.8) give identical score 54.19. Unsharp has zero impact on composite metric.

### NTSC-h sweep
| h | Score (NTSC) |
|---|---|
| **4 (best)** | **46.98** |
| 5 | 46.95 |
| 8 (old) | 46.75 |
| 10 | 46.42 |
| 15 | 45.52 |

🏆 **Optimal NTSC h=4** (+0.23 over h=8)

## 4. New Preset Experiments

| Combo | Score | Time | Note |
|---|---|---|---|
| temporal-premium (orig) | 54.24 | 323ms | Baseline |
| TP+RollingGuide | 54.28 | 296ms | ≈ baseline |
| NLM+GF+Chroma | 54.27 | 238ms | Non-temporal alternative |
| TP+US(0.3) | 54.18 | 378ms | ≈ baseline (within noise) |
| Rolling+Detail | 53.10 | 50ms | Fast alternative |
| Wavelet+GF+Chroma | 51.17 | 191ms | Lower quality |
| CB+DB+Chroma | 43.60 | 46ms | Cross-bilat too weak |

**Finding:** No new combination significantly beats temporal-premium. The existing presets are well-optimized.

## 5. Changes Made

### config.py
- **temporal-premium**: Updated h=8→h=10, eps=50→eps=10, chroma_denoise=0.2→0.1
- **New preset: temporal-ntsc**: h=4 optimized for NTSC-heavy content (+chroma_denoise=0.5)

### main.py
- Registered `temporal-ntsc` in PRESETS dict (now 36 presets)

### run_ablation.py
- Updated `ALL_PRESETS` from 15→36 presets (added all v3.2/v3.3 presets)

### scripts/autonomous_improve.py
- Fixed hardcoded paths → workspace-relative resolution
- Added `--repo-dir` argument for flexible execution

## 6. Current Best Presets (v3.3+)

| Preset | Score | Speed | Best for |
|---|---|---|---|
| **temporal-premium** | **55.34** | ~180ms | Best overall quality |
| chroma-focus | 54.63 | ~35ms | Fast quality |
| super-premium-fast | 52.17 | ~38ms | Good balance |
| temporal-ntsc | 46.98 | ~185ms | NTSC-heavy content |
| ntsc-plus | 48.02 | ~176ms | Best NTSC (wavelet-based) |
| ultralight | 41.76 | ~5ms | Real-time 30fps+ |

## 7. Next Steps (Iter10)

- Try temporal_nlm_multi with video-specific temporal window sweep
- Investigate depth-wise separable NLM for speed improvement
- Explore fusion of temporal-premium + wavelet for mixed noise types
- Multi-image benchmark across all 3 input files
