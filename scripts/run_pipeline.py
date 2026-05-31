#!/usr/bin/env python3
"""
정적 이미지 파이프라인 실행기 (모듈식 v2.0)
"""
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import filters
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.degradation import degrade_image, add_ntsc_noise
from smu_sig_prossessing.evaluation import (
    evaluate, save_comparison, save_histogram_comparison, print_summary
)

BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR = os.path.join(BASE, "input")
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

H, W = 480, 640


# ─── Test image generation (same as before) ─────────────────────────

def generate_test_images():
    images = {}
    # 1) synthetic_color
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            img[y, x] = [int(x / W * 255), int(y / H * 255), 128]
    cv2.rectangle(img, (100, 100), (250, 250), (255, 255, 255), 2)
    cv2.circle(img, (450, 300), 80, (0, 0, 255), -1)
    cv2.line(img, (50, 400), (590, 100), (255, 255, 0), 3)
    cv2.putText(img, "TEST", (250, 380), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    images["synthetic_color"] = img
    # 2) portrait_mock
    img2 = np.random.randint(60, 200, (H, W, 3), dtype=np.uint8)
    cv2.ellipse(img2, (320, 240), (120, 150), 0, 0, 360, (180, 140, 120), -1)
    cv2.circle(img2, (280, 200), 15, (40, 40, 40), -1)
    cv2.circle(img2, (360, 200), 15, (40, 40, 40), -1)
    cv2.ellipse(img2, (320, 270), (40, 20), 0, 0, 180, (60, 40, 40), -1)
    for i in range(0, W, 40):
        cv2.line(img2, (i, 0), (i, H), (100, 100, 150), 1)
    images["portrait_mock"] = img2
    # 3) high_contrast
    img3 = np.zeros((H, W, 3), dtype=np.uint8)
    img3[100:200, 100:540] = 255
    img3[250:350, 150:490] = (0, 200, 200)
    for x in range(300, 380, 2):
        img3[380:460, x] = 255
    images["high_contrast"] = img3
    # 4) low_light
    img4 = np.random.randint(10, 50, (H, W, 3), dtype=np.uint8)
    cv2.circle(img4, (320, 240), 100, (30, 40, 50), -1)
    cv2.circle(img4, (320, 240), 50, (50, 60, 70), -1)
    images["low_light"] = img4
    # 5) color_bias
    img5 = np.zeros((H, W, 3), dtype=np.uint8)
    img5[:H//3, :] = [200, 50, 50]
    img5[H//3:2*H//3, :] = [50, 180, 50]
    img5[2*H//3:, :] = [50, 50, 200]
    cv2.circle(img5, (320, 240), 80, (180, 180, 180), -1)
    images["color_bias"] = img5
    for name, im in images.items():
        cv2.imwrite(os.path.join(INPUT_DIR, f"{name}.png"), im)
    return images


# ─── Pipeline comparison runner ─────────────────────────────────────

def run_single_filter_evaluation(originals, degraded_map):
    """Run each filter individually on degraded images and report PSNR/SSIM."""
    results = {}
    for name, original in originals.items():
        degraded = degraded_map[name]
        img_results = []

        # Phase 1-A: Median
        med3 = filters.median_filter(degraded, ksize=3)
        img_results.append(evaluate(original, med3, "Median k=3"))
        save_comparison(original, degraded, med3,
                        os.path.join(OUTPUT_DIR, f"{name}_median_k3.png"))

        med5 = filters.median_filter(degraded, ksize=5)
        img_results.append(evaluate(original, med5, "Median k=5"))
        save_comparison(original, degraded, med5,
                        os.path.join(OUTPUT_DIR, f"{name}_median_k5.png"))

        # Phase 1-B: Wiener (NO Gaussian LP — Wiener only)
        wf = filters.wiener_filter(degraded, noise_var=625)
        img_results.append(evaluate(original, wf, "Wiener var=625"))
        save_comparison(original, degraded, wf,
                        os.path.join(OUTPUT_DIR, f"{name}_wiener_625.png"))

        # Additional methods: NLM + Bilateral
        nlm = filters.nlm_filter(degraded, h=10)
        img_results.append(evaluate(original, nlm, "NLM h=10"))
        save_comparison(original, degraded, nlm,
                        os.path.join(OUTPUT_DIR, f"{name}_nlm_h10.png"))

        bil = filters.bilateral_filter(degraded, d=9, sigma_color=75, sigma_space=75)
        img_results.append(evaluate(original, bil, "Bilateral d=9"))
        save_comparison(original, degraded, bil,
                        os.path.join(OUTPUT_DIR, f"{name}_bilateral_9.png"))

        # Phase 1-C: FFT Notch
        fft = filters.fft_notch_filter(degraded, threshold_percentile=99.5)
        img_results.append(evaluate(original, fft, "FFT Notch 99.5%"))
        save_comparison(original, degraded, fft,
                        os.path.join(OUTPUT_DIR, f"{name}_fft_notch_99.5.png"))

        # Phase 2: Color/contrast (on best denoised base)
        best_base = filters.wiener_filter(filters.median_filter(degraded, 3), noise_var=400)

        gam = filters.gamma_correction(best_base, gamma=2.0)
        img_results.append(evaluate(original, gam, "Gamma=2.0"))
        save_comparison(original, degraded, gam,
                        os.path.join(OUTPUT_DIR, f"{name}_gamma_2.0.png"))
        save_histogram_comparison(best_base, gam,
                                  os.path.join(OUTPUT_DIR, f"{name}_hist_gamma_2.0.png"))

        he = filters.histogram_equalization_yuv(best_base)
        img_results.append(evaluate(original, he, "HistEq YUV"))
        save_comparison(original, degraded, he,
                        os.path.join(OUTPUT_DIR, f"{name}_histeq_yuv.png"))
        save_histogram_comparison(best_base, he,
                                  os.path.join(OUTPUT_DIR, f"{name}_hist_histeq_yuv.png"))

        ch = filters.channel_correction(best_base)
        img_results.append(evaluate(original, ch, "Channel Correction"))
        save_comparison(original, degraded, ch,
                        os.path.join(OUTPUT_DIR, f"{name}_channel_corr.png"))
        save_histogram_comparison(best_base, ch,
                                  os.path.join(OUTPUT_DIR, f"{name}_hist_channel_corr.png"))

        # Deblur test (Wiener deconvolution)
        deblur = filters.deblur_wiener(best_base, kernel_size=5, noise_var=0.01)
        img_results.append(evaluate(original, deblur, "Deblur Wiener"))
        save_comparison(original, degraded, deblur,
                        os.path.join(OUTPUT_DIR, f"{name}_deblur_wiener.png"))

        # Unsharp mask (detail recovery)
        sharp = filters.unsharp_mask(best_base, strength=1.0, radius=1.0)
        img_results.append(evaluate(original, sharp, "Unsharp Mask"))
        save_comparison(original, degraded, sharp,
                        os.path.join(OUTPUT_DIR, f"{name}_unsharp.png"))

        results[name] = img_results

    return results


def run_full_pipelines(originals, degraded_map):
    """Run all preset pipelines and compare."""
    presets = {
        "wiener_only": PipelineConfig.wiener_only(),
        "edge_preserving": PipelineConfig.edge_preserving(),
        "aggressive": PipelineConfig.aggressive(),
        "research_best": PipelineConfig.research_best(),
    }

    all_results = []
    for preset_name, cfg in presets.items():
        print(f"\n  {'─' * 50}")
        print(f"  Pipeline: {cfg.label}")
        print(f"  {'─' * 50}")
        for name, original in originals.items():
            degraded = degraded_map[name]
            result = pl.apply_pipeline(degraded, cfg)
            r = evaluate(original, result, f"[{name}] {cfg.label}")
            all_results.append(r)
            save_comparison(original, degraded, result,
                            os.path.join(OUTPUT_DIR, f"{name}_pipeline_{preset_name}.png"),
                            labels=("Original", "Degraded", cfg.label))
            save_histogram_comparison(degraded, result,
                                       os.path.join(OUTPUT_DIR, f"{name}_hist_{preset_name}.png"))

    print_summary(all_results)


def main():
    print("=" * 60)
    print("📷 Modular Image Processing Pipeline v2.0")
    print("   Wiener-only denoising + modular filter control")
    print("=" * 60)

    # Phase 0: Generate images
    print("\n[Phase 0] Generating test images...")
    originals = generate_test_images()

    # Degrade using NTSC + basic noise
    print("\n[Phase 0] Applying degradation (NTSC analog + synthetic)...")
    degraded_map = {}
    for name, img in originals.items():
        d = degrade_image(img, use_ntsc=False)
        # Save both basic and NTSC versions
        d_ntsc = degrade_image(img, use_ntsc=True, ntsc_intensity="medium")
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_degraded.png"), d)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_ntsc_degraded.png"), d_ntsc)
        degraded_map[name] = d

    # Phase 1-2: Individual filter evaluation
    print("\n" + "=" * 60)
    print("[Phase 1-2] Individual Filter Evaluation")
    print("=" * 60)
    results = run_single_filter_evaluation(originals, degraded_map)

    # Summary per image
    print("\n" + "=" * 60)
    print("📊 Results by Image Type")
    print("=" * 60)
    for name, img_results in results.items():
        print(f"\n[{name}]")
        for r in img_results:
            print(f"  {r['label']:30s}  PSNR={r['psnr']:6.2f}  SSIM={r['ssim']:.4f}")

    # Full pipeline presets
    print("\n" + "=" * 60)
    print("[Pipeline] Preset Pipeline Comparison")
    print("=" * 60)
    run_full_pipelines(originals, degraded_map)

    print(f"\n✅ All outputs → {OUTPUT_DIR}/")
    print(f"   Inputs  → {INPUT_DIR}/")


if __name__ == "__main__":
    main()
