#!/usr/bin/env python3
"""
정성적 & 정량적 평가 실행기

정성적 평가:
  - Google Drive 샘플 (1분 분량) → NTSC 아티팩트 추출 → 시각적 비교
정량적 평가:
  - Google Photos 샘플 (1분 분량) → PSNR/SSIM 측정 → 수치 비교

Usage:
    python scripts/run_evaluation.py                           # 기본 (합성)
    python scripts/run_evaluation.py --qualitative <video>     # 정성적
    python scripts/run_evaluation.py --quantitative <video>    # 정량적
    python scripts/run_evaluation.py --all                     # 모두
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing import filters
from smu_sig_prossessing.degradation import (
    degrade_image, add_ntsc_noise,
    add_gaussian_noise, add_impulse_noise, add_periodic_noise
)
from smu_sig_prossessing.evaluation import (
    evaluate, save_comparison, save_four_comparison,
    save_histogram_comparison, print_summary
)

BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE, "output")
SAMPLES_DIR = os.path.join(BASE, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

H, W = 480, 640


def extract_frames_from_video(video_path: str, max_frames: int = 1800,
                              target_dir: str | None = None) -> list[np.ndarray]:
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or count >= max_frames:
            break
        frames.append(frame)
        count += 1
        if count % 100 == 0:
            print(f"  extracted {count} frames...")
    cap.release()
    print(f"  total: {len(frames)} frames from {video_path}")
    return frames


# ─── 정성적 평가 (Qualitative) ──────────────────────────────────────

def qualitative_evaluation(video_path: str | None = None):
    """
    정성적 평가: 실제 아날로그 영상에 파이프라인 적용 후 시각적 비교.
    Google Drive (https://drive.google.com/file/d/1Yt9malgEFHyRFCHFGCQJxm7OdNSNLD3H)
    에서 1분 샘플 → 프레임 추출 → 각 파이프라인 적용 → 비교 이미지 생성.
    """
    print("=" * 60)
    print("🎨 정성적 평가 (Qualitative Evaluation)")
    print("=" * 60)
    print("\n  대상: 실제 아날로그 DVR / CCTV / FPV 영상")
    print("  목표: 육안으로 확인 가능한 화질 개선")
    print("  출력: 처리 전/후 나란히 비교 이미지\n")

    if video_path and os.path.exists(video_path):
        print(f"[1/3] Loading video: {video_path}")
        frames = extract_frames_from_video(video_path, max_frames=1800)
    else:
        print("[1/3] No video provided. Generating synthetic test frames...")
        frames = [_generate_test_frame(i / 90) for i in range(90)]
        # Apply NTSC noise for realistic artifacts
        frames = [add_ntsc_noise(f, intensity="heavy") for f in frames]

    # 대표 프레임 선정 (처음, 중간, 끝)
    print(f"[2/3] Processing {min(10, len(frames))} representative frames...")
    step = max(1, len(frames) // 10)
    rep_frames = frames[::step][:10]

    # 여러 파이프라인 적용
    pipelines = {
        "wiener_only": PipelineConfig.wiener_only(),
        "edge_preserving": PipelineConfig.edge_preserving(),
        "research_best": PipelineConfig.research_best(),
    }

    for i, frame in enumerate(rep_frames):
        results = {"degraded": frame}
        for name, cfg in pipelines.items():
            results[name] = pl.apply_pipeline(frame, cfg)

        # 4-way comparison (degraded | wiener | edge | research)
        if len(rep_frames) <= 10:
            from smu_sig_prossessing.evaluation import save_four_comparison
            # Create extended comparison: degraded → filtered versions
            h, w = frame.shape[:2]
            n_cols = 4
            canvas = np.zeros((h, w * n_cols, 3), dtype=np.uint8)
            canvas[:, :w] = frame  # degraded input
            canvas[:, w:2*w] = results["wiener_only"]
            canvas[:, 2*w:3*w] = results["edge_preserving"]
            canvas[:, 3*w:4*w] = results["research_best"]

            lbls = ["Input (Degraded)", "Wiener Pipeline", "Edge-Preserving", "Research Best"]
            for j, lbl in enumerate(lbls):
                cv2.putText(canvas, lbl, (j * w + 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            path = os.path.join(OUTPUT_DIR, f"qualitative_frame_{i:03d}_comparison.png")
            cv2.imwrite(path, canvas)
            print(f"  saved: {path}")

    print(f"\n[3/3] ✅ 정성적 평가 완료 → {OUTPUT_DIR}/qualitative_*.png")


# ─── 정량적 평가 (Quantitative) ─────────────────────────────────────

def quantitative_evaluation(video_path: str | None = None):
    """
    정량적 평가: PSNR/SSIM 수치 측정.
    Google Photos (https://photos.app.goo.gl/XrXXiejisjcFhoiH8)
    에서 1분 샘플 → 프레임별 PSNR/SSIM 측정 → 표로 정리.
    """
    print("=" * 60)
    print("📊 정량적 평가 (Quantitative Evaluation)")
    print("=" * 60)
    print("\n  대상: 원본(Clean) + 열화(Degraded) + 복원(Restored)")
    print("  지표: PSNR (dB), SSIM")
    print("  비교: Wiener-only vs Edge-Preserving vs Research Best\n")

    if video_path and os.path.exists(video_path):
        print(f"[1/3] Loading video: {video_path}")
        frames = extract_frames_from_video(video_path, max_frames=1800)
        # If only one video (degraded), we need separate clean reference
        print("  ⚠ Using single video — treating as degraded input only.")
        clean_frames = None
        degraded_frames = frames
    else:
        print("[1/3] Generating synthetic evaluation data...")
        clean_frames = [_generate_test_frame(i / 90) for i in range(90)]
        # Degrade with NTSC for realistic artifacts
        degraded_frames = [degrade_image(f, use_ntsc=True, ntsc_intensity="medium")
                           for f in clean_frames]

    print(f"[2/3] Running pipelines on {len(degraded_frames)} frames...")

    cfg_list = {
        "Wiener Pipeline": PipelineConfig.wiener_only(),
        "Edge-Preserving": PipelineConfig.edge_preserving(),
        "Research Best": PipelineConfig.research_best(),
    }

    metrics = {}
    for label, cfg in cfg_list.items():
        print(f"\n  Processing: {label}...")
        p_list, s_list = [], []
        for i, deg in enumerate(degraded_frames):
            r = pl.apply_pipeline(deg, cfg)
            # For PSNR comparison, compare degraded→restored if no clean reference
            ref = clean_frames[i] if clean_frames else deg
            # Use degraded as reference to measure improvement
            p = psnr(deg, r)  # How much did the processing clean the image?
            s = ssim(deg, r, channel_axis=-1)
            p_list.append(p)
            s_list.append(s)

        metrics[label] = {
            "psnr_mean": float(np.mean(p_list)),
            "psnr_std": float(np.std(p_list)),
            "ssim_mean": float(np.mean(s_list)),
            "ssim_std": float(np.std(s_list)),
        }

    # If we have clean reference, also compare clean vs restored
    if clean_frames:
        for label, cfg in cfg_list.items():
            p_list, s_list = [], []
            for i, (clean, deg) in enumerate(zip(clean_frames, degraded_frames)):
                r = pl.apply_pipeline(deg, cfg)
                p = psnr(clean, r)
                s = ssim(clean, r, channel_axis=-1)
                p_list.append(p)
                s_list.append(s)
            metrics[f"{label} (vs clean)"] = {
                "psnr_mean": float(np.mean(p_list)),
                "psnr_std": float(np.std(p_list)),
                "ssim_mean": float(np.mean(s_list)),
                "ssim_std": float(np.std(s_list)),
            }

    # Report
    print(f"\n[3/3] ✅ 정량적 평가 결과")
    print(f"\n{'=' * 60}")
    print("📊 Quantitative Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  {'Method':40s}  {'PSNR (dB)':>14s}  {'SSIM':>10s}")
    print(f"  {'-' * 40}  {'-' * 14}  {'-' * 10}")
    for label, m in metrics.items():
        print(f"  {label:40s}  {m['psnr_mean']:7.2f} ± {m['psnr_std']:.2f}  {m['ssim_mean']:.4f} ± {m['ssim_std']:.4f}")

    return metrics


def _generate_test_frame(t: float) -> np.ndarray:
    """Generate a synthetic test frame (shared between eval functions)."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            hue_shift = int(t * 180)
            r = int((x / W * 180 + hue_shift) % 256)
            g = int((y / H * 180 + hue_shift * 0.5) % 256)
            b = int(128 + 60 * np.sin(t * 2 * np.pi))
            img[y, x] = [b, g, r]
    cx = int(100 + t * (W - 200))
    cy = int(H // 2 + 80 * np.sin(t * 4 * np.pi))
    cv2.circle(img, (cx, cy), 40, (255, 200, 100), -1)
    cv2.circle(img, (cx, cy), 40, (255, 255, 255), 2)
    return img


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SMU Signal Processing Evaluation")
    parser.add_argument("--qualitative", "-q", nargs="?", const=True,
                        help="Qualitative eval (optionally: video path)")
    parser.add_argument("--quantitative", "-qt", nargs="?", const=True,
                        help="Quantitative eval (optionally: video path)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Run both qualitative and quantitative")
    args = parser.parse_args()

    run_all = args.all or (len(sys.argv) == 1)

    if run_all or args.qualitative:
        vid_path = args.qualitative if isinstance(args.qualitative, str) else None
        qualitative_evaluation(vid_path)

    if run_all or args.quantitative:
        vid_path = args.quantitative if isinstance(args.quantitative, str) else None
        quantitative_evaluation(vid_path)


if __name__ == "__main__":
    main()
