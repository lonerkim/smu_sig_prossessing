#!/usr/bin/env python3
"""
비디오 기반 파이프라인 테스트
- 합성 테스트 영상 생성 (컬러 파레트 회전 + 이동 피사체)
- 프레임별 열화 적용 (Gaussian + Impulse + Color bias + Brightness + Periodic)
- Phase 1~2 파이프라인 프레임 단위 적용
- 결과 영상 저장 + PSNR/SSIM 평가
"""

import os
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

BASE = os.path.expanduser("~/video-pipeline")
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

H, W, FPS, FRAMES = 480, 640, 30, 90  # 3초 @ 30fps

# ─── 열화 함수 ───

def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def add_impulse_noise(img, prob=0.03):
    noisy = img.copy()
    mask = np.random.random(img.shape[:2])
    noisy[mask < prob / 2] = 0
    noisy[mask > 1 - prob / 2] = 255
    return noisy

def add_color_bias(img, r_gain=1.3, g_gain=0.7, b_gain=1.1):
    biased = img.astype(np.float32)
    biased[:, :, 2] *= r_gain
    biased[:, :, 1] *= g_gain
    biased[:, :, 0] *= b_gain
    return np.clip(biased, 0, 255).astype(np.uint8)

def reduce_brightness(img, gamma_val=0.5):
    table = np.array([(i / 255.0) ** (1.0 / gamma_val) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def add_periodic_noise(img, freq=25, amplitude=30):
    noisy = img.astype(np.float32)
    rows, cols = img.shape[:2]
    x = np.arange(cols)
    y = np.arange(rows)
    xx, yy = np.meshgrid(x, y)
    pattern = amplitude * np.sin(2 * np.pi * freq / cols * xx + 2 * np.pi * freq / rows * yy)
    noisy += pattern[:, :, np.newaxis]
    return np.clip(noisy, 0, 255).astype(np.uint8)

def degrade_frame(img):
    d = img.copy()
    d = add_gaussian_noise(d, sigma=25)
    d = add_impulse_noise(d, prob=0.03)
    d = add_color_bias(d, r_gain=1.3, g_gain=0.7, b_gain=1.1)
    d = reduce_brightness(d, gamma_val=0.5)
    d = add_periodic_noise(d, freq=25, amplitude=30)
    return d


# ─── 복원 함수 ───

def median_filter(img, ksize=3):
    return cv2.medianBlur(img, ksize)

def gaussian_lowpass(img, sigma=3.0):
    return cv2.GaussianBlur(img, (0, 0), sigma)

def fft_notch_filter(img, threshold_percentile=99.5):
    if len(img.shape) == 3:
        result = np.zeros_like(img)
        for c in range(3):
            result[:, :, c] = _notch_channel(img[:, :, c], threshold_percentile)
        return result
    return _notch_channel(img, threshold_percentile)

def _notch_channel(channel, threshold_percentile):
    f = np.fft.fft2(channel.astype(np.float64))
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift)
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2
    threshold = np.percentile(magnitude, threshold_percentile)
    mask = magnitude > threshold
    mask[max(0, crow - 3):crow + 4, max(0, ccol - 3):ccol + 4] = False
    f_shift[mask] = 0
    result = np.fft.ifftshift(f_shift)
    result = np.fft.ifft2(result)
    return np.clip(np.abs(result), 0, 255).astype(np.uint8)

def gamma_correction(img, gamma=1.8):
    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def histogram_equalization_yuv(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def channel_correction(img):
    result = img.astype(np.float32)
    means = [np.mean(result[:, :, c]) for c in range(3)]
    target = np.mean(means)
    for c in range(3):
        if means[c] > 0:
            scale = np.clip(target / means[c], 0.7, 1.3)
            result[:, :, c] *= scale
    return np.clip(result, 0, 255).astype(np.uint8)


def denoise_only(frame):
    """Phase 1 only: noise removal"""
    f = median_filter(frame, 3)
    f = gaussian_lowpass(f, sigma=3.0)
    f = fft_notch_filter(f, threshold_percentile=99.5)
    return f

def full_pipeline(frame):
    """Phase 1 + Phase 2"""
    f = median_filter(frame, 3)
    f = gaussian_lowpass(f, sigma=3.0)
    f = fft_notch_filter(f, threshold_percentile=99.5)
    f = channel_correction(f)
    f = gamma_correction(f, gamma=1.8)
    f = histogram_equalization_yuv(f)
    return f


# ─── 메인 ───

def main():
    print("=" * 60)
    print("Video Pipeline Test — Synthetic Video + Artificial Noise")
    print(f"  Resolution: {W}x{H}, {FPS}fps, {FRAMES} frames ({FRAMES/FPS:.1f}s)")
    print("=" * 60)

    # 1) 합성 원본 영상 생성
    print("\n[1/4] Generating synthetic clean video...")
    clean_path = os.path.join(OUTPUT_DIR, "video_clean.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_clean = cv2.VideoWriter(clean_path, fourcc, FPS, (W, H))

    clean_frames = []
    for i in range(FRAMES):
        t = i / FRAMES
        # 배경: 시간에 따라 변하는 그라데이션
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(H):
            for x in range(W):
                hue_shift = int(t * 180)
                r = int((x / W * 180 + hue_shift) % 256)
                g = int((y / H * 180 + hue_shift * 0.5) % 256)
                b = int(128 + 60 * np.sin(t * 2 * np.pi))
                img[y, x] = [b, g, r]

        # 이동하는 피사체 (원)
        cx = int(100 + t * (W - 200))
        cy = int(H // 2 + 80 * np.sin(t * 4 * np.pi))
        cv2.circle(img, (cx, cy), 40, (255, 200, 100), -1)
        cv2.circle(img, (cx, cy), 40, (255, 255, 255), 2)

        # 텍스트 오버레이
        cv2.putText(img, f"Frame {i:03d}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        clean_frames.append(img)
        writer_clean.write(img)

    writer_clean.release()
    print(f"  saved: {clean_path}")

    # 2) 열화 적용
    print("\n[2/4] Applying degradation per frame...")
    degraded_path = os.path.join(OUTPUT_DIR, "video_degraded.mp4")
    writer_deg = cv2.VideoWriter(degraded_path, fourcc, FPS, (W, H))

    degraded_frames = []
    for i, frame in enumerate(clean_frames):
        d = degrade_frame(frame)
        degraded_frames.append(d)
        writer_deg.write(d)
        if (i + 1) % 30 == 0:
            print(f"  degraded {i+1}/{FRAMES} frames")

    writer_deg.release()
    print(f"  saved: {degraded_path}")

    # 3) Phase 1 only (denoise)
    print("\n[3/4] Running Phase 1 pipeline (denoise only)...")
    denoise_path = os.path.join(OUTPUT_DIR, "video_denoised_p1.mp4")
    writer_denoise = cv2.VideoWriter(denoise_path, fourcc, FPS, (W, H))

    p1_psnr_list = []
    p1_ssim_list = []
    for i, (clean, degraded) in enumerate(zip(clean_frames, degraded_frames)):
        restored = denoise_only(degraded)
        writer_denoise.write(restored)

        p = psnr(clean, restored)
        s = ssim(clean, restored, channel_axis=-1)
        p1_psnr_list.append(p)
        p1_ssim_list.append(s)

        if (i + 1) % 30 == 0:
            avg_p = np.mean(p1_psnr_list[-30:])
            avg_s = np.mean(p1_ssim_list[-30:])
            print(f"  frame {i+1}/{FRAMES} — recent avg PSNR={avg_p:.2f} SSIM={avg_s:.4f}")

    writer_denoise.release()
    print(f"  saved: {denoise_path}")

    # 4) Full pipeline (Phase 1 + 2)
    print("\n[4/4] Running Full pipeline (Phase 1 + Phase 2)...")
    full_path = os.path.join(OUTPUT_DIR, "video_full_pipeline.mp4")
    writer_full = cv2.VideoWriter(full_path, fourcc, FPS, (W, H))

    full_psnr_list = []
    full_ssim_list = []
    for i, (clean, degraded) in enumerate(zip(clean_frames, degraded_frames)):
        restored = full_pipeline(degraded)
        writer_full.write(restored)

        p = psnr(clean, restored)
        s = ssim(clean, restored, channel_axis=-1)
        full_psnr_list.append(p)
        full_ssim_list.append(s)

        if (i + 1) % 30 == 0:
            avg_p = np.mean(full_psnr_list[-30:])
            avg_s = np.mean(full_ssim_list[-30:])
            print(f"  frame {i+1}/{FRAMES} — recent avg PSNR={avg_p:.2f} SSIM={avg_s:.4f}")

    writer_full.release()
    print(f"  saved: {full_path}")

    # ── 결과 요약 ──
    print("\n" + "=" * 60)
    print("📊 Video Results Summary")
    print("=" * 60)

    # degraded baseline
    deg_psnr = [psnr(c, d) for c, d in zip(clean_frames, degraded_frames)]
    deg_ssim = [ssim(c, d, channel_axis=-1) for c, d in zip(clean_frames, degraded_frames)]

    print(f"\n  {'Method':30s}  {'Avg PSNR':>10s}  {'Avg SSIM':>10s}  {'Min PSNR':>10s}  {'Max SSIM':>10s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for label, plist, slist in [
        ("Degraded (no processing)", deg_psnr, deg_ssim),
        ("Phase 1 (denoise only)", p1_psnr_list, p1_ssim_list),
        ("Full pipeline (P1+P2)", full_psnr_list, full_ssim_list),
    ]:
        print(f"  {label:30s}  {np.mean(plist):8.2f} dB  {np.mean(slist):10.4f}  {np.min(plist):8.2f} dB  {np.max(slist):10.4f}")

    # 대표 프레임 비교 이미지 저장 (중간 프레임)
    mid = FRAMES // 2
    for idx in [0, mid, FRAMES - 1]:
        h, w = H, W
        canvas = np.zeros((h, w * 4, 3), dtype=np.uint8)
        canvas[:, :w] = clean_frames[idx]
        canvas[:, w:2*w] = degraded_frames[idx]
        canvas[:, 2*w:3*w] = denoise_only(degraded_frames[idx])
        canvas[:, 3*w:] = full_pipeline(degraded_frames[idx])

        labels = ["Clean", "Degraded", "Phase1 Denoise", "Full Pipeline"]
        for j, lbl in enumerate(labels):
            cv2.putText(canvas, lbl, (j * w + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        path = os.path.join(OUTPUT_DIR, f"video_frame{idx:03d}_comparison.png")
        cv2.imwrite(path, canvas)
        print(f"\n  frame comparison: {path}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
