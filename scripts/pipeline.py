#!/usr/bin/env python3
"""
아날로그 영상 잡음 완화 파이프라인 — Phase 0 ~ Phase 2
기준 이미지 생성 → 인위 열화 → 잡음 제거 → 색상/대비 보정 → PSNR/SSIM 평가
"""

import os
import sys
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

BASE = os.path.expanduser("~/video-pipeline")
INPUT_DIR = os.path.join(BASE, "input")
OUTPUT_DIR = os.path.join(BASE, "output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

H, W = 480, 640  # 표준 해상도


# ─── Phase 0: 기준 이미지 + 인위 열화 ────────────────────────────

def generate_test_images():
    """다양한 특성을 가진 합성 기준 이미지 5장 생성"""
    images = {}

    # 1) 컬러 그라데이션 + 형태
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            img[y, x] = [int(x / W * 255), int(y / H * 255), 128]
    cv2.rectangle(img, (100, 100), (250, 250), (255, 255, 255), 2)
    cv2.circle(img, (450, 300), 80, (0, 0, 255), -1)
    cv2.line(img, (50, 400), (590, 100), (255, 255, 0), 3)
    cv2.putText(img, "TEST", (250, 380), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    images["synthetic_color"] = img

    # 2) Lena 유사 — 랜덤 텍스처 + 피사체
    img2 = np.random.randint(60, 200, (H, W, 3), dtype=np.uint8)
    # 부드러운 피부 톤 영역
    cv2.ellipse(img2, (320, 240), (120, 150), 0, 0, 360, (180, 140, 120), -1)
    cv2.circle(img2, (280, 200), 15, (40, 40, 40), -1)
    cv2.circle(img2, (360, 200), 15, (40, 40, 40), -1)
    cv2.ellipse(img2, (320, 270), (40, 20), 0, 0, 180, (60, 40, 40), -1)
    # 배경 패턴
    for i in range(0, W, 40):
        cv2.line(img2, (i, 0), (i, H), (100, 100, 150), 1)
    images["portrait_mock"] = img2

    # 3) 고대비 엣지 테스트 (ringing/dot crawl 확인용)
    img3 = np.zeros((H, W, 3), dtype=np.uint8)
    img3[100:200, 100:540] = 255
    img3[250:350, 150:490] = (0, 200, 200)
    # 미세 스트라이프 패턴 (cross-color 유발)
    for x in range(300, 380, 2):
        img3[380:460, x] = 255
    images["high_contrast"] = img3

    # 4) 어두운 영상 (low-light 시나리오)
    img4 = np.random.randint(10, 50, (H, W, 3), dtype=np.uint8)
    cv2.circle(img4, (320, 240), 100, (30, 40, 50), -1)
    cv2.circle(img4, (320, 240), 50, (50, 60, 70), -1)
    images["low_light"] = img4

    # 5) 색 편향 영상 (채도 높은 영역)
    img5 = np.zeros((H, W, 3), dtype=np.uint8)
    img5[:H//3, :] = [200, 50, 50]       # 빨강 과다
    img5[H//3:2*H//3, :] = [50, 180, 50] # 초록 과다
    img5[2*H//3:, :] = [50, 50, 200]     # 파랑 과다
    cv2.circle(img5, (320, 240), 80, (180, 180, 180), -1)
    images["color_bias"] = img5

    # 저장
    for name, im in images.items():
        path = os.path.join(INPUT_DIR, f"{name}.png")
        cv2.imwrite(path, im)
        print(f"  saved: {path}")

    return images


# ─── 열화 함수 ───

def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def add_impulse_noise(img, prob=0.05):
    noisy = img.copy()
    mask = np.random.random(img.shape[:2])
    noisy[mask < prob / 2] = 0        # salt
    noisy[mask > 1 - prob / 2] = 255  # pepper
    return noisy

def reduce_brightness(img, gamma_val=0.4):
    """어둡게 + 저대비"""
    table = np.array([(i / 255.0) ** (1.0 / gamma_val) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def add_color_bias(img, r_gain=1.3, g_gain=0.8, b_gain=1.1):
    biased = img.astype(np.float32)
    biased[:, :, 2] *= r_gain  # R
    biased[:, :, 1] *= g_gain  # G
    biased[:, :, 0] *= b_gain  # B
    return np.clip(biased, 0, 255).astype(np.uint8)

def add_periodic_noise(img, freq=30, amplitude=50):
    """사인파 주기적 잡음 (diagonal banding 시뮬)"""
    noisy = img.astype(np.float32)
    rows, cols = img.shape[:2]
    x = np.arange(cols)
    y = np.arange(rows)
    xx, yy = np.meshgrid(x, y)
    pattern = amplitude * np.sin(2 * np.pi * freq / cols * xx + 2 * np.pi * freq / rows * yy)
    noisy += pattern[:, :, np.newaxis]
    return np.clip(noisy, 0, 255).astype(np.uint8)


def degrade_image(img):
    """종합 열화 — 모든 잡음 누적 적용"""
    d = img.copy()
    d = add_gaussian_noise(d, sigma=25)
    d = add_impulse_noise(d, prob=0.03)
    d = add_color_bias(d, r_gain=1.3, g_gain=0.7, b_gain=1.1)
    d = reduce_brightness(d, gamma_val=0.5)
    d = add_periodic_noise(d, freq=25, amplitude=30)
    return d


# ─── Phase 1: 잡음 제거 ─────────────────────────────────────────

def median_filter(img, ksize=3):
    return cv2.medianBlur(img, ksize)

def gaussian_lowpass(img, sigma=1.5):
    return cv2.GaussianBlur(img, (0, 0), sigma)

def wiener_filter(img, noise_var=625):
    """간단한 위너 필터 (주파수 도메인)"""
    if len(img.shape) == 3:
        result = np.zeros_like(img)
        for c in range(3):
            result[:, :, c] = _wiener_channel(img[:, :, c], noise_var)
        return result
    return _wiener_channel(img, noise_var)

def _wiener_channel(channel, noise_var):
    f = np.fft.fft2(channel.astype(np.float64))
    f_shift = np.fft.fftshift(f)
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2

    # 신호 파워 스펙트럼 추정
    power = np.abs(f_shift) ** 2
    signal_power = np.mean(power)

    # 위너 필터: H(w) = S(w) / (S(w) + N(w))
    h = signal_power / (power + noise_var)
    h = np.clip(h, 0, 1)

    result = f_shift * h
    result = np.fft.ifftshift(result)
    result = np.fft.ifft2(result)
    return np.clip(np.abs(result), 0, 255).astype(np.uint8)

def fft_notch_filter(img, threshold_percentile=99.5):
    """2D FFT → 노이즈 피크 제거 → IFFT"""
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

    # DC 성분 보호
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2

    # 노이즈 피크 탐지
    threshold = np.percentile(magnitude, threshold_percentile)
    mask = magnitude > threshold
    # DC 및 낮은 주파수 보존
    mask[max(0, crow - 3):crow + 4, max(0, ccol - 3):ccol + 4] = False

    # 피크 제거
    f_shift[mask] = 0

    result = np.fft.ifftshift(f_shift)
    result = np.fft.ifft2(result)
    return np.clip(np.abs(result), 0, 255).astype(np.uint8)


# ─── Phase 2: 색상 및 대비 보정 ──────────────────────────────────

def gamma_correction(img, gamma=1.8):
    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def log_transform(img, c=40):
    img_f = img.astype(np.float64)
    result = c * np.log1p(img_f)
    return np.clip(result, 0, 255).astype(np.uint8)

def histogram_equalization_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def histogram_equalization_yuv(img):
    """YUV 공간에서 Y채널만 평활화"""
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def channel_correction(img):
    """RGB 채널별 밝기 분포 분석 → 편향 보정"""
    result = img.astype(np.float32)
    means = [np.mean(result[:, :, c]) for c in range(3)]
    target = np.mean(means)
    for c in range(3):
        if means[c] > 0:
            scale = target / means[c]
            scale = np.clip(scale, 0.7, 1.3)  # 과도 보정 방지
            result[:, :, c] *= scale
    return np.clip(result, 0, 255).astype(np.uint8)


# ─── 평가 ────────────────────────────────────────────────────────

def evaluate(original, processed, label=""):
    if original.shape != processed.shape:
        # 채널 수 맞추기
        if len(original.shape) == 2 and len(processed.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        elif len(original.shape) == 3 and len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    p = psnr(original, processed)
    s = ssim(original, processed, channel_axis=-1 if len(original.shape) == 3 else None)
    print(f"  {label:30s}  PSNR={p:.2f} dB   SSIM={s:.4f}")
    return {"label": label, "psnr": round(p, 2), "ssim": round(s, 4)}


def save_comparison(original, degraded, processed, name, phase_label):
    """전/후 비교 이미지 (나란히)"""
    h, w = original.shape[:2]
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, :w] = original
    canvas[:, w:2*w] = degraded
    canvas[:, 2*w:] = processed
    # 라벨
    cv2.putText(canvas, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(canvas, "Degraded", (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(canvas, phase_label, (2*w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    path = os.path.join(OUTPUT_DIR, f"{name}_{phase_label.replace(' ', '_').replace('/', '_')}.png")
    cv2.imwrite(path, canvas)
    return path

def save_histogram_comparison(original, processed, name, phase_label):
    """히스토그램 전/후 비교"""
    fig_w = 600
    fig_h = 400
    canvas = np.zeros((fig_h, fig_w * 2, 3), dtype=np.uint8)

    for idx, (img, label) in enumerate([(original, "Before"), (processed, "After")]):
        x_off = idx * fig_w
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]  # BGR
        for c in range(3):
            hist = cv2.calcHist([img], [c], None, [256], [0, 256])
            hist = hist / hist.max() * 150 if hist.max() > 0 else hist
            pts = []
            for i in range(256):
                px = x_off + 20 + int(i * (fig_w - 40) / 256)
                py = 300 - int(hist[i][0])
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, colors[c], 1)
        cv2.putText(canvas, label, (x_off + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    path = os.path.join(OUTPUT_DIR, f"{name}_hist_{phase_label.replace(' ', '_').replace('/', '_')}.png")
    cv2.imwrite(path, canvas)
    return path


# ─── 메인 실행 ────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 0: 기준 이미지 생성 + 인위 열화")
    print("=" * 60)

    originals = generate_test_images()

    results = {}

    for name, original in originals.items():
        print(f"\n{'─' * 50}")
        print(f"이미지: {name}")
        print(f"{'─' * 50}")

        degraded = degrade_image(original)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_degraded.png"), degraded)

        # 열화된 이미지 평가
        evaluate(original, degraded, "Degraded (base)")

        img_results = []

        # ── Phase 1-A: 미디언 필터 ──
        print(f"\n  Phase 1-A: Median Filter")
        med3 = median_filter(degraded, 3)
        r = evaluate(original, med3, "Median k=3")
        save_comparison(original, degraded, med3, name, "median_k3")
        img_results.append(r)

        med5 = median_filter(degraded, 5)
        r = evaluate(original, med5, "Median k=5")
        save_comparison(original, degraded, med5, name, "median_k5")
        img_results.append(r)

        # ── Phase 1-B: 로우패스 + 위너 ──
        print(f"\n  Phase 1-B: Lowpass + Wiener")
        lp = gaussian_lowpass(degraded, sigma=1.5)
        r = evaluate(original, lp, "Gaussian LP sigma=1.5")
        save_comparison(original, degraded, lp, name, "gauss_lp_1.5")
        img_results.append(r)

        lp2 = gaussian_lowpass(degraded, sigma=3.0)
        r = evaluate(original, lp2, "Gaussian LP sigma=3.0")
        save_comparison(original, degraded, lp2, name, "gauss_lp_3.0")
        img_results.append(r)

        wf = wiener_filter(degraded, noise_var=625)
        r = evaluate(original, wf, "Wiener var=625")
        save_comparison(original, degraded, wf, name, "wiener_625")
        img_results.append(r)

        # ── Phase 1-C: FFT notch ──
        print(f"\n  Phase 1-C: FFT Notch Filter")
        fft = fft_notch_filter(degraded, threshold_percentile=99.5)
        r = evaluate(original, fft, "FFT Notch 99.5%")
        save_comparison(original, degraded, fft, name, "fft_notch_99.5")
        img_results.append(r)

        # ── Phase 2-A: 감마 보정 ──
        print(f"\n  Phase 2-A: Gamma Correction")
        # 열화 이미지에 위상1 최적 필터 적용 후 보정
        # 여기서는 median+wiener 조합을 기본으로 사용
        best_denoised = wiener_filter(median_filter(degraded, 3), noise_var=400)

        gam = gamma_correction(best_denoised, gamma=2.0)
        r = evaluate(original, gam, "Gamma=2.0 (after denoise)")
        save_comparison(original, degraded, gam, name, "gamma_2.0")
        save_histogram_comparison(best_denoised, gam, name, "gamma_2.0")
        img_results.append(r)

        gam15 = gamma_correction(best_denoised, gamma=1.5)
        r = evaluate(original, gam15, "Gamma=1.5 (after denoise)")
        save_comparison(original, degraded, gam15, name, "gamma_1.5")
        img_results.append(r)

        # ── Phase 2-B: 로그 연산 ──
        print(f"\n  Phase 2-B: Log Transform")
        log_img = log_transform(best_denoised, c=40)
        r = evaluate(original, log_img, "Log c=40 (after denoise)")
        save_comparison(original, degraded, log_img, name, "log_c40")
        save_histogram_comparison(best_denoised, log_img, name, "log_c40")
        img_results.append(r)

        # ── Phase 2-C: 히스토그램 평활화 ──
        print(f"\n  Phase 2-C: Histogram Equalization")
        he_gray = histogram_equalization_gray(best_denoised)
        r = evaluate(original, he_gray, "HistEq (gray)")
        save_comparison(original, degraded, he_gray, name, "histeq_gray")
        img_results.append(r)

        he_yuv = histogram_equalization_yuv(best_denoised)
        r = evaluate(original, he_yuv, "HistEq (YUV-Y)")
        save_comparison(original, degraded, he_yuv, name, "histeq_yuv")
        save_histogram_comparison(best_denoised, he_yuv, name, "histeq_yuv")
        img_results.append(r)

        # ── Phase 2-D: 채널별 보정 ──
        print(f"\n  Phase 2-D: Channel Correction")
        ch_corr = channel_correction(best_denoised)
        r = evaluate(original, ch_corr, "Channel Correction")
        save_comparison(original, degraded, ch_corr, name, "channel_corr")
        save_histogram_comparison(best_denoised, ch_corr, name, "channel_corr")
        img_results.append(r)

        results[name] = img_results

    # ── 전체 파이프라인 (최적 조합) ──
    print("\n" + "=" * 60)
    print("Full Pipeline: Median(3) → Wiener → FFT Notch → Channel → Gamma → HistEq(YUV)")
    print("=" * 60)

    for name, original in originals.items():
        degraded = degrade_image(original)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_degraded.png"), degraded)

        step = median_filter(degraded, 3)
        step = wiener_filter(step, noise_var=400)
        step = fft_notch_filter(step, threshold_percentile=99.5)
        step = channel_correction(step)
        step = gamma_correction(step, gamma=1.8)
        step = histogram_equalization_yuv(step)

        r = evaluate(original, step, f"Full Pipeline")
        save_comparison(original, degraded, step, name, "full_pipeline")
        save_histogram_comparison(degraded, step, name, "full_pipeline")

    # ── 결과 요약 ──
    print("\n" + "=" * 60)
    print("📊 결과 요약표")
    print("=" * 60)
    for name, img_results in results.items():
        print(f"\n[{name}]")
        for r in img_results:
            print(f"  {r['label']:30s}  PSNR={r['psnr']:6.2f}  SSIM={r['ssim']:.4f}")

    print(f"\n✅ 모든 결과 이미지 → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
