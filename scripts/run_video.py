#!/usr/bin/env python3
"""
비디오 기반 파이프라인 (모듈식 v2.0)
— 합성 영상 생성 → 프레임별 열화 → 파이프라인 적용 → PSNR/SSIM 평가
"""
import os
import sys
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.degradation import degrade_image

BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

H, W, FPS, FRAMES = 480, 640, 30, 90  # 3초 @ 30fps


def create_synthetic_video():
    """Generate synthetic test video with moving objects."""
    frames = []
    for i in range(FRAMES):
        t = i / FRAMES
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
        cv2.putText(img, f"Frame {i:03d}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        frames.append(img)
    return frames


def write_video(path, frames):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()
    return path


def main():
    print("=" * 60)
    print("🎬 Modular Video Pipeline v2.0")
    print(f"   {W}x{H}, {FPS}fps, {FRAMES} frames ({FRAMES/FPS:.1f}s)")
    print("=" * 60)

    # 1) Clean video
    print("\n[1/4] Generating synthetic clean video...")
    clean_frames = create_synthetic_video()
    write_video(os.path.join(OUTPUT_DIR, "video_clean.mp4"), clean_frames)
    print(f"  saved: {OUTPUT_DIR}/video_clean.mp4")

    # 2) Degraded video
    print("\n[2/4] Applying degradation (NTSC + synthetic noise)...")
    degraded_frames = [degrade_image(f, use_ntsc=True, ntsc_intensity="medium")
                       for f in clean_frames]
    write_video(os.path.join(OUTPUT_DIR, "video_degraded.mp4"), degraded_frames)
    print(f"  saved: {OUTPUT_DIR}/video_degraded.mp4")

    # 3) Pipeline: Wiener-only (recommended)
    wiener_cfg = PipelineConfig.wiener_only()
    print(f"\n[3/4] Running: {wiener_cfg.label}...")
    restored_w = []
    p_list = []
    s_list = []
    for i, (clean, deg) in enumerate(zip(clean_frames, degraded_frames)):
        r = pl.apply_pipeline(deg, wiener_cfg)
        restored_w.append(r)
        p_list.append(psnr(clean, r))
        s_list.append(ssim(clean, r, channel_axis=-1))
        if (i + 1) % 30 == 0:
            print(f"  frame {i+1}/{FRAMES} — PSNR={np.mean(p_list[-30:]):.2f} SSIM={np.mean(s_list[-30:]):.4f}")

    write_video(os.path.join(OUTPUT_DIR, "video_wiener_pipeline.mp4"), restored_w)
    print(f"  saved: {OUTPUT_DIR}/video_wiener_pipeline.mp4")

    # 4) Pipeline: Edge-preserving
    ep_cfg = PipelineConfig.edge_preserving()
    print(f"\n[4/4] Running: {ep_cfg.label}...")
    restored_ep = []
    ep_p_list = []
    ep_s_list = []
    for i, (clean, deg) in enumerate(zip(clean_frames, degraded_frames)):
        r = pl.apply_pipeline(deg, ep_cfg)
        restored_ep.append(r)
        ep_p_list.append(psnr(clean, r))
        ep_s_list.append(ssim(clean, r, channel_axis=-1))
        if (i + 1) % 30 == 0:
            print(f"  frame {i+1}/{FRAMES} — PSNR={np.mean(ep_p_list[-30:]):.2f} SSIM={np.mean(ep_s_list[-30:]):.4f}")

    write_video(os.path.join(OUTPUT_DIR, "video_edge_preserving.mp4"), restored_ep)
    print(f"  saved: {OUTPUT_DIR}/video_edge_preserving.mp4")

    # ── Results Summary ──
    deg_psnr = [psnr(c, d) for c, d in zip(clean_frames, degraded_frames)]
    deg_ssim = [ssim(c, d, channel_axis=-1) for c, d in zip(clean_frames, degraded_frames)]

    print("\n" + "=" * 60)
    print("📊 Video Results Summary")
    print("=" * 60)
    print(f"\n  {'Method':35s}  {'Avg PSNR':>10s}  {'Avg SSIM':>10s}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}")
    for label, plist, slist in [
        ("Degraded (no processing)", deg_psnr, deg_ssim),
        (f"Wiener Pipeline", p_list, s_list),
        (f"Edge-Preserving", ep_p_list, ep_s_list),
    ]:
        print(f"  {label:35s}  {np.mean(plist):8.2f} dB  {np.mean(slist):10.4f}")

    # Comparison images
    for idx in [0, FRAMES // 2, FRAMES - 1]:
        h, w = H, W
        canvas = np.zeros((h, w * 5, 3), dtype=np.uint8)
        canvas[:, :w] = clean_frames[idx]
        canvas[:, w:2*w] = degraded_frames[idx]
        canvas[:, 2*w:3*w] = restored_w[idx]
        canvas[:, 3*w:4*w] = restored_ep[idx]
        labels = ["Clean", "Degraded", "Wiener", "Edge-Pres", "Research Best"]
        for j, lbl in enumerate(labels):
            cv2.putText(canvas, lbl, (j * w + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"video_frame{idx:03d}_comparison.png"), canvas)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
