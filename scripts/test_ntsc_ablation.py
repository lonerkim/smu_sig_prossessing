#!/usr/bin/env python3
"""Test presets on real analog video frame with NTSC-heavy degradation."""
import cv2, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.degradation import degrade_image
from smu_sig_prossessing.evaluation import calculate_psnr, calculate_ssim

cap = cv2.VideoCapture('input/analog_whoop_footage.mp4')
ret, frame = cap.read()
cap.release()
if not ret:
    print("FAIL")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"Frame: {w}x{h}")

# Test NTSC-heavy degrade
print("\n=== NTSC-heavy degrade ===")
degraded = degrade_image(frame, use_ntsc=True, ntsc_intensity='heavy')
deg_psnr = calculate_psnr(frame, degraded)
deg_ssim = calculate_ssim(frame, degraded)
print(f"  Degraded  PSNR={deg_psnr:.2f}  SSIM={deg_ssim:.4f}")

# Test presets
presets = [
    ("edge-preserve", PipelineConfig.edge_preserve()),
    ("optimized-fast", PipelineConfig.optimized_fast()),
    ("optimized-quality", PipelineConfig.optimized_quality()),
    ("guided-denoise", PipelineConfig.guided_denoise()),
    ("wavelet-denoise", PipelineConfig.wavelet_denoise()),
    ("tv-denoise", PipelineConfig.tv_denoise_preset()),
]

print("\n=== Restoration ===")
for name, cfg in presets:
    t0 = time.time()
    restored = pl.apply_pipeline(degraded, cfg)
    elapsed = time.time() - t0
    p = calculate_psnr(frame, restored)
    s = calculate_ssim(frame, restored)
    flt = " → ".join(s.name for s in cfg.stages if s.enabled)
    print(f"  {name:25s}  PSNR={p:6.2f}  SSIM={s:.4f}  {elapsed:.3f}s  [{flt[:60]}...]")

# Also test basic degrade for comparison
print("\n=== Basic degrade strength=0.5 ===")
degraded2 = degrade_image(frame, use_ntsc=False, strength=0.5)
print(f"  Degraded  PSNR={calculate_psnr(frame, degraded2):.2f}  SSIM={calculate_ssim(frame, degraded2):.4f}")
for name, cfg in presets[:4]:
    t0 = time.time()
    restored = pl.apply_pipeline(degraded2, cfg)
    elapsed = time.time() - t0
    p = calculate_psnr(frame, restored)
    s = calculate_ssim(frame, restored)
    print(f"  {name:25s}  PSNR={p:6.2f}  SSIM={s:.4f}  {elapsed:.3f}s")
