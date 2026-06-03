#!/usr/bin/env python3
"""Compare all presets side-by-side on one analog video frame."""
import cv2, os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
video_path = os.path.join(BASE, "input", "analog_whoop_footage.mp4")
out_dir = os.path.join(BASE, "output")

PRESETS = {
    "edge-preserve": PipelineConfig.edge_preserve(),
    "guided-denoise": PipelineConfig.guided_denoise(),
    "wavelet-denoise": PipelineConfig.wavelet_denoise(),
    "tv-denoise": PipelineConfig.tv_denoise_preset(),
    "fast-denoise": PipelineConfig.fast_denoise(),
    "wiener-denoise": PipelineConfig.wiener_denoise(),
}

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()
if not ret:
    print("FAIL: cannot read video")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"Frame: {w}x{h}")

# Build comparison canvas
n = len(PRESETS)
cell_w = w
cell_h = h + 40
canvas = np.zeros((cell_h * 2, cell_w * 4, 3), dtype=np.uint8)  # 2 rows × 4 cols

# First 4 in row 0, rest in row 1
names = list(PRESETS.keys())
for idx, name in enumerate(names):
    row = idx // 4
    col = idx % 4
    cfg = PRESETS[name]
    t0 = time.time()
    restored = pl.apply_pipeline(frame, cfg)
    elapsed = time.time() - t0
    y0 = row * cell_h
    x0 = col * cell_w
    canvas[y0:y0+h, x0:x0+w] = cv2.resize(restored, (w, h))
    label = f"{name}  {elapsed:.2f}s"
    cv2.putText(canvas, label, (x0 + 5, y0 + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    print(f"  {name:20s}  {elapsed:.3f}s")

out_path = os.path.join(out_dir, "analog_preset_comparison.png")
cv2.imwrite(out_path, canvas)
print(f"\nSaved → {out_path}  ({canvas.shape[1]}x{canvas.shape[0]})")
