#!/usr/bin/env python3
"""Test washing effect and compare preset improvements."""
import cv2, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl

def make_cfg(label, stages):
    cfg = PipelineConfig(label=label)
    for fn, kwargs in stages:
        cfg.add(fn, **kwargs)
    return cfg

presets = {
    "max-quality (original)": PipelineConfig.max_quality(),
    "max-quality+CLAHE": make_cfg("M+CLAHE", [
        ("median", {"ksize": 3}),
        ("bilateral", {"d": 19, "sigma_color": 150, "sigma_space": 150}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("histogram_eq_clahe", {"clip_limit": 1.5, "tile_size": 8}),
    ]),
    "guided+CLAHE": make_cfg("G+CLAHE", [
        ("median", {"ksize": 3}),
        ("guided_filter", {"radius": 3, "eps": 100.0}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("histogram_eq_clahe", {"clip_limit": 1.5, "tile_size": 8}),
        ("unsharp_mask", {"strength": 0.2, "radius": 0.5, "threshold": 5}),
    ]),
    "wavelet+CLAHE": make_cfg("W+CLAHE", [
        ("wavelet", {"wavelet": "db4", "level": 2, "threshold_mode": "soft"}),
        ("bilateral", {"d": 5, "sigma_color": 20, "sigma_space": 20}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.15}),
        ("histogram_eq_clahe", {"clip_limit": 1.5, "tile_size": 8}),
    ]),
    "guided+wavelet+CLAHE": make_cfg("G+W+CLAHE", [
        ("median", {"ksize": 3}),
        ("guided_filter", {"radius": 3, "eps": 100.0}),
        ("wavelet", {"wavelet": "db4", "level": 2, "threshold_mode": "soft"}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
        ("histogram_eq_clahe", {"clip_limit": 1.5, "tile_size": 8}),
        ("unsharp_mask", {"strength": 0.2, "radius": 0.5, "threshold": 5}),
    ]),
    "optimized-fast": PipelineConfig.optimized_fast(),
    "edge-preserve": PipelineConfig.edge_preserve(),
}

cap = cv2.VideoCapture("input/analog_whoop_footage.mp4")
frames = []
for i in range(200):
    ret, f = cap.read()
    if not ret: break
    if i in [30, 60, 90, 120]:
        frames.append((i, f))
cap.release()
print(f"Testing {len(frames)} frames\n")

for fidx, frame in frames:
    h, w = frame.shape[:2]
    print(f"Frame {fidx} ({w}x{h}):")
    print(f"  {'Preset':30s} {'Time':>8s} {'Wash':>6s} {'Edge':>6s}")
    print(f"  {'-'*30} {'-'*8} {'-'*6} {'-'*6}")

    for name, cfg in presets.items():
        t0 = time.time()
        restored = pl.apply_pipeline(frame, cfg)
        elapsed = time.time() - t0

        # Washing metric: ratio of highlight preservation (higher = less washing)
        orig_top = frame.astype(np.float32)
        rest_top = restored.astype(np.float32)
        # Mean of top 5% brightest pixels (luminance)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_r = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY).astype(np.float32)
        thresh = np.percentile(gray, 95)
        mask = gray >= thresh
        if mask.sum() > 100:
            orig_hl = gray[mask].mean()
            rest_hl = gray_r[mask].mean()
            wash = rest_hl / orig_hl if orig_hl > 0 else 1.0
        else:
            wash = 1.0

        # Edge retention
        def edge_mag(img):
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            return np.sqrt(gx**2 + gy**2)
        e_orig = edge_mag(frame)
        e_rest = edge_mag(restored)
        edge = float(np.sum(e_rest) / max(np.sum(e_orig), 1e-10))

        flt = " → ".join(s.name for s in cfg.stages if s.enabled)
        print(f"  {name:30s} {elapsed:7.3f}s {wash:5.3f}  {edge:5.3f}  [{flt[:55]}]")
    print()

# Save comparison grid for frame 30
print("\nSaving comparison grid...")
if frames:
    _, f0 = frames[0]
    images = []
    labels = []
    for name, cfg in presets.items():
        restored = pl.apply_pipeline(f0, cfg)
        images.append(restored)
        labels.append(name)

    n = len(images)
    h, w = f0.shape[:2]
    scale = min(1.0, 1200 / (w * 2))
    dh, dw = (int(h*scale), int(w*scale)) if scale < 1.0 else (h, w)
    cell_w, cell_h = dw, dh + 35
    cols = 3
    rows = (n + cols - 1) // cols
    canvas = np.zeros((rows * cell_h + 10, cols * cell_w + 10, 3), dtype=np.uint8)

    for idx, (img, label) in enumerate(zip(images, labels)):
        r, c = divmod(idx, cols)
        y0 = r * cell_h + 5
        x0 = c * cell_w + 5
        resized = cv2.resize(img, (dw, dh))
        canvas[y0:y0+dh, x0:x0+dw] = resized
        cv2.putText(canvas, label[:30], (x0+5, y0+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    out_path = "output/video_quality_comparison.png"
    cv2.imwrite(out_path, canvas)
    print(f"Saved → {out_path}")
