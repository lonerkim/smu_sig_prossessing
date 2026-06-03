#!/usr/bin/env python3
"""
Iteration 4: Parameter tuning + cascade experiments + temporal video testing.
Saves all results to /mnt/nfs-hermes/artifacts/
"""
from __future__ import annotations

import csv
import os
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.degradation import degrade_image
from smu_sig_prossessing.evaluation import calculate_psnr, calculate_ssim
from smu_sig_prossessing.filters import FILTER_REGISTRY, reset_temporal_state

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
ARTIFACTS = "/mnt/nfs-hermes/artifacts"
os.makedirs(ARTIFACTS, exist_ok=True)

# Load test image
TEST_IMG = os.path.join(PROJ_ROOT, "input", "test_small.jpg")
if not os.path.exists(TEST_IMG):
    img = cv2.imread(os.path.join(PROJ_ROOT, "input", "REAL_WORLD_PICTURE.jpg"))
    img = cv2.resize(img, (1600, 740))
    cv2.imwrite(TEST_IMG, img)
original = cv2.imread(TEST_IMG)

# Degrade
degraded_basic = degrade_image(original, use_ntsc=False, strength=0.5)

# ─── Helper ────────────────────────────────────────────────────────

def edge_retention(orig, proc):
    def _edge_mag(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx**2 + gy**2)
    return float(np.sum(_edge_mag(proc)) / max(np.sum(_edge_mag(orig)), 1e-10))

def run_cfg(degraded, cfg, label, original=None):
    t0 = time.time()
    restored = pl.apply_pipeline(degraded, cfg)
    elapsed = time.time() - t0
    r = {"label": label, "time_s": round(elapsed, 4), "filters": " → ".join(s.name for s in cfg.stages if s.enabled)}
    if original is not None:
        o, r_img = original.copy(), restored.copy()
        if o.shape != r_img.shape:
            r_img = cv2.resize(r_img, (o.shape[1], o.shape[0]))
        r["psnr"] = round(calculate_psnr(o, r_img), 2)
        r["ssim"] = round(calculate_ssim(o, r_img), 4)
        r["edge"] = round(edge_retention(o, r_img), 3)
    return r, restored

def save_grid(images, labels, path, orig=None, degraded_img=None):
    """Build a grid comparison image."""
    n = len(images)
    if orig is not None:
        n += 2  # both orig and degraded_img get a slot
    h, w = images[0].shape[:2]
    scale = min(1.0, 1200 / (w * 2))
    if scale < 1.0:
        dh, dw = int(h * scale), int(w * scale)
    else:
        dh, dw = h, w

    cell_w = dw
    cell_h = dh + 35
    cols = 2
    rows = (n + 1) // 2
    canvas = np.zeros((rows * cell_h + 10, cols * cell_w + 10, 3), dtype=np.uint8)

    def _place(img, idx, label=""):
        r = idx // 2
        c = idx % 2
        y0 = r * cell_h + 5
        x0 = c * cell_w + 5
        resized = cv2.resize(img, (dw, dh))
        canvas[y0:y0+dh, x0:x0+dw] = resized
        if label:
            cv2.putText(canvas, label, (x0+5, y0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        return y0 + dh

    idx = 0
    if orig is not None:
        _place(orig, idx, "Original")
        idx += 1
        _place(degraded_img, idx, "Degraded")
        idx += 1

    for img, label in zip(images, labels):
        _place(img, idx, str(label)[:40])
        idx += 1

    cv2.imwrite(path, canvas)
    return path

def save_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        if results:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

def save_report(results, path, title=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write("| Label | PSNR | SSIM | Edge | Time (s) |\n")
        f.write("|-------|------|------|------|----------|\n")
        for r in results:
            psnr = f"{r['psnr']:.2f}" if "psnr" in r else "—"
            ssim = f"{r['ssim']:.4f}" if "ssim" in r else "—"
            edge = f"{r['edge']:.2f}" if "edge" in r else "—"
            f.write(f"| {r['label']} | {psnr} | {ssim} | {edge} | {r['time_s']:.4f} |\n")


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Bilateral parameter sweep (in optimized-fast context)
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("EXP 1: Bilateral param sweep")
print("=" * 60)
results_e1 = []
for sigma in [30, 50, 75, 100, 125, 150]:
    d = max(3, int(sigma / 15) * 2 + 1)  # d scales with sigma
    cfg = PipelineConfig(label=f"Bilateral σ={sigma}, d={d}")
    cfg.add("median", ksize=3)
    cfg.add("bilateral", d=d, sigma_color=sigma, sigma_space=sigma)
    cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
    r, _ = run_cfg(degraded_basic, cfg, f"bilateral.sigma={sigma}", original)
    results_e1.append(r)
    print(f"  σ={sigma:3d}  PSNR={r.get('psnr','—'):>6}  SSIM={r.get('ssim','—'):>.4f}  {r['time_s']:.4f}s")

save_csv(results_e1, f"{ARTIFACTS}/exp1_bilateral_sweep.csv")
save_report(results_e1, f"{ARTIFACTS}/exp1_bilateral_sweep.md", "Bilateral sigma_color sweep")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Wavelet level + threshold mode sweep
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 2: Wavelet level + threshold mode sweep")
print("=" * 60)
results_e2 = []
for level in [1, 2, 3, 4]:
    for mode in ["soft", "hard"]:
        cfg = PipelineConfig(label=f"Wavelet L={level} {mode}")
        cfg.add("wavelet", wavelet="db4", level=level, threshold_mode=mode)
        cfg.add("bilateral", d=5, sigma_color=20, sigma_space=20)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        r, _ = run_cfg(degraded_basic, cfg, f"wavelet.L{level}.{mode}", original)
        results_e2.append(r)
        print(f"  L={level} {mode:5s}  PSNR={r.get('psnr','—'):>6}  SSIM={r.get('ssim','—'):>.4f}  {r['time_s']:.4f}s")

save_csv(results_e2, f"{ARTIFACTS}/exp2_wavelet_sweep.csv")
save_report(results_e2, f"{ARTIFACTS}/exp2_wavelet_sweep.md", "Wavelet level + threshold mode sweep")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Guided filter radius + eps sweep
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 3: Guided filter radius + eps sweep")
print("=" * 60)
results_e3 = []
for radius in [2, 3, 5, 8]:
    for eps in [50, 100, 200, 500]:
        cfg = PipelineConfig(label=f"Guided r={radius} ε={eps}")
        cfg.add("guided_filter", radius=radius, eps=float(eps))
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)
        r, _ = run_cfg(degraded_basic, cfg, f"guided.r{radius}.e{eps}", original)
        results_e3.append(r)
        print(f"  r={radius} ε={eps:4d}  PSNR={r.get('psnr','—'):>6}  SSIM={r.get('ssim','—'):>.4f}  {r['time_s']:.4f}s")

save_csv(results_e3, f"{ARTIFACTS}/exp3_guided_sweep.csv")
save_report(results_e3, f"{ARTIFACTS}/exp3_guided_sweep.md", "Guided filter radius + eps sweep")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 4: TV denoise weight sweep
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 4: TV denoise weight sweep")
print("=" * 60)
results_e4 = []
for weight in [0.02, 0.05, 0.08, 0.1, 0.2, 0.5]:
    cfg = PipelineConfig(label=f"TV w={weight}")
    cfg.add("tv_denoise", weight=weight, max_num_iter=80)
    cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
    cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)
    r, _ = run_cfg(degraded_basic, cfg, f"tv.w{weight}", original)
    results_e4.append(r)
    print(f"  w={weight:.2f}  PSNR={r.get('psnr','—'):>6}  SSIM={r.get('ssim','—'):>.4f}  {r['time_s']:.4f}s")

save_csv(results_e4, f"{ARTIFACTS}/exp4_tv_sweep.csv")
save_report(results_e4, f"{ARTIFACTS}/exp4_tv_sweep.md", "TV denoise weight sweep")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Cascade comparisons — filter order matters
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 5: Cascade / order experiments")
print("=" * 60)
results_e5 = []
images_e5 = []

cascades = OrderedDict({
    "wavelet→bilateral": lambda c: (c.add("wavelet", wavelet="db4", level=3, threshold_mode="soft"),
                                     c.add("bilateral", d=5, sigma_color=20, sigma_space=20),
                                     c.add("channel_correction", clamp_min=0.85, clamp_max=1.15),
                                     c.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)),
    "bilateral→wavelet": lambda c: (c.add("bilateral", d=9, sigma_color=75, sigma_space=75),
                                     c.add("wavelet", wavelet="db4", level=2, threshold_mode="soft"),
                                     c.add("channel_correction", clamp_min=0.85, clamp_max=1.15),
                                     c.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)),
    "wavelet→guided": lambda c: (c.add("wavelet", wavelet="db4", level=2, threshold_mode="soft"),
                                  c.add("guided_filter", radius=3, eps=100.0),
                                  c.add("channel_correction", clamp_min=0.85, clamp_max=1.15)),
    "guided→wavelet": lambda c: (c.add("guided_filter", radius=3, eps=100.0),
                                  c.add("wavelet", wavelet="db4", level=2, threshold_mode="soft"),
                                  c.add("channel_correction", clamp_min=0.85, clamp_max=1.15)),
    "tv→bilateral": lambda c: (c.add("tv_denoise", weight=0.08, max_num_iter=80),
                                c.add("bilateral", d=5, sigma_color=30, sigma_space=30),
                                c.add("channel_correction", clamp_min=0.85, clamp_max=1.15)),
    "bilateral→tv": lambda c: (c.add("bilateral", d=7, sigma_color=50, sigma_space=50),
                                c.add("tv_denoise", weight=0.08, max_num_iter=80),
                                c.add("channel_correction", clamp_min=0.85, clamp_max=1.15)),
    "median→bilateral→wavelet→channel": lambda c: (
        c.add("median", ksize=3), c.add("bilateral", d=9, sigma_color=75, sigma_space=75),
        c.add("wavelet", wavelet="db4", level=2, threshold_mode="soft"),
        c.add("channel_correction", clamp_min=0.85, clamp_max=1.25)),
    "median→wavelet→bilateral→channel": lambda c: (
        c.add("median", ksize=3), c.add("wavelet", wavelet="db4", level=2, threshold_mode="soft"),
        c.add("bilateral", d=9, sigma_color=75, sigma_space=75),
        c.add("channel_correction", clamp_min=0.85, clamp_max=1.25)),
})

for label, builder in cascades.items():
    cfg = PipelineConfig(label=label)
    builder(cfg)
    r, restored = run_cfg(degraded_basic, cfg, label, original)
    results_e5.append(r)
    images_e5.append(restored)
    print(f"  {label:40s}  PSNR={r.get('psnr','—'):>6}  SSIM={r.get('ssim','—'):>.4f}  {r['time_s']:.4f}s")

save_csv(results_e5, f"{ARTIFACTS}/exp5_cascade.csv")
save_report(results_e5, f"{ARTIFACTS}/exp5_cascade.md", "Cascade / filter order experiments")
save_grid(images_e5, [r['label'] for r in results_e5], f"{ARTIFACTS}/exp5_cascade_grid.png",
          orig=original, degraded_img=degraded_basic)

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 6: Best-of-each-family comparison (grid)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 6: Best-of-each-family + full preset re-evaluation")
print("=" * 60)
results_e6 = []
images_e6 = []

def make_cfg(label, stages):
    cfg = PipelineConfig(label=label)
    for fn_name, kwargs in stages:
        cfg.add(fn_name, **kwargs)
    return cfg

preset_configs = [
    ("optimized-fast (best)", PipelineConfig.optimized_fast()),
    ("optimized-quality", PipelineConfig.optimized_quality()),
    ("wavelet L4 soft", make_cfg("WavL4S", [
        ("wavelet", {"wavelet": "db4", "level": 4, "threshold_mode": "soft"}),
        ("bilateral", {"d": 5, "sigma_color": 20, "sigma_space": 20}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.15}),
        ("unsharp_mask", {"strength": 0.2, "radius": 0.5, "threshold": 5}),
    ])),
    ("bilateral σ=150", make_cfg("Bil150", [
        ("median", {"ksize": 3}),
        ("bilateral", {"d": 19, "sigma_color": 150, "sigma_space": 150}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ])),
    ("guided r3 e100", make_cfg("GuidedR3", [
        ("guided_filter", {"radius": 3, "eps": 100.0}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.15}),
        ("unsharp_mask", {"strength": 0.2, "radius": 0.5, "threshold": 5}),
    ])),
    ("TV w=0.20", make_cfg("TVw20", [
        ("tv_denoise", {"weight": 0.2, "max_num_iter": 80}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.15}),
        ("unsharp_mask", {"strength": 0.2, "radius": 0.5, "threshold": 5}),
    ])),
    ("wavelet→guided", make_cfg("W→G", [
        ("wavelet", {"wavelet": "db4", "level": 2, "threshold_mode": "soft"}),
        ("guided_filter", {"radius": 3, "eps": 100.0}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.15}),
    ])),
    ("median→bilateral→wavelet→channel", make_cfg("M→B→W→C", [
        ("median", {"ksize": 3}),
        ("bilateral", {"d": 9, "sigma_color": 75, "sigma_space": 75}),
        ("wavelet", {"wavelet": "db4", "level": 2, "threshold_mode": "soft"}),
        ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.25}),
    ])),
    ("edge-preserve (baseline)", PipelineConfig.edge_preserve()),
]

for name, cfg in preset_configs:
    r, restored = run_cfg(degraded_basic, cfg, name, original)
    results_e6.append(r)
    images_e6.append(restored)
    print(f"  {name:40s}  PSNR={r.get('psnr','—'):>6}  SSIM={r.get('ssim','—'):>.4f}  {r['time_s']:.4f}s")

save_csv(results_e6, f"{ARTIFACTS}/exp6_best_of_family.csv")
save_report(results_e6, f"{ARTIFACTS}/exp6_best_of_family.md", "Best-of-family comparison")
save_grid(images_e6, [r['label'] for r in results_e6], f"{ARTIFACTS}/exp6_best_family_grid.png",
          orig=original, degraded_img=degraded_basic)

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 7: Temporal video test — 30 frames with st-video
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 7: Temporal video denoising (30 frames)")
print("=" * 60)

video_path = os.path.join(PROJ_ROOT, "input", "analog_whoop_footage.mp4")
cap = cv2.VideoCapture(video_path)
if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = min(30, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # Test single-frame (spatial-only) vs temporal-aware denoising
    temporal_presets = OrderedDict([
        ("spatial-only (edge-preserve)", PipelineConfig.edge_preserve()),
        ("temporal-averaging (5 frames)", make_cfg("Tavg5", [
            ("temporal_average", {"n_frames": 5}),
            ("bilateral", {"d": 5, "sigma_color": 30, "sigma_space": 30}),
            ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.15}),
        ])),
        ("motion-compensated", make_cfg("Tmc", [
            ("temporal_motion", {"strength": 0.3}),
            ("bilateral", {"d": 5, "sigma_color": 30, "sigma_space": 30}),
            ("channel_correction", {"clamp_min": 0.85, "clamp_max": 1.15}),
        ])),
        ("spatio-temporal (st-video)", PipelineConfig.spatial_temporal_video()),
    ])

    # Process 30 frames with each preset, measure frame-by-frame
    for name, cfg in temporal_presets.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        reset_temporal_state()
        
        cfg = temporal_presets[name]
        frames = []
        times = []
        for f_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.time()
            restored = pl.apply_pipeline(frame, cfg)
            elapsed = time.time() - t0
            times.append(elapsed)
            if f_idx < 5:  # save first 5 frames for comparison grid
                frames.append(restored)
        
        avg_time = np.mean(times)
        print(f"  {name:35s}  avg {avg_time:.4f}s/frame  ({len(times)} frames)")
        
        # Save sample frames comparison
        if len(frames) >= 5:
            labels = [f"Frame {i}" for i in range(5)]
            safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            save_grid(frames[:5], labels, f"{ARTIFACTS}/exp7_{safe}_frames.png")

    cap.release()
else:
    print("  Cannot open video")

print("\n" + "=" * 60)
print(f"All results saved to {ARTIFACTS}/")
print("=" * 60)
