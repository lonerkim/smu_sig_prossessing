#!/usr/bin/env python3
"""
BM4D parameter sweep — optimize sigma_psd, temporal_window, max_frames.
Tests on real analog_whoop_footage.mp4 with BRISQUE + NIQE + full metrics.
"""
import sys, os, time, json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.config import PipelineConfig, FilterConfig
from smu_sig_prossessing.filters import reset_temporal_state
from smu_sig_prossessing.auto_evaluation import AutoEvaluator, MetricResult

INPUT = "input/analog_whoop_footage.mp4"
N_FRAMES = 8  # enough for max temporal window
OUT_DIR = "output/bm4d_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

# Load frames
cap = cv2.VideoCapture(INPUT)
frames = []
for _ in range(N_FRAMES):
    ret, f = cap.read()
    if ret:
        frames.append(f)
cap.release()
print(f"Loaded {len(frames)} frames from {INPUT}")

ev = AutoEvaluator()

# Parameter grid
params_grid = [
    # (sigma_psd, temporal_window, max_frames)
    (10, 2, 5),
    (10, 3, 5),
    (15, 2, 5),
    (15, 3, 5),
    (15, 3, 8),   # current default
    (15, 4, 8),
    (20, 2, 5),
    (20, 3, 5),
    (20, 3, 8),
    (20, 4, 8),
    (25, 2, 5),
    (25, 3, 5),
    (30, 2, 5),
    (30, 3, 8),
]

results = []

for sigma_psd, temporal_window, max_frames in params_grid:
    # Build config with just BM4D + minimal post
    cfg = PipelineConfig(label=f"BM4D σ={sigma_psd} tw={temporal_window} mf={max_frames}")
    cfg.add("bm4d_volume", sigma_psd=sigma_psd, temporal_window=temporal_window,
            max_frames=max_frames)
    cfg.add("chroma_denoise", strength=0.2)
    cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
    cfg.add("unsharp_mask", strength=0.15, radius=0.5, threshold=5)

    # Process
    reset_temporal_state()
    times = []
    for i, f in enumerate(frames):
        t0 = time.time()
        p = pl.apply_pipeline(f, cfg)
        dt = (time.time() - t0) * 1000
        times.append(dt)

    avg_time = np.mean(times)
    # Evaluate last frame (after temporal buffer filled)
    r = ev.evaluate(frames[-1], p, label=cfg.label, verbose=False)
    m = {m.name: m.value for m in r.metrics}

    result = {
        "sigma_psd": sigma_psd,
        "temporal_window": temporal_window,
        "max_frames": max_frames,
        "composite_score": r.composite_score,
        "psnr": round(m.get("psnr", 0), 2),
        "ssim": round(m.get("ssim", 0), 4),
        "color_fidelity": round(m.get("color_fidelity", 0), 2),
        "edge_retention": round(m.get("edge_retention", 0), 3),
        "noise_level": round(m.get("noise_level", 0), 1),
        "detail_recovery": round(m.get("detail_recovery", 0), 3),
        "artifact_score": round(m.get("artifact_score", 0), 2),
        "vif": round(m.get("vif", 0), 4),
        "niqe": round(m.get("niqe", 0), 2),
        "brisque": round(m.get("brisque", 0), 2),
        "avg_time_ms": round(avg_time, 1),
    }
    results.append(result)

    print(f"σ={sigma_psd:2d} tw={temporal_window} mf={max_frames}  "
          f"Score={r.composite_score:5.2f}  NIQE={result['niqe']:5.2f}  "
          f"BRISQUE={result['brisque']:5.2f}  PSNR={result['psnr']:5.2f}  "
          f"Time={avg_time:6.1f}ms")

# Sort by composite score descending
results.sort(key=lambda r: r["composite_score"], reverse=True)

# Save results
path = os.path.join(OUT_DIR, "bm4d_sweep.json")
with open(path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {path}")

# Print ranking
print(f"\n{'='*100}")
print(f"{'Rank':>4s} {'σ':>4s} {'tw':>3s} {'mf':>3s}  {'Score':>7s} {'PSNR':>7s} "
      f"{'SSIM':>7s} {'NIQE':>7s} {'BRISQUE':>7s} {'ΔE':>6s} {'Time':>7s}")
print(f"{'='*100}")
for i, r in enumerate(results):
    print(f"{i+1:3d}.  {r['sigma_psd']:3d}  {r['temporal_window']:2d}  {r['max_frames']:2d}  "
          f"{r['composite_score']:6.2f}  {r['psnr']:6.2f}  {r['ssim']:6.4f}  "
          f"{r['niqe']:6.2f}  {r['brisque']:6.2f}  {r['color_fidelity']:5.2f}  "
          f"{r['avg_time_ms']:6.1f}ms")

# Also save as CSV
csv_path = os.path.join(OUT_DIR, "bm4d_sweep.csv")
with open(csv_path, "w") as f:
    keys = results[0].keys()
    f.write(",".join(keys) + "\n")
    for r in results:
        f.write(",".join(str(r[k]) for k in keys) + "\n")
print(f"CSV saved to {csv_path}")
