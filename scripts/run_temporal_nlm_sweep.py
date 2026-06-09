#!/usr/bin/env python3
"""
Temporal NLM Multi parameter sweep — optimize h, h_color, temporal_window.
Tests on real analog_whoop_footage.mp4 with full metrics including BRISQUE.
"""
import sys, os, time, json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.config import PipelineConfig, FilterConfig
from smu_sig_prossessing.filters import reset_temporal_state
from smu_sig_prossessing.auto_evaluation import AutoEvaluator

INPUT = "input/analog_whoop_footage.mp4"
N_FRAMES = 8  # enough to fill temporal buffer
OUT_DIR = "output/temporal_nlm_sweep"
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

# Parameter grid: (h, h_color, temporal_window, max_frames)
params_grid = [
    # Gentle denoising
    (5, 5, 2, 5),
    (5, 5, 3, 7),
    # Medium (current default-ish)
    (8, 8, 2, 5),
    (8, 8, 3, 5),
    (8, 8, 3, 7),
    # Stronger
    (10, 10, 2, 5),
    (10, 10, 3, 5),
    (10, 8, 3, 5),  # stronger luma than chroma
    (12, 8, 2, 5),
    (12, 8, 3, 5),
    # Aggressive
    (15, 10, 2, 5),
    (15, 10, 3, 5),
    (15, 15, 2, 5),
    # Different ratios
    (8, 12, 3, 5),  # stronger chroma
    (10, 15, 3, 5),
]

results = []

for h, h_color, temporal_window, max_frames in params_grid:
    # Build config with temporal_nlm_multi + minimal post-processing
    cfg = PipelineConfig(label=f"TNLM h={h} hc={h_color} tw={temporal_window} mf={max_frames}")
    cfg.add("temporal_nlm_multi", h=h, h_color=h_color,
            temporal_window=temporal_window, max_frames=max_frames)
    cfg.add("guided_filter", radius=3, eps=50.0)
    cfg.add("chroma_denoise", strength=0.2)
    cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
    cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
    cfg.add("unsharp_mask", strength=0.1, radius=0.5, threshold=10)

    # Process all frames
    reset_temporal_state()
    times = []
    last_out = None
    for i, f in enumerate(frames):
        t0 = time.time()
        p = pl.apply_pipeline(f, cfg)
        dt = (time.time() - t0) * 1000
        times.append(dt)
        last_out = p

    avg_time = np.mean(times)
    # Evaluate on last frame
    r = ev.evaluate(frames[-1], last_out, label=cfg.label, verbose=False)
    m = {m.name: m.value for m in r.metrics}

    result = {
        "h": h,
        "h_color": h_color,
        "temporal_window": temporal_window,
        "max_frames": max_frames,
        "composite_score": r.composite_score,
        "psnr": round(m.get("psnr", 0), 2),
        "ssim": round(m.get("ssim", 0), 4),
        "color_fidelity": round(m.get("color_fidelity", 0), 2),
        "edge_retention": round(m.get("edge_retention", 0), 3),
        "detail_recovery": round(m.get("detail_recovery", 0), 3),
        "artifact_score": round(m.get("artifact_score", 0), 2),
        "vif": round(m.get("vif", 0), 4),
        "niqe": round(m.get("niqe", 0), 2),
        "brisque": round(m.get("brisque", 0), 2),
        "avg_time_ms": round(avg_time, 1),
    }
    results.append(result)

    print(f"h={h:2d} hc={h_color:2d} tw={temporal_window} mf={max_frames}  "
          f"Score={r.composite_score:5.2f}  NIQE={result['niqe']:5.2f}  "
          f"BRISQUE={result['brisque']:5.2f}  PSNR={result['psnr']:5.2f}  "
          f"Time={avg_time:5.1f}ms")

# Sort by BRISQUE (lower is better for no-reference quality)
results_by_brisque = sorted(results, key=lambda r: r["brisque"])
# Sort by composite
results_by_composite = sorted(results, key=lambda r: r["composite_score"], reverse=True)

# Save results
path = os.path.join(OUT_DIR, "temporal_nlm_sweep.json")
with open(path, "w") as f:
    json.dump({"by_brisque": results_by_brisque, "by_composite": results_by_composite}, f, indent=2)
print(f"\nResults saved to {path}")

# Print rankings
print(f"\n{'='*100}")
print(f"RANKED BY BRISQUE (lower = better perceptual quality)")
print(f"{'='*100}")
print(f"{'Rank':>4s} {'h':>3s} {'hc':>3s} {'tw':>3s} {'mf':>3s}  "
      f"{'Score':>7s} {'PSNR':>7s} {'NIQE':>7s} {'BRISQUE':>7s} {'Time':>7s}")
for i, r in enumerate(results_by_brisque[:10]):
    print(f"{i+1:3d}.  {r['h']:2d}  {r['h_color']:2d}  {r['temporal_window']:2d}  {r['max_frames']:2d}  "
          f"{r['composite_score']:6.2f}  {r['psnr']:6.2f}  "
          f"{r['niqe']:6.2f}  {r['brisque']:6.2f}  {r['avg_time_ms']:5.1f}ms")

print(f"\n{'='*100}")
print(f"RANKED BY COMPOSITE SCORE (higher = better overall)")
print(f"{'='*100}")
for i, r in enumerate(results_by_composite[:10]):
    print(f"{i+1:3d}.  {r['h']:2d}  {r['h_color']:2d}  {r['temporal_window']:2d}  {r['max_frames']:2d}  "
          f"{r['composite_score']:6.2f}  {r['psnr']:6.2f}  "
          f"{r['niqe']:6.2f}  {r['brisque']:6.2f}  {r['avg_time_ms']:5.1f}ms")

# Also save CSV
csv_path = os.path.join(OUT_DIR, "temporal_nlm_sweep.csv")
with open(csv_path, "w") as f:
    keys = list(results[0].keys())
    f.write(",".join(keys) + "\n")
    for r in results:
        f.write(",".join(str(r[k]) for k in keys) + "\n")
print(f"\nCSV: {csv_path}")
