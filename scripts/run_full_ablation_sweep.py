#!/usr/bin/env python3
"""
Full Ablation Sweep — Run ALL presets on analog_whoop_footage.mp4,
collect PSNR, SSIM, composite scores, timing, and save to JSON table.

Usage:
    .venv/bin/python3 run_full_ablation_sweep.py
    .venv/bin/python3 run_full_ablation_sweep.py --skip-bm3d  # skip BM3D-based presets
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import PRESETS
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.auto_evaluation import AutoEvaluator
from smu_sig_prossessing.filters import reset_temporal_state

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO = os.path.join(BASE, "input", "analog_whoop_footage.mp4")
OUTPUT_DIR = os.path.join(BASE, "output", "ablation_sweep")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BM3D_PRESETS = {"bm3d-denoise", "bm3d-fast", "retinex-bm3d", "retinex-bm3d-msrcr"}
TIMEOUT_SEC = 120  # per preset

evaluator = AutoEvaluator()


def run_single_preset(
    preset_name: str,
    frames: list[np.ndarray],
    skip_bm3d: bool,
) -> dict:
    """Run a single preset on the frames and return metrics."""
    if skip_bm3d and preset_name in BM3D_PRESETS:
        return {
            "preset": preset_name,
            "status": "skipped",
            "reason": "BM3D (skipped via --skip-bm3d)",
        }

    cfg = PRESETS.get(preset_name)
    if cfg is None:
        return {"preset": preset_name, "status": "error", "reason": "unknown preset"}

    try:
        reset_temporal_state()

        # Measure timing on a single frame
        t0 = time.perf_counter()
        restored = pl.apply_pipeline(frames[0].copy(), cfg)
        elapsed = time.perf_counter() - t0

        # Evaluate against the original (degraded=none, so original=noisy input)
        result = evaluator.evaluate(
            frames[0], restored,
            label=preset_name,
            degraded=frames[0],
            verbose=False,
        )

        # Collect metrics
        m = {mt.name: mt.value for mt in result.metrics}

        entry = {
            "preset": preset_name,
            "status": "ok",
            "composite_score": round(result.composite_score, 2),
            "psnr": round(m.get("psnr", 0), 2),
            "ssim": round(m.get("ssim", 0), 4),
            "color_fidelity_dE": round(m.get("color_fidelity", 0), 2),
            "edge_retention": round(m.get("edge_retention", 0), 3),
            "noise_level": round(m.get("noise_level", 0), 1),
            "detail_recovery": round(m.get("detail_recovery", 0), 3),
            "artifact_score": round(m.get("artifact_score", 0), 2),
            "vif": round(m.get("vif", 0), 4),
            "niqe": round(m.get("niqe", 0), 2),
            "time_sec": round(elapsed, 4),
            "time_ms": round(elapsed * 1000, 1),
            "filters": " → ".join(s.name for s in cfg.stages if s.enabled),
        }
        print(
            f"  {preset_name:25s}  "
            f"Score={entry['composite_score']:6.2f}  "
            f"PSNR={entry['psnr']:5.2f}  "
            f"SSIM={entry['ssim']:.4f}  "
            f"ΔE={entry['color_fidelity_dE']:.2f}  "
            f"VIF={entry['vif']:.4f}  "
            f"NIQE={entry['niqe']:.2f}  "
            f"{entry['time_ms']:7.1f}ms"
        )
        return entry

    except Exception as e:
        return {
            "preset": preset_name,
            "status": "error",
            "reason": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Full ablation sweep — all 38 presets on analog_whoop_footage.mp4"
    )
    parser.add_argument("--skip-bm3d", action="store_true",
                        help="Skip BM3D-based presets (slow)")
    parser.add_argument("--presets", type=str, default=None,
                        help="Comma-separated preset list (default: all)")
    args = parser.parse_args()

    # Load input video — read first 5 frames
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {INPUT_VIDEO}")
        sys.exit(1)

    frames = []
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"📹 Loaded {len(frames)} frames from {os.path.basename(INPUT_VIDEO)}")
    print(f"   Resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")
    print(f"   Degrade: none (evaluating on real analog footage)")
    print()

    # Determine which presets to run
    if args.presets:
        preset_list = [p.strip() for p in args.presets.split(",")]
    else:
        preset_list = sorted(PRESETS.keys())

    # Remove 'adaptive' preset from sweep (uses AdaptivePipeline, not apply_pipeline)
    preset_list = [p for p in preset_list if p != "adaptive"]

    print(f"🧪 Running {len(preset_list)} presets...")
    if args.skip_bm3d:
        print(f"   (BM3D presets will be skipped)")
    print()

    results = {}
    for pname in preset_list:
        entry = run_single_preset(pname, frames, args.skip_bm3d)
        results[pname] = entry

    # Build output data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "input": os.path.basename(INPUT_VIDEO),
        "num_frames": len(frames),
        "degrade": "none",
        "preset_count": len(preset_list),
        "results": results,
    }

    # Sort by composite score desc for ranking
    ranked = sorted(
        [r for r in results.values() if r.get("status") == "ok"],
        key=lambda x: x.get("composite_score", 0),
        reverse=True,
    )

    # Print ranking
    print(f"\n{'=' * 100}")
    print("🏆 RANKING (by Composite Score)")
    print(f"{'=' * 100}")
    print(f"{'Rank':<6} {'Preset':<26} {'Score':<8} {'PSNR':<7} {'SSIM':<8} "
          f"{'ΔE':<7} {'VIF':<7} {'NIQE':<7} {'Time(ms)':<9} {'Filters'}")
    print("-" * 100)
    for i, r in enumerate(ranked):
        print(
            f"  #{i+1:<2}  {r['preset']:<26} "
            f"{r['composite_score']:<8.2f} "
            f"{r['psnr']:<7.2f} "
            f"{r['ssim']:<8.4f} "
            f"{r['color_fidelity_dE']:<7.2f} "
            f"{r['vif']:<7.4f} "
            f"{r['niqe']:<7.2f} "
            f"{r['time_ms']:<9.1f} "
            f"{r['filters'][:35]}"
        )

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, f"full_ablation_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n📄 JSON results → {json_path}")

    # Also save a compact CSV
    csv_path = os.path.join(OUTPUT_DIR, f"full_ablation_{timestamp}.csv")
    with open(csv_path, "w") as f:
        headers = [
            "rank", "preset", "composite_score", "psnr", "ssim",
            "color_fidelity_dE", "edge_retention", "noise_level",
            "detail_recovery", "artifact_score", "vif", "niqe",
            "time_ms", "status",
        ]
        f.write(",".join(headers) + "\n")
        for i, r in enumerate(ranked):
            row = [
                str(i + 1),
                r["preset"],
                str(r.get("composite_score", "")),
                str(r.get("psnr", "")),
                str(r.get("ssim", "")),
                str(r.get("color_fidelity_dE", "")),
                str(r.get("edge_retention", "")),
                str(r.get("noise_level", "")),
                str(r.get("detail_recovery", "")),
                str(r.get("artifact_score", "")),
                str(r.get("vif", "")),
                str(r.get("niqe", "")),
                str(r.get("time_ms", "")),
                r.get("status", "ok"),
            ]
            f.write(",".join(row) + "\n")
    print(f"📄 CSV results  → {csv_path}")

    print(f"\n✅ Full ablation sweep complete — {len(ranked)}/{len(preset_list)} presets succeeded")


if __name__ == "__main__":
    main()
