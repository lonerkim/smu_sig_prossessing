#!/usr/bin/env python3
"""
Multi-video validation — Test top presets on 5+ representative analog videos.

This validates that NIQE/BRISQUE rankings are consistent across different
types of analog FPV footage (different cameras, lighting, noise levels).

Usage:
    .venv/bin/python3 run_multi_video_benchmark.py
"""
from __future__ import annotations

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
INPUT_DIR = os.path.join(BASE, "input")
OUTPUT_DIR = os.path.join(BASE, "output", "multi_video_benchmark")
os.makedirs(OUTPUT_DIR, exist_ok=True)

evaluator = AutoEvaluator()

# Top presets to evaluate (by NIQE, BRISQUE, and balance)
TOP_PRESETS = [
    # NIQE-focused
    "optimal-bior4",
    "optimal-balanced",
    "nlm-chroma",
    "analog-clean",
    "cross-chroma-detail",
    # BRISQUE-focused
    "chroma-focus",
    "fast-guided-chroma",
    "grey-premium",
    "temporal-premium",
    # Fast + quality
    "bm4d-temporal",
    "fast-denoise",
    "optimal-ultrafast",
]

# Representative videos (different cameras/resolutions)
TEST_VIDEOS = [
    "analog_whoop_footage.mp4",           # 854x480, whoop FPV
    "VID00001-00.00.14.901-00.00.49.342-seg1.mp4",  # Analog capture 1
    "VID00002-00.00.57.100-00.01.31.895-seg2.mp4",  # Analog capture 2
    "VID00003-00.00.08.857-00.01.48.535-seg1.mp4",  # Large analog capture
    "VID00006-00.00.13.921-00.00.53.715-seg1.mp4",  # Different camera
]


def get_first_frames(video_path: str, n: int = 3) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def evaluate_preset(preset_name: str, frames: list[np.ndarray]) -> dict:
    cfg = PRESETS.get(preset_name)
    if cfg is None:
        return {"preset": preset_name, "status": "error", "reason": "unknown"}
    if preset_name == "adaptive":
        return {"preset": preset_name, "status": "skipped", "reason": "adaptive"}

    try:
        reset_temporal_state()
        t0 = time.perf_counter()
        restored = pl.apply_pipeline(frames[0].copy(), cfg)
        elapsed = time.perf_counter() - t0

        result = evaluator.evaluate(
            frames[0], restored, label=preset_name, degraded=frames[0], verbose=False
        )
        m = {mt.name: mt.value for mt in result.metrics}

        return {
            "preset": preset_name,
            "status": "ok",
            "niqe": round(m.get("niqe", 0), 2),
            "brisque": round(m.get("brisque", 0), 2),
            "composite_score": round(result.composite_score, 2),
            "psnr": round(m.get("psnr", 0), 2),
            "ssim": round(m.get("ssim", 0), 4),
            "time_ms": round(elapsed * 1000, 1),
        }
    except Exception as e:
        return {"preset": preset_name, "status": "error", "reason": str(e)}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        "timestamp": ts,
        "presets": TOP_PRESETS,
        "videos": [],
    }

    for vname in TEST_VIDEOS:
        vpath = os.path.join(INPUT_DIR, vname)
        if not os.path.exists(vpath):
            # Search for partial match
            candidates = [f for f in os.listdir(INPUT_DIR) if vname.replace(".mp4", "") in f]
            if candidates:
                vpath = os.path.join(INPUT_DIR, candidates[0])
                vname = candidates[0]
            else:
                print(f"⚠  Video not found: {vname}")
                continue

        frames = get_first_frames(vpath, 3)
        if not frames:
            print(f"⚠  Cannot read frames from {vname}")
            continue

        h, w = frames[0].shape[:2]
        print(f"\n{'=' * 90}")
        print(f"📹 {vname}  ({w}x{h})")
        print(f"{'=' * 90}")

        video_results = {"video": vname, "width": w, "height": h, "results": []}

        for pname in TOP_PRESETS:
            entry = evaluate_preset(pname, frames)
            video_results["results"].append(entry)
            if entry["status"] == "ok":
                print(
                    f"  {pname:30s}  "
                    f"NIQE={entry['niqe']:5.2f}  "
                    f"BRISQUE={entry['brisque']:5.2f}  "
                    f"Score={entry['composite_score']:5.2f}  "
                    f"PSNR={entry['psnr']:5.2f}  "
                    f"SSIM={entry['ssim']:.4f}  "
                    f"{entry['time_ms']:7.1f}ms"
                )
            else:
                print(f"  {pname:30s}  ❌ {entry.get('reason', 'error')}")

        # Per-video ranking
        valid = [r for r in video_results["results"] if r["status"] == "ok"]
        for metric in ("niqe", "brisque"):
            valid.sort(key=lambda x: x.get(metric, 999))
            print(f"\n  Ranking by {metric}:")
            for i, r in enumerate(valid[:5]):
                print(f"    #{i+1} {r['preset']:30s} {metric}={r[metric]:.2f}")

        all_results["videos"].append(video_results)

    # ── Cross-video consistency analysis ───────────────────────────
    print(f"\n{'=' * 90}")
    print("📊 CROSS-VIDEO CONSISTENCY ANALYSIS")
    print(f"{'=' * 90}")

    # For each preset, compute mean and std of NIQE across all videos
    preset_scores: dict[str, dict] = {}
    for video_entry in all_results["videos"]:
        for r in video_entry["results"]:
            if r["status"] != "ok":
                continue
            p = r["preset"]
            if p not in preset_scores:
                preset_scores[p] = {"niqe": [], "brisque": [], "time_ms": []}
            preset_scores[p]["niqe"].append(r["niqe"])
            preset_scores[p]["brisque"].append(r["brisque"])
            preset_scores[p]["time_ms"].append(r["time_ms"])

    print(f"\n{'Preset':30s} {'Mean NIQE':<10} {'Std NIQE':<10} {'Mean BRISQUE':<14} {'Std BRISQUE':<12} {'Mean(ms)':<10}")
    print(f"{'─' * 90}")

    for pname in TOP_PRESETS:
        if pname not in preset_scores:
            continue
        d = preset_scores[pname]
        mean_n = np.mean(d["niqe"])
        std_n = np.std(d["niqe"])
        mean_b = np.mean(d["brisque"])
        std_b = np.std(d["brisque"])
        mean_t = np.mean(d["time_ms"])
        print(
            f"  {pname:30s} "
            f"{mean_n:<10.2f} {std_n:<10.3f} "
            f"{mean_b:<14.2f} {std_b:<12.3f} "
            f"{mean_t:<10.1f}"
        )

    # Save JSON
    out_path = os.path.join(OUTPUT_DIR, f"multi_video_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n📄 Results saved → {out_path}")

    # Summary by ranking
    print(f"\n{'=' * 90}")
    print("🏆 OVERALL RANKING (mean NIQE across all videos)")
    print(f"{'=' * 90}")
    sorted_presets = sorted(
        [(p, d) for p, d in preset_scores.items()],
        key=lambda x: np.mean(x[1]["niqe"]),
    )
    for i, (pname, d) in enumerate(sorted_presets):
        print(
            f"  #{i+1:<2} {pname:30s} "
            f"NIQE={np.mean(d['niqe']):.2f}±{np.std(d['niqe']):.3f}  "
            f"BRISQUE={np.mean(d['brisque']):.2f}±{np.std(d['brisque']):.3f}  "
            f"{np.mean(d['time_ms']):.1f}ms"
        )


if __name__ == "__main__":
    main()
