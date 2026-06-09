#!/usr/bin/env python3
"""
Comprehensive Benchmark — Evaluate ALL presets by no-reference metrics (NIQE, BRISQUE).

This script fixes the flaw in the ablation sweep: since we're evaluating on real
analog footage (no clean reference), PSNR/SSIM against the noisy input are
misleading.  Instead, we rank by NIQE and BRISQUE (lower = better perceptual quality).

Usage:
    .venv/bin/python3 run_comprehensive_benchmark.py
    .venv/bin/python3 run_comprehensive_benchmark.py --presets optimal-balanced,fast-denoise,temporal-premium
    .venv/bin/python3 run_comprehensive_benchmark.py --multi-video  # test on 5 videos
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
INPUT_DIR = os.path.join(BASE, "input")
OUTPUT_DIR = os.path.join(BASE, "output", "comprehensive_benchmark")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Videos to test (sorted by size, representative set)
TEST_VIDEOS = [
    "analog_whoop_footage.mp4",      # 854x480, 64s
    "VID00001-00.00.14.901-00.00.49.342-seg1.mp4",  # analog
    "VID00002-00.00.57.100-00.01.31.895-seg2.mp4",  # analog
    "VID00003-00.00.08.857-00.01.48.535-seg1.mp4",  # analog (large)
    "VID00006-00.00.13.921-00.00.53.715-seg1.mp4",  # analog
    "VID00007-00.00.10.699-00.00.46.107-seg1.mp4",  # analog
    "digital_whoop_footage.mp4",     # 1440x1080, 118s
]

evaluator = AutoEvaluator()


def get_first_frames(video_path: str, n: int = 3) -> list[np.ndarray]:
    """Read first N frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def run_preset_eval(
    preset_name: str,
    frames: list[np.ndarray],
) -> dict:
    """Evaluate a single preset on given frames."""
    cfg = PRESETS.get(preset_name)
    if cfg is None:
        return {"preset": preset_name, "status": "error", "reason": "unknown preset"}

    if preset_name == "adaptive":
        # Adaptive needs special handling
        return {"preset": preset_name, "status": "skipped", "reason": "adaptive pipeline"}

    try:
        reset_temporal_state()

        # Time measurement on first frame
        t0 = time.perf_counter()
        restored = pl.apply_pipeline(frames[0].copy(), cfg)
        elapsed = time.perf_counter() - t0

        # Full evaluation against original (noisy) reference
        result = evaluator.evaluate(
            frames[0], restored,
            label=preset_name,
            degraded=frames[0],
            verbose=False,
        )

        m = {mt.name: mt.value for mt in result.metrics}

        entry = {
            "preset": preset_name,
            "status": "ok",
            "composite_score": round(result.composite_score, 2),
            "psnr": round(m.get("psnr", 0), 2),
            "ssim": round(m.get("ssim", 0), 4),
            "niqe": round(m.get("niqe", 0), 2),
            "brisque": round(m.get("brisque", 0), 2),
            "color_fidelity_dE": round(m.get("color_fidelity", 0), 2),
            "edge_retention": round(m.get("edge_retention", 0), 3),
            "noise_level": round(m.get("noise_level", 0), 1),
            "detail_recovery": round(m.get("detail_recovery", 0), 3),
            "artifact_score": round(m.get("artifact_score", 0), 2),
            "vif": round(m.get("vif", 0), 4),
            "time_ms": round(elapsed * 1000, 1),
            "tags": {
                "fast": elapsed < 0.033,
                "realtime": elapsed < 0.033,
                "interactive": elapsed < 0.5,
            },
            "filters": " → ".join(s.name for s in cfg.stages if s.enabled),
        }
        return entry

    except Exception as e:
        return {"preset": preset_name, "status": "error", "reason": str(e)}


def evaluate_presets(
    preset_list: list[str],
    frames: list[np.ndarray],
    video_name: str,
) -> list[dict]:
    """Run evaluation for all presets, return results list."""
    print(f"\n{'=' * 90}")
    print(f"📹 Video: {video_name}  ({frames[0].shape[1]}x{frames[0].shape[0]})")
    print(f"🧪 Evaluating {len(preset_list)} presets...")
    print(f"{'=' * 90}")

    results = []
    for pname in preset_list:
        entry = run_preset_eval(pname, frames)
        results.append(entry)

        if entry.get("status") == "ok":
            print(
                f"  {pname:30s}  "
                f"NIQE={entry['niqe']:5.2f}  "
                f"BRISQUE={entry['brisque']:5.2f}  "
                f"Score={entry['composite_score']:5.2f}  "
                f"PSNR={entry['psnr']:5.2f}  "
                f"SSIM={entry['ssim']:.4f}  "
                f"{entry['time_ms']:7.1f}ms"
            )
        elif entry.get("status") == "error":
            print(f"  {pname:30s}  ❌ {entry.get('reason', 'unknown error')}")
        elif entry.get("status") == "skipped":
            print(f"  {pname:30s}  ⏭️  {entry.get('reason', 'skipped')}")

    return results


def print_rankings(results: list[dict], metric: str = "niqe"):
    """Print ranked results by a given metric."""
    valid = [r for r in results if r.get("status") == "ok"]
    reverse = metric in ("composite_score", "psnr", "ssim", "vif", "edge_retention", "detail_recovery")
    valid.sort(key=lambda x: x.get(metric, 999), reverse=reverse)

    print(f"\n{'─' * 90}")
    print(f"🏆 RANKING BY {metric.upper()} {'(higher=better)' if reverse else '(lower=better)'}")
    print(f"{'─' * 90}")
    print(f"{'Rank':<6} {'Preset':<30} {metric:<12} {'Score':<8} {'PSNR':<7} {'SSIM':<8} {'Time(ms)':<9}")
    print(f"{'─' * 90}")
    for i, r in enumerate(valid):
        print(
            f"  #{i+1:<2}  {r['preset']:<30} "
            f"{r.get(metric, 0):<12.2f} "
            f"{r['composite_score']:<8.2f} "
            f"{r['psnr']:<7.2f} "
            f"{r['ssim']:<8.4f} "
            f"{r['time_ms']:<9.1f}"
        )
    return valid


def save_results(all_results: dict[str, list[dict]], output_path: str):
    """Save all results to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n📄 Results saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark — rank presets by NIQE/BRISQUE no-reference quality"
    )
    parser.add_argument("--presets", type=str, default=None,
                        help="Comma-separated preset names (default: all)")
    parser.add_argument("--multi-video", action="store_true",
                        help="Test on multiple representative videos")
    parser.add_argument("--video", type=str, default=None,
                        help="Specific video file to use")
    parser.add_argument("--sample", type=int, default=3,
                        help="Number of frames to evaluate (default: 3)")
    args = parser.parse_args()

    # Determine preset list
    if args.presets:
        preset_list = [p.strip() for p in args.presets.split(",")]
    else:
        preset_list = sorted(PRESETS.keys())

    # Remove adaptive from automated sweep
    preset_list = [p for p in preset_list if p != "adaptive"]

    # Determine videos to test
    if args.video:
        videos = [args.video]
    elif args.multi_video:
        videos = TEST_VIDEOS
    else:
        videos = ["analog_whoop_footage.mp4"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        "timestamp": timestamp,
        "preset_count": len(preset_list),
        "video_count": len(videos),
        "sample_frames": args.sample,
        "results": {},
    }

    for vname in videos:
        vpath = os.path.join(INPUT_DIR, vname)
        if not os.path.exists(vpath):
            # Try searching for partial match
            candidates = [f for f in os.listdir(INPUT_DIR) if vname in f]
            if candidates:
                vpath = os.path.join(INPUT_DIR, candidates[0])
                vname = candidates[0]
            else:
                print(f"⚠  Video not found: {vname}")
                continue

        frames = get_first_frames(vpath, args.sample)
        if not frames:
            print(f"⚠  Cannot read frames from {vname}")
            continue

        results = evaluate_presets(preset_list, frames, vname)
        all_results["results"][vname] = results

        # Print rankings by key metrics
        print_rankings(results, "niqe")
        print_rankings(results, "brisque")
        print_rankings(results, "composite_score")

        # Print top-10 summary
        valid_nr = [r for r in results if r.get("status") == "ok"]
        valid_nr.sort(key=lambda x: x.get("niqe", 999))
        print(f"\n{'=' * 90}")
        print("📊 TOP 10 BY NIQE (no-reference quality, lower=better)")
        print(f"{'=' * 90}")
        for i, r in enumerate(valid_nr[:10]):
            print(
                f"  #{i+1:<2}  {r['preset']:30s}  "
                f"NIQE={r['niqe']:5.2f}  "
                f"BRISQUE={r['brisque']:5.2f}  "
                f"Score={r['composite_score']:5.2f}  "
                f"PSNR={r['psnr']:5.2f}  "
                f"SSIM={r['ssim']:.4f}  "
                f"{r['time_ms']:7.1f}ms"
            )

        valid_bq = [r for r in results if r.get("status") == "ok"]
        valid_bq.sort(key=lambda x: x.get("brisque", 999))
        print(f"\n📊 TOP 10 BY BRISQUE (no-reference quality, lower=better)")
        print(f"{'=' * 90}")
        for i, r in enumerate(valid_bq[:10]):
            print(
                f"  #{i+1:<2}  {r['preset']:30s}  "
                f"BRISQUE={r['brisque']:5.2f}  "
                f"NIQE={r['niqe']:5.2f}  "
                f"Score={r['composite_score']:5.2f}  "
                f"PSNR={r['psnr']:5.2f}  "
                f"SSIM={r['ssim']:.4f}  "
                f"{r['time_ms']:7.1f}ms"
            )

    # Save full results
    out_path = os.path.join(OUTPUT_DIR, f"benchmark_{timestamp}.json")
    save_results(all_results, out_path)

    # Also save summary CSV
    csv_path = os.path.join(OUTPUT_DIR, f"benchmark_{timestamp}_summary.csv")
    with open(csv_path, "w") as f:
        f.write("video,preset,status,niqe,brisque,composite_score,psnr,ssim,time_ms\n")
        for vname, results in all_results["results"].items():
            for r in results:
                f.write(
                    f"{vname},{r.get('preset','')},{r.get('status','')},"
                    f"{r.get('niqe','')},{r.get('brisque','')},"
                    f"{r.get('composite_score','')},{r.get('psnr','')},"
                    f"{r.get('ssim','')},{r.get('time_ms','')}\n"
                )
    print(f"📄 CSV summary → {csv_path}")


if __name__ == "__main__":
    main()
