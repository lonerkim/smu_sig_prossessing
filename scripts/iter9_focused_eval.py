#!/usr/bin/env python3
"""
Iter9 — Focused batch evaluation: 27 frames × 10 top presets.
Fast, focused, generates comprehensive report.
"""
import sys, os, time, csv, json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.degradation import degrade_image
from smu_sig_prossessing.auto_evaluation import AutoEvaluator
from main import PRESETS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output", "iter9_batch")
os.makedirs(OUTPUT, exist_ok=True)

TOP_PRESETS = [
    "temporal-premium", "chroma-focus", "super-premium-fast",
    "rolling-premium", "video-enhanced", "ntsc-plus",
    "temporal-ntsc", "ultralight", "fast-premium",
    "optimized-quality",
]

def extract_frames(video_path, n_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    step = max(1, total // n_frames)
    frames = []
    for i in range(min(n_frames, total)):
        pos = min(i * step, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        if len(frames) >= n_frames:
            break
    cap.release()
    return frames[:n_frames]

def resize_if_needed(img, max_w=800):
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def main():
    print("=" * 70)
    print("ITER9 — Focused Batch Evaluation")
    print("=" * 70)

    # 1) Load frames (27 total)
    frames = []
    names = []

    analog = extract_frames(os.path.join(BASE, "input", "analog_whoop_footage.mp4"), 13)
    digital = extract_frames(os.path.join(BASE, "input", "digital_whoop_footage.mp4"), 12)
    
    print(f"\n📽️  analog: {len(analog)} frames, digital: {len(digital)} frames")

    for i, f in enumerate(analog):
        frames.append(resize_if_needed(f))
        names.append(f"analog_{i:02d}")
    for i, f in enumerate(digital):
        frames.append(resize_if_needed(f))
        names.append(f"digital_{i:02d}")

    for img_name in ["test_small.jpg", "REAL_WORLD_PICTURE.jpg"]:
        img = cv2.imread(os.path.join(BASE, "input", img_name))
        if img is not None:
            frames.append(resize_if_needed(img))
            names.append(img_name.replace(".jpg", ""))
            print(f"  still: {img_name} ({img.shape[1]}x{img.shape[0]})")

    print(f"  Total: {len(frames)} frames")

    # 2) Process each preset on each frame
    results = {}  # (name, preset) -> (time, Score, PSNR, SSIM, NIQE, DE)
    evaluator = AutoEvaluator()

    for fi, fname in enumerate(names):
        frame = frames[fi]
        h, w = frame.shape[:2]
        if h % 2 != 0 or w % 2 != 0:
            frame = cv2.resize(frame, (w - w % 2, h - h % 2))
        
        deg = degrade_image(frame, use_ntsc=False, strength=0.5)
        
        for preset_name in TOP_PRESETS:
            if preset_name not in PRESETS:
                continue
            try:
                cfg = PRESETS[preset_name]
                t0 = time.perf_counter()
                restored = pl.apply_pipeline(deg, cfg)
                elapsed = time.perf_counter() - t0
                
                res = evaluator.evaluate(frame, restored, label=f"{fname}@{preset_name}",
                                         degraded=deg, verbose=False)
                
                n = res.get("niqe")
                niqe_val = n.value if n else 0
                de_val = res.get("color_fidelity").value if res.get("color_fidelity") else 0
                
                results[(fname, preset_name)] = {
                    "score": res.composite_score,
                    "psnr": res.get("psnr").value,
                    "ssim": res.get("ssim").value,
                    "niqe": niqe_val,
                    "de": de_val,
                    "time_ms": elapsed * 1000,
                }
                
                pct = ((fi * len(TOP_PRESETS) + TOP_PRESETS.index(preset_name) + 1) * 100) // (len(frames) * len(TOP_PRESETS))
                print(f"\r  [{pct:3d}%] {fname:20s} → {preset_name:25s} Score={res.composite_score:.1f}  {elapsed*1000:.0f}ms  ", end="", flush=True)
            except Exception as e:
                print(f"\n  ⚠ {fname} / {preset_name}: {e}")

    print()

    # 3) Aggregate
    preset_scores = {p: [] for p in TOP_PRESETS}
    for (fname, pname), data in results.items():
        preset_scores[pname].append(data)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV
    csv_path = os.path.join(OUTPUT, f"iter9_batch_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Frame", "Preset", "Score", "PSNR", "SSIM", "NIQE", "DE", "Time_ms"])
        for fname in names:
            for pname in TOP_PRESETS:
                if (fname, pname) in results:
                    d = results[(fname, pname)]
                    wr.writerow([fname, pname, f"{d['score']:.2f}", f"{d['psnr']:.2f}",
                                f"{d['ssim']:.4f}", f"{d['niqe']:.2f}", f"{d['de']:.2f}", f"{d['time_ms']:.0f}"])
    print(f"  📄 {csv_path}")

    # Aggregated results
    agg = {}
    for pname in TOP_PRESETS:
        scores = preset_scores[pname]
        if not scores:
            continue
        agg[pname] = {
            "avg_score": np.mean([s["score"] for s in scores]),
            "avg_psnr": np.mean([s["psnr"] for s in scores]),
            "avg_ssim": np.mean([s["ssim"] for s in scores]),
            "avg_niqe": np.mean([s["niqe"] for s in scores]),
            "avg_time_ms": np.mean([s["time_ms"] for s in scores]),
        }
    ranked = sorted(agg.items(), key=lambda x: x[1]["avg_score"], reverse=True)

    # Markdown report
    md_path = os.path.join(OUTPUT, f"iter9_batch_{ts}.md")
    with open(md_path, "w") as f:
        f.write(f"# Iter9 Batch Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Test frames:** {len(frames)} (analog: {len(analog)}, digital: {len(digital)}, still: {len(frames) - len(analog) - len(digital)})\n")
        f.write(f"**Presets tested:** {len(TOP_PRESETS)}\n\n")

        f.write(f"## Overall Ranking (avg across {len(frames)} frames)\n\n")
        f.write(f"| Rank | Preset | AvgScore | AvgPSNR | AvgSSIM | AvgNIQE | AvgTime |\n")
        f.write(f"|------|--------|----------|---------|---------|---------|--------|\n")
        for i, (name, a) in enumerate(ranked):
            m = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else ""
            f.write(f"| {i+1} | {m} {name} | **{a['avg_score']:.1f}** | {a['avg_psnr']:.1f} | {a['avg_ssim']:.4f} | {a['avg_niqe']:.2f} | {a['avg_time_ms']:.0f}ms |\n")

        f.write(f"\n## Best Preset Per Frame\n\n")
        f.write(f"| Frame | Best Preset | Score | PSNR | SSIM | NIQE |\n")
        f.write(f"|-------|-------------|-------|------|------|------|\n")
        for fname in names:
            best = None
            best_score = -1
            best_data = None
            for pname in TOP_PRESETS:
                if (fname, pname) in results:
                    d = results[(fname, pname)]
                    if d["score"] > best_score:
                        best_score = d["score"]
                        best = pname
                        best_data = d
            if best and best_data:
                f.write(f"| {fname} | {best} | {best_data['score']:.1f} | {best_data['psnr']:.1f} | {best_data['ssim']:.4f} | {best_data['niqe']:.2f} |\n")

    print(f"  📄 {md_path}")
    
    print(f"\n" + "=" * 70)
    print("📊 RANKING")
    print("=" * 70)
    for i, (name, a) in enumerate(ranked):
        m = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"  {m} #{i+1}: {name:<28s} Score={a['avg_score']:.2f}  PSNR={a['avg_psnr']:.2f}  NIQE={a['avg_niqe']:.2f}  {a['avg_time_ms']:.0f}ms")
    
    print(f"\n✅ Done — results in {OUTPUT}/")

if __name__ == "__main__":
    main()
