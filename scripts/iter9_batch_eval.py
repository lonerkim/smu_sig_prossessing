#!/usr/bin/env python3
"""
Iter9 — Batch evaluation across 27 frames extracted from videos.
1. Extract 27 evenly-spaced frames from both whoop footage videos
2. Process with top presets
3. Auto-evaluate (PSNR/SSIM/NIQE/etc.)
4. Generate comprehensive report
"""
import sys, os, time, csv, json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.degradation import degrade_image
from smu_sig_prossessing.auto_evaluation import AutoEvaluator, EvalResult

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(REPO, "input")
OUTPUT = os.path.join(REPO, "output", "iter9_batch")
os.makedirs(OUTPUT, exist_ok=True)

# All 36 presets — we only run top performers on all frames
ALL_PRESETS = {
    "edge-preserve": PipelineConfig.edge_preserve(),
    "fast-denoise": PipelineConfig.fast_denoise(),
    "nlm-denoise": PipelineConfig.nlm_denoise(),
    "wiener-denoise": PipelineConfig.wiener_denoise(),
    "wavelet-denoise": PipelineConfig.wavelet_denoise(),
    "guided-denoise": PipelineConfig.guided_denoise(),
    "tv-denoise": PipelineConfig.tv_denoise_preset(),
    "aniso-denoise": PipelineConfig.aniso_denoise(),
    "dct-denoise": PipelineConfig.dct_denoise(),
    "st-video": PipelineConfig.spatial_temporal_video(),
    "optimized-fast": PipelineConfig.optimized_fast(),
    "optimized-quality": PipelineConfig.optimized_quality(),
    "max-quality": PipelineConfig.max_quality(),
    "video-enhanced": PipelineConfig.video_enhanced(),
    "aggressive": PipelineConfig.aggressive(),
    "research-best": PipelineConfig.research_best(),
    "analog-clean": PipelineConfig.analog_clean(),
    "analog-heavy": PipelineConfig.analog_heavy(),
    "adaptive": PipelineConfig.adaptive(),
    "video-ultra": PipelineConfig.video_ultra(),
    "ntsc-plus": PipelineConfig.ntc_plus(),
    "fast-premium": PipelineConfig.fast_premium(),
    "bm3d-denoise": PipelineConfig.bm3d_denoise(),
    "retinex-enhance": PipelineConfig.retinex_enhance(),
    "retinex-bm3d": PipelineConfig.retinex_bm3d(),
    "bm3d-fast": PipelineConfig.bm3d_fast(),
    "retinex-msrcr": PipelineConfig.retinex_msrcr_enhance(),
    "retinex-bm3d-msrcr": PipelineConfig.retinex_bm3d_msrcr(),
    "super-premium": PipelineConfig.super_premium(),
    "super-premium-fast": PipelineConfig.super_premium_fast(),
    "rolling-premium": PipelineConfig.rolling_premium(),
    "temporal-premium": PipelineConfig.temporal_premium(),
    "temporal-ntsc": PipelineConfig.temporal_ntsc(),
    "bm4d-temporal": PipelineConfig.bm4d_temporal(),
    "ultralight": PipelineConfig.ultralight(),
    "chroma-focus": PipelineConfig.chroma_focus(),
}

# Top presets from ablation results
TOP_PRESETS = [
    "temporal-premium", "chroma-focus", "super-premium-fast",
    "bm4d-temporal", "rolling-premium", "super-premium",
    "video-enhanced", "video-ultra", "fast-premium",
    "optimized-quality", "research-best", "ntsc-plus",
    "temporal-ntsc", "ultralight",
]

def extract_frames(video_path: str, n_frames: int) -> list[tuple[int, np.ndarray]]:
    """Extract evenly-spaced frames from video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    step = max(1, total // n_frames)
    frames = []
    for i in range(0, total, step):
        if len(frames) >= n_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append((i, frame))
    cap.release()
    # If we got fewer frames than requested, supplement with last frames
    if len(frames) < n_frames and total > 0:
        for i in range(max(0, total - n_frames + len(frames)), total):
            if len(frames) >= n_frames:
                break
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append((i, frame))
            cap.release()
    return frames[:n_frames]

def process_and_evaluate(origin: np.ndarray, preset_name: str, label_suffix: str = "") -> tuple[float, EvalResult]:
    """Run a preset and evaluate."""
    cfg = ALL_PRESETS[preset_name]
    # Degrade (basic, strength 0.5) for clean images
    degraded = degrade_image(origin, use_ntsc=False, strength=0.5)
    t0 = time.perf_counter()
    restored = pl.apply_pipeline(degraded, cfg)
    elapsed = time.perf_counter() - t0
    label = f"{preset_name}{label_suffix}"
    ev = AutoEvaluator()
    result = ev.evaluate(origin, restored, label=label, degraded=degraded, verbose=False)
    return elapsed, result

def main():
    print("=" * 70)
    print("Iter9 — Batch Evaluation: 27 frames x top presets")
    print("=" * 70)

    # 1) Extract 27 frames: 13 from analog, 12 from digital + 2 images
    print("\n📽️  Extracting frames from videos...")
    analog_frames = extract_frames(os.path.join(INPUT, "analog_whoop_footage.mp4"), 13)
    digital_frames = extract_frames(os.path.join(INPUT, "digital_whoop_footage.mp4"), 12)
    print(f"   analog_whoop: {len(analog_frames)} frames")
    print(f"   digital_whoop: {len(digital_frames)} frames")

    # Also use existing images
    still_frames = []
    for img_name in ["test_small.jpg", "REAL_WORLD_PICTURE.jpg"]:
        img = cv2.imread(os.path.join(INPUT, img_name))
        if img is not None:
            still_frames.append((img_name, img))
            print(f"   still: {img_name} ({img.shape[1]}x{img.shape[0]})")

    all_frames = []
    for fidx, (frame_num, frame) in enumerate(analog_frames):
        all_frames.append((f"analog_f{fidx:02d}", frame))
    for fidx, (frame_num, frame) in enumerate(digital_frames):
        all_frames.append((f"digital_f{fidx:02d}", frame))
    for name, img in still_frames:
        all_frames.append((name.replace(".jpg", ""), img))

    print(f"\n   Total test frames: {len(all_frames)}")

    # 2) Run all 36 presets on first frame (small) for ranking
    print("\n" + "=" * 70)
    print("Phase 1: Full 36-preset ranking on test_small.jpg...")
    print("=" * 70)

    small_img = cv2.imread(os.path.join(INPUT, "test_small.jpg"))
    full_ranking = []
    for pname, cfg in ALL_PRESETS.items():
        try:
            degraded = degrade_image(small_img, use_ntsc=False, strength=0.5)
            t0 = time.perf_counter()
            restored = pl.apply_pipeline(degraded, cfg)
            elapsed = time.perf_counter() - t0
            ev = AutoEvaluator()
            res = ev.evaluate(small_img, restored, label=pname, degraded=degraded, verbose=False)
            
            n = res.get("niqe")
            niqe_val = n.value if n else 0
            full_ranking.append((pname, res.composite_score, res.get("psnr").value, res.get("ssim").value, niqe_val, elapsed))
            print(f"  {pname:30s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  NIQE={niqe_val:.2f}  {elapsed*1000:.0f}ms")
        except Exception as e:
            print(f"  {pname:30s}  ERROR: {e}")

    # Sort and display top 15
    full_ranking.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'🏆 ALL 36 PRESETS RANKING (test_small.jpg)':^70}")
    print(f"{'Rank':<5} {'Preset':<30} {'Score':<8} {'PSNR':<8} {'SSIM':<8} {'NIQE':<8} {'Time':<8}")
    print("-" * 70)
    for i, (name, score, psnr, ssim, niqe, elapsed) in enumerate(full_ranking):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"  {medal} #{i+1:<2} {name:<30s} {score:<8.2f} {psnr:<8.2f} {ssim:<8.4f} {niqe:<8.2f} {elapsed*1000:<8.0f}ms")

    # 3) Top 14 presets on all 27 frames
    print(f"\n{'=' * 70}")
    print(f"Phase 2: Top {len(TOP_PRESETS)} presets × {len(all_frames)} frames")
    print("=" * 70)

    all_results = {}  # (frame_name, preset_name) -> (elapsed, EvalResult)
    
    for fname, frame in all_frames:
        print(f"\n  📷 Processing: {fname} ({(len(all_frames) - all_frames.index((fname, frame)))} remaining)")
        # Ensure divisible by 2 for some filters
        h, w = frame.shape[:2]
        if h % 2 != 0 or w % 2 != 0:
            frame = cv2.resize(frame, (w - w % 2, h - h % 2))
        
        for pname in TOP_PRESETS:
            try:
                degraded = degrade_image(frame, use_ntsc=False, strength=0.5)
                t0 = time.perf_counter()
                restored = pl.apply_pipeline(degraded, ALL_PRESETS[pname])
                elapsed = time.perf_counter() - t0
                ev = AutoEvaluator()
                res = ev.evaluate(frame, restored, label=f"{pname}@{fname}", degraded=degraded, verbose=False)
                all_results[(fname, pname)] = (elapsed, res)
            except Exception as e:
                print(f"    ⚠ {pname} on {fname}: {e}")
    
    # 4) Generate comprehensive reports
    print(f"\n{'=' * 70}")
    print("Generating reports...")
    print("=" * 70)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Aggregate per-preset across all frames
    preset_agg = {}
    for pname in TOP_PRESETS:
        scores = []
        psnrs = []
        ssims = []
        niqes = []
        times = []
        for fname, _ in all_frames:
            if (fname, pname) in all_results:
                elapsed, res = all_results[(fname, pname)]
                n = res.get("niqe")
                niqe_val = n.value if n else 0
                scores.append(res.composite_score)
                psnrs.append(res.get("psnr").value)
                ssims.append(res.get("ssim").value)
                niqes.append(niqe_val)
                times.append(elapsed)
        if scores:
            preset_agg[pname] = {
                "avg_score": np.mean(scores),
                "med_score": np.median(scores),
                "std_score": np.std(scores),
                "avg_psnr": np.mean(psnrs),
                "avg_ssim": np.mean(ssims),
                "avg_niqe": np.mean(niqes),
                "avg_time_ms": np.mean(times) * 1000,
            }

    # Sort by avg score
    ranked = sorted(preset_agg.items(), key=lambda x: x[1]["avg_score"], reverse=True)

    # CSV report
    csv_path = os.path.join(OUTPUT, f"iter9_batch_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Preset", "AvgScore", "MedScore", "StdScore", "AvgPSNR", "AvgSSIM", "AvgNIQE", "AvgTime_ms", "Frames"])
        for i, (name, agg) in enumerate(ranked):
            writer.writerow([i+1, name, f"{agg['avg_score']:.2f}", f"{agg['med_score']:.2f}", f"{agg['std_score']:.2f}", f"{agg['avg_psnr']:.2f}", f"{agg['avg_ssim']:.4f}", f"{agg['avg_niqe']:.2f}", f"{agg['avg_time_ms']:.1f}", len(all_frames)])
    print(f"  📄 CSV → {csv_path}")

    # JSON
    json_path = os.path.join(OUTPUT, f"iter9_batch_{ts}.json")
    with open(json_path, "w") as f:
        json.dump({"timestamp": ts, "n_frames": len(all_frames), "ranking": [
            {"rank": i+1, "preset": name, **agg} for i, (name, agg) in enumerate(ranked)
        ]}, f, indent=2)
    print(f"  📄 JSON → {json_path}")

    # Markdown report
    md_path = os.path.join(OUTPUT, f"iter9_batch_{ts}.md")
    with open(md_path, "w") as f:
        f.write(f"# Iter9 Batch Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Test frames:** {len(all_frames)} (analog: {len(analog_frames)}, digital: {len(digital_frames)}, still: {len(still_frames)})\n")
        f.write(f"**Presets:** {len(TOP_PRESETS)} top performers\n\n")
        f.write(f"## Overall Ranking (avg across {len(all_frames)} frames)\n\n")
        f.write(f"| Rank | Preset | AvgScore | ±Std | AvgPSNR | AvgSSIM | AvgNIQE | AvgTime |\n")
        f.write(f"|------|--------|----------|------|---------|---------|---------|---------|\n")
        for i, (name, agg) in enumerate(ranked):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else ""
            f.write(f"| {i+1} | {medal} {name} | **{agg['avg_score']:.1f}** | ±{agg['std_score']:.1f} | {agg['avg_psnr']:.1f} | {agg['avg_ssim']:.4f} | {agg['avg_niqe']:.2f} | {agg['avg_time_ms']:.0f}ms |\n")
        f.write("\n")
        
        # Per-frame table (first video frames)
        f.write(f"## Per-Frame Detail (first 10 frames)\n\n")
        f.write(f"| Frame |")
        for pname, _ in ranked[:5]:
            f.write(f" Score({pname[:12]}) |")
        f.write("\n|")
        f.write("---|" * (1 + min(5, len(ranked))) + "\n")
        for fname, _ in all_frames[:10]:
            f.write(f"| {fname} |")
            for pname, _ in ranked[:5]:
                if (fname, pname) in all_results:
                    _, res = all_results[(fname, pname)]
                    f.write(f" {res.composite_score:.1f} |")
                else:
                    f.write(" — |")
            f.write("\n")
        
        # Best for each frame
        f.write(f"\n## Best Preset Per Frame\n\n")
        f.write(f"| Frame | Best Preset | Score | PSNR | SSIM | NIQE |\n")
        f.write(f"|-------|-------------|-------|------|------|------|\n")
        for fname, _ in all_frames:
            best_p = None
            best_score = -1
            best_res = None
            for pname in TOP_PRESETS:
                if (fname, pname) in all_results:
                    _, res = all_results[(fname, pname)]
                    if res.composite_score > best_score:
                        best_score = res.composite_score
                        best_p = pname
                        best_res = res
            if best_p and best_res:
                n = best_res.get("niqe")
                nv = n.value if n else 0
                f.write(f"| {fname} | {best_p} | {best_res.composite_score:.1f} | {best_res.get('psnr').value:.1f} | {best_res.get('ssim').value:.4f} | {nv:.2f} |\n")
        
        # All 36 presets ranking on small image
        f.write(f"\n## Full 36-Preset Ranking (test_small.jpg)\n\n")
        f.write(f"| Rank | Preset | Score | PSNR | SSIM | NIQE | Time |\n")
        f.write(f"|------|--------|-------|------|------|------|------|\n")
        for i, (name, score, psnr, ssim, niqe, elapsed) in enumerate(full_ranking):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else ""
            f.write(f"| {i+1} | {medal} {name} | {score:.1f} | {psnr:.1f} | {ssim:.4f} | {niqe:.2f} | {elapsed*1000:.0f}ms |\n")
    
    print(f"  📄 Report → {md_path}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Test frames:    {len(all_frames)}")
    print(f"Presets tested: {len(TOP_PRESETS)} (top) + {len(ALL_PRESETS)} (full)")
    for i, (name, agg) in enumerate(ranked[:5]):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"  {medal} #{i+1}: {name:<30s} AvgScore={agg['avg_score']:.2f}  PSNR={agg['avg_psnr']:.2f}  SSIM={agg['avg_ssim']:.4f}  NIQE={agg['avg_niqe']:.2f}")
    print(f"\n📄 Reports saved to: {OUTPUT}")

if __name__ == "__main__":
    main()
