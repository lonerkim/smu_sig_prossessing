#!/usr/bin/env python3
"""
Iter9: Full ablation sweep — run all presets on all input videos,
collect PSNR/SSIM/speed/NR metrics, output JSON + ranking table.
"""
import subprocess, json, time, os, sys, glob
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).resolve().parent.parent  # scripts/ → project root
PYTHON = str(PROJECT / ".venv" / "bin" / "python3")
MAIN = str(PROJECT / "main.py")
INPUT_DIR = PROJECT / "input"
OUT_DIR = PROJECT / "output" / "ablation_sweep"
RESULTS_FILE = PROJECT / "output" / "iter9_sweep_results.json"

PRESETS = [
    "edge-preserve", "fast-denoise", "nlm-denoise", "wiener-denoise",
    "wavelet-denoise", "guided-denoise", "tv-denoise", "aniso-denoise",
    "dct-denoise", "st-video", "optimized-fast", "optimized-quality",
    "max-quality", "video-enhanced", "video-ultra", "ntsc-plus",
    "fast-premium", "aggressive", "research-best", "analog-clean",
    "analog-heavy", "adaptive", "bm3d-denoise", "retinex-enhance",
    "retinex-bm3d", "bm3d-fast", "retinex-msrcr", "retinex-bm3d-msrcr",
    "super-premium", "super-premium-fast", "rolling-premium",
    "temporal-premium", "bm4d-temporal", "ultralight", "chroma-focus"
]

# Use a representative subset for speed
QUICK_VIDEOS = ["analog_whoop_footage.mp4"]
FULL_VIDEOS = sorted([f.name for f in INPUT_DIR.glob("*.mp4")])

def run_preset(video, preset, sample=3):
    """Run a single preset and capture output."""
    cmd = [
        PYTHON, MAIN,
        "-p", str(INPUT_DIR / video),
        "--preset", preset,
        "--degrade", "none",
        "--sample", str(sample)
    ]
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - t0
        
        output = result.stdout + result.stderr
        # Parse metrics from output
        metrics = {"time_sec": round(elapsed, 2), "exit_code": result.returncode}
        
        for line in output.split("\n"):
            if "PSNR" in line:
                try: metrics["psnr"] = float(line.split(":")[-1].strip().split()[0])
                except: pass
            if "SSIM" in line:
                try: metrics["ssim"] = float(line.split(":")[-1].strip().split()[0])
                except: pass
            if "NIQE" in line or "niqe" in line.lower():
                try: metrics["niqe"] = float(line.split(":")[-1].strip().split()[0])
                except: pass
            if "score" in line.lower():
                try: metrics["score"] = float(line.split(":")[-1].strip().split()[0])
                except: pass
            if "frame" in line.lower() and "sec" in line.lower():
                try: metrics["fps"] = float(line.split("=")[-1].strip().split()[0])
                except: pass
        
        return metrics
    except subprocess.TimeoutExpired:
        return {"time_sec": 300, "exit_code": -1, "error": "timeout"}
    except Exception as e:
        return {"time_sec": time.time() - t0, "exit_code": -2, "error": str(e)}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    videos = QUICK_VIDEOS if mode == "quick" else FULL_VIDEOS
    sample = 3 if mode == "quick" else 5
    
    print(f"🔍 Iter9 Ablation Sweep — {mode} mode")
    print(f"   Videos: {len(videos)}, Presets: {len(PRESETS)}, Sample: {sample} frames")
    print(f"   Total runs: {len(videos) * len(PRESETS)}")
    print()
    
    all_results = {}
    total = len(videos) * len(PRESETS)
    done = 0
    
    for video in videos:
        video_results = {}
        for preset in PRESETS:
            done += 1
            print(f"[{done}/{total}] {preset} ← {video} ... ", end="", flush=True)
            metrics = run_preset(video, preset, sample)
            video_results[preset] = metrics
            status = "✅" if metrics.get("exit_code") == 0 else "❌"
            print(f"{status} {metrics.get('time_sec', '?')}s" + 
                  (f" PSNR={metrics.get('psnr', '?')}" if 'psnr' in metrics else ""))
        
        all_results[video] = video_results
    
    # Save raw results
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "videos": videos,
            "results": all_results
        }, f, indent=2)
    
    # Generate ranking
    print("\n" + "="*60)
    print("📊 RANKING BY COMPOSITE SCORE")
    print("="*60)
    
    preset_scores = {}
    for preset in PRESETS:
        scores = []
        times = []
        for video, vresults in all_results.items():
            m = vresults.get(preset, {})
            if m.get("exit_code") != 0: continue
            s = m.get("psnr", 0) * 0.4
            s += m.get("ssim", 0) * 40  # scale SSIM to similar range
            s -= m.get("niqe", 0) * 2   # lower NIQE is better
            scores.append(s)
            times.append(m.get("time_sec", 999))
        if scores:
            preset_scores[preset] = {
                "avg_score": round(sum(scores)/len(scores), 2),
                "avg_time": round(sum(times)/len(times), 2),
                "runs": len(scores)
            }
    
    ranked = sorted(preset_scores.items(), key=lambda x: -x[1]["avg_score"])
    for i, (name, data) in enumerate(ranked[:15], 1):
        print(f"  {i:2d}. {name:<25s} score={data['avg_score']:<8.2f} time={data['avg_time']:.1f}s")
    
    print(f"\n💾 Full results saved to {RESULTS_FILE}")
    return ranked

if __name__ == "__main__":
    main()
