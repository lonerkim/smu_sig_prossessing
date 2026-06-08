#!/usr/bin/env python3
"""
Iter9: Incremental ablation sweep with timeout protection.
Runs presets one-by-one, saves results incrementally.
"""
import subprocess, json, time, os, sys, signal
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).resolve().parent.parent
PYTHON = str(PROJECT / ".venv" / "bin" / "python3")
MAIN = str(PROJECT / "main.py")
INPUT_DIR = PROJECT / "input"
RESULTS_FILE = PROJECT / "output" / "iter9_sweep_results.json"

# Fast presets first, slow ones at end
FAST_PRESETS = [
    "fast-denoise", "guided-denoise", "st-video", "optimized-fast",
    "ultralight", "wiener-denoise", "dct-denoise", "analog-clean",
    "analog-heavy", "chroma-focus", "fast-premium", "edge-preserve",
    "wavelet-denoise", "tv-denoise", "aniso-denoise", "nlm-denoise",
    "optimized-quality", "max-quality", "ntsc-plus", "adaptive",
]
MEDIUM_PRESETS = [
    "video-enhanced", "video-ultra", "aggressive", "research-best",
    "retinex-enhance", "retinex-msrcr", "rolling-premium",
    "super-premium-fast", "bm3d-fast",
]
SLOW_PRESETS = [
    "bm3d-denoise", "retinex-bm3d", "retinex-bm3d-msrcr",
    "super-premium", "temporal-premium", "bm4d-temporal",
]

PRESETS = FAST_PRESETS + MEDIUM_PRESETS + SLOW_PRESETS

VIDEO = "analog_whoop_footage.mp4"
SAMPLE = 3
PER_PRESET_TIMEOUT = 120  # seconds max per preset

def load_existing():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"timestamp": "", "results": {}}

def save_results(data):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["timestamp"] = datetime.now().isoformat()
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def run_preset(preset):
    cmd = [
        PYTHON, MAIN,
        "-p", str(INPUT_DIR / VIDEO),
        "--preset", preset,
        "--degrade", "none",
        "--sample", str(SAMPLE)
    ]
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=PER_PRESET_TIMEOUT,
            cwd=str(PROJECT)
        )
        elapsed = time.time() - t0
        output = result.stdout + result.stderr
        
        metrics = {"time_sec": round(elapsed, 2), "exit_code": result.returncode}
        
        # Parse output for metrics
        for line in output.split("\n"):
            lower = line.lower()
            if "psnr" in lower and "=" in line:
                try:
                    val = line.split("=")[-1].strip().split()[0].replace("dB","")
                    metrics["psnr"] = float(val)
                except: pass
            if "ssim" in lower and "=" in line:
                try:
                    metrics["ssim"] = float(line.split("=")[-1].strip().split()[0])
                except: pass
            if ("niqe" in lower or "score" in lower) and "=" in line:
                try:
                    val = line.split("=")[-1].strip().split()[0]
                    if "score" in lower:
                        metrics["score"] = float(val)
                    else:
                        metrics["niqe"] = float(val)
                except: pass
        
        # Count output files as success indicator
        if result.returncode == 0:
            out_frames = list((PROJECT / "output").glob("analog_whoop_footage_sample_*.png"))
            metrics["output_frames"] = len(out_frames)
        
        return metrics
    except subprocess.TimeoutExpired:
        return {"time_sec": PER_PRESET_TIMEOUT, "exit_code": -1, "error": "timeout"}
    except Exception as e:
        return {"time_sec": round(time.time() - t0, 2), "exit_code": -2, "error": str(e)}

def main():
    data = load_existing()
    done_presets = set(data.get("results", {}).get(VIDEO, {}).keys())
    remaining = [p for p in PRESETS if p not in done_presets]
    
    print(f"🔍 Iter9 Ablation Sweep")
    print(f"   Done: {len(done_presets)}, Remaining: {len(remaining)}, Total: {len(PRESETS)}")
    print(f"   Timeout per preset: {PER_PRESET_TIMEOUT}s")
    print()
    
    if VIDEO not in data["results"]:
        data["results"][VIDEO] = {}
    
    for i, preset in enumerate(remaining):
        print(f"[{len(done_presets)+i+1}/{len(PRESETS)}] {preset} ... ", end="", flush=True)
        metrics = run_preset(preset)
        data["results"][VIDEO][preset] = metrics
        save_results(data)  # incremental save
        
        status = "✅" if metrics.get("exit_code") == 0 else "❌"
        extra = ""
        if "psnr" in metrics: extra += f" PSNR={metrics['psnr']:.1f}"
        if "ssim" in metrics: extra += f" SSIM={metrics['ssim']:.3f}"
        if "score" in metrics: extra += f" score={metrics['score']:.1f}"
        if "error" in metrics: extra += f" ({metrics['error']})"
        print(f"{status} {metrics.get('time_sec', '?')}s{extra}")
    
    # Final ranking
    print("\n" + "="*60)
    print("📊 RANKING (by speed-adjusted score)")
    print("="*60)
    
    results = data["results"].get(VIDEO, {})
    ranked = []
    for name, m in results.items():
        if m.get("exit_code") != 0: continue
        score = m.get("psnr", 0) * 0.4 + m.get("ssim", 0) * 40
        speed_penalty = max(0, m.get("time_sec", 0) - 5) * 0.1
        ranked.append((name, round(score - speed_penalty, 2), m))
    
    ranked.sort(key=lambda x: -x[1])
    for i, (name, score, m) in enumerate(ranked[:20], 1):
        print(f"  {i:2d}. {name:<25s} adj={score:<7.2f} t={m.get('time_sec',0):.1f}s")

if __name__ == "__main__":
    main()
