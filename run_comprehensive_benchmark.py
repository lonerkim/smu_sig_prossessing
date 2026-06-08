#!/usr/bin/env python3
"""Comprehensive benchmark with all 8 metrics (CIEDE2000, VIF, etc.)"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.degradation import degrade_image
from smu_sig_prossessing.auto_evaluation import AutoEvaluator
from main import PRESETS

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output", "optimization")
os.makedirs(OUT_DIR, exist_ok=True)

evaluator = AutoEvaluator()

def main():
    img = cv2.imread(os.path.join(BASE, "input", "test_small.jpg"))
    if img is None:
        print("ERROR: cannot load image"); return
    small = cv2.resize(img, (400, 185))
    degraded = degrade_image(small, use_ntsc=False, strength=0.5)

    skip = {"adaptive"}
    results = []
    for pname in sorted(PRESETS.keys()):
        if pname in skip: continue
        try:
            cfg = PRESETS[pname]
            t0 = time.perf_counter()
            restored = pl.apply_pipeline(degraded, cfg)
            t = time.perf_counter() - t0
            res = evaluator.evaluate(small, restored, label=pname, degraded=degraded, verbose=False)
            res.notes = f"{t:.4f}"
            results.append(res)
            print(f"  {pname:25s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
                  f"SSIM={res.get('ssim').value:.4f}  ΔE={res.get('color_fidelity').value:.2f}  "
                  f"VIF={res.get('vif').value:.4f}  {t*1000:5.1f}ms")
        except Exception as e:
            print(f"  {pname:25s}  ERROR: {e}")

    results.sort(key=lambda r: r.composite_score, reverse=True)
    
    # Save CSV
    csv_path = os.path.join(OUT_DIR, "auto_eval_ranking.csv")
    import csv as csv_mod
    with open(csv_path, "w", newline="") as f:
        wtr = csv_mod.writer(f)
        wtr.writerow(["rank", "preset", "composite", "psnr", "ssim", "ciede2000",
                       "edge_ret", "noise_lvl", "detail_rec", "artifact", "vif", "time_s"])
        for i, r in enumerate(results):
            m = {m.name: m.value for m in r.metrics}
            wtr.writerow([i+1, r.label, round(r.composite_score, 2)] +
                        [round(m.get(k, 0), 4) for k in 
                         ["psnr","ssim","color_fidelity","edge_retention",
                          "noise_level","detail_recovery","artifact_score","vif"]] +
                        [r.notes])
    print(f"\nSaved: {csv_path}")
    
    # Print top 10
    print(f"\n{'='*80}")
    print("TOP 10 PRESETS (by Composite Score with CIEDE2000 + VIF)")
    print(f"{'Rank':<5} {'Preset':<25} {'Score':<7} {'PSNR':<7} {'SSIM':<7} {'ΔE²⁰⁰⁰':<7} {'VIF':<7} {'Time':<7}")
    print("-"*65)
    for i, r in enumerate(results[:10]):
        m = {m.name: m.value for m in r.metrics}
        print(f"  #{i+1:<2} {r.label:<25} {r.composite_score:<7.2f} "
              f"{m.get('psnr',0):<7.2f} {m.get('ssim',0):<7.4f} "
              f"{m.get('color_fidelity',0):<7.2f} {m.get('vif',0):<7.4f} "
              f"{float(r.notes)*1000:<7.1f}ms")

    # BM3D sweep
    print(f"\n{'='*80}")
    print("BM3D SWEEP (via bm3d filter):")
    bm3d_results = []
    for s in [5, 10, 15, 20, 25, 30]:
        try:
            from smu_sig_prossessing.config import PipelineConfig
            cfg = PipelineConfig(label=f"bm3d_{s}")
            cfg.add("bm3d", sigma_psd=s)
            cfg.add("channel_correction")
            t0 = time.perf_counter()
            restored = pl.apply_pipeline(degraded, cfg)
            t = time.perf_counter() - t0
            res = evaluator.evaluate(small, restored, label=f"bm3d_{s}", degraded=degraded, verbose=False)
            bm3d_results.append((s, res, t))
            print(f"  σ={s:3d}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
                  f"SSIM={res.get('ssim').value:.4f}  {t*1000:5.1f}ms")
        except Exception as e:
            print(f"  σ={s:3d}  ERROR: {e}")

    # NTSC test (now fixed)
    print(f"\n{'='*80}")
    print("NTSC HEAVY (dimension mismatch fixed):")
    ntsc_deg = degrade_image(small, use_ntsc=True, ntsc_intensity="heavy")
    for pname in ["wavelet-denoise", "bm3d-denoise", "video-enhanced", 
                  "optimized-fast", "edge-preserve", "retinex-bm3d"]:
        try:
            cfg = PRESETS[pname]
            restored = pl.apply_pipeline(ntsc_deg, cfg)
            res = evaluator.evaluate(small, restored, label=pname, degraded=ntsc_deg, verbose=False)
            print(f"  {pname:25s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
                  f"SSIM={res.get('ssim').value:.4f}  ΔE={res.get('color_fidelity').value:.2f}")
        except Exception as e:
            print(f"  {pname:25s}  ERROR: {e}")

    # Strength ablation
    print(f"\n{'='*80}")
    print("STRENGTH SCALING:")
    for s in [0.3, 0.5, 0.7, 1.0]:
        deg = degrade_image(small, use_ntsc=False, strength=s)
        for pname in ["optimized-fast", "video-enhanced", "bm3d-denoise", "retinex-bm3d"]:
            try:
                cfg = PRESETS[pname]
                restored = pl.apply_pipeline(deg, cfg)
                res = evaluator.evaluate(small, restored, degraded=deg, verbose=False)
                print(f"  s={s:.1f} {pname:25s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}")
            except Exception as e:
                print(f"  s={s:.1f} {pname:25s}  ERROR: {e}")

    print(f"\n✅ Benchmark complete. Results in: {csv_path}")

if __name__ == "__main__":
    main()
