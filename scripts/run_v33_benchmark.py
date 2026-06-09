#!/usr/bin/env python3
"""v3.3 benchmark — focus on new presets, skip slow BM3D variants."""
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

    # Skip BM3D-heavy presets that take 8+ seconds each
    slow_presets = {"bm3d-denoise", "bm3d-fast", "retinex-bm3d", "retinex-bm3d-msrcr", "bm4d-temporal"}
    skip = {"adaptive"} | slow_presets

    results = []
    for pname in sorted(PRESETS.keys()):
        if pname in skip:
            print(f"  {pname:25s}  SKIP (slow/adaptive)")
            continue
        try:
            cfg = PRESETS[pname]
            t0 = time.perf_counter()
            restored = pl.apply_pipeline(degraded, cfg)
            t = time.perf_counter() - t0
            res = evaluator.evaluate(small, restored, label=pname, degraded=degraded, verbose=False)
            res.notes = f"{t:.4f}"
            results.append(res)
            arrow = "🆕" if pname in ["super-premium", "super-premium-fast", "rolling-premium",
                                        "temporal-premium", "ultralight", "chroma-focus"] else "  "
            print(f"  {arrow} {pname:25s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
                  f"SSIM={res.get('ssim').value:.4f}  ΔE={res.get('color_fidelity').value:.2f}  "
                  f"VIF={res.get('vif').value:.4f}  {t*1000:5.1f}ms")
        except Exception as e:
            print(f"  ❌ {pname:25s}  ERROR: {e}")
            import traceback; traceback.print_exc()

    results.sort(key=lambda r: r.composite_score, reverse=True)
    
    print(f"\n{'='*80}")
    print("TOP 10 PRESETS (v3.3 benchmark)")
    print(f"{'Rank':<5} {'Preset':<25} {'Score':<7} {'PSNR':<7} {'SSIM':<7} {'ΔE²⁰⁰⁰':<7} {'VIF':<7} {'Time':<7}")
    print("-"*65)
    for i, r in enumerate(results[:10]):
        m = {m.name: m.value for m in r.metrics}
        is_new = "🆕" if r.label in ["super-premium", "super-premium-fast", "rolling-premium",
                                       "temporal-premium", "ultralight", "chroma-focus"] else "  "
        print(f"  {is_new} #{i+1:<2} {r.label:<25} {r.composite_score:<7.2f} "
              f"{m.get('psnr',0):<7.2f} {m.get('ssim',0):<7.4f} "
              f"{m.get('color_fidelity',0):<7.2f} {m.get('vif',0):<7.4f} "
              f"{float(r.notes)*1000:<7.1f}ms")
    
    # BM4D and BM3D variants - tested separately
    print(f"\n{'='*80}")
    print("SLOW PRESETS (tested separately):")
    for pname in ["bm4d-temporal", "bm3d-denoise", "bm3d-fast"]:
        try:
            cfg = PRESETS[pname]
            t0 = time.perf_counter()
            restored = pl.apply_pipeline(degraded, cfg)
            t = time.perf_counter() - t0
            res = evaluator.evaluate(small, restored, label=pname, degraded=degraded, verbose=False)
            print(f"  {pname:25s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
                  f"SSIM={res.get('ssim').value:.4f}  ΔE={res.get('color_fidelity').value:.2f}  "
                  f"{t*1000:5.1f}ms")
        except Exception as e:
            print(f"  ❌ {pname:25s}  ERROR: {e}")

    # NTSC test with new presets
    print(f"\n{'='*80}")
    print("NTSC HEAVY (new presets):")
    ntsc_deg = degrade_image(small, use_ntsc=True, ntsc_intensity="heavy")
    for pname in ["temporal-premium", "super-premium", "super-premium-fast", 
                  "rolling-premium", "chroma-focus",
                  "video-enhanced", "wavelet-denoise"]:
        try:
            cfg = PRESETS[pname]
            restored = pl.apply_pipeline(ntsc_deg, cfg)
            res = evaluator.evaluate(small, restored, label=pname, degraded=ntsc_deg, verbose=False)
            print(f"  🆕 {pname:25s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}  "
                  f"SSIM={res.get('ssim').value:.4f}  ΔE={res.get('color_fidelity').value:.2f}")
        except Exception as e:
            print(f"  ❌ {pname:25s}  ERROR: {e}")
    
    # Strength 0.3 test (easier noise)
    print(f"\n{'='*80}")
    print("STRENGTH=0.3 (new presets on mild noise):")
    mild_deg = degrade_image(small, use_ntsc=False, strength=0.3)
    for pname in ["super-premium", "super-premium-fast", "temporal-premium", "fast-premium", "video-enhanced"]:
        try:
            cfg = PRESETS[pname]
            restored = pl.apply_pipeline(mild_deg, cfg)
            res = evaluator.evaluate(small, restored, degraded=mild_deg, verbose=False)
            print(f"  🆕 {pname:25s}  Score={res.composite_score:6.2f}  PSNR={res.get('psnr').value:5.2f}")
        except Exception as e:
            print(f"  ❌ {pname:25s}  ERROR: {e}")

    print(f"\n✅ v3.3 benchmark complete")
    print(f"Results: {os.path.join(OUT_DIR, 'auto_eval_ranking.csv')}")

if __name__ == "__main__":
    main()
