#!/usr/bin/env python3
"""
Iter9 — Verify updated presets.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.auto_evaluation import AutoEvaluator
from smu_sig_prossessing.degradation import degrade_image
from main import PRESETS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
evaluator = AutoEvaluator()

img = cv2.imread(os.path.join(BASE, "input", "test_small.jpg"))
small = cv2.resize(img, (400, 185))
degraded = degrade_image(small, use_ntsc=False, strength=0.5)
ntsc_deg = degrade_image(small, use_ntsc=True, ntsc_intensity="heavy")

# Test temporal-ntsc + updated temporal-premium
test_sets = {
    "basic(0.5)": {"deg": degraded, "presets": ["temporal-premium", "temporal-ntsc", "chroma-focus", "wavelet-denoise", "video-enhanced"]},
    "NTSC-heavy": {"deg": ntsc_deg, "presets": ["temporal-ntsc", "temporal-premium", "chroma-focus", "wavelet-denoise", "ntsc-plus"]},
}

for test_name, cfg in test_sets.items():
    print(f"\n{'='*60}")
    print(f"  {test_name}")
    print(f"{'='*60}")
    print(f"{'Preset':<25s} {'Score':<8s} {'PSNR':<8s} {'SSIM':<8s} {'DE':<8s} {'Time':<8s}")
    print("-" * 65)
    for pname in cfg["presets"]:
        try:
            t0 = time.perf_counter()
            restored = pl.apply_pipeline(cfg["deg"], PRESETS[pname])
            t = time.perf_counter() - t0
            res = evaluator.evaluate(small, restored, label=pname, degraded=cfg["deg"], verbose=False)
            print(f"  {pname:<25s} {res.composite_score:<8.2f} "
                  f"{res.get('psnr').value:<8.2f} {res.get('ssim').value:<8.4f} "
                  f"{res.get('color_fidelity').value:<8.2f} {t*1000:<8.1f}ms")
        except Exception as e:
            print(f"  ❌ {pname:<25s} ERROR: {e}")

print(f"\n✅ Verification done.")
