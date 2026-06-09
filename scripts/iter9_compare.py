#!/usr/bin/env python3
"""
Generate a self-contained HTML page comparing all presets on a test image
with score overlays (PSNR, SSIM, NIQE).
"""
import sys, os, base64, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.degradation import degrade_image
from smu_sig_prossessing.auto_evaluation import AutoEvaluator
from main import PRESETS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")
os.makedirs(OUTPUT, exist_ok=True)

def img_to_data_url(img):
    """Convert OpenCV image (BGR) to base64 data URL."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64 = base64.b64encode(buf).decode()
    return f"data:image/jpeg;base64,{b64}"

def main():
    img = cv2.imread(os.path.join(BASE, "input", "test_small.jpg"))
    if img is None:
        print("ERROR: cannot load test_small.jpg")
        return
    small = cv2.resize(img, (400, 185))
    degraded = degrade_image(small, use_ntsc=False, strength=0.5)

    evaluator = AutoEvaluator()
    rows = []
    excluded = {"adaptive"}  # skip adaptive (needs special handling)

    for pname in sorted(PRESETS.keys()):
        if pname in excluded:
            continue
        try:
            cfg = PRESETS[pname]
            restored = pl.apply_pipeline(degraded, cfg)
            res = evaluator.evaluate(small, restored, label=pname, degraded=degraded, verbose=False)
            n = res.get("niqe")
            niqe_val = n.value if n else 0
            rows.append((
                pname,
                img_to_data_url(small),
                img_to_data_url(degraded),
                img_to_data_url(restored),
                res.composite_score,
                res.get("psnr").value,
                res.get("ssim").value,
                niqe_val,
                round(float(res.get("color_fidelity").value), 2) if res.get("color_fidelity") else 0,
            ))
        except Exception as e:
            print(f"  ⚠ {pname}: {e}")

    rows.sort(key=lambda r: r[4], reverse=True)

    # Build HTML
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Iter9 — Preset Comparison (test_small.jpg)</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #0d1117; color: #c9d1d9; padding: 20px; }
h1 { color: #58a6ff; margin-bottom: 5px; }
.subtitle { color: #8b949e; margin-bottom: 20px; }
table { border-collapse: collapse; width: 100%; }
th { position: sticky; top: 0; background: #161b22; padding: 10px 8px;
     text-align: center; font-size: 13px; color: #8b949e; border-bottom: 2px solid #30363d; }
td { padding: 6px 8px; text-align: center; border-bottom: 1px solid #21262d; vertical-align: top; }
tr:hover { background: #161b22; }
img { max-width: 400px; height: auto; border-radius: 4px; }
.label { font-weight: 600; color: #58a6ff; text-align: left; }
.score { font-weight: 700; }
.score-low { color: #f85149; }
.score-mid { color: #d29922; }
.score-high { color: #3fb950; }
.rank-badge { display: inline-block; width: 24px; height: 24px; line-height: 24px;
              border-radius: 50%; text-align: center; font-size: 12px; font-weight: 700; }
.gold { background: #d29922; color: #0d1117; }
.silver { background: #8b949e; color: #0d1117; }
.bronze { background: #a371f7; color: #0d1117; }
</style>
</head>
<body>
<h1>🖼️ Iter9 — All Preset Comparison</h1>
<p class="subtitle">test_small.jpg (400x185) | basic degrade strength=0.5 | Sorted by Composite Score</p>
<table>
<thead><tr>
<th>Rank</th><th>Preset</th><th>Original</th><th>Degraded</th><th>Restored</th>
<th>Score</th><th>PSNR</th><th>SSIM</th><th>NIQE</th><th>ΔE</th>
</tr></thead>
<tbody>
"""
    for i, (name, orig_url, deg_url, rest_url, score, psnr, ssim, niqe, de) in enumerate(rows):
        rank_class = ""
        rank_badge = str(i+1)
        if i == 0:
            rank_class = "gold"
            rank_badge = "🥇"
        elif i == 1:
            rank_class = "silver"
            rank_badge = "🥈"
        elif i == 2:
            rank_class = "bronze"
            rank_badge = "🥉"

        def score_class(v, higher_better=True):
            if higher_better:
                if v >= 55: return "score-high"
                if v >= 45: return "score-mid"
                return "score-low"
            else:
                if v <= 8: return "score-high"
                if v <= 14: return "score-mid"
                return "score-low"

        html += f"<tr>"
        html += f'<td><span class="rank-badge {rank_class}">{rank_badge}</span></td>'
        html += f'<td class="label">{name}</td>'
        html += f'<td><img src="{orig_url}" alt="original"></td>'
        html += f'<td><img src="{deg_url}" alt="degraded"></td>'
        html += f'<td><img src="{rest_url}" alt="restored"></td>'
        html += f'<td class="score {score_class(score)}">{score:.1f}</td>'
        html += f'<td class="{score_class(psnr)}">{psnr:.1f}</td>'
        html += f'<td class="{score_class(ssim)}">{ssim:.4f}</td>'
        html += f'<td class="{score_class(niqe, False)}">{niqe:.2f}</td>'
        html += f'<td class="{score_class(de, False)}">{de:.1f}</td>'
        html += f"</tr>\n"

    html += """</tbody></table>
</body>
</html>"""

    path = os.path.join(OUTPUT, "iter9_comparison.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"✅ Comparison page → {path}")
    print(f"   {len(rows)} presets compared")
    print(f"   🥇 1st: {rows[0][0]}")
    print(f"   🥈 2nd: {rows[1][0]}")
    print(f"   🥉 3rd: {rows[2][0]}")

if __name__ == "__main__":
    main()
