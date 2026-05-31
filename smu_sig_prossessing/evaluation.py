"""
Evaluation utilities — PSNR/SSIM, visual comparison, histogram analysis.
"""
from __future__ import annotations

import os
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ─── Metrics ────────────────────────────────────────────────────────

def calculate_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """PSNR between two images."""
    return float(psnr(original, processed))


def calculate_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """SSIM between two images."""
    return float(ssim(original, processed, channel_axis=-1
                      if len(original.shape) == 3 else None))


def evaluate(original: np.ndarray, processed: np.ndarray,
             label: str = "", verbose: bool = True) -> dict:
    """Full evaluation returning PSNR and SSIM."""
    # Match shapes
    if original.shape != processed.shape:
        if len(original.shape) == 2 and len(processed.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        elif len(original.shape) == 3 and len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    p = psnr(original, processed)
    s = ssim(original, processed, channel_axis=-1
             if len(original.shape) == 3 else None)
    if verbose:
        print(f"  {label:30s}  PSNR={p:.2f} dB   SSIM={s:.4f}")
    return {"label": label, "psnr": round(float(p), 2),
            "ssim": round(float(s), 4)}


# ─── Visualization ──────────────────────────────────────────────────

def save_comparison(original: np.ndarray, degraded: np.ndarray,
                    processed: np.ndarray, path: str,
                    labels: tuple | None = None) -> str:
    """Side-by-side comparison: original | degraded | processed."""
    if labels is None:
        labels = ("Original", "Degraded", "Processed")

    h, w = original.shape[:2]
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, :w] = original
    canvas[:, w:2*w] = degraded
    canvas[:, 2*w:] = processed

    for i, label in enumerate(labels):
        cv2.putText(canvas, label, (i * w + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, canvas)
    return path


def save_four_comparison(original: np.ndarray, degraded: np.ndarray,
                         denoised: np.ndarray, final: np.ndarray,
                         path: str) -> str:
    """Four-way comparison: original | degraded | denoised | final."""
    h, w = original.shape[:2]
    canvas = np.zeros((h, w * 4, 3), dtype=np.uint8)
    canvas[:, :w] = original
    canvas[:, w:2*w] = degraded
    canvas[:, 2*w:3*w] = denoised
    canvas[:, 3*w:] = final

    labels = ["Original", "Degraded", "Denoised", "Final"]
    for i, label in enumerate(labels):
        cv2.putText(canvas, label, (i * w + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, canvas)
    return path


def save_histogram_comparison(before: np.ndarray, after: np.ndarray,
                               path: str) -> str:
    """RGB histogram comparison: before vs after."""
    fig_w = 600
    fig_h = 400
    canvas = np.zeros((fig_h, fig_w * 2, 3), dtype=np.uint8)

    for idx, (img, label) in enumerate([(before, "Before"), (after, "After")]):
        x_off = idx * fig_w
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]  # BGR
        for c in range(3):
            hist = cv2.calcHist([img], [c], None, [256], [0, 256])
            hist = hist / hist.max() * 150 if hist.max() > 0 else hist
            pts = []
            for i in range(256):
                px = x_off + 20 + int(i * (fig_w - 40) / 256)
                py = 300 - int(hist[i][0])
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, colors[c], 1)
        cv2.putText(canvas, label, (x_off + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, canvas)
    return path


# ─── Batch evaluation ───────────────────────────────────────────────

def evaluate_batch(originals: dict[str, np.ndarray],
                   processed: dict[str, np.ndarray],
                   config_label: str = "") -> list[dict]:
    """Evaluate multiple images and return results table."""
    all_results = []
    for name in originals:
        if name not in processed:
            continue
        r = evaluate(originals[name], processed[name],
                     f"{config_label} [{name}]")
        all_results.append(r)
    return all_results


def print_summary(results: list[dict]) -> None:
    """Print a formatted summary table."""
    print(f"\n{'=' * 60}")
    print("📊 Results Summary")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  {r['label']:30s}  PSNR={r['psnr']:6.2f}  SSIM={r['ssim']:.4f}")
    if results:
        avg_psnr = np.mean([r["psnr"] for r in results])
        avg_ssim = np.mean([r["ssim"] for r in results])
        print(f"\n  {'AVERAGE':30s}  PSNR={avg_psnr:6.2f}  SSIM={avg_ssim:.4f}")
