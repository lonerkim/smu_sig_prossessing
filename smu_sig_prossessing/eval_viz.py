"""
평가 결과 시각화 — radar chart, bar chart, 비교 그리드.
matplotlib 없이 순수 OpenCV로 렌더링 (의존성 최소화).
"""
from __future__ import annotations

import os
import math
import numpy as np
import cv2
from typing import Optional
from .auto_evaluation import EvalResult


# ─── Color palette ────────────────────────────────────────────────

COLORS = [
    (100, 200, 100),   # green
    (100, 150, 255),   # orange
    (255, 150, 100),   # blue
    (255, 100, 255),   # pink
    (100, 255, 255),   # yellow
    (200, 100, 100),   # red
    (150, 150, 255),   # light blue
    (255, 255, 100),   # cyan
]

BG_COLOR = (30, 30, 30)       # dark background
TEXT_COLOR = (220, 220, 220)
GRID_COLOR = (60, 60, 60)
ACCENT_COLOR = (80, 180, 80)  # green accent


def _put_text(img: np.ndarray, text: str, pos: tuple,
              font_scale: float = 0.5, color: tuple = TEXT_COLOR,
              thickness: int = 1):
    """Helper for text rendering."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


# ─── Radar Chart ──────────────────────────────────────────────────

def draw_radar_chart(results: list[EvalResult],
                     output_path: str,
                     size: int = 800) -> str:
    """
    다중 preset 비교 radar chart.
    축: PSNR, SSIM, Color, Edge, Detail (5각형).
    Noise와 Artifact는 "높을수록 좋음"으로 반전.
    """
    # 사용할 메트릭 (radar에 표시할 5개 핵심)
    radar_metrics = [
        ("psnr", "PSNR", "higher"),
        ("ssim", "SSIM", "higher"),
        ("color_fidelity", "Color", "lower"),     # 반전
        ("edge_retention", "Edge", "higher"),
        ("detail_recovery", "Detail", "higher"),
    ]

    n_axes = len(radar_metrics)
    canvas = np.full((size, size, 3), BG_COLOR, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    radius = size // 2 - 80

    # 각 메트릭의 min/max 정규화 범위
    ranges = {
        "psnr": (10, 35),
        "ssim": (0.3, 1.0),
        "color_fidelity": (0, 20),  # 반전
        "edge_retention": (0, 2.0),
        "detail_recovery": (0, 2.0),
    }

    # 축 그리기
    angles = [2 * math.pi * i / n_axes - math.pi / 2 for i in range(n_axes)]
    for i, angle in enumerate(angles):
        x = int(cx + radius * math.cos(angle))
        y = int(cy + radius * math.sin(angle))
        cv2.line(canvas, (cx, cy), (x, y), GRID_COLOR, 1)
        # 라벨
        lx = int(cx + (radius + 30) * math.cos(angle))
        ly = int(cy + (radius + 30) * math.sin(angle))
        _put_text(canvas, radar_metrics[i][1], (lx - 20, ly + 5), 0.4)

    # 동심원 (25%, 50%, 75%, 100%)
    for pct in [0.25, 0.5, 0.75, 1.0]:
        r = int(radius * pct)
        pts = []
        for angle in angles:
            px = int(cx + r * math.cos(angle))
            py = int(cy + r * math.sin(angle))
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [pts], True, GRID_COLOR, 1)

    # 각 preset 폴리곤
    for idx, result in enumerate(results[:8]):  # max 8
        color = COLORS[idx % len(COLORS)]
        points = []
        for i, (mkey, mlabel, best) in enumerate(radar_metrics):
            m = result.get(mkey)
            if m is None:
                val = 0.5
            else:
                lo, hi = ranges[mkey]
                norm = (m.value - lo) / max(hi - lo, 1e-6)
                norm = np.clip(norm, 0, 1)
                if best == "lower":
                    norm = 1.0 - norm  # 반전
                val = norm

            r = radius * val
            px = int(cx + r * math.cos(angles[i]))
            py = int(cy + r * math.sin(angles[i]))
            points.append([px, py])

        pts = np.array(points, dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
        cv2.polylines(canvas, [pts], True, color, 2)

    # 범례
    ly = size - 25 * len(results) - 20
    for idx, result in enumerate(results[:8]):
        color = COLORS[idx % len(COLORS)]
        cv2.rectangle(canvas, (15, ly + idx * 22), (30, ly + idx * 22 + 14), color, -1)
        _put_text(canvas, f"{result.label} ({result.composite_score:.1f})",
                  (35, ly + idx * 22 + 13), 0.4)

    # 타이틀
    _put_text(canvas, "Quality Radar", (size // 2 - 50, 25), 0.6, ACCENT_COLOR, 2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, canvas)
    return output_path


# ─── Bar Chart ────────────────────────────────────────────────────

def draw_bar_chart(results: list[EvalResult],
                   output_path: str,
                   width: int = 1200, height: int = 600) -> str:
    """
    메트릭별 바 차트 (preset별 그룹바).
    """
    metric_keys = ["psnr", "ssim", "color_fidelity", "edge_retention",
                   "noise_level", "detail_recovery", "artifact_score"]
    metric_labels = ["PSNR\n(dB)", "SSIM", "ΔE", "Edge", "Noise", "Detail", "Artifact"]
    # 정규화 범위
    norm_ranges = {
        "psnr": (10, 35),
        "ssim": (0, 1),
        "color_fidelity": (0, 20),
        "edge_retention": (0, 2),
        "noise_level": (0, 2000),
        "detail_recovery": (0, 2),
        "artifact_score": (0, 30),
    }

    canvas = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)

    n_metrics = len(metric_keys)
    n_presets = len(results)
    margin_left = 60
    margin_bottom = 100
    margin_top = 60
    margin_right = 30
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    group_w = chart_w / n_metrics
    bar_w = min(group_w * 0.7 / max(n_presets, 1), 40)
    bar_gap = 2

    # Y축 (0~1 normalized)
    cv2.line(canvas, (margin_left, margin_top),
             (margin_left, height - margin_bottom), GRID_COLOR, 1)
    for pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = int(margin_top + chart_h * (1 - pct))
        cv2.line(canvas, (margin_left, y), (width - margin_right, y), GRID_COLOR, 1)
        _put_text(canvas, f"{pct:.0%}", (5, y + 5), 0.35, (150, 150, 150))

    # 바 그리기
    for mi, mkey in enumerate(metric_keys):
        gx = margin_left + mi * group_w
        # X축 라벨
        label_lines = metric_labels[mi].split("\n")
        for li, line in enumerate(label_lines):
            _put_text(canvas, line,
                      (int(gx + group_w / 2 - 15), height - margin_bottom + 20 + li * 18),
                      0.35)

        lo, hi = norm_ranges[mkey]
        for pi, result in enumerate(results[:8]):
            m = result.get(mkey)
            if m is None:
                norm_val = 0
            else:
                norm_val = (m.value - lo) / max(hi - lo, 1e-6)
                norm_val = np.clip(norm_val, 0, 1)

            bar_h = int(chart_h * norm_val)
            x1 = int(gx + (group_w - bar_w * n_presets) / 2 + pi * (bar_w + bar_gap))
            y1 = margin_top + chart_h - bar_h
            y2 = margin_top + chart_h

            color = COLORS[pi % len(COLORS)]
            cv2.rectangle(canvas, (x1, y1), (x1 + int(bar_w), y2), color, -1)
            # 값 표시
            if m is not None:
                val_str = f"{m.value:.1f}" if m.value > 10 else f"{m.value:.2f}"
                _put_text(canvas, val_str, (x1 - 2, y1 - 5), 0.28, color)

    # 범례
    for pi, result in enumerate(results[:8]):
        color = COLORS[pi % len(COLORS)]
        lx = margin_left + pi * 130
        cv2.rectangle(canvas, (lx, height - 25), (lx + 12, height - 13), color, -1)
        _put_text(canvas, result.label, (lx + 16, height - 14), 0.35)

    # 타이틀
    _put_text(canvas, "Metric Comparison (Normalized)", (width // 2 - 120, 25),
              0.6, ACCENT_COLOR, 2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, canvas)
    return output_path


# ─── Comparison Grid ──────────────────────────────────────────────

def draw_comparison_grid(images: dict[str, np.ndarray],
                         reference: np.ndarray,
                         output_path: str,
                         max_cols: int = 4,
                         zoom_region: Optional[tuple] = None) -> str:
    """
    여러 결과물을 그리드 배치.

    Parameters
    ----------
    images : {label: image} dict
    reference : 원본 이미지 (첫 번째 셀에 배치)
    zoom_region : (x, y, w, h) 확대 영역 (선택)
    """
    n = len(images) + 1  # +1 for reference
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    h, w = reference.shape[:2]
    # resize if too large
    max_cell_w = 400
    if w > max_cell_w:
        scale = max_cell_w / w
        w = int(w * scale)
        h = int(h * scale)

    cell_h = h + 30  # +30 for label
    canvas_h = rows * cell_h + 10
    canvas_w = cols * (w + 10) + 10
    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)

    def _place(img: np.ndarray, label: str, idx: int):
        r, c = divmod(idx, cols)
        resized = cv2.resize(img, (w, h))
        x0 = c * (w + 10) + 5
        y0 = r * cell_h + 5
        canvas[y0:y0 + h, x0:x0 + w] = resized
        color = TEXT_COLOR
        if idx == 0:
            color = ACCENT_COLOR
        _put_text(canvas, label, (x0 + 5, y0 + h + 22), 0.4, color)

    # Reference
    _place(reference, "Reference", 0)

    # Results
    for i, (label, img) in enumerate(images.items()):
        if img.shape[:2] != reference.shape[:2]:
            img = cv2.resize(img, (reference.shape[1], reference.shape[0]))
        _place(img, label, i + 1)

    # Zoom inset (optional)
    if zoom_region is not None:
        zx, zy, zw, zh = zoom_region
        for idx in range(n):
            src = reference if idx == 0 else list(images.values())[idx - 1]
            src_r = cv2.resize(src, (w, h))
            # scale coordinates
            sx = int(zx * w / reference.shape[1])
            sy = int(zy * h / reference.shape[0])
            sw = int(zw * w / reference.shape[1])
            sh = int(zh * h / reference.shape[0])
            region = src_r[max(0, sy):sy + sh, max(0, sx):sx + sw]
            if region.size > 0:
                region = cv2.resize(region, (zw, zh), interpolation=cv2.INTER_NEAREST)
                # Place in top-right corner of cell
                r, c = divmod(idx, cols)
                x0 = c * (w + 10) + 5 + w - zw - 5
                y0 = r * cell_h + 5 + 5
                canvas[y0:y0 + zh, x0:x0 + zw] = region
                cv2.rectangle(canvas, (x0, y0), (x0 + zw, y0 + zh), (0, 255, 255), 1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, canvas)
    return output_path


# ─── Qualitative Evaluation Sheet ─────────────────────────────────

def create_eval_sheet(origin: np.ndarray,
                      degraded: np.ndarray,
                      results: dict[str, np.ndarray],
                      output_path: str,
                      zoom_regions: list[tuple] | None = None) -> str:
    """
    정성 평가용 시트 — 각 preset의 결과를 나란히 배치 + zoom 영역.
    유저가 직접 보고 코멘트 달 수 있도록 구성.
    """
    h, w = origin.shape[:2]
    # 축소
    scale = min(1.0, 400 / w)
    sw, sh = int(w * scale), int(h * scale)

    n = len(results) + 2  # origin + degraded + results
    cols = min(5, n)
    rows = math.ceil(n / cols)

    cell_h = sh + 35
    canvas = np.full((rows * cell_h + 50, cols * (sw + 10) + 10, 3),
                     BG_COLOR, dtype=np.uint8)

    items = [("✅ Original", origin), ("❌ Degraded", degraded)]
    for label, img in results.items():
        items.append((f"🔧 {label}", img))

    for idx, (label, img) in enumerate(items):
        r, c = divmod(idx, cols)
        resized = cv2.resize(img, (sw, sh))
        x0 = c * (sw + 10) + 5
        y0 = r * cell_h + 5
        canvas[y0:y0 + sh, x0:x0 + sw] = resized
        color = ACCENT_COLOR if idx == 0 else (100, 100, 255) if idx == 1 else TEXT_COLOR
        _put_text(canvas, label, (x0 + 5, y0 + sh + 22), 0.35, color)

    # Zoom insets
    if zoom_regions:
        for zidx, (zx, zy, zw, zh) in enumerate(zoom_regions[:3]):
            zoom_size = min(150, sw // 2)
            for idx in range(len(items)):
                src = items[idx][1]
                region = src[max(0, zy):zy + zh, max(0, zx):zx + zw]
                if region.size == 0:
                    continue
                region = cv2.resize(region, (zoom_size, zoom_size),
                                    interpolation=cv2.INTER_NEAREST)
                r, c = divmod(idx, cols)
                x0 = c * (sw + 10) + 5 + sw - zoom_size - 5
                y0 = r * cell_h + 5 + zidx * (zoom_size + 5) + 5
                if y0 + zoom_size < r * cell_h + sh + 5:
                    canvas[y0:y0 + zoom_size, x0:x0 + zoom_size] = region
                    cv2.rectangle(canvas, (x0, y0),
                                  (x0 + zoom_size, y0 + zoom_size),
                                  (0, 255, 255), 1)

    _put_text(canvas, "Qualitative Evaluation Sheet",
              (canvas.shape[1] // 2 - 120, 25), 0.5, ACCENT_COLOR, 2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, canvas)
    return output_path
