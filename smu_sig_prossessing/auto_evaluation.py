"""
자동 정량 평가 모듈 — PSNR/SSIM 외 8개 메트릭 자동 산출.

메트릭 구성:
  1. PSNR            — Peak Signal-to-Noise Ratio (dB, 높을수록 좋음)
  2. SSIM            — Structural Similarity (0~1, 높을수록 좋음)
  3. Color Fidelity  — CIEDE2000 ΔE 평균 (0=완전일치, 낮을수록 좋음)
  4. Edge Retention  — Canny edge 픽셀 비율 (1.0=원본과 동일 edge량)
  5. Noise Level     — Laplacian 분산 (낮을수록 깨끗함)
  6. Detail Recovery — 고주파 에너지 보존율 (1.0=완전 보존)
  7. Artifact Score  — ringing/blocking 의심 점수 (낮을수록 좋음)
  8. VIF             — Visual Information Fidelity (0~1, 높을수록 좋음)

Composite Quality Score = 가중 평균 (0~100 점)

사용법:
    from smu_sig_prossessing.auto_evaluation import AutoEvaluator
    ev = AutoEvaluator()
    scores = ev.evaluate(origin, processed, label="optimized-fast")
    ev.compare_presets(origin, degraded, results_dict)
"""
from __future__ import annotations

import os
import json
import csv
import numpy as np
import cv2
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

# NIQE integration — add scripts dir to path for standalone use
import sys as _sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_scripts_dir = os.path.join(_project_root, "scripts")
if _scripts_dir not in _sys.path:
    _sys.path.insert(0, _scripts_dir)


@dataclass
class MetricResult:
    """단일 메트릭 결과."""
    name: str
    value: float
    unit: str
    description: str
    best: str = "lower"  # "lower"=낮을수록 좋음, "higher"=높을수록 좋음

    @property
    def display(self) -> str:
        arrow = "↑" if self.best == "higher" else "↓"
        return f"{self.name}: {self.value:.3f}{arrow} ({self.unit})"


@dataclass
class EvalResult:
    """전체 평가 결과 (6개 메트릭 + composite)."""
    label: str = ""
    timestamp: str = ""
    metrics: list[MetricResult] = field(default_factory=list)
    composite_score: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metrics"] = [asdict(m) for m in self.metrics]
        return d

    def get(self, metric_name: str) -> Optional[MetricResult]:
        for m in self.metrics:
            if m.name == metric_name:
                return m
        return None


class AutoEvaluator:
    """
    자동 정량 평가기.

    origin(원본)과 processed(처리 결과)를 비교하여 8개 메트릭 산출.
    degrade 없이 실제 아날로그 영상을 평가할 때는 origin 대신
    reference 이미지를 넣으면 됨.
    """

    # 가중치 (합=1.0) — 프로젝트 특성에 맞게 조정 가능
    WEIGHTS = {
        "psnr": 0.1300,
        "ssim": 0.1700,
        "color_fidelity": 0.1300,
        "edge_retention": 0.1300,
        "noise_level": 0.1300,
        "detail_recovery": 0.0850,
        "artifact_score": 0.0850,
        "vif": 0.0450,
        "niqe": 0.0950,
    }

    def __init__(self, weights: dict | None = None):
        if weights:
            self.WEIGHTS = weights

    # ─── Core Metrics ─────────────────────────────────────────────

    def _psnr(self, a: np.ndarray, b: np.ndarray) -> float:
        """PSNR (dB). 높을수록 좋음."""
        mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
        if mse == 0:
            return 100.0
        return float(10 * np.log10(255.0 ** 2 / mse))

    def _ssim(self, a: np.ndarray, b: np.ndarray) -> float:
        """SSIM (0~1). 높을수록 좋음. luminance만 비교."""
        from skimage.metrics import structural_similarity
        a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if len(a.shape) == 3 else a
        b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if len(b.shape) == 3 else b
        return float(structural_similarity(a_gray, b_gray, data_range=255))

    def _color_fidelity(self, origin: np.ndarray, processed: np.ndarray) -> float:
        """
        색 충실도 — CIEDE2000 ΔE 평균.
        0 = 완전 일치, 낮을수록 좋음.
        일반적으로 ΔE < 1 = 지각 불가, < 3 = 허용, > 6 = 현저.
        CIEDE2000은 CIE76보다 인간 시각 인지에 더 부합함.
        """
        from skimage.color import deltaE_ciede2000

        lab_o = cv2.cvtColor(origin, cv2.COLOR_BGR2LAB).astype(np.float64)
        lab_p = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB).astype(np.float64)
        # Scale to CIELAB space
        lab_o[:, :, 0] = lab_o[:, :, 0] * 100.0 / 255.0  # L 0-100
        lab_p[:, :, 0] = lab_p[:, :, 0] * 100.0 / 255.0
        lab_o[:, :, 1] -= 128  # a -128..127
        lab_p[:, :, 1] -= 128
        lab_o[:, :, 2] -= 128  # b -128..127
        lab_p[:, :, 2] -= 128

        de = deltaE_ciede2000(lab_o, lab_p)
        return float(np.mean(de))

    def _brightness_ratio(self, origin: np.ndarray, processed: np.ndarray) -> float:
        """
        휘도 비율 — processed / origin 평균 휘도.
        1.0 = 동일, >1 = 밝아짐, <1 = 어두워짐.
        """
        lum_o = np.mean(cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY).astype(np.float64))
        lum_p = np.mean(cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY).astype(np.float64))
        if lum_o < 1e-6:
            return 1.0
        return float(lum_p / lum_o)

    def _edge_retention(self, origin: np.ndarray, processed: np.ndarray) -> float:
        """
        에지 보존율 — Canny edge 픽셀수 비율.
        1.0 = 원본과 동일 edge량, >1 = edge 강조, <1 = edge 손실.
        """
        gray_o = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY) if len(origin.shape) == 3 else origin
        gray_p = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed

        # adaptive threshold for Canny (Otsu)
        thresh_o_val = float(cv2.threshold(gray_o, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
        thresh_p_val = float(cv2.threshold(gray_p, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])

        edge_o = cv2.Canny(gray_o, max(thresh_o_val * 0.5, 30), thresh_o_val)
        edge_p = cv2.Canny(gray_p, max(thresh_p_val * 0.5, 30), thresh_p_val)

        count_o = np.count_nonzero(edge_o)
        count_p = np.count_nonzero(edge_p)
        if count_o == 0:
            return 1.0
        return float(count_p / count_o)

    def _noise_level(self, img: np.ndarray) -> float:
        """
        노이즈 레벨 — Laplacian 분산.
        낮을수록 깨끗함. 텍스처가 많은 영상에서는 높게 나올 수 있음.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())

    def _detail_recovery(self, origin: np.ndarray, processed: np.ndarray) -> float:
        """
        디테일 복원율 — 고주파 에너지 보존.
        1.0 = 원본 디테일 완전 보존, >1 = 샤프닝, <1 = 블러링.
        """
        gray_o = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_p = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # 고주파 = 원본 - lowpass
        low_o = cv2.GaussianBlur(gray_o, (0, 0), 3.0)
        low_p = cv2.GaussianBlur(gray_p, (0, 0), 3.0)
        hf_o = gray_o - low_o
        hf_p = gray_p - low_p

        energy_o = np.mean(hf_o ** 2)
        energy_p = np.mean(hf_p ** 2)
        if energy_o < 1e-6:
            return 1.0
        return float(np.sqrt(energy_p / energy_o))

    def _artifact_score(self, origin: np.ndarray, processed: np.ndarray) -> float:
        """
        아티팩트 점수 — 처리 과정에서 발생한 인위적 패턴 감지.
        ringing/blocking/overshooting 의심도. 낮을수록 좋음.

        원리:
        1. ringing = edge 근처에서 oscillation 감지
        2. blocking = 8x8 블록 경계 불연속 감지
        3. overshooting = edge 방향으로 pixel value가 원본보다 과도하게 튐
        """
        gray_o = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_p = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # 1) 차이 맵에서 ringing 감지
        diff = gray_p - gray_o
        # edge 근처에서만 패턴이 있으면 ringing 의심
        edge_map = cv2.Canny(cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY), 50, 150)
        # edge 주변 확장 (dilate)
        kernel = np.ones((5, 5), np.uint8)
        edge_region = cv2.dilate(edge_map, kernel, iterations=1).astype(bool)

        if edge_region.sum() > 0:
            edge_diff = np.abs(diff[edge_region])
            ringing_score = np.mean(edge_diff)
        else:
            ringing_score = 0.0

        # 2) blocking artifact — 8px 간격 수평/수직 edge 강도
        h, w = gray_p.shape
        block_score = 0.0
        if h > 16 and w > 16:
            vert_sum = 0.0
            for by in range(8, min(h - 1, 64), 8):
                vert_sum += float(np.mean(np.abs(gray_p[by, :] - gray_p[by - 1, :])))
            horiz_sum = 0.0
            for bx in range(8, min(w - 1, 64), 8):
                horiz_sum += float(np.mean(np.abs(gray_p[:, bx] - gray_p[:, bx - 1])))
            n_checks = (min(h - 1, 64) // 8 - 1) + (min(w - 1, 64) // 8 - 1)
            block_score = (vert_sum + horiz_sum) / max(n_checks, 1)

        # 3) overshooting — processed가 origin보다 극단적으로 밝거나 어두운 영역
        overshoot = np.mean(np.abs(diff) > 50) * 100  # percentage

        return float(ringing_score * 0.4 + block_score * 0.3 + overshoot * 0.3)

    def _vif(self, origin: np.ndarray, processed: np.ndarray) -> float:
        """
        Visual Information Fidelity (VIF) — simplified wavelet-based.

        Computes the ratio of mutual information between reference and
        distorted images in 4-level DWT subbands. Higher = more information
        preserved from the reference. Range: ~0-1 (theoretically unbounded
        but practically 0-1 for similar images).

        Simplified approach: decompose both images with Daubechies 9/7
        wavelet, compute the correlation (as a proxy for mutual information)
        in each subband, and aggregate into a single fidelity score.
        """
        import pywt

        gray_o = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_p = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # 4-level DWT with Daubechies 9/7 wavelet
        coeffs_o = pywt.wavedec2(gray_o, 'db9', level=4)
        coeffs_p = pywt.wavedec2(gray_p, 'db9', level=4)

        # Approximate subband: cA (low-pass)
        cA_o = coeffs_o[0]
        cA_p = coeffs_p[0]

        # Detail subbands: cH, cV, cD at each level
        scores = []
        # Include approximation subband correlation
        if cA_o.size > 1 and cA_p.size > 1:
            # Normalized cross-correlation as information preservation proxy
            cA_o_flat = cA_o.ravel()
            cA_p_flat = cA_p.ravel()
            if np.std(cA_o_flat) > 1e-8 and np.std(cA_p_flat) > 1e-8:
                cA_corr = float(np.corrcoef(cA_o_flat, cA_p_flat)[0, 1])
                scores.append(max(0, cA_corr))

        # Detail subbands at each level
        for level_idx, detail_tuple in enumerate(coeffs_o[1:], 1):
            detail_p_tuple = coeffs_p[level_idx]
            for band_idx in range(3):  # HL, LH, HH
                band_o = detail_tuple[band_idx].ravel()
                band_p = detail_p_tuple[band_idx].ravel()
                if band_o.size > 1 and np.std(band_o) > 1e-8 and np.std(band_p) > 1e-8:
                    corr = float(np.corrcoef(band_o, band_p)[0, 1])
                    # Weight higher levels (coarser scales) more heavily
                    weight = 1.0 + level_idx * 0.25
                    scores.append(max(0, corr) * weight)

        if not scores:
            return 0.0

        return float(np.mean(scores))

    def _niqe(self, img: np.ndarray, patch_size: int = 96,
              stride: int = 48) -> float:
        """
        NIQE (Natural Image Quality Evaluator) — no-reference quality metric.
        Lower = better perceptual quality. Natural images typically 2.5–4.5.

        Computes MSCN (Mean Subtracted Contrast Normalized) coefficients,
        extracts patch-level quality-aware features, and measures the
        multivariate Gaussian distance to pristine natural scene statistics.
        """
        from calculate_niqe import compute_niqe
        return compute_niqe(img, patch_size, stride)

    # ─── Composite Score ──────────────────────────────────────────

    def compute_composite(self, result: EvalResult) -> float:
        """
        종합 품질 점수 (0~100).
        각 메트릭을 0~100으로 정규화 후 가중 평균.
        """
        scores = {}

        for m in result.metrics:
            if m.name == "psnr":
                # 10~40 dB → 0~100 (보통 15~30 범위)
                scores["psnr"] = np.clip((m.value - 10) / 30 * 100, 0, 100)
            elif m.name == "ssim":
                # 0~1 → 0~100
                scores["ssim"] = np.clip(m.value * 100, 0, 100)
            elif m.name == "color_fidelity":
                # ΔE: 0~20 → 100~0 (낮을수록 좋음)
                scores["color_fidelity"] = np.clip((1 - m.value / 20) * 100, 0, 100)
            elif m.name == "edge_retention":
                # 0~2 → 0~100, 1.0=50점 기준
                scores["edge_retention"] = np.clip(m.value * 50, 0, 100)
            elif m.name == "noise_level":
                # Laplacian var: 0~2000 → 100~0
                scores["noise_level"] = np.clip((1 - m.value / 2000) * 100, 0, 100)
            elif m.name == "detail_recovery":
                # 0~2 → 0~100, 1.0=50점
                scores["detail_recovery"] = np.clip(m.value * 50, 0, 100)
            elif m.name == "artifact_score":
                # 0~30 → 100~0
                scores["artifact_score"] = np.clip((1 - m.value / 30) * 100, 0, 100)
            elif m.name == "vif":
                # 0~1 → 0~100 (theoretically can exceed 1, clamp at 100)
                scores["vif"] = np.clip(m.value * 100, 0, 100)
            elif m.name == "niqe":
                # NIQE: 0~15 → 100~0 (lower is better, natural ~2.5-4.5)
                scores["niqe"] = np.clip((1 - m.value / 15) * 100, 0, 100)

        total = 0.0
        for key, weight in self.WEIGHTS.items():
            if key in scores:
                total += scores[key] * weight

        return round(total, 2)

    # ─── Main Evaluate ────────────────────────────────────────────

    def evaluate(self, origin: np.ndarray, processed: np.ndarray,
                 label: str = "", degraded: np.ndarray | None = None,
                 verbose: bool = True) -> EvalResult:
        """
        전체 자동 평가 수행.

        Parameters
        ----------
        origin : 원본 (clean) 이미지
        processed : 파이프라인 처리 결과
        label : preset/실험 이름
        degraded : (선택) degraded 이미지 — noise_level 비교용
        verbose : 콘솔 출력 여부

        Returns
        -------
        EvalResult with 8 metrics + composite score
        """
        # shape 맞추기
        if origin.shape[:2] != processed.shape[:2]:
            h, w = origin.shape[:2]
            processed = cv2.resize(processed, (w, h))

        result = EvalResult(
            label=label,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # 9개 메트릭 산출 (8 + NIQE)
        metrics = [
            MetricResult("psnr", self._psnr(origin, processed), "dB",
                         "Peak Signal-to-Noise Ratio", "higher"),
            MetricResult("ssim", self._ssim(origin, processed), "",
                         "Structural Similarity Index", "higher"),
            MetricResult("color_fidelity", self._color_fidelity(origin, processed), "ΔE",
                         "CIEDE2000 color difference (lower=better)", "lower"),
            MetricResult("edge_retention", self._edge_retention(origin, processed), "ratio",
                         "Canny edge count ratio (1.0=identical)", "higher"),
            MetricResult("noise_level", self._noise_level(processed), "Lap.var",
                         "Laplacian variance (lower=cleaner)", "lower"),
            MetricResult("detail_recovery", self._detail_recovery(origin, processed), "ratio",
                         "High-frequency energy ratio (1.0=preserved)", "higher"),
            MetricResult("artifact_score", self._artifact_score(origin, processed), "score",
                         "Ringing+blocking+overshoot score (lower=better)", "lower"),
            MetricResult("vif", self._vif(origin, processed), "",
                         "Visual Information Fidelity (higher=better)", "higher"),
            MetricResult("niqe", self._niqe(processed), "score",
                         "NIQE no-reference quality (lower=better)", "lower"),
        ]

        # degraded 노이즈 레벨도 참고용으로 추가
        if degraded is not None:
            noise_deg = self._noise_level(degraded)
            metrics.append(MetricResult("noise_level_degraded", noise_deg, "Lap. var",
                                        "Degraded noise level (reference)", "lower"))

        result.metrics = metrics
        result.composite_score = self.compute_composite(result)

        if verbose:
            self._print_result(result)

        return result

    def _print_result(self, r: EvalResult):
        """결과를 콤팩트 테이블 형식으로 출력."""
        print(f"\n{'═' * 65}")
        print(f"📊 Auto Evaluation: {r.label}")
        print(f"{'═' * 65}")
        for m in r.metrics:
            arrow = "↑" if m.best == "higher" else "↓"
            unit_str = f" [{m.unit}]" if m.unit else ""
            print(f"  {m.name:20s} {m.value:8.4f} {arrow}{unit_str}")
        print(f"{'─' * 65}")
        print(f"  {'COMPOSITE':20s} {r.composite_score:8.2f} /100")
        print(f"{'═' * 65}")

    # ─── Batch Compare ────────────────────────────────────────────

    def compare_presets(self, origin: np.ndarray,
                        results: dict[str, np.ndarray],
                        degraded: np.ndarray | None = None,
                        verbose: bool = True) -> list[EvalResult]:
        """
        여러 preset 결과를 한번에 비교 평가.

        Parameters
        ----------
        origin : 원본 이미지
        results : {preset_name: processed_image} dict
        degraded : (선택) degraded 이미지
        verbose : 콘솔 출력

        Returns
        -------
        list of EvalResult, sorted by composite_score desc
        """
        all_results = []
        for name, img in results.items():
            r = self.evaluate(origin, img, label=name,
                              degraded=degraded, verbose=verbose)
            all_results.append(r)

        # composite 내림차순 정렬
        all_results.sort(key=lambda x: x.composite_score, reverse=True)

        if verbose and len(all_results) > 1:
            print(f"\n{'═' * 70}")
            print(f"📊 RANKING (by Composite Score)")
            print(f"{'═' * 70}")
            for i, r in enumerate(all_results):
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                print(f"  {medal} #{i+1} {r.label:30s}  Score: {r.composite_score:.2f}")
            print(f"{'═' * 70}")

        return all_results

    # ─── Export ────────────────────────────────────────────────────

    @staticmethod
    def export_csv(results: list[EvalResult], path: str):
        """결과를 CSV로 저장."""
        if not results:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        metric_names = [m.name for m in results[0].metrics]
        headers = ["label", "timestamp", "composite_score"] + metric_names

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in results:
                row = {"label": r.label, "timestamp": r.timestamp,
                       "composite_score": r.composite_score}
                for m in r.metrics:
                    row[m.name] = round(m.value, 4)
                writer.writerow(row)
        return path

    @staticmethod
    def export_json(results: list[EvalResult], path: str):
        """결과를 JSON으로 저장."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = [r.to_dict() for r in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def export_markdown(results: list[EvalResult], path: str,
                        title: str = "Auto Evaluation Report"):
        """결과를 Markdown 리포트로 저장."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        lines = [f"# {title}", ""]
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Ranking
        lines.append("## Ranking")
        lines.append("")
        lines.append("| # | Preset | Composite | PSNR | SSIM | ΔE | Edge | Noise | Detail | Artifact | VIF |")
        lines.append("|---|--------|-----------|------|------|----|------|-------|--------|----------|-----|")
        for i, r in enumerate(results):
            m = {m.name: m.value for m in r.metrics}
            lines.append(
                f"| {i+1} | {r.label} | **{r.composite_score:.1f}** | "
                f"{m.get('psnr', 0):.2f} | {m.get('ssim', 0):.4f} | "
                f"{m.get('color_fidelity', 0):.2f} | {m.get('edge_retention', 0):.3f} | "
                f"{m.get('noise_level', 0):.1f} | {m.get('detail_recovery', 0):.3f} | "
                f"{m.get('artifact_score', 0):.2f} | {m.get('vif', 0):.4f} |"
            )
        lines.append("")

        # Per-preset detail
        lines.append("## Detail per Preset")
        lines.append("")
        for r in results:
            lines.append(f"### {r.label} — Score: {r.composite_score:.1f}/100")
            lines.append("")
            for m in r.metrics:
                arrow = "↑" if m.best == "higher" else "↓"
                lines.append(f"- **{m.name}**: {m.value:.4f} {arrow} ({m.unit})")
            if r.notes:
                lines.append(f"- Notes: {r.notes}")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path
