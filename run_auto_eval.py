#!/usr/bin/env python3
"""
자동 정량+정성 평가 CLI — preset별 비교, 리포트, 시각화 한번에 생성.

Usage:
    # 전체 preset 비교 (이미지)
    python run_auto_eval.py -i input/REAL_WORLD_PICTURE.jpg --degrade basic --strength 0.5

    # 특정 preset만
    python run_auto_eval.py -i input/test_small.jpg --presets optimized-fast,video-enhanced,wavelet-denoise

    # 비디오 첫 N프레임 평가
    python run_auto_eval.py -i input/analog_whoop_footage.mp4 --degrade none --sample 5

    # 실제 아날로그 영상 (degrade 없이)
    python run_auto_eval.py -i input/analog_whoop_footage.mp4 --degrade none --sample 3

    # 정성 평가 시트만 생성
    python run_auto_eval.py -i input/test_small.jpg --qualitative-only

Output:
    output/eval/
      auto_eval_{timestamp}.csv      — 정량 데이터
      auto_eval_{timestamp}.md       — Markdown 리포트
      auto_eval_{timestamp}_radar.png — Radar chart
      auto_eval_{timestamp}_bar.png   — Bar chart
      auto_eval_{timestamp}_grid.png  — 비교 그리드
      auto_eval_{timestamp}_qual.png  — 정성 평가 시트
      qualitative_notes.md           — 정성 코멘트 템플릿
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import zipfile
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.degradation import degrade_image
from smu_sig_prossessing.auto_evaluation import AutoEvaluator, EvalResult
from smu_sig_prossessing.eval_viz import (
    draw_radar_chart, draw_bar_chart,
    draw_comparison_grid, create_eval_sheet,
)
from smu_sig_prossessing.filters import reset_temporal_state

# ─── Preset map (main.py와 동일) ──────────────────────────────────

PRESETS: dict[str, PipelineConfig] = {
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
}

DEGRADE_MODES = {"none", "basic", "ntsc-light", "ntsc-medium", "ntsc-heavy"}

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output", "eval")


def make_degraded(origin: np.ndarray, mode: str, strength: float) -> np.ndarray:
    """Apply degradation."""
    if mode == "none":
        return origin.copy()
    if mode.startswith("ntsc"):
        intensity = mode.split("-")[1]
        return degrade_image(origin, use_ntsc=True, ntsc_intensity=intensity)
    return degrade_image(origin, use_ntsc=False, strength=strength)


def run_evaluation(input_path: str, presets: list[str],
                   degrade_mode: str, strength: float,
                   sample_frames: int = 0,
                   qualitative_only: bool = False,
                   output_dir: str | None = None,
                   make_zip: bool = False):
    """Main evaluation runner."""
    out_dir = output_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluator = AutoEvaluator()

    # ── Load input ────────────────────────────────────────────────
    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    if is_video:
        if sample_frames <= 0:
            sample_frames = 5  # default for video eval
        print(f"\n🎬 Video mode — evaluating first {sample_frames} frame(s)")
        files = _eval_video(input_path, presets, degrade_mode, strength,
                    sample_frames, evaluator, out_dir, ts, qualitative_only)
    else:
        print(f"\n🖼️  Image mode — {input_path}")
        files = _eval_image(input_path, presets, degrade_mode, strength,
                    evaluator, out_dir, ts, qualitative_only)

    # ── ZIP packaging ─────────────────────────────────────────────
    if make_zip and files:
        zip_path = os.path.join(out_dir, f"auto_eval_{ts}_{_stem(input_path)}.zip")
        _pack_zip(files, zip_path)
        return zip_path

    return out_dir


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _pack_zip(file_paths: list[str], zip_path: str):
    """결과 파일들을 ZIP으로 패키징 (파일 내부 경로는 basename만)."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in file_paths:
            if os.path.isfile(fp):
                zf.write(fp, arcname=os.path.basename(fp))
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n📦 ZIP → {zip_path} ({size_mb:.1f} MB, {len(file_paths)} files)")


def _eval_image(input_path: str, presets: list[str],
                degrade_mode: str, strength: float,
                evaluator: AutoEvaluator, out_dir: str, ts: str,
                qual_only: bool):
    """이미지 평가."""
    origin = cv2.imread(input_path)
    if origin is None:
        print(f"⚠ Cannot read: {input_path}")
        sys.exit(1)

    name = os.path.splitext(os.path.basename(input_path))[0]
    print(f"   Image: {name}  {origin.shape[1]}x{origin.shape[0]}")
    print(f"   Degrade: {degrade_mode}  strength={strength:.1f}")
    print(f"   Presets: {len(presets)}")

    # Degrade
    degraded = make_degraded(origin, degrade_mode, strength)

    # Process all presets
    results_images: dict[str, np.ndarray] = {}
    eval_results: list[EvalResult] = []

    for preset_name in presets:
        cfg = PRESETS.get(preset_name)
        if cfg is None:
            print(f"  ⚠ Unknown preset: {preset_name}")
            continue
        print(f"\n  🔧 Processing: {preset_name}...")
        processed = pl.apply_pipeline(degraded, cfg)

        if not qual_only:
            r = evaluator.evaluate(origin, processed, label=preset_name,
                                   degraded=degraded, verbose=True)
            eval_results.append(r)
        results_images[preset_name] = processed

    generated_files: list[str] = []

    if not qual_only and eval_results:
        generated_files.extend(_save_reports(eval_results, out_dir, ts, name))

    # 시각화 (항상 생성)
    print(f"\n📊 Generating visualizations...")

    # 비교 그리드
    grid_path = os.path.join(out_dir, f"auto_eval_{ts}_{name}_grid.png")
    draw_comparison_grid(results_images, origin, grid_path)
    print(f"   Grid → {grid_path}")
    generated_files.append(grid_path)

    # 정성 평가 시트
    qual_path = os.path.join(out_dir, f"auto_eval_{ts}_{name}_qual.png")
    create_eval_sheet(origin, degraded, results_images, qual_path)
    print(f"   Qual → {qual_path}")
    generated_files.append(qual_path)

    # Radar + Bar (정량 결과 있을 때만)
    if eval_results:
        radar_path = os.path.join(out_dir, f"auto_eval_{ts}_{name}_radar.png")
        draw_radar_chart(eval_results, radar_path)
        print(f"   Radar → {radar_path}")
        generated_files.append(radar_path)

        bar_path = os.path.join(out_dir, f"auto_eval_{ts}_{name}_bar.png")
        draw_bar_chart(eval_results, bar_path)
        print(f"   Bar → {bar_path}")
        generated_files.append(bar_path)

    # 정성 코멘트 템플릿 생성
    tmpl = _create_qual_template(results_images.keys(), out_dir, name, ts)
    if tmpl:
        generated_files.append(tmpl)

    print(f"\n✅ Evaluation complete — {len(presets)} presets")
    return generated_files


def _eval_video(input_path: str, presets: list[str],
                degrade_mode: str, strength: float,
                n_frames: int,
                evaluator: AutoEvaluator, out_dir: str, ts: str,
                qual_only: bool):
    """비디오 평가 (첫 N프레임)."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"⚠ Cannot open: {input_path}")
        sys.exit(1)

    name = os.path.splitext(os.path.basename(input_path))[0]
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Video: {name}  {w}x{h}  {fps}fps")

    # 첫 프레임으로 평가
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("⚠ Cannot read first frame")
        sys.exit(1)

    degraded = make_degraded(first_frame, degrade_mode, strength)

    results_images: dict[str, np.ndarray] = {}
    eval_results: list[EvalResult] = []

    for preset_name in presets:
        cfg = PRESETS.get(preset_name)
        if cfg is None:
            continue
        reset_temporal_state()
        print(f"  🔧 Processing: {preset_name}...")
        processed = pl.apply_pipeline(degraded, cfg)

        if not qual_only:
            ref = first_frame if degrade_mode == "none" else first_frame
            r = evaluator.evaluate(ref, processed, label=preset_name,
                                   degraded=degraded, verbose=True)
            eval_results.append(r)
        results_images[preset_name] = processed

    generated_files: list[str] = []

    if not qual_only and eval_results:
        generated_files.extend(_save_reports(eval_results, out_dir, ts, name))

    # 시각화
    grid_path = os.path.join(out_dir, f"auto_eval_{ts}_{name}_grid.png")
    draw_comparison_grid(results_images, first_frame, grid_path)
    print(f"   Grid → {grid_path}")
    generated_files.append(grid_path)

    qual_path = os.path.join(out_dir, f"auto_eval_{ts}_{name}_qual.png")
    create_eval_sheet(first_frame, degraded, results_images, qual_path)
    print(f"   Qual → {qual_path}")
    generated_files.append(qual_path)

    if eval_results:
        radar_path = os.path.join(out_dir, f"auto_eval_{ts}_{name}_radar.png")
        draw_radar_chart(eval_results, radar_path)
        print(f"   Radar → {radar_path}")
        generated_files.append(radar_path)

        bar_path = os.path.join(out_dir, f"auto_eval_{ts}_{name}_bar.png")
        draw_bar_chart(eval_results, bar_path)
        print(f"   Bar → {bar_path}")
        generated_files.append(bar_path)

    tmpl = _create_qual_template(results_images.keys(), out_dir, name, ts)
    if tmpl:
        generated_files.append(tmpl)

    print(f"\n✅ Evaluation complete — {len(presets)} presets")
    return generated_files


def _save_reports(results: list[EvalResult], out_dir: str, ts: str, name: str) -> list[str]:
    """CSV + JSON + Markdown 리포트 저장. Returns list of generated file paths."""
    # Sort by composite
    results.sort(key=lambda x: x.composite_score, reverse=True)
    paths: list[str] = []

    csv_path = os.path.join(out_dir, f"auto_eval_{ts}_{name}.csv")
    AutoEvaluator.export_csv(results, csv_path)
    print(f"\n   CSV → {csv_path}")
    paths.append(csv_path)

    json_path = os.path.join(out_dir, f"auto_eval_{ts}_{name}.json")
    AutoEvaluator.export_json(results, json_path)
    print(f"   JSON → {json_path}")
    paths.append(json_path)

    md_path = os.path.join(out_dir, f"auto_eval_{ts}_{name}.md")
    AutoEvaluator.export_markdown(results, md_path,
                                  title=f"Auto Evaluation — {name}")
    print(f"   MD → {md_path}")
    paths.append(md_path)
    return paths


def _create_qual_template(preset_names, out_dir: str, name: str, ts: str):
    """정성 평가 코멘트 템플릿 — 유저가 직접 보고 코멘트."""
    path = os.path.join(out_dir, f"qualitative_notes_{ts}_{name}.md")
    lines = [
        f"# 정성 평가 노트 — {name}",
        f"",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Image: {name}",
        f"",
        f"## 평가 기준",
        f"각 preset에 대해 1~5점 평가 + 자유 코멘트",
        f"",
        f"| 점수 | 의미 |",
        f"|------|------|",
        f"| 1 | 매우 나쁨 (원본보다 심하게 열화)|",
        f"| 2 | 나쁨 (개선 효과 미미)|",
        f"| 3 | 보통 (일부 개선, 일부 악화)|",
        f"| 4 | 좋음 (전반적으로 개선)|",
        f"| 5 | 매우 좋음 (선명하고 자연스러움)|",
        f"",
    ]

    for pname in preset_names:
        lines.extend([
            f"## {pname}",
            f"",
            f"- **색상**: _/5 — 코멘트: ",
            f"- **선명도**: _/5 — 코멘트: ",
            f"- **노이즈 제거**: _/5 — 코멘트: ",
            f"- **전체 인상**: _/5 — 코멘트: ",
            f"- **총평**: ",
            f"",
        ])

    lines.extend([
        f"## 종합 의견",
        f"",
        f"- 최고 preset: ",
        f"- 최악 preset: ",
        f"- 특이사항: ",
        f"",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"   Qual template → {path}")
    return path


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="자동 정량+정성 평가 — preset 비교, 리포트, 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input image/video path")
    parser.add_argument("--presets", type=str, default=None,
                        help="Comma-separated preset names (default: all)")
    parser.add_argument("--degrade", type=str, default="basic",
                        choices=list(DEGRADE_MODES),
                        help="Degradation mode (default: basic)")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="Degradation strength 0.0–1.0 (default: 0.5)")
    parser.add_argument("--sample", type=int, default=0, metavar="N",
                        help="Process N frames for video (default: 5)")
    parser.add_argument("--qualitative-only", action="store_true",
                        help="Generate only qualitative sheet (no metrics)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: output/eval/)")
    parser.add_argument("--zip", action="store_true",
                        help="Pack all results into a .zip file")

    args = parser.parse_args()

    # Resolve presets
    if args.presets:
        preset_list = [p.strip() for p in args.presets.split(",")]
        unknown = [p for p in preset_list if p not in PRESETS]
        if unknown:
            print(f"⚠ Unknown presets: {unknown}")
            print(f"   Available: {list(PRESETS.keys())}")
            sys.exit(1)
    else:
        preset_list = list(PRESETS.keys())

    result = run_evaluation(
        input_path=args.input,
        presets=preset_list,
        degrade_mode=args.degrade,
        strength=args.strength,
        sample_frames=args.sample,
        qualitative_only=args.qualitative_only,
        output_dir=args.output,
        make_zip=args.zip,
    )

    # --zip일 경우 결과 zip 경로 출력
    if args.zip and isinstance(result, str) and result.endswith(".zip"):
        print(f"\n📎 ZIP ready: {result}")


if __name__ == "__main__":
    main()
