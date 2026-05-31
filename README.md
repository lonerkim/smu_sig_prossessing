# 아날로그 영상 잡음 완화 파이프라인 v2.0

영상처리 기말 프로젝트 — 6팀 (김승민, 김원석)

## 프로젝트 개요
아날로그 FPV DVR 영상에서 시인성을 해치는 잡음·색왜곡·대비저하를 기본 영상처리 기법으로 완화하는 범용 파이프라인.

**v2.0 주요 개선**:
- 🎬 **NTSC 아날로그 시뮬레이터 통합** (zhuker/ntsc) — realistic dot crawl, ringing, color bleeding
- 🔧 **필터 모듈화** — 각 단계를 ON/OFF 가능한 FilterConfig로 분리
- 🎯 **Wiener 필터 단독 사용** — Gaussian Lowpass 제거 (불필요한 blur 최소화)
- 🧪 **추가 연구 기법** — NLM, Bilateral, CLAHE, Unsharp Mask, Wiener Deconvolution
- 📊 **정성/정량 평가 도구** — PSNR/SSIM + 4-way visual comparison

## 구조
```
smu_sig_prossessing/        — 핵심 패키지 (모듈식)
  config.py                 — PipelineConfig (필터 ON/OFF + 파라미터)
  filters.py                — 필터 레지스트리 (16개 필터)
  pipeline.py               — 파이프라인 러너
  degradation.py            — 열화 (기본 + NTSC)
  evaluation.py             — PSNR/SSIM + 시각화
  ntsc_plugin.py            — zhuker/ntsc

scripts/
  run_pipeline.py           — 정적 이미지 파이프라인
  run_video.py              — 비디오 파이프라인
  run_evaluation.py         — 정성/정량 평가

plan/
  Project_plan.md           — 프로젝트 계획서
  Implementation_Plan.md    — 구현 계획서
```

## 간단 실행

```bash
# 정적 이미지 파이프라인 (기본)
python main.py

# 비디오 파이프라인
python main.py --video

# 필터 목록 보기
python main.py --list-filters
```

## 상세 실행

```bash
# 정성적 평가 (실제 영상 필요)
python scripts/run_evaluation.py --qualitative /path/to/video.mp4

# 정량적 평가 (실제 영상 필요)
python scripts/run_evaluation.py --quantitative /path/to/video.mp4
```

## PipelineConfig 프리셋 비교

| 프리셋 | 필터 구성 | 속도 | 품질 |
|--------|-----------|------|------|
| Wiener Only | Median→Wiener→FFT→Channel→Gamma→HistEq | ⚡빠름 | ✅ 좋음 |
| Edge-Preserving | Median→NLM→Wiener→FFT→Channel→Gamma→HistEq | 🐢느림 | ⭐ 탁월 |
| Aggressive | Median(k=5)→Bilateral→Wiener→FFT→Channel→Gamma→HistEq | 보통 | ✅ 좋음 (매우 noisy) |
| Research Best | Median→NLM→Wiener→FFT→Bilateral→Channel→Gamma→HistEq | 🐢느림 | ⭐ 최고 |

## 의존성
- Python 3.11+
- opencv-python-headless
- scikit-image
- scipy
- numpy
