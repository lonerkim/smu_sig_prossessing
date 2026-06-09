# 아날로그 영상 잡음 완화 파이프라인 v4.0

영상처리 기말 프로젝트 — 6팀 (김승민 · 김원석)

아날로그 FPV DVR 영상에서 시인성을 해치는 잡음·색왜곡·대비저하를 **41개 필터·54개 프리셋**으로 완화하는 모듈식 영상처리 파이프라인.

---

## ✨ v4.0 하이라이트

- 🏆 **Composite 61.38** — `grey-premium`, 11-metric 기준 최고
- 🧠 **Noise-Aware Adaptive Pipeline** — 노이즈 레벨 + 모션 자동 감지 → preset/params 자동 선택
- 🖼️ **Super-Resolution** — Lanczos4/Cubic/EDSR 2x 업스케일 필터
- 📊 **11-metric evaluation** — MS-SSIM 포함, NIQE+BRISQUE+MS-SSIM 무참조 3종
- 🎯 **41 filters / 54 presets** — 13 Phase, 모든 노이즈 유형 대응
- ⚡ **Ultralight 4.4ms / 227fps** — 실시간 처리
- 📺 **NTSC 시뮬레이션** — zhuker/ntsc 통합

---

## 📁 프로젝트 구조

```
smu_sig_prossessing/          — 핵심 패키지
├── __init__.py               — 패키지 임포트
├── __main__.py               — CLI 진입점
├── config.py                 — PipelineConfig + 54개 프리셋
├── filters.py                — 41개 필터 레지스트리
├── pipeline.py               — 파이프라인 러너
├── adaptive.py               — 노이즈 자동감지 + 모션분기 파이프라인
├── noise_estimator.py        — 노이즈 타입/강도 추정
├── noise_estimation.py       — 노이즈 레벨 기반 적응형 파라미터 추정
├── auto_evaluation.py        — 11종 메트릭 자동평가 (PSNR~MS-SSIM)
├── eval_viz.py               — 평가 시각화
├── evaluation.py             — PSNR/SSIM 유틸리티
├── degradation.py            — 열화 모듈 (기본 + NTSC)
└── ntsc_plugin.py            — zhuker/ntsc (ringPattern.npy 포함)

calculate_niqe.py              — NIQE 독립 모듈 (patch-based MVG distance)
main.py                        — CLI 메인 (argparse)
generate_report.py             — 벤치마크 리포트 생성
scripts/                       — 벤치마크/평가 스크립트 모음
plan/                          — 프로젝트 계획서 + 설계 변경 이력
```

---

## 🔧 41개 필터

| 필터 | 설명 | 필터 | 설명 |
|------|------|------|------|
| `median` | 미디언 필터 | `nlm` | Non-Local Means |
| `bilateral` | 양방향 필터 | `cross_bilateral` | Cross 양방향 |
| `guided_filter` | 가이드 필터 | `domain_transform` | 도메인 변환 |
| `rolling_guidance` | 롤링 가이던스 | `wavelet` | 웨이블릿 (BayesShrink) |
| `bm3d` | BM3D 협업 필터링 | `bm3d_denoise` | BM3D (고급) |
| `bm4d_volume` | BM4D 시공간 | `tv_denoise` | Total Variation |
| `anisotropic_diffusion` | 이방성 확산 | `wiener` | 위너 필터 |
| `deblur_wiener` | 위너 디컨볼루션 | `fft_notch` | FFT 노치 필터 |
| `vertical_notch` | 수직 노치 | `scanline_remove` | 스캔라인 제거 |
| `gamma` | 감마 보정 | `log_transform` | 로그 변환 |
| `histogram_eq` | 히스토그램 평활화 | `histogram_eq_clahe` | CLAHE |
| `adaptive_equalize` | 적응형 평활화 | `channel_correction` | RGB 채널 보정 |
| `chroma_denoise` | 크로마 노이즈 제거 | `unsharp_mask` | 언샤프 마스크 |
| `detail_boost` | 디테일 강조 | `retinex` | Retinex SSR |
| `retinex_msrcr` | Multi-Scale Retinex | `grey_edge` | Grey-Edge 색보정 |
| `flicker_stabilize` | 플리커 안정화 | `deinterlace` | 디인터레이스 |
| `temporal_average` | 시간 평균 | `temporal_motion` | 모션 보상 시간 |
| `temporal_nlm_multi` | 다중 NLM 시간 | `temporal_spatial` | 시공간 융합 |
| `nlm_gray` | NLM (그레이) | `gaussian_lowpass` | 가우시안 LP |
| `histogram_eq_gray` | 그레이 히스토그램 | `patch_collaborative` | 패치 협업 |
| `super_resolve` | Super-Resolution (Lanczos4/Cubic/EDSR 2x) |

---

## 🎯 v4.0 Benchmark TOP 10 (test_small.jpg, basic 0.5, 11-metric)

| # | Preset | Composite | PSNR | SSIM | MS-SSIM | Speed |
|---|--------|-----------|------|------|---------|-------|
| 🥇 | **grey-premium** | **61.38** | 19.19 | 0.6263 | 0.9674 | 171ms |
| 🥈 | grey-guided-chroma | **59.72** | 19.10 | 0.6016 | 0.9651 | **11.7ms** |
| 🥉 | temporal-premium | **59.34** | 19.09 | 0.6479 | 0.9702 | 157ms |
| 4 | temporal-ntsc | **58.72** | 19.06 | 0.6160 | 0.9668 | 170ms |
| 5 | chroma-focus | **58.55** | 18.60 | 0.5966 | 0.9642 | 35ms |
| 6 | fast-guided-chroma | **57.57** | 19.05 | 0.6038 | 0.9637 | **9.3ms** |
| 7 | grey-fast | **57.33** | 19.00 | 0.6141 | 0.9655 | 143ms |
| 8 | chroma-guided-bior4 | **56.60** | 19.01 | 0.6246 | 0.9681 | 141ms |
| 9 | rolling-premium | **56.37** | 18.98 | 0.6821 | 0.9720 | 35ms |
| 10 | fast-premium | **55.95** | 18.32 | 0.5938 | 0.9628 | 142ms |

*11-metric composite 기준. No-reference: NIQE 7.23 (chroma-bior4-detail), BRISQUE 34.04 (temporal-bior4).*

---

## 🚀 설치

```bash
# 가상환경 생성
python -m venv .venv && source .venv/bin/activate

# 의존성 설치
pip install opencv-python-headless scikit-image scipy numpy PyWavelets bm3d bm4d brisque sewar opencv-contrib-python
```

---

## 📖 사용법

### 실제 아날로그 영상 처리

```bash
# 추천 프리셋으로 처리
python main.py -p "input/analog_footage.mp4" --preset chroma-bior4-detail --degrade none

# 실시간 처리 (227fps)
python main.py -p "input/analog_footage.mp4" --preset optimal-ultrafast --degrade none

# 자동 노이즈 감지 + Adaptive Pipeline
python main.py -p "input/analog_footage.mp4" --preset adaptive --degrade none

# Super-Resolution 업스케일
python main.py -p "input/analog_footage.mp4" --preset grey-premium --degrade none --sr 2x

# 영상 파일로 출력
python main.py -p "input/analog_footage.mp4" --preset analog-clean --degrade none --output-video

# 5프레임만 미리보기
python main.py -p "input/analog_footage.mp4" --preset wavelet-denoise --degrade none --sample 5
```

### 합성 열화 + 복원 테스트

```bash
# 기본 열화 (Gaussian + Impulse + Color Bias + 밝기)
python main.py -p "input/photo.jpg" --preset optimal-balanced --degrade basic --strength 0.5

# NTSC 열화 (실제 아날로그 아티팩트)
python main.py -p "input/photo.jpg" --preset ntsc-plus --degrade ntsc-heavy
```

### 유틸리티

```bash
# 전체 프리셋 목록
python main.py --help

# 필터 레지스트리 확인
python main.py --list-filters
```

---

## 📊 평가 지표

**11종 자동평가 메트릭** (`auto_evaluation.py`):

| 지표 | 설명 | 참조 필요 |
|------|------|-----------|
| PSNR | Peak Signal-to-Noise Ratio | ✅ |
| SSIM | Structural Similarity | ✅ |
| MS-SSIM | Multi-Scale SSIM | ✅ |
| VIF | Visual Information Fidelity | ✅ |
| NIQE | Natural Image Quality Evaluator | ❌ |
| BRISQUE | Blind Referenceless Spatial Quality | ❌ |
| ΔE (CIEDE2000) | 색차 | ✅ |
| Edge Retention | 엣지 보존율 | ✅ |
| Noise Level | 잔여 노이즈 | ❌ |
| Color Fidelity | 색 충실도 | ✅ |
| Artifact Score | 아티팩트 점수 | ❌ |

---

## 🔬 버전 히스토리

| 버전 | 주요 변경 |
|------|-----------|
| **v4.0** | Super-Resolution (Lanczos4/Cubic/EDSR 2x), MS-SSIM 메트릭, Noise-Aware Adaptive Pipeline, Composite 61.38 (grey-premium), 41 filters / 54 presets |
| **v3.7** | 4개 최적화 프리셋 (chroma-bior4-detail, temporal-bior4), 멀티비디오 교차검증, adaptive 모션인식 분기 |
| **v3.6** | BRISQUE 메트릭, temporal NLM 최적화 (BRISQUE 59.87), adaptive 파이프라인 |
| **v3.5** | 디인터레이스 필터, 전체 프리셋 ablation sweep, 27영상 배치처리 |
| **v3.0** | BM3D, Retinex, 광학플로우 시간 denoising, ablation 프레임워크 |
| **v2.0** | NTSC 시뮬레이터 통합, 모듈식 PipelineConfig, 위너 디컨볼루션 |
| **v1.0** | 기본 파이프라인 (median, wiener, NLM, bilateral) |

---

## 📚 참고

- [zhuker/ntsc](https://github.com/zhuker/ntsc) — NTSC 아날로그 비디오 시뮬레이터
- [BM3D](https://github.com/sters BM3D) — Block-Matching 3D denoising
- [OpenCV NLM](https://docs.opencv.org/master/d5/d69/tutorial_py_non_local_means.html)
- [PyWavelets](https://pywavelets.readthedocs.io/) — BayesShrink 웨이블릿 임계처리
- [sewar](https://github.com/andrewek/sewar) — MS-SSIM, VIF 등 full-reference metrics
