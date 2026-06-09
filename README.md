# 아날로그 영상 잡음 완화 파이프라인 v3.7

영상처리 기말 프로젝트 — 6팀 (김승민 · 김원석)

아날로그 FPV DVR 영상에서 시인성을 해치는 잡음·색왜곡·대비저하를 **40개 필터·51개 프리셋**으로 완화하는 모듈식 영상처리 파이프라인.

---

## ✨ v3.7 하이라이트

- 🧠 **Adaptive Pipeline** — 노이즈 종류 자동 감지 + 모션 인식 분기 (fast 49ms / quality 712ms)
- 🏆 **NIQE 7.23** — `chroma-bior4-detail` 프리셋, 무참조 품질 평가 기준 최고
- ⚡ **12.5ms / 80fps** — `optimal-ultrafast`, 실시간 처리 가능
- 🎬 **BRISQUE 34.04** — `temporal-bior4`, 멀티비디오 벤치마크 최고 체감품질
- 📊 **5영상 교차검증** — 26개 실제 아날로그 영상으로 벤치마크 완료
- 🔬 **BM3D/BM4D** — Block-Matching 3D/4D 협업 필터링 (state-of-the-art)
- 📺 **NTSC 시뮬레이션** — zhuker/ntsc 통합 (dot crawl, ringing, color bleeding)

---

## 📁 프로젝트 구조

```
smu_sig_prossessing/          — 핵심 패키지
├── __init__.py               — 패키지 임포트
├── __main__.py               — CLI 진입점
├── config.py                 — PipelineConfig + 51개 프리셋
├── filters.py                — 40개 필터 레지스트리
├── pipeline.py               — 파이프라인 러너
├── adaptive.py               — 노이즈 자동감지 + 모션분기 파이프라인
├── noise_estimator.py        — 노이즈 타입/강도 추정
├── auto_evaluation.py        — 10종 메트릭 자동평가 (PSNR~BRISQUE)
├── eval_viz.py               — 평가 시각화
├── evaluation.py             — PSNR/SSIM 유틸리티
├── degradation.py            — 열화 모듈 (기본 + NTSC)
└── ntsc_plugin.py            — zhuker/ntsc (ringPattern.npy 포함)

main.py                       — CLI 메인 (argparse)
generate_report.py            — 벤치마크 리포트 생성
scripts/                      — 벤치마크/평가 스크립트 모음
plan/                         — 프로젝트 계획서 + 설계 변경 이력
```

---

## 🔧 40개 필터

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

---

## 🎯 추천 프리셋 TOP 10

| 프리셋 | 특징 | NIQE↓ | BRISQUE↓ | 속도 |
|--------|------|-------|----------|------|
| **chroma-bior4-detail** | 🏆 최고 체감품질 | **7.23** | 77.2 | 1030ms |
| **optimal-bior4** | 웨이블릿 최적화 | 7.26 | 75.6 | 884ms |
| **optimal-balanced** | 품질/속도 균형 | 7.28 | 74.9 | 911ms |
| **nlm-chroma** | NLM+크로마 | 7.33 | 62.6 | 775ms |
| **analog-clean** | 아날로그 전용 | 7.33 | 67.8 | 839ms |
| **temporal-bior4** | 🎬 최고 BRISQUE | 7.43 | **53.5** | 1737ms |
| **fast-guided-chroma** | ⚡ 빠른 품질 | 7.63 | 55.4 | 60ms |
| **optimal-ultrafast** | 🚀 실시간 | 7.64 | 64.0 | **12.5ms** |
| **fast-denoise** | 범용 빠름 | 7.57 | 71.1 | 22ms |
| **adaptive** | 🧠 자동감지 | — | 60.2 | 가변 |

*NIQE/BRISQUE는 낮을수록 좋음. 854×480 아날로그 FPV 영상 기준.*

---

## 🚀 설치

```bash
# 가상환경 생성
python -m venv .venv && source .venv/bin/activate

# 의존성 설치
pip install opencv-python-headless scikit-image scipy numpy PyWavelets bm3d bm4d brisque
```

---

## 📖 사용법

### 실제 아날로그 영상 처리

```bash
# 추천 프리셋으로 처리
python main.py -p "input/analog_footage.mp4" --preset chroma-bior4-detail --degrade none

# 실시간 처리 (80fps)
python main.py -p "input/analog_footage.mp4" --preset optimal-ultrafast --degrade none

# 자동 노이즈 감지
python main.py -p "input/analog_footage.mp4" --preset adaptive --degrade none

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

**10종 자동평가 메트릭** (`auto_evaluation.py`):

| 지표 | 설명 | 참조 필요 |
|------|------|-----------|
| PSNR | Peak Signal-to-Noise Ratio | ✅ |
| SSIM | Structural Similarity | ✅ |
| VIF | Visual Information Fidelity | ✅ |
| NIQE | Natural Image Quality Evaluator | ❌ |
| BRISQUE | Blind Referenceless SPatial Quality | ❌ |
| ΔE (CIEDE2000) | 색차 | ✅ |
| Edge Retention | 엣지 보존율 | ✅ |
| Noise Level | 잔여 노이즈 | ❌ |
| Color Fidelity | 색 충실도 | ✅ |
| Artifact Score | 아티팩트 점수 | ❌ |

---

## 🔬 버전 히스토리

| 버전 | 주요 변경 |
|------|-----------|
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
