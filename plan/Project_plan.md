# 아날로그 영상 잡음 및 색 왜곡 완화를 위한 범용 영상처리 파이프라인 설계

**기말 프로젝트 과제계획서**  
**6팀 | 지능IoT융합전공 김승민 · 컴퓨터과학과 김원석**  
**과목: 영상처리 기말 프로젝트**  
**제출일: 2026년 6월 9일 (v4.0)**

---

## 📋 문서 버전 정보
- **버전**: v4.0 (최종 릴리즈)
- **작성자**: 김승민, 김원석
- **마지막 업데이트**: 2026-06-09
- **상태**: ✅ **v4.0 릴리즈 완료** (41 filters, 54 presets, 11 metrics)

---

## 1. 🎯 프로젝트 개요

### 프로젝트명
**아날로그 영상 잡음 및 색 왜곡 완화를 위한 범용 영상처리 파이프라인 설계**

### 연구 배경 및 필요성
일상에서 사용되는 영상 장비 중 일부는 최신 디지털 카메라처럼 항상 선명한 영상을 제공하지 않는다. 오래된 CCTV, 차량 후방카메라, 저가형 AV 카메라, 무선 영상 송수신 장치, **아날로그 DVR(Digital Video Recorder)** 영상 등에서는 다음과 같은 문제가 나타날 수 있다:
- 화면이 거칠게 보임 (잡음)
- 색이 틀어짐 (색상 왜곡)
- 밝기 및 대비가 불안정 (시인성 저하)

### 프로젝트 목표
본 프로젝트는 **저화질 영상에서 시인성을 해치는 잡음·색왜곡·대비저하를 수업에서 배운 영상처리 기법으로 완화**하는 범용 파이프라인을 설계·구현·평가하는 것을 목표로 한다.

### 최종 성과 (v4.0)
- 🎯 **41개 필터** (13 Phase) — median, wiener, NLM, bilateral, wavelet, BM3D/BM4D, guided filter, rolling guidance, Grey-Edge, deinterlace, super_resolve 등
- 🚀 **54개 preset** — grey-premium (Composite **61.38**), grey-guided-chroma (59.72, 11.7ms), temporal-premium (59.34)
- 📊 **11개 평가 메트릭** — PSNR/SSIM/**MS-SSIM**/VIF/ΔE/Edge/Noise/Detail/Artifact/NIQE/BRISQUE
- 🧠 **Noise-Aware Adaptive Pipeline** — 노이즈 레벨 + 모션 자동 감지 → preset/params 자동 선택
- 🖼️ **Super-Resolution** — Lanczos4/Cubic/EDSR 2x 업스케일 필터
- ⚡ **실시간 처리** — ultralight 4.4ms (227fps), optimal-ultrafast 2.3ms (435fps)
- 🧠 **No-reference quality metrics** — NIQE best 7.23 (chroma-bior4-detail), BRISQUE best 34.04 (temporal-bior4)
- 🔄 **자동 개선 루프** — 2시간마다 benchmark + experiments + commit (cron)

---

## 2. 📊 실험 데이터 구성

### 2.1 데이터 유형

#### Type A: 정량 평가용 실험 데이터
- **목적**: PSNR/SSIM/MS-SSIM/NIQE/BRISQUE 등 11종 메트릭 정량 평가
- **구성**: 깨끗한 디지털 이미지 5종 + 인위적 열화 적용 (기본 + NTSC)
- **특징**: 원본 영상 보존 → 처리 전후 비교 가능

#### Type B: 실제 적용용 데이터
- **목적**: 현실적인 영상 개선 효과 확인
- **구성**: 실제 아날로그 FPV DVR 영상 27개 비디오 (Google Drive 샘플)
- **특징**: 다양한 카메라, 조명, 노이즈 레벨 (Laplacian var 296~9023)

### 2.2 인위 열화 유형

| 열화 유형 | 설명 | 파라미터 | 소스 |
|-----------|------|----------|------|
| Gaussian Noise | 화면 전체에 깔린 거친 잡음 | σ = 25 | 기본 |
| Impulse Noise | 점처럼 튀는 Salt & Pepper 잡음 | p = 0.03 | 기본 |
| Color Bias | RGB 채널별 편향 | R:1.3, G:0.7, B:1.1 | 기본 |
| Brightness Reduction | 어둡고 저대비 | γ = 0.5 | 기본 |
| Periodic Noise | 주기적 사선 잡음 | freq=25, amp=30 | 기본 |
| **NTSC Dot Crawl** | 3.58MHz 크로마 누설 → 루마 | — | **zhuker/ntsc** |
| **NTSC Ringing** | 저가 필터에 의한 엣지 에코 | α=0.3~0.99 | **zhuker/ntsc** |
| **NTSC Color Bleeding** | Y/C 지연 오차에 의한 색번짐 | horiz=1~6 | **zhuker/ntsc** |
| **NTSC Chroma Noise** | 채도 영역의 컬러 노이즈 | 0~16384 | **zhuker/ntsc** |
| **VHS Head Switching** | VHS 헤드 전환 노이즈 | — | **zhuker/ntsc** |

---

## 3. 🔧 아키텍처 (v4.0)

### 3.1 패키지 구조

```
smu_sig_prossessing/         — 핵심 패키지 (모듈식)
├── __init__.py              — 패키지 임포트
├── __main__.py              — python -m 진입점 (process/eval/list-filters)
├── config.py                — PipelineConfig + FilterConfig (54개 preset)
├── filters.py               — 필터 레지스트리 (41개 필터)
├── pipeline.py              — 파이프라인 러너 (설정 기반 순차 실행)
├── degradation.py           — 열화 모듈 (기본 + NTSC)
├── evaluation.py            — PSNR/SSIM/MS-SSIM + 시각화
├── auto_evaluation.py       — 자동 11메트릭 + Composite Score (MS-SSIM 포함)
├── eval_viz.py              — 시각화 (radar/bar/grid/정성시트)
├── adaptive.py              — Adaptive pipeline with motion detection
├── noise_estimator.py       — 노이즈 타입/강도 추정
├── noise_estimation.py      — 노이즈 레벨 기반 적응형 파라미터 추정
└── ntsc_plugin.py           — zhuker/ntsc 카피 (ringPattern.npy 포함)

calculate_niqe.py            — NIQE 독립 모듈 (patch-based MVG distance)
main.py                      — CLI 진입점 (54개 preset 지원)
run_v33_benchmark.py         — 통합 벤치마크 스크립트
scripts/
├── iter_loop.py             — 2시간 자동 개선 루프
├── run_multi_video_benchmark.py — 5개 비디오 크로스 검증
├── run_filter_interaction.py/v2/v3 — Filter interaction analysis
├── iter9_batch_eval.py      — 27프레임 일괄 평가
├── iter9_compare.py         — HTML 비교 페이지 생성
└── ... (기타 실험 스크립트)
```

### 3.2 파이프라인 아키텍처

```
┌─────────────┐    ┌──────────────┐    ┌───────────────────┐    ┌──────────────────────┐
│  입력 영상   │───→│  열화 (NTSC  │───→│  모듈식 필터       │───→│  11메트릭 평가       │
│ (Clean/DVR) │    │  + 기본)     │    │  PipelineConfig   │    │ PSNR/SSIM/MS-SSIM/   │
└─────────────┘    └──────────────┘    └───────────────────┘    │ NIQE/BRISQUE/VIF/... │
                                              │                  └──────────────────────┘
                                     ┌────────┴────────┐
                                     │  FilterConfig 1  │
                                     │  FilterConfig 2  │
                                     │  FilterConfig N  │
                                     └─────────────────┘
```

### 3.3 PipelineConfig (필터 모듈화)

각 필터는 `FilterConfig(name, enabled, params)`로 독립적으로 제어:

```python
cfg = PipelineConfig(label="My Custom Pipeline")
cfg.add("median", ksize=3)                 # ON
cfg.add("guided_filter", radius=3, eps=100.0)  # ON
cfg.add("wavelet", wavelet="db4", level=2) # ON
cfg.add("grey_edge", strength=0.22)        # v3.4: Grey-Edge color constancy
cfg.add("channel_correction")             # ON (default params)
cfg.add("histogram_eq_clahe", enabled=False)  # OFF
cfg.add("super_resolve", scale=2, method="lanczos4")  # v4.0: Super-Resolution
```

### 3.4 v4.0 Architecture Changes

#### Noise-Aware Adaptive Pipeline
- `noise_estimation.py` — 입력 영상의 노이즈 레벨 (Laplacian variance) 및 타입 추정
- `adaptive.py` — 추정된 노이즈에 따라 preset과 파라미터 자동 선택
  - Low noise → ultralight preset (4.4ms)
  - Medium noise → balanced preset
  - High noise → quality preset (grey-premium 등)
  - Motion detection → temporal branch 분기

#### Super-Resolution Filter
- `super_resolve` 필터 (`filters.py`에 추가)
  - Lanczos4 2x (fast, 기본)
  - Cubic 2x (balanced)
  - EDSR 2x (quality, 가중치 파일 필요)
- 파이프라인 마지막 단계에서 적용, 저해상도 아날로그 영상의 디테일 복원

#### MS-SSIM Metric
- `evaluation.py` — `sewar` 라이브러리 활용 MS-SSIM 계산
- `auto_evaluation.py` — Composite Score에 MS-SSIM 포함 (11-metric)
- 3개의 무참조 메트릭 (NIQE, BRISQUE, MS-SSIM) → 참조 필요 없는 평가 강화

### 3.5 최종 Benchmark (test_small.jpg, basic 0.5, 11-metric)

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

No-reference metric 기준:
- 🥇 **NIQE**: chroma-bior4-detail (**7.23**)
- 🥇 **BRISQUE**: temporal-bior4 (**34.04**)
- ⚡ **실시간**: ultralight (**4.4ms**, 227fps)

---

## 4. 🎯 개선 사항 (v1.0 → v4.0)

| 항목 | v3.8 | v4.0 | 효과 |
|------|------|------|------|
| **필터 수** | 40개 | **41개** (+1) | Super-Resolution 업스케일 지원 |
| **Preset 수** | 50개 | **54개** (+4) | Adaptive + SR preset 확장 |
| **평가 메트릭** | 10개 | **11개** (+MS-SSIM) | 다중 스케일 구조 유사도 평가 |
| **Super-Resolution** | 없음 | Lanczos4/Cubic/EDSR 2x | 저해상도 영상 디테일 복원 |
| **Adaptive Pipeline** | Motion-aware branch | **Noise-Aware** + motion | 노이즈 레벨 기반 preset 자동 선택 |
| **Composite Score** | 59.04 (v3.8 grey-premium) | **61.38** (v4.0 grey-premium) | **+3.96% 향상** |
| **NTSC 열화** | 유지 | 유지 + Super-Resolution preset | 더 넓은 범위 커버 |
| **MS-SSIM metric** | 없음 | 추가 (sewar 라이브러리) | 다중 스케일 평가 가능 |
| **자동 개선** | 2시간 cron loop | 유지 + MS-SSIM 포함 | 지속적 성능 개선 |

---

## 5. 📈 평가 방법

### 5.1 정량적 평가 (Quantitative) — 11개 메트릭

| 메트릭 | 단위 | 방향 | 설명 |
|--------|------|------|------|
| PSNR | dB | ↑ | Peak Signal-to-Noise Ratio |
| SSIM | — | ↑ | Structural Similarity |
| MS-SSIM | — | ↑ | Multi-Scale Structural Similarity |
| Color Fidelity | ΔE | ↓ | CIEDE2000 색차 |
| Edge Retention | ratio | ↑ | Canny edge 비율 (1.0=원본 동일) |
| Noise Level | Lap. var | ↓ | Laplacian 분산 |
| Detail Recovery | ratio | ↑ | 고주파 에너지 보존율 |
| Artifact Score | score | ↓ | Ringing+blocking+overshooting |
| VIF | — | ↑ | Visual Information Fidelity |
| NIQE | score | ↓ | Natural Image Quality Evaluator (0~20) |
| BRISQUE | score | ↓ | Blind/Referenceless Image Spatial Quality Evaluator (0~100) |

**Composite Score** = 가중 평균 (11개 메트릭 정규화 후 합산, range 0~100)

### 5.2 정성적 평가 (Qualitative)
- **대상**: 실제 아날로그 FPV DVR 영상 27개
- **방법**: 처리 전/후 이미지 나란히 비교 (iter9_comparison.html)
- **분석**: NTSC 아티팩트 제거율, 색상 자연스러움, 엣지 보존 정도
- **도구**: `python run_auto_eval.py --qualitative-only`

---

## 6. 📋 실행 방법

```bash
# 기본 이미지 처리
python main.py -p "input/*.jpg" --preset grey-premium

# 실제 아날로그 영상 처리 (degrade 없음)
python main.py -p "input/analog_whoop_footage.mp4" --preset grey-premium --degrade none

# Super-Resolution 업스케일 적용
python main.py -p "input/analog_whoop_footage.mp4" --preset grey-premium --degrade none --sr 2x

# 모든 preset 벤치마크
python run_v33_benchmark.py

# 특정 preset 자동 평가 (11 메트릭)
python run_auto_eval.py -i input/test_small.jpg --presets grey-premium,chroma-focus

# 다중 비디오 크로스 검증
python scripts/run_multi_video_benchmark.py

# 필터 목록 확인
python main.py --list-filters
```

---

## 7. 📚 참고 자료

자세한 참고 문헌은 `docs/reference.md` 참조.

### 외부 라이브러리
- [OpenCV](https://opencv.org) — Apache 2.0 (영상 처리 기반)
- [scikit-image](https://scikit-image.org) — BSD 3-Clause (TV denoising)
- [PyWavelets](https://pywavelets.readthedocs.io) — MIT (웨이블릿 변환)
- [BM3D/BM4D](https://github.com/meric7784/bm3d) — MIT (블록 매칭 필터링)
- [BRISQUE](https://github.com/bukalapak/pybrisque) — MIT (무참조 품질 평가)
- [sewar](https://github.com/andrewek/sewar) — MIT (MS-SSIM, VIF 등 full-reference metrics)
- [opencv-contrib-python](https://github.com/opencv/opencv_contrib) — Apache 2.0 (추가 OpenCV 모듈)
- [NTSC 시뮬레이터 (zhuker/ntsc)](https://github.com/zhuker/ntsc) — MIT (아날로그 비디오 시뮬레이션)

### 핵심 참고 논문
- Buades et al. (2005) — Non-Local Means
- Dabov et al. (2007) — BM3D
- Tomasi & Manduchi (1998) — Bilateral Filter
- He et al. (2010) — Guided Filter
- Zhang et al. (2014) — Rolling Guidance Filter
- van de Weijer et al. (2007) — Grey-Edge Color Constancy
- Wang et al. (2003) — MS-SSIM (Multi-Scale SSIM)
- Mittal et al. (2012/2013) — BRISQUE / NIQE
- Perona & Malik (1990) — Anisotropic Diffusion
- Rudin et al. (1992) — Total Variation

### 프로젝트 문서
- `docs/explanations.md` — 40개 필터 상세 설명 (원리/작동/구현)
- `docs/reference.md` — 오픈소스 및 논문 레퍼런스
- `plan/design_change_log.md` — 전체 설계 변경 이력
