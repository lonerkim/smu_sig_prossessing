# 아날로그 영상 잡음 완화 — 구현 계획서 (v2.0)

> **6팀** | 지능IoT융합전공 김승민 · 컴퓨터과학과 김원석
> 과목: 영상처리 기말 프로젝트
> 업데이트: 2026-05-31

---

## 1. 프로젝트 목표 (한 줄)

아날로그 FPV DVR 영상에서 시인성을 해치는 **잡음·색왜곡·대비저하**를 수업 기법으로 완화하되, **가우시안 로우패스와 위너 필터를 동시에 사용하지 않고 위너 필터만 사용**하며, **NTSC 아티팩트 시뮬레이터를 열화에 통합**하고, **모든 필터를 모듈화(ON/OFF 가능)** 한 범용 파이프라인.

**범위 (Scope)**
- ✅ 할 것: NTSC 통합 열화, 위너 중심 디노이징, NLM/Bilateral/CLAHE/Unsharp/Deblur 추가, PipelineConfig 모듈화, PSNR/SSIM 평가, 정성/정량 평가
- ❌ 안 할 것: 심층학습 디노이징, 완전히 손실된 프레임 복원, 실시간 처리

---

## 2. 아키텍처 (v2.0)

```
smu_sig_prossessing/
├── __init__.py           # 패키지
├── config.py             # PipelineConfig + FilterConfig (모듈식 설정)
├── filters.py            # 필터 레지스트리 (16개 필터 등록)
├── pipeline.py           # 파이프라인 러너
├── degradation.py        # 열화 (기본 5종 + NTSC 5종)
├── evaluation.py         # PSNR/SSIM + 시각화
└── ntsc_plugin.py        # zhuker/ntsc

scripts/
├── run_pipeline.py       # 정적 이미지 파이프라인
├── run_video.py          # 비디오 파이프라인
└── run_evaluation.py     # 정성/정량 평가
```

---

## 3. Phase별 구현 상태

### Phase 0: 열화 (✅ 100%)
- [x] 기본 합성 열화 5종 (Gaussian, Impulse, Color bias, Brightness, Periodic)
- [x] **NTSC 통합** (zhuker/ntsc): light / medium / heavy / vhs 프리셋
- [x] 종합 `degrade_image(use_ntsc=True/False)`

### Phase 1: 잡음 제거 (✅ 100%)
- [x] Median Filter (k=3, k=5)
- [x] **Wiener Filter (주파수 도메인)** — noise_var 조절 가능
- [x] ~~Gaussian Lowpass~~ → **제거** (Wiener만 사용)
- [x] **NLM (Non-Local Means)** — edge-preserving denoising
- [x] **Bilateral Filter** — edge-preserving smoothing
- [x] FFT Notch Filter (periodic noise)

### Phase 2: 색상/대비 보정 (✅ 100%)
- [x] Gamma Correction (γ 조절)
- [x] Log Transform
- [x] Histogram Equalization (YUV, Gray, CLAHE)
- [x] Channel Correction (RGB 평균 보정)

### Phase 3: 디블러/블러 최소화 (✅ 100%)
- [x] **Unsharp Mask** — denoising 후 선명도 복원
- [x] **Wiener Deconvolution** — 주파수 도메인 디블러

### Phase 4: 평가 (✅ 100%)
- [x] 정량 평가: PSNR/SSIM 프레임별 측정 → 평균/표준편차
- [x] 정성 평가: 4-way comparison 이미지 (Degraded/Wiener/Edge/Research)
- [x] 합성 + 실제 영상 모두 지원

---

## 4. PipelineConfig 프리셋

| 프리셋 | 필터 순서 | 특징 |
|--------|-----------|------|
| **Wiener Only** (권장) | Median→Wiener→FFT Notch→Channel→Gamma→HistEq | 가장 빠르고 안정적 |
| **Edge-Preserving** | Median→NLM→Wiener→FFT Notch→Channel→Gamma→HistEq | 엣지 보존 탁월, 느림 |
| **Aggressive** | Median(k=5)→Bilateral→Wiener→FFT Notch→Channel→Gamma→HistEq | 매우 noisy한 영상용 |
| **Research Best** | Median→NLM→Wiener→FFT Notch→Bilateral→Channel→Gamma→HistEq | 최고 품질 (무거움) |

---

## 5. 평가 계획

| 대상 | 방법 | 지표 | 도구 |
|------|------|------|------|
| 합성 영상 | 정량 평가 | PSNR, SSIM | scripts/run_evaluation.py |
| 실제 영상 (Google Drive) | 정성 평가 | 시각 비교 (4-way) | scripts/run_evaluation.py --qualitative |
| 실제 영상 (Google Photos) | 정량 평가 | PSNR, SSIM | scripts/run_evaluation.py --quantitative |

---

## 6. 의존성
- Python 3.11+
- opencv-python-headless
- scikit-image
- scipy
- numpy
