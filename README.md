# 아날로그 영상 잡음 완화 파이프라인 v3.0

영상처리 기말 프로젝트 — 6팀 (김승민, 김원석)

## 프로젝트 개요
아날로그 FPV DVR 영상에서 시인성을 해치는 잡음·색왜곡·대비저하를 기본 영상처리 기법으로 완화하는 범용 파이프라인.

**v3.0 주요 개선**:
- 🆕 **Spatio-temporal denoising** — optical flow motion compensation, frame averaging
- 🎯 **7개 신규 필터** (총 23개) — guided filter, TV denoising, anisotropic diffusion, block DCT, temporal
- ⚡ **Ablation 최적화 preset** — optimized-fast (PSNR 19.00, edge-preserve 대비 18× 빠름)
- 🔬 **Batch ablation framework** — `run_ablation.py`로 자동 preset/param/filter ON-OFF 테스트
- 📋 **Heuristic preview** — `--sample N` 플래그로 전체 처리 전 N프레임 미리보기
- 🖼️ **실제 아날로그 아티팩트**로 기본 degrade 변경 (diagonal periodic noise 제거 → horizontal line noise + dropout)

## 구조
```
smu_sig_prossessing/        — 핵심 패키지 (모듈식)
  __init__.py               — 패키지 임포트
  config.py                 — PipelineConfig (14개 preset)
  filters.py                — 필터 레지스트리 (23개 필터)
  pipeline.py               — 파이프라인 러너
  degradation.py            — 열화 (기본 + NTSC, horizontal line noise + dropout)
  evaluation.py             — PSNR/SSIM + 시각화
  ntsc_plugin.py            — zhuker/ntsc (ringPattern.npy 포함)

input/                      — 입력 데이터 (실제 영상 + 이미지)
  REAL_WORLD_PICTURE.jpg    — 실제 풍경 사진 (4000×1848)
  analog_whoop_footage.mp4  — 아날로그 FPV 드론 영상 (64초, 854×480)
  digital_whoop_footage.mp4 — 디지털 FPV 드론 영상 (118초, 1440×1080)
  test_small.jpg            — 축소 테스트 이미지 (1600×740)

output/
  raw/                      — 열화된 복사본 (degraded)
  processed/                — 파이프라인 적용 결과
  ablation/                 — Ablation study 결과 (CSV + MD + grid)

scripts/
  run_ablation.py           — Batch preset/filter ON-OFF/param tuning 평가
  compare_presets.py        — 프레임별 preset 비교 그리드 생성
  iteration4.py             — Iteration 4: param tuning + cascade + temporal test

plan/
  Project_plan.md           — 프로젝트 계획서
  design_change_log.md      — 설계 변경 이력 (전체 실험 데이터 포함)
```

## 설치

```bash
pip install opencv-python-headless scikit-image scipy numpy pywavelets
```

## 사용법

### 기본 — 실제 영상 처리 (degrade 없이)

```bash
# 실제 아날로그 영상 처리 (degrade 없음)
python main.py -p "input/analog_whoop_footage.mp4" --preset video-enhanced --degrade none

# 실제 아날로그 영상 + wavelet (NTSC 아티팩트에 강함)
python main.py -p "input/analog_whoop_footage.mp4" --preset wavelet-denoise --degrade none
```

### 기본 — 합성 degrade + 복원 테스트

```bash
# 단일 이미지
python main.py -p "input/REAL_WORLD_PICTURE.jpg" --preset optimized-fast --strength 0.5

# 모든 이미지 한번에 처리
python main.py -p "input/*.jpg" --preset optimized-fast

# 비디오 처리
python main.py -p "input/digital_whoop_footage.mp4" --preset optimized-fast
```

### Heuristic 미리보기 (전체 처리 전 샘플 확인)

```bash
# 비디오의 첫 5프레임만 처리하여 비교 이미지 생성
python main.py -p "input/analog_whoop_footage.mp4" --preset video-enhanced --degrade none --sample 5
```

### Degrade 종류

| 플래그 | 설명 |
|--------|------|
| `--degrade none` | **실제 영상 처리** — degrade 생략 (권장) |
| `--degrade basic` | 합성 노이즈 (Gaussian + impulse + color + horizontal line + dropout) |
| `--degrade ntsc-light/medium/heavy` | NTSC 아날로그 시뮬레이션 (dot crawl, ringing, chroma noise) |

`--strength 0.0~1.0`로 basic degrade 강도 조절 (기본값 0.5).

### 필터 목록 보기

```bash
python main.py --list-filters
```

## 프리셋 목록 (14개)

### 권장 Preset

| Preset | 필터 구성 | PSNR | 속도 | 용도 |
|--------|----------|------|------|------|
| **video-enhanced** 🆕 | Med→Guided→Wavelet→Channel→CLAHE→Unsharp | **18.8** | **0.12s** | **아날로그 비디오 권장** (washing 없음) |
| **optimized-fast** 🏆 | Med→Bilateral(σ=110)→Channel | **19.00** | **0.12s** | **일반 목적 최적** (NLM 제거, 18× 속도) |
| **optimized-quality** | Med→Bilateral→Wavelet→Channel→Unsharp | 18.83 | 0.28s | 고품질 융합 |
| **wavelet-denoise** | Wavelet→Bilateral→Channel→Unsharp | 28.13(NTSC) | **0.11s** | **NTSC 아날로그 영상 최적** |

### 기타 Preset

| Preset | 필터 구성 | 특징 |
|--------|----------|------|
| edge-preserve | Med→NLM→Bilateral→Channel→Unsharp | 기본 (NLM redundant) |
| max-quality | Med→Bilateral(σ=150)→Channel→CLAHE | 최고 PSNR, 약간 느림 |
| guided-denoise | Guided→Channel→Unsharp | 빠름, edge 보존 우수 |
| tv-denoise | TV→Channel→Unsharp | small noise 제거 |
| fast-denoise | Bilateral→Channel→Unsharp | **0.05s** 실시간 |
| st-video | Temporal→Bilateral→Guided→Channel→Unsharp | Spatio-temporal 비디오 |
| nlm-denoise | NLM→Bilateral→Channel→Unsharp | NLM 중심 |
| wiener-denoise | Med→Wiener→FFT→Channel→Unsharp→Gamma | 주기적 잡음용 |
| aggressive | Med→NLM→Bilateral→Wiener→Channel→Unsharp | 극한 노이즈 |
| research-best | Med→NLM→Bilateral→Wiener→Channel→Unsharp | 최고 SSIM |

### 성능 참고 (854×480 프레임)

| Preset | 시간/프레임 | Real-time 가능? |
|--------|------------|----------------|
| fast-denoise | **0.02s** | ✅ 50fps |
| temporal-averaging | **0.015s** | ✅ **66fps** |
| optimized-fast | **0.08s** | ⚠️ 12fps |
| video-enhanced | **0.12s** | ⚠️ 8fps |
| wavelet-denoise | **0.11s** | ⚠️ 9fps |
| edge-preserve | 0.78s | ❌ 1.3fps |

## Degrade 노이즈 종류

### Basic degrade (기본)
- Gaussian noise (σ ∝ strength)
- Salt & pepper impulse noise
- Color bias (R↑, G↓, B↑)
- Brightness reduction (gamma)
- **Horizontal scan-line noise** ← realistic analog artifact
- **Dropout streaks** ← realistic VHS tape artifact

Diagonal periodic noise는 일반적인 아날로그 아티팩트가 아니므로 **제거**되었습니다.
대신 `horizontal_line_noise` (CRT 스캔라인 간섭) + `dropout` (VHS 테이프 손상)으로 대체.

### NTSC degrade (zhuker/ntsc)
- Dot crawl
- Ringing
- Color bleeding (horizontal/vertical)
- Chroma noise + phase noise
- Frequency noise
- VHS head switching (vhs 모드)

## Ablation Study

```bash
# 모든 preset 비교
python run_ablation.py

# 특정 preset만 비교
python run_ablation.py --presets optimized-fast,video-enhanced,wavelet-denoise

# Filter ON/OFF ablation (edge-preserve 기준)
python run_ablation.py --ablation median,nlm,bilateral,channel_correction

# Param tuning (bilateral sigma 스윕)
python run_ablation.py --param-tune "bilateral.sigma_color,50,75,100,125,150"

# NTSC degrade 테스트
python run_ablation.py --degrade ntsc-heavy --presets wavelet-denoise,optimized-fast

# 결과물
# output/ablation/{image_name}/ablation_{timestamp}.csv  (데이터)
# output/ablation/{image_name}/ablation_{timestamp}.md   (보고서)
# output/ablation/{image_name}/grid_{timestamp}.png       (시각 비교)
```

핵심 발견:
- **NLM은 median+bilateral과 완전 중복** → 제거해도 PSNR 동일, 25× 속도 향상
- **Bilateral σ=110~150**이 최적 PSNR (σ↑ = PSNR↑, edge↓)
- **Wavelet**은 NTSC 아티팩트에 가장 강함 (SSIM 0.8795)
- **Guided filter**가 bilateral보다 edge 보존 우수 (washing 없음)
- **필터 순서**: guided→wavelet > wavelet→guided > wavelet→bilateral > bilateral→wavelet

## 자동 정량+정성 평가 (v3.1)

```bash
# 전체 preset 정량+정성 평가 (7개 메트릭 + Composite Score + 시각화)
python run_auto_eval.py -i input/test_small.jpg --degrade basic --strength 0.5

# 특정 preset만
python run_auto_eval.py -i input/test_small.jpg --presets optimized-fast,video-enhanced

# 실제 아날로그 영상
python run_auto_eval.py -i input/analog_whoop_footage.mp4 --degrade none --sample 3

# 정성 평가 시트만 (메트릭 계산 없이)
python run_auto_eval.py -i input/test_small.jpg --qualitative-only
```

### 7개 자동 메트릭

| 메트릭 | 단위 | 방향 | 설명 |
|--------|------|------|------|
| PSNR | dB | ↑ | Peak Signal-to-Noise Ratio |
| SSIM | — | ↑ | Structural Similarity |
| Color Fidelity | ΔE | ↓ | CIE76 LAB 색차 |
| Edge Retention | ratio | ↑ | Canny edge 비율 (1.0=원본 동일) |
| Noise Level | Lap. var | ↓ | Laplacian 분산 |
| Detail Recovery | ratio | ↑ | 고주파 에너지 보존율 |
| Artifact Score | score | ↓ | Ringing+blocking+overshooting |

### 산출물

```
output/eval/
  auto_eval_{ts}_{name}.csv       — 정량 데이터
  auto_eval_{ts}_{name}.json      — JSON
  auto_eval_{ts}_{name}.md        — Markdown 리포트
  auto_eval_{ts}_{name}_radar.png — Radar chart
  auto_eval_{ts}_{name}_bar.png   — Bar chart
  auto_eval_{ts}_{name}_grid.png  — 비교 그리드
  auto_eval_{ts}_{name}_qual.png  — 정성 평가 시트
  qualitative_notes_{ts}.md       — 정성 코멘트 템플릿
```

## 등록된 필터 (23개)

```
  anisotropic_diffusion     — Perona-Malik anisotropic diffusion
  bilateral                 — Bilateral filter (edge-preserving smoothing)
  channel_correction        — RGB 채널 평균 보정
  deblur_wiener             — Wiener deconvolution (디블러)
  fft_notch                 — FFT 노치 필터 (주기적 잡음)
  gamma                     — 감마 보정
  gaussian_lowpass          — Gaussian LP (참고용)
  guided_filter             — Guided filter (edge-preserving local linear model)
  histogram_eq              — 히스토그램 평활화 (YUV)
  histogram_eq_clahe        — CLAHE (적응형)
  histogram_eq_gray         — 그레이스케일 히스토그램 평활화
  log_transform             — 로그 변환
  median                    — 미디언 필터 (impulse 잡음)
  nlm                       — Non-Local Means (edge 보존)
  nlm_gray                  — NLM on luminance only
  patch_collaborative       — Block DCT thresholding
  temporal_average          — Sliding multi-frame averaging
  temporal_motion           — Farneback optical flow temporal denoising
  temporal_spatial          — Combined bilateral + motion compensation
  tv_denoise                — Total Variation ROF (Chambolle)
  unsharp_mask              — 언샤프 마스크
  wavelet                   — Wavelet denoising (db4, VisuShrink)
  wiener                    — Wiener filter (주파수 도메인)
```

## PipelineConfig 프로그래밍 예제

```python
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
import cv2

# 나만의 파이프라인 만들기
cfg = PipelineConfig(label="My Custom Pipeline")
cfg.add("median", ksize=3)
cfg.add("guided_filter", radius=3, eps=100.0)
cfg.add("wavelet", wavelet="db4", level=2, threshold_mode="soft")
cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
cfg.add("histogram_eq_clahe", clip_limit=1.5, tile_size=8)

img = cv2.imread('input/REAL_WORLD_PICTURE.jpg')
result = pl.apply_pipeline(img, cfg)
cv2.imwrite('output/processed/my_result.png', result)
```

## Temporal Filter 사용 (비디오)

Temporal filter는 프레임 간 상태를 유지합니다. 새 비디오 처리 시 자동 리셋:

```python
from smu_sig_prossessing.filters import reset_temporal_state

# 새 비디오 시작 전
reset_temporal_state()

# temporal_motion 필터가 포함된 pipeline 사용
cfg = PipelineConfig(label="Motion Compensated")
cfg.add("temporal_motion", strength=0.3)
cfg.add("bilateral", d=5, sigma_color=30, sigma_space=30)
cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
```

## 의존성
- Python 3.11+
- opencv-python-headless (4.9+)
- scikit-image
- scipy
- numpy
- PyWavelets

## 참고
- [zhuker/ntsc](https://github.com/zhuker/ntsc) — NTSC 아날로그 비디오 시뮬레이터
- 설계 변경 이력: [plan/design_change_log.md](plan/design_change_log.md)
- 기말 프로젝트 — 상명대학교 영상처리 (6팀 김승민, 김원석)
