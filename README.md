# 아날로그 영상 잡음 완화 파이프라인 v2.0

영상처리 기말 프로젝트 — 6팀 (김승민, 김원석)

## 프로젝트 개요
아날로그 FPV DVR 영상에서 시인성을 해치는 잡음·색왜곡·대비저하를 기본 영상처리 기법으로 완화하는 범용 파이프라인.

**v2.0 주요 개선**:
- 🎬 **NTSC 아날로그 시뮬레이터 통합** (zhuker/ntsc) — realistic dot crawl, ringing, color bleeding
- 🔧 **필터 모듈화** — 각 단계를 ON/OFF 가능한 FilterConfig로 분리
- 🎯 **Wiener 필터 단독 사용** — Gaussian Lowpass 제거 (불필요한 blur 최소화)
- 🧪 **추가 연구 기법** — NLM, Bilateral, CLAHE, Unsharp Mask, Wiener Deconvolution
- 📂 **CLI 기반 실제 파일 처리** — `-p "glob"` + `--preset` 플래그

## 구조
```
smu_sig_prossessing/        — 핵심 패키지 (모듈식)
  __init__.py               — 패키지 임포트
  config.py                 — PipelineConfig (필터 ON/OFF + 파라미터)
  filters.py                — 필터 레지스트리 (16개 필터)
  pipeline.py               — 파이프라인 러너
  degradation.py            — 열화 (기본 + NTSC)
  evaluation.py             — PSNR/SSIM + 시각화
  ntsc_plugin.py            — zhuker/ntsc (ringPattern.npy 포함)

input/                      — 입력 데이터 (실제 영상 + 이미지)
  REAL_WORLD_PICTURE.jpg    — 실제 풍경 사진 (4000×1848)
  analog_whoop_footage.mp4  — 아날로그 FPV 드론 영상 (64초, 854×480)
  digital_whoop_footage.mp4 — 디지털 FPV 드론 영상 (118초, 1440×1080)

output/
  raw/                      — 원본 입력 복사본 (degraded product)
  processed/                — 파이프라인 적용 결과

plan/
  Project_plan.md           — 프로젝트 계획서 (v2.0)
  Implementation_Plan.md    — 구현 계획서 (v2.0)
```

## 설치

```bash
pip install opencv-python-headless scikit-image scipy numpy
```

## 사용법

### 기본 — 경로 패턴 + 프리셋

```bash
# 단일 이미지 처리
python main.py -p "input/REAL_WORLD_PICTURE.jpg" --preset wiener-only

# 모든 이미지 한번에 처리
python main.py -p "input/*.jpg" --preset edge-preserving

# 비디오 처리 (주의: 프레임당 ~0.3초 소요)
python main.py -p "input/analog_whoop_footage.mp4" --preset wiener-only

# 여러 확장자 동시 처리
python main.py -p "input/*.{jpg,mp4}" --preset research-best

# 필터 목록 보기
python main.py --list-filters
```

### 프리셋 목록

| 프리셋 | 플래그 | 필터 구성 | 속도 | 용도 |
|--------|--------|-----------|------|------|
| **Wiener Only** (권장) | `--preset wiener-only` | Median→Wiener→FFT→Channel→Gamma→HistEq | ⚡빠름 | 일반적인 잡음 제거 |
| **Edge-Preserving** | `--preset edge-preserving` | Median→NLM→Wiener→FFT→Channel→Gamma→HistEq | 🐢느림 | 엣지 보존이 중요할 때 |
| **Aggressive** | `--preset aggressive` | Median(k=5)→Bilateral→Wiener→FFT→Channel→Gamma→HistEq | 보통 | 매우 noisy한 영상 |
| **Research Best** | `--preset research-best` | Median→NLM→Wiener→FFT→Bilateral→Channel→Gamma→HistEq | 🐢느림 | 최고 품질 |

### 출력 구조

```
output/
  raw/
    REAL_WORLD_PICTURE.png          — 입력 원본 (RAW)
    analog_whoop_footage.mp4         — 입력 비디오 원본
  processed/
    REAL_WORLD_PICTURE.png          — 파이프라인 적용 결과
    analog_whoop_footage.mp4         — 파이프라인 적용 결과 비디오
```

### PipelineConfig 프로그래밍 예제

```python
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
import cv2

# 나만의 파이프라인 만들기
my_cfg = PipelineConfig(label="My Custom Pipeline")
my_cfg.add("median", ksize=3)                           # ON
my_cfg.add("wiener", noise_var=625)                     # ON
my_cfg.add("fft_notch", enabled=False)                  # OFF (주기적 잡음 없음)
my_cfg.add("channel_correction")                        # ON (기본 파라미터)
my_cfg.add("nlm", h=8, template_window=7)               # ON
my_cfg.add("gamma", gamma=1.5)                          # ON
my_cfg.add("histogram_eq_clahe", clip_limit=2.0)        # ON (CLAHE)

img = cv2.imread('input/REAL_WORLD_PICTURE.jpg')
result = pl.apply_pipeline(img, my_cfg)
cv2.imwrite('output/processed/my_result.png', result)
```

## 등록된 필터 (16개)

```
  bilateral              — Bilateral filter (edge-preserving smoothing)
  channel_correction     — RGB 채널 평균 보정
  deblur_wiener          — Wiener deconvolution (디블러)
  fft_notch              — FFT 노치 필터 (주기적 잡음)
  gamma                  — 감마 보정
  gaussian_lowpass       — Gaussian LP (참고용)
  histogram_eq           — 히스토그램 평활화 (YUV)
  histogram_eq_clahe     — CLAHE (적응형)
  histogram_eq_gray      — 그레이스케일 히스토그램 평활화
  log_transform          — 로그 변환
  median                 — 미디언 필터 (impulse 잡음)
  nlm                    — Non-Local Means (엣지 보존)
  nlm_gray               — NLM on luminance only
  unsharp_mask           — 언샤프 마스크 (블러 복원)
  wiener                 — 위너 필터 (주파수 도메인, 권장)
```

## 성능 참고

| 작업 | 해상도 | 처리 시간 |
|------|--------|-----------|
| 이미지 (wiener-only) | 4000×1848 | ~2초 |
| 이미지 (edge-preserving) | 4000×1848 | ~8초 |
| 비디오 프레임 (wiener-only) | 854×480 | ~0.3초/프레임 |
| 비디오 프레임 (edge-preserving) | 854×480 | ~1.1초/프레임 |

※ FFT 기반 Wiener 필터가 주요 병목입니다. 전체 비디오 처리 시 시간이 오래 걸릴 수 있습니다.

## 의존성
- Python 3.11+
- opencv-python-headless
- scikit-image
- scipy
- numpy

## 참고
- [zhuker/ntsc](https://github.com/zhuker/ntsc) — NTSC 아날로그 비디오 시뮬레이터
- 기말 프로젝트 — 상명대학교 영상처리 (6팀 김승민, 김원석)
