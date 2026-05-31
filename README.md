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
  __init__.py               — 패키지 임포트
  config.py                 — PipelineConfig (필터 ON/OFF + 파라미터)
  filters.py                — 필터 레지스트리 (16개 필터)
  pipeline.py               — 파이프라인 러너
  degradation.py            — 열화 (기본 + NTSC)
  evaluation.py             — PSNR/SSIM + 시각화
  ntsc_plugin.py            — zhuker/ntsc (ringPattern.npy 포함)

scripts/
  run_pipeline.py           — 정적 이미지 파이프라인 (합성 + 실제 이미지)
  run_video.py              — 비디오 파이프라인
  run_evaluation.py         — 정성적/정량적 평가 도구

input/                      — 입력 데이터 (실제 영상 + 이미지)
  REAL_WORLD_PICTURE.jpg    — 실제 풍경 사진 (4000×1848)
  analog_whoop_footage.mp4  — 아날로그 FPV 드론 영상 (64초, 854×480)
  digital_whoop_footage.mp4 — 디지털 FPV 드론 영상 (118초, 1440×1080)

plan/
  Project_plan.md           — 프로젝트 계획서 (v2.0)
  Implementation_Plan.md    — 구현 계획서 (v2.0)
```

## 설치

```bash
# 의존성 설치
pip install opencv-python-headless scikit-image scipy numpy

# 또는 pyproject.toml 기반
pip install .
```

## 사용법

### 기본 실행

```bash
# 정적 이미지 파이프라인 (합성 이미지 5종 생성 + 평가)
python main.py

# 비디오 파이프라인 (합성 비디오 생성 + 평가)
python main.py --video

# 등록된 필터 목록 확인
python main.py --list-filters
```

### 실제 데이터 평가

```bash
# 정성적 평가 — 실제 아날로그 영상 → 시각적 비교 이미지 생성
# (analog_whoop_footage.mp4: 아날로그 FPV, 64초, 1분 샘플)
python -c "
import cv2, os
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl

cap = cv2.VideoCapture('input/analog_whoop_footage.mp4')
for idx in [0, 500, 1000, 1500]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret: continue
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w*4, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    canvas[:, w:2*w] = pl.apply_pipeline(frame, PipelineConfig.wiener_only())
    canvas[:, 2*w:3*w] = pl.apply_pipeline(frame, PipelineConfig.edge_preserving())
    canvas[:, 3*w:] = pl.apply_pipeline(frame, PipelineConfig.research_best())
    cv2.imwrite(f'output/qualitative_frame_{idx}.png', canvas)
cap.release()
"

# 정량적 평가 — 디지털 영상에 인위 열화 후 PSNR/SSIM 측정
# (digital_whoop_footage.mp4: 디지털 FPV, 118초, 2분 샘플)
python -c "
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing import pipeline as pl
from smu_sig_prossessing.degradation import degrade_image
import cv2, numpy as np

cap = cv2.VideoCapture('input/digital_whoop_footage.mp4')
clean_frames = []
for i in range(0, 3000, 200):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, f = cap.read()
    if ret: clean_frames.append(cv2.resize(f, (640, 480)))
cap.release()

degraded = [degrade_image(f) for f in clean_frames]

for name, cfg in [('Wiener', PipelineConfig.wiener_only()),
                  ('Edge-Preserving', PipelineConfig.edge_preserving())]:
    p, s = [], []
    for c, d in zip(clean_frames, degraded):
        r = pl.apply_pipeline(d, cfg)
        p.append(psnr(c, r)); s.append(ssim(c,r,channel_axis=-1))
    print(f'{name}: PSNR={np.mean(p):.2f}±{np.std(p):.2f} dB, SSIM={np.mean(s):.4f}')
"
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

# 적용
img = cv2.imread('input/REAL_WORLD_PICTURE.jpg')
result = pl.apply_pipeline(img, my_cfg)

# 특정 필터만 끄기
my_cfg.disable("nlm")
result_no_nlm = pl.apply_pipeline(img, my_cfg)

# 파라미터 변경
my_cfg.get("wiener").params["noise_var"] = 400
```

## PipelineConfig 프리셋

| 프리셋 | 필터 구성 | 속도 | 용도 |
|--------|-----------|------|------|
| **Wiener Only** (권장) | Median→Wiener→FFT→Channel→Gamma→HistEq | ⚡빠름 | 일반적인 잡음 제거 |
| **Edge-Preserving** | Median→NLM→Wiener→FFT→Channel→Gamma→HistEq | 🐢느림 | 엣지 보존이 중요할 때 |
| **Aggressive** | Median(k=5)→Bilateral→Wiener→FFT→Channel→Gamma→HistEq | 보통 | 매우 noisy한 영상 |
| **Research Best** | Median→NLM→Wiener→FFT→Bilateral→Channel→Gamma→HistEq | 🐢느림 | 최고 품질 |

## 등록된 필터 (16개)

| 필터명 | 설명 | 주요 파라미터 |
|--------|------|-------------|
| `median` | 미디언 필터 (impulse 잡음) | `ksize=3` |
| `wiener` | 위너 필터 (주파수 도메인, **권장**) | `noise_var=625` |
| `nlm` | Non-Local Means (엣지 보존) | `h=10`, `template_window=7` |
| `bilateral` | Bilateral Filter (엣지 보존 평활화) | `d=9`, `sigma_color=75` |
| `fft_notch` | FFT 노치 필터 (주기적 잡음) | `threshold_percentile=99.5` |
| `gamma` | 감마 보정 | `gamma=1.8` |
| `histogram_eq` | 히스토그램 평활화 (YUV) | — |
| `histogram_eq_clahe` | CLAHE (적응형) | `clip_limit=2.0` |
| `channel_correction` | RGB 채널 평균 보정 | `clamp_min=0.7` |
| `unsharp_mask` | 언샤프 마스크 (블러 복원) | `strength=1.0` |
| `deblur_wiener` | Wiener 디컨볼루션 (디블러) | `kernel_size=5` |
| `log_transform` | 로그 변환 | `c=40` |
| `gaussian_lowpass` | 가우시안 로우패스 (참고용) | `sigma=1.5` |

## 평가 결과 (실제 데이터)

### 정량적 평가 — digital_whoop_footage.mp4
```
Method                  PSNR vs Clean    Denoise Strength
──────────────────────────────────────────────────────────
Degraded (baseline)     10.57±0.45 dB    (reference)
Denoise Only (Wiener)    4.28±0.87 dB    5.86±0.97 dB
Full Pipeline            8.28±0.85 dB    7.10±0.35 dB   ← Best
```

### 정성적 평가 — analog_whoop_footage.mp4
- 10개 대표 프레임 × 3개 파이프라인 (Wiener / Edge-Preserving / Research Best)
- 4-way comparison 이미지 → `output/analog_qualitative_frame*.png`

## 의존성
- Python 3.11+
- opencv-python-headless
- scikit-image
- scipy
- numpy

## 참고
- [zhuker/ntsc](https://github.com/zhuker/ntsc) — NTSC 아날로그 비디오 시뮬레이터
- 기말 프로젝트 — 상명대학교 영상처리 (6팀 김승민, 김원석)
