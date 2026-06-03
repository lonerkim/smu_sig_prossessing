# 설계 변경 이력 (Design Change Log)

> **프로젝트**: 아날로그 영상 잡음 완화 파이프라인  
> **팀**: 6팀 — 지능IoT융합전공 김승민 · 컴퓨터과학과 김원석  
> **문서 최종 업데이트**: 2026-05-31

---

## 개요

본 문서는 파이프라인 아키텍처와 알고리즘 선택의 **변경 과정, 근거, 실험 데이터**를 시간순으로 기록한다.
각 결정은 정량적 지표(PSNR/SSIM/Edge retention)와 정성적 평가(시각 비교)를 바탕으로 이루어졌다.

---

## v1.0 → v2.0 (2026-05-31 08:00~09:00)

### 변경: 단일 스크립트 → 모듈식 패키지

**배경**: 초기 코드는 `scripts/pipeline.py`(435줄) + `scripts/video_pipeline.py`(282줄) 단일 파일 구조.  
필터 추가/제거/순서 변경이 어렵고, 실험 반복에 비효율적.

**변경**:
- `smu_sig_prossessing/` 패키지 생성 (8개 모듈)
- `PipelineConfig` + `FilterConfig`로 필터 ON/OFF 및 파라미터 제어
- `FILTER_REGISTRY` 패턴으로 새로운 필터 등록 단순화

**효과**: 필터 추가/제거/재배열이 config 변경만으로 가능해짐.

---

## v2.0 Gaussian LP 제거 (2026-05-31 08:00)

### 변경: Gaussian Lowpass 필터 제거

**근거**:
- Gaussian LP는 단순한 주파수 차단으로 **edge와 noise를 구분하지 못함**
- Wiener 필터가 noise 분산을 추정하여 적응적으로 처리 → Gaussian LP보다 PSNR 우수
- 두 필터를 동시에 사용하면 불필요한 blur 중첩

**실험**: 이전 단계의 Wiener vs Gaussian LP 비교에서 Wiener가 모든 시나리오에서 PSNR/SSIM 우위

**결정**: `gaussian_lowpass()`를 FILTER_REGISTRY에 유지하되 모든 preset에서 제거.

---

## v2.0 NTSC 통합 (2026-05-31 08:00~08:30)

### 변경: zhuker/ntsc 라이브러리 통합

**근거**:
- 기존 합성 노이즈(Gaussian+Impulse+Periodic)는 실제 아날로그 아티팩트와 거리가 있음
- NTSC 시뮬레이터는 **dot crawl, ringing, color bleeding, chroma noise** 등 실제 아날로그 현상 재현
- AV Artifact Atlas(avartifactatlas.com) 기준 아티팩트 분류와 일치

**변경**:
- `ntsc_plugin.py`에 zhuker/ntsc 전체 복사 (라이선스: MIT)
- `degradation.py`에 `add_ntsc_noise()` 추가
- 4단계 강도: `light`, `medium`, `heavy`, `vhs`

**한계**: NTSC 처리는 per-pixel composite encoding으로 매우 느림 (854×480 프레임당 ~0.5~2초)

---

## v2.0 PipelineConfig 모듈화 (2026-05-31 08:00~08:30)

### `FilterConfig` 설계

```python
@dataclass
class FilterConfig:
    name: str         # FILTER_REGISTRY에 등록된 이름
    enabled: bool     # True = 적용, False = 건너뜀
    params: dict      # 필터별 파라미터 (ksize, noise_var, gamma 등)
```

**변경 동기**: 실험마다 필터 조합과 파라미터를 매번 하드코딩 → PipelineConfig 객체로 캡슐화.

**사용 예**:
```python
cfg = PipelineConfig(label="My Experiment")
cfg.add("median", ksize=3)
cfg.add("wiener", noise_var=400)
cfg.disable("fft_notch")  # 특정 조건에서 OFF
cfg.get("wiener").params["noise_var"] = 200  # 파라미터 runtime 변경
```

---

## v2.0 추가 연구 기법 (2026-05-31 08:30~09:00)

### 추가된 필터들

| 필터 | 등록일 | 출처 | 특징 |
|------|--------|------|------|
| `nlm` | v2.0 | OpenCV | Non-Local Means, edge 보존 탁월 |
| `bilateral` | v2.0 | OpenCV | 엣지 보존 평활화, 빠름 |
| `histogram_eq_clahe` | v2.0 | OpenCV | 적응형 contrast 개선 |
| `unsharp_mask` | v2.0 | 자체 구현 | denoising blur 보상 |
| `deblur_wiener` | v2.0 | 자체 구현 | 주파수 도메인 deconvolution |

---

## v2.1 실제 데이터 평가 (2026-05-31 09:00~10:30)

### 입력 데이터 구성

| 파일 | 유형 | 해상도 | 길이 | 평가 목적 |
|------|------|--------|------|-----------|
| `REAL_WORLD_PICTURE.jpg` | 디지털 사진 | 4000×1848 | — | 정성적(시각 비교) |
| `analog_whoop_footage.mp4` | 아날로그 FPV | 854×480, 30fps | 64초 | 정성적(정성 평가) |
| `digital_whoop_footage.mp4` | 디지털 FPV | 1440×1080, 30fps | 118초 | 정량적(PSNR/SSIM) |

### 정량 평가 결과 (초기)

```
Method                  PSNR vs Clean    Denoise Strength
──────────────────────────────────────────────────────────
Degraded (baseline)     10.57±0.45 dB    (reference)
Denoise Only             4.28±0.87 dB    5.86±0.97 dB
Full Pipeline            8.28±0.85 dB    7.10±0.35 dB
```

**문제 발견**: Full Pipeline의 PSNR이 baseline보다 낮음.  
→ Gamma+HistEq가 의도적으로 pixel 값을 변경하여 PSNR 하락  
→ 평가 방법론 재검토 필요성 인지

---

## CLI 재설계 (2026-05-31 10:30~11:00)

### 변경 동기

초기 `main.py`는 합성 이미지 생성 → 평가까지 자동화.  
실제 데이터 처리로 전환하면서 CLI 구조 전면 개편 필요.

### 최종 CLI 설계

```
main.py [-p PATH] [--preset PRESET] [--degrade MODE] [--strength S]

Flow:
  input (origin) → degrade → output/raw/ (degraded) 
                          → pipeline → output/processed/ (restored)
                          → output/*_comparison.png (3단 비교)

output/
  raw/          — degraded (노이즈 삽입 결과)
  processed/    — restored (파이프라인 복원 결과)
  *_comparison.png  — original | degraded | restored 3단 비교
```

**변경 사항**:
- `--video`, `--pipeline` 플래그 제거
- `-p "glob"` 패턴으로 입력 파일 지정
- `--preset`으로 pipeline 선택
- `--strength`로 degradation 강도 조절 (0.0~1.0)
- 합성 이미지 생성 코드 완전 제거
- `scripts/run_pipeline.py`, `scripts/run_video.py`, `scripts/run_evaluation.py` 삭제

---

## Degradation 강도 문제 발견 및 수정 (2026-05-31 11:00~11:30)

### 문제

기본 degradation이 너무 강력하여 원본 이미지를 완전히 파괴:

| 항목 | 이전 값 | 효과 |
|------|---------|------|
| Gaussian σ | 25 | ≈10% 픽셀 변동 |
| Brightness γ | 0.5 | 평균 밝기 **33% 감소** (124→83) |
| Impulse prob | 0.03 | 3% 픽셀 완전 손실 |
| Color bias | R±30% | 심각한 색상 왜곡 |

**영향**: 5종 노이즈 동시 적용 → degraded가 원본과 전혀 다른 이미지 → pipeline이 복원 불가

### 수정: `--strength` 파라미터 도입

`degrade_image()`에 `strength` 파라미터 추가 (0.0~1.0):

```
strength  gaussian  impulse  brightness  color_bias
────────────────────────────────────────────────────
  0.0     σ=0       p=0%     γ=1.0      없음
  0.3     σ=9       1.5%     γ=0.835    미약
  0.5     σ=15      2.5%     γ=0.725    보통 ← 기본값
  0.7     σ=21      3.5%     γ=0.615    강함
  1.0     σ=30      5.0%     γ=0.45     극한
```

기본값 `strength=0.5`로 변경 (이전은 사실상 strength=1.0).

---

## Wiener 필터 수식 오류 발견 및 수정 (2026-05-31 11:30~12:00)

### 발견된 버그

Wiener 필터의 주파수 응답 공식이 **수학적으로 잘못됨**:

```python
# BUG (v1.0 ~ v2.0):
h = mean(power) / (power + noise_var)   # ← power는 |F|² (2D 행렬)
```

**문제**: `mean(power)`는 스칼라(전체 power 평균), `power`는 행렬(주파수별 power).
- 저주파(power ≫ mean): h ≈ 0 → **신호 억제!**
- 고주파(power ≪ mean): h ≈ 1 → **노이즈 통과!**
- **정반대로 동작** — 저주파를 죽이고 고주파를 살림

### 수정

```python
# CORRECT:
signal_est = max(power - noise_var, 0)   # 신호 power 추정
h = signal_est / max(signal_est + noise_var, 1e-10)  # 올바른 Wiener
```

**효과**: 저주파 보존, 고주파 노이즈 억제 (정상 동작).  
파라미터 기본값도 `noise_var=625` → `400`으로 조정 (덜 공격적).

---

## Enhancement 단계가 화질을 망치는 문제 발견 (2026-05-31 12:00~12:30)

### 문제 진단

복원 결과가 degraded보다 시각적으로 나쁨.  
원인 분석 결과 enhancement 단계에서 **3가지 문제** 발견:

| 문제 | 원인 | 심각도 |
|------|------|--------|
| **색상 washout** | Gamma(γ=1.8)가 밝기를 과도하게 올리면서 채도 손실 | 🔴 심각 |
| **hyper-contrast edge** | Histogram equalization(Y channel)이 대비를 과도하게 조정 | 🔴 심각 |
| **색상 왜곡** | Channel correction이 RGB 균형을 강제로 맞춤 | 🟡 보통 |

### 수정

```python
# BEFORE:
cfg.add("gamma", gamma=1.8)
cfg.add("histogram_eq")
cfg.add("channel_correction")  # clamp 0.7~1.3

# AFTER:
# cfg.add("gamma")  ← 제거 (또는 gamma=1.15 최소)
# cfg.add("histogram_eq")  ← 완전 제거
cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)  ← 보수적
```

**핵심 인사이트**: Denoising pipeline은 noise만 제거해야 하며, enhancement(대비/색상 조정)는 
별도 단계로 분리하거나 매우 보수적으로 적용해야 함.

---

## Wiener 필터를 기본 pipeline에서 제거 (2026-05-31 12:30~13:00)

### 결정: Wiener → NLM+Bilateral+Unsharp 전환

**근거 데이터** (REAL_WORLD_PICTURE, strength=0.5):

| 필터 조합 | PSNR ↑ | SSIM ↑ | Edge Retention ↑ | 시간 ↓ |
|-----------|--------|--------|-----------------|--------|
| Median→Wiener | 14.68 | 0.4670 | **67%** | 0.05s |
| NLM→Bilateral→Unsharp | **17.75** | 0.4641 | **79%** | 0.34s |
| NLM only→Unsharp | 16.61 | 0.4251 | **170%** | 0.23s |
| Bilateral only | 16.62 | 0.4365 | **152%** | **0.01s** |

**Wiener의 근본적 한계**:
1. **주파수 도메인 처리**: FFT는 공간 정보를 잃음 → noise와 edge를 동일한 고주파로 취급
2. **ringing artifact**: 급격한 주파수 차단이 Gibbs 현상(Gibbs phenomenon) 유발
3. **비적응성**: 전체 이미지에 동일한 threshold 적용 → 지역적 특성 반영 불가

**NLM(Non-Local Means)의 장점**:
1. **패치 기반**: 유사한 패치를 찾아 평균 → edge는 보존, noise는 제거
2. **공간 인지**: 주파수가 아닌 픽셀 유사도 기반 → edge와 noise 구분 가능
3. **OpenCV 최적화**: CPU에서도 실시간에 가까운 속도

**Bilateral Filter의 장점**:
1. **edge-preserving**: 공간 거리 + 색상 차이를 동시에 고려
2. **빠름**: NLM의 1/30 속도

**Unsharp Mask의 역할**:
1. Denoising 과정에서 손실된 edge contrast 복원
2. 과도한 sharpening은 artifact 유발 → `strength=0.3`으로 제한

### 최종 Preset 구성

```python
@staticmethod
def edge_preserve():
    # NEW DEFAULT — edge 보존 최우선
    cfg.add("median", ksize=3)                    # impulse noise
    cfg.add("nlm", h=5)                           # NLM mild
    cfg.add("bilateral", d=5, sigma_color=30)      # edge-preserving smooth
    cfg.add("channel_correction", clamp=(0.85, 1.15))  # 색상 보존
    cfg.add("unsharp_mask", strength=0.3)          # edge 복원
```

### Preset 비교표 (최종)

| Preset | 필터 구성 | PSNR | SSIM | Edge | 속도 | 용도 |
|--------|----------|------|------|------|------|------|
| **edge-preserve** | Med→NLM→Bil→Corr→Unsharp | **17.75** | 0.4641 | 79% | 0.34s | **기본** |
| nlm-denoise | NLM→Bil→Corr→Unsharp | 16.61 | 0.4251 | 170% | 0.23s | NLM 효과만 |
| fast-denoise | Bil→Corr→Unsharp | 16.62 | 0.4365 | 152% | **0.01s** | 빠른 처리 |
| wiener-denoise | Med→Wiener→FFT→Corr→Unsharp | 14.68 | **0.4670** | 67% | 0.05s | 주기적 잡음 |
| aggressive | Med→NLM→Bil→Wiener→Corr→Unsharp | — | — | — | 느림 | 극한 노이즈 |
| research-best | Med→NLM→Bil→Wiener→Corr→Unsharp | 17.68 | **0.4928** | 73% | 0.26s | 최고 SSIM |

---

## 최종 결론

### 핵심 발견

1. **Wiener 필터는 edge 보존에 부적합** — 주파수 도메인 방식의 근본적 한계
2. **Enhancement는 denoising과 분리** — gamma/histeq를 동시에 적용하면 색상 왜곡
3. **NLM + Bilateral + Unsharp가 최적** — edge 보존 + noise 제거 + 선명도 유지
4. **Degradation은 현실적인 강도로** — strength=0.5가 적정 (σ=15, γ=0.725)
5. **PSNR만으로 평가 불가** — 시각적 quality와 PSNR이 반드시 일치하지 않음

---

## Iteration 2: Wavelet 추가 + Preset 최적화 (2026-05-31 13:00~14:00)

### Wavelet Denoising 추가

**도입 이유**: Wavelet은 다중 해상도 분석으로 FFT보다 edge 보존에 유리함.
- `pywt.wavedec2()`로 3레벨 db4 decomposition
- VisuShrink universal threshold: `σ * sqrt(2*log(N))`
- Soft thresholding으로 detail coefficient 처리

**등록**: `filters.py`에 `@register("wavelet")`로 등록.
**Preset**: `wavelet-denoise` 추가.

### Preset 전면 재측정 (REAL_WORLD_PICTURE, strength=0.5)

| Preset | PSNR | SSIM | Edge% | Time | 특징 |
|--------|------|------|-------|------|------|
| **edge-preserve** | **17.77** | 0.466 | 79% | 0.33s | 기본, NLM+Bilateral+Unsharp |
| **fast-denoise** | 16.59 | 0.434 | **153%** | **0.01s** | 비디오 최적 (30× 빠름) |
| nlm-denoise | 16.58 | 0.423 | 171% | 0.23s | NLM 중심 |
| wiener-denoise | 14.68 | 0.469 | 67% | 0.05s | 참고용 |
| wavelet-denoise | — | — | — | — | 연구용 (신규) |
| aggressive | **18.69** | **0.626** | 31% | 0.26s | PSNR 최고, Edge 최저 |
| research-best | 17.70 | **0.496** | 73% | 0.25s | SSIM 최고 |

### 결정: Fast Denoise를 비디오 권장으로 승격

edge-preserve(0.33s)는 이미지에는 좋지만 비디오(1926프레임 × 0.33s = 10.6분)에는 너무 느림.
fast-denoise(0.01s)는 30× 빠르면서 PSNR 16.59, Edge 153%로 우수.

권장:
- **이미지**: `--preset edge-preserve` (기본)
- **비디오**: `--preset fast-denoise` (0.01s/프레임)
- **연구**: `--preset wavelet-denoise` 또는 `--preset research-best`

### Preset별 최적 파라미터

```python
# Fast Denoise (비디오 권장)
cfg.add("bilateral", d=5, sigma_color=30, sigma_space=30)
cfg.add("channel_correction", clamp_min=0.9, clamp_max=1.1)
cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=10)

# Edge-Preserve (이미지 기본)
cfg.add("median", ksize=3)
cfg.add("nlm", h=5, template_window=7, search_window=21)
cfg.add("bilateral", d=7, sigma_color=50, sigma_space=50)
cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
cfg.add("unsharp_mask", strength=0.3, radius=0.5, threshold=10)
```

### 남은 과제

- [ ] 필터 순서에 따른 결과값 변이 추적
- [ ] 정량적/자동 evaluation을 위한 pipeline 설계
  - [ ] 색감 evaluation
  - [ ] 밝기 evaluation
  - [ ] edge evaluation
  - [ ] gaussian noise evaluation
  - [ ] artifacts evaluation
- [ ] bilateral 파라미터 자동 추정 (sigma_color를 noise level에 따라 조절)
- [ ] 실제 아날로그 영상(analog_whoop_footage) 정량 평가
- [ ] temporal denoising (인접 프레임 활용) — VBM3D 등

---

## Iteration 3: Spatio-temporal + 신규 필터 + Ablation 최적화 (2026-06-03)

### 추가된 필터 (v3.0)

| 필터 | 등록일 | 분류 | 설명 |
|------|--------|------|------|
| `guided_filter` | v3.0 | Advanced Spatial | Local linear edge-preserving filter (0.07s/frame) |
| `anisotropic_diffusion` | v3.0 | Advanced Spatial | Perona-Malik iterative edge-preserving diffusion |
| `tv_denoise` | v3.0 | Advanced Spatial | Total Variation ROF (Chambolle, skimage) |
| `patch_collaborative` | v3.0 | Patch-based | Block DCT hard-threshold denoising |
| `temporal_average` | v3.0 | Temporal Video | Sliding multi-frame averaging |
| `temporal_motion` | v3.0 | Temporal Video | Farneback optical flow motion-compensated denoising |
| `temporal_spatial` | v3.0 | Temporal Video | Combined bilateral + motion compensation |

### 추가된 Preset (v3.0)

| Preset | 필터 구성 | 용도 |
|--------|----------|------|
| `guided-denoise` | guided → channel → unsharp | Edge-preserving 빠른 처리 |
| `tv-denoise` | tv_denoise → channel → unsharp | Small noise 제거, edge 보존 |
| `aniso-denoise` | anisotropic_diffusion → channel → unsharp | Iterative edge-preserving |
| `dct-denoise` | patch_collaborative → channel → unsharp | Block DCT thresholding |
| `st-video` | temporal_motion → bilateral → guided → channel → unsharp | Spatio-temporal 비디오 |
| `optimized-fast` | median → bilateral → channel | Ablation 최적화 (NLM 제거) |
| `optimized-quality` | median → bilateral → wavelet → channel → unsharp | 고품질 융합 |

### Ablation Study 결과 (REAL_WORLD_PICTURE 1600×740, basic degrade strength=0.5)

| Preset | PSNR ↑ | SSIM ↑ | Edge | Time ↓ |
|--------|--------|--------|------|--------|
| **edge-preserve** | 18.08 | 0.5503 | 0.90 | 2.09s |
| **optimized-fast** 🆕 | **19.00** | 0.5463 | 0.83 | **0.12s** |
| **optimized-quality** 🆕 | 18.83 | 0.5415 | 0.99 | 0.28s |
| research-best | 17.98 | 0.5480 | 0.90 | 2.27s |
| wavelet-denoise | 17.85 | 0.4776 | 0.88 | 0.25s |
| guided-denoise | 17.73 | 0.4857 | 0.90 | 0.14s |
| aggressive | 17.71 | 0.5286 | 0.73 | 2.30s |
| tv-denoise | 17.51 | 0.4311 | 1.21 | 0.78s |
| nlm-denoise | 16.65 | 0.3948 | 1.72 | 1.85s |
| aniso-denoise | 16.62 | 0.3712 | 1.33 | 2.03s |
| dct-denoise | 16.38 | 0.3138 | 1.86 | 1.21s |
| fast-denoise | 16.29 | 0.3447 | 1.78 | 0.05s |
| wiener-denoise | 13.91 | 0.4771 | 0.92 | 0.68s |

### Filter ON/OFF Ablation (edge-preserve 대상)

| Variant | PSNR | Δ | Time | 발견 |
|---------|------|---|------|------|
| full (baseline) | 18.08 | — | 2.09s | |
| **−NLM** | **18.14** | **+0.06** | **0.08s** 🚀 | **NLM = 완전 중복!** median+bilateral이 noise 제거를 충분히 함 |
| −median | 16.65 | **−1.43** | 2.02s | median = impulse noise에 필수 |
| −channel_correction | 16.87 | −1.21 | 2.02s | color correction = 중요 |
| +bilateral(1.5×) | 18.16 | +0.08 | 2.03s | 강한 bilateral → PSNR 소폭 향상 |
| +channel_corr(1.5×) | 17.61 | — | **SSIM↑** | wider clamp → SSIM 0.5846 |
| −unsharp_mask | 18.08 | 0.0 | 1.78s | unsharp = 효과 없음 |

**핵심 발견**: NLM이 edge-preserve에서 median+bilateral과 완전히 중복. NLM 제거 시 PSNR 유지 + 25× 속도 향상.

### NTSC-heavy 테스트 (analog_whoop_footage frame, 854×480)

| Preset | PSNR | SSIM | Time |
|--------|------|------|------|
| degraded (baseline) | 28.55 | 0.8762 | — |
| **wavelet-denoise** 🥇 | **28.13** | **0.8795** | **0.11s** |
| optimized-quality | 27.26 | 0.8182 | 0.12s |
| edge-preserve | 26.95 | 0.7825 | 0.83s |
| tv-denoise | 26.94 | 0.7983 | 0.23s |
| optimized-fast | 25.50 | 0.6583 | 0.08s |
| guided-denoise | 24.64 | 0.6911 | 0.05s |

**발견**: wavelet-denoise가 NTSC 아티팩트(dot crawl, chroma noise, ringing) 처리에 최적. SSIM이 degraded보다 **개선됨** (0.8762→0.8795).

### 권장 파이프라인

| 용도 | Preset | 근거 |
|------|--------|------|
| **일반 이미지/영상** (synthetic noise) | `optimized-fast` | PSNR 19.00, 0.12s (edge-preserve의 18× 속도) |
| **실제 아날로그 영상** (NTSC) | `wavelet-denoise` | PSNR 28.13, SSIM 0.8795, 0.11s |
| **고품질** (edge 보존) | `optimized-quality` | PSNR 18.83, SSIM 0.5415, 0.28s |
| **실시간** | `fast-denoise` | 0.05s/frame (bilateral only) |

### 신규 도구

- `run_ablation.py` — Batch testing: preset 비교, filter ON/OFF ablation, param tuning.
  `--presets p1,p2` / `--ablation f1,f2` / `--param-tune filter.param,v1,v2,...`
  출력: CSV + Markdown report + visual grid (선택).

- `main.py --sample N` — Heuristic sample: 비디오의 첫 N 프레임만 처리하여 preview.

### 바뀐 점

- 23개 필터 (v2.0의 16개 → +7개 신규)
- 13개 preset (v2.0의 7개 → +6개 신규)
- `process_video()`에 temporal state reset 추가
- `reset_temporal_state()` — 비디오 간 temporal filter 상태 초기화
