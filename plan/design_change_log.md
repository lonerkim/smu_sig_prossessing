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
- 14개 preset (v2.0의 7개 → +7개 신규)
- `process_video()`에 temporal state reset 추가
- `reset_temporal_state()` — 비디오 간 temporal filter 상태 초기화

---

## Iteration 4: Parameter tuning + cascade comparisons + temporal video test (2026-06-03)

### 실험 개요

파라미터 스윕, 필터 순서(cascade) 최적화, temporal denoising 실제 비디오 평가를 수행.
모든 결과는 `/mnt/nfs-hermes/artifacts/`에 저장됨.

### EXP 1: Bilateral sigma_color 스윕 (optimized-fast base)

| σ | d | PSNR ↑ | SSIM ↑ | Edge | Time ↓ |
|---|----|--------|--------|------|--------|
| 30 | 5 | 18.85 | 0.5336 | 1.07 | 0.03s |
| 50 | 7 | 18.91 | 0.5454 | 1.00 | 0.06s |
| 75 | 9 | 19.02 | 0.5571 | 0.88 | 0.12s |
| 100 | 11 | 19.06 | 0.5595 | 0.81 | 0.16s |
| **125** | **17** | **19.17** | **0.5814** | **0.70** | **0.26s** |
| **150** | **19** | **19.25** | **0.6142** | **0.59** | **0.45s** |

→ σ=125~150이 최적. σ=150: PSNR=**19.25** (전체 최고). 높은 σ는 속도와 edge retention을 희생.

### EXP 2: Wavelet level + threshold mode 스윕

| Config | PSNR | SSIM | Edge | Time |
|--------|------|------|------|------|
| L1 soft | 17.72 | 0.4312 | 1.52 | 0.19s |
| L2 soft | 17.94 | 0.4889 | 1.08 | 0.21s |
| **L4 soft** | **18.07** | **0.5154** | **0.76** | **0.22s** |
| L1-L4 hard | 17.0-17.4 | 0.39-0.40 | 1.25-1.62 | 0.17-0.21s |

→ Soft thresholding이 hard보다 모든 레벨에서 우월. L4 soft가 최고 PSNR (18.07).

### EXP 3: Guided filter radius + eps 스윕

| Config | PSNR | SSIM | Edge | Time |
|--------|------|------|------|------|
| r=2 | 17.35 | 0.3878 | 1.35 | 0.12s |
| **r=3** | **18.08** | **0.5079** | **1.05** | **0.13s** |
| r=5 | 18.01 | 0.5183 | 0.80 | 0.14s |
| r=8 | 17.71 | 0.5002 | 0.62 | 0.14s |

→ eps는 거의 영향 없음. radius=3이 최적 (PSNR=18.08). radius=5는 SSIM 더 높음.

### EXP 4: TV denoise weight 스윕

| w | PSNR | SSIM | Edge | Time |
|---|------|------|------|------|
| 0.02 | 16.68 | 0.3609 | 1.67 | 1.05s |
| 0.08 | 17.52 | 0.4323 | 1.20 | 0.78s |
| **0.20** | **18.18** | **0.5134** | **0.88** | **0.95s** |
| 0.50 | 18.21 | 0.5723 | 0.52 | 3.25s |

→ w=0.20이 최적 PSNR/속도 균형. w=0.50은 미미한 PSNR 향상에 3.4× 느림.

### EXP 5: Cascade (필터 순서) 실험

| 순서 | PSNR | SSIM | Edge | 발견 |
|------|------|------|------|------|
| wavelet→bilateral | 17.85 | 0.4776 | 0.88 | wavelet first > bilateral first |
| bilateral→wavelet | 17.36 | 0.4032 | 1.31 | bilateral이 wavelet보다 먼저 오면 PSNR 하락 |
| wavelet→guided | 17.91 | 0.5042 | 0.95 | |
| guided→wavelet | **17.99** | **0.5098** | **1.02** | guided first > wavelet first |
| median→bilateral→wavelet→channel | **18.86** | **0.5418** | **0.92** | **최고 cascade** |
| median→wavelet→bilateral→channel | 18.86 | 0.5402 | 0.89 | 순서 교체 = 동일 |

→ **median→bilateral→wavelet→channel** = PSNR 18.86 (optimized-quality과 동등), 시간 0.27s.

### EXP 6: Best-of-family 최종 비교

| Preset | PSNR | SSIM | Edge | Time |
|--------|------|------|------|------|
| **bilateral σ=150 + median** | **19.19** | **0.5930** | 0.63 | **0.33s** |
| **optimized-fast** | 18.99 | 0.5462 | 0.83 | **0.12s** |
| optimized-quality | 18.83 | 0.5415 | 0.99 | 0.27s |
| median→bil→wavelet→channel | 18.86 | 0.5418 | 0.92 | 0.28s |
| TV w=0.20 | 18.18 | 0.5134 | 0.88 | 0.97s |
| guided r3 e100 | 18.08 | 0.5079 | 1.05 | 0.14s |
| edge-preserve (baseline) | 18.08 | 0.5499 | 0.90 | 2.13s |

### EXP 7: Temporal video (30 frames, analog_whoop_footage)

| Preset | 평균 시간/프레임 | 특징 |
|--------|----------------|------|
| temporal-averaging (5 frames) | **0.0153s** 🚀 | 가장 빠름, real-time 가능 |
| motion-compensated | **0.1178s** | Farneback optical flow + blending |
| spatio-temporal (st-video) | **0.1476s** | temporal_motion + bilateral + guided |
| spatial-only (edge-preserve) | 0.7543s | NLM 병목, temporal 없음 |

→ Temporal averaging (0.015s/frame)은 **real-time 60fps 처리 가능**!
→ Motion-compensated는 0.12s/frame으로 8fps, spatial-only 대비 6× 빠름.

### 최종 권장

| 용도 | Preset | PSNR | 시간/프레임 |
|------|--------|------|------------|
| **최고 화질** (synthetic) | median + bilateral σ=150 | **19.25 dB** | 0.45s |
| **일반 목적** | optimized-fast | 18.99 dB | 0.12s |
| **아날로그 영상** (NTSC) | wavelet-denoise | 28.13 dB | 0.11s |
| **실시간** (60fps) | temporal-averaging + bilateral | ~16-17 dB | **0.015s** |
| **압도적 화질/비용비** | optimized-fast | 18.99 dB | 0.12s |

### 추가된 Preset

- `optimized-fast` — median(3) + bilateral(d=11, σ=110) + channel
- `optimized-quality` — median + bilateral + wavelet (L2 soft) + channel + unsharp

### artifacts 구조

```
/mnt/nfs-hermes/artifacts/
  exp1_bilateral_sweep.csv/.md
  exp2_wavelet_sweep.csv/.md
  exp3_guided_sweep.csv/.md
  exp4_tv_sweep.csv/.md
  exp5_cascade.csv/.md + _grid.png
  exp6_best_of_family.csv/.md + _grid.png
  exp7_*_frames.png  (temporal frame comparisons)
```

---

## Iteration 5: Degrade 현실화 + Washing fix + Video preset (2026-06-03)

### Degrade 개선: Diagonal periodic noise 제거

**문제**: `add_periodic_noise()`가 대각선 사인파 패턴을 추가 — 이는 실제 아날로그 영상에서 일반적이지 않은 아티팩트.
파이프라인이 이 패턴을 처리하기 어려워했음.

**변경**:
- `add_periodic_noise()`는 유지하되 **degrade_image의 기본 구성에서 제거** (참고용으로 남김)
- `add_horizontal_line_noise()` — CRT 스캔라인 간섭 시뮬레이션 (random horizontal line shift)
- `add_dropout()` — VHS 테이프 손상 시뮬레이션 (random white/black horizontal streaks)

이제 basic degrade 구성:
```
Gaussian(σ) → Impulse(p) → Color bias → Brightness(gamma) → Horizontal line noise → Dropout
```

### Washing 문제 해결

**문제**: `max-quality` preset의 bilateral(σ=150)이 밝은 영역(highlights)을 과도하게 평활화하여 washing 발생.
Edge retention도 0.42로 매우 낮음.

**원인 진단** (854×480 real analog frame 테스트):

| Preset | Wash Ratio | Edge | 설명 |
|--------|-----------|------|------|
| max-quality (original) | **0.949** ↓ | **0.423** ↓↓ | Highlight washing + edge 붕괴 |
| optimized-fast (σ=110) | 0.972 | 0.589 | Mild washing |
| **guided+CLAHE** 🏆 | **1.016** ✓ | **1.306** ✓✓ | **Washing 없음, edge 3× 향상** |
| edge-preserve | 0.979 | 0.657 | 중간 |

Wash ratio = 1.0 = 완벽 보존, < 1.0 = highlight dimming.

**수정**:
- `max-quality`에 `histogram_eq_clahe(clip=1.5)` 추가 — local contrast 복원
- 신규 `video-enhanced` preset 추가 — guided filter + wavelet + CLAHE (washing 0, edge 1.3)

### 신규 Preset

| Preset | 구성 | Wash | Edge | 시간 |
|--------|------|------|------|------|
| `video-enhanced` | Med→Guided→Wavelet→Channel→CLAHE→Unsharp | **1.016** | **1.306** | **0.12s** |

### 바뀐 점

- Degrade: diagonal periodic noise 제거, horizontal line noise + dropout 으로 대체
- max-quality preset: CLAHE 추가로 washing 방지
- video-enhanced preset 신규 등록
- README 전면 업데이트 (v3.0 문서화)

---

## Iteration 6: 자동 정량+정성 평가 파이프라인 (2026-06-06)

### 추가된 모듈

| 모듈 | 설명 |
|------|------|
| `auto_evaluation.py` | 7개 메트릭 자동 산출 + Composite Score |
| `eval_viz.py` | Radar chart, Bar chart, Comparison grid, 정성 평가 시트 |
| `run_auto_eval.py` | CLI: preset별 자동 평가 → CSV/JSON/MD + 시각화 |

### 자동 평가 메트릭 (7개)

| # | 메트릭 | 단위 | 방향 | 설명 |
|---|--------|------|------|------|
| 1 | PSNR | dB | ↑ | Peak Signal-to-Noise Ratio |
| 2 | SSIM | — | ↑ | Structural Similarity (luminance) |
| 3 | Color Fidelity | ΔE | ↓ | CIE76 LAB color difference |
| 4 | Edge Retention | ratio | ↑ | Canny edge pixel ratio (1.0=원본 동일) |
| 5 | Noise Level | Lap. var | ↓ | Laplacian variance (낮을수록 깨끗) |
| 6 | Detail Recovery | ratio | ↑ | 고주파 에너지 보존율 (1.0=완전 보존) |
| 7 | Artifact Score | score | ↓ | Ringing + blocking + overshoot 감지 |

### Composite Quality Score (0~100)

가중 평균 — 프로젝트 특성에 맞춰 조정 가능:

```
PSNR × 0.15 + SSIM × 0.20 + Color × 0.15 + Edge × 0.15
+ Noise × 0.15 + Detail × 0.10 + Artifact × 0.10
```

### 실험 결과

#### test_small.jpg (1600×740, basic degrade strength=0.5)

| # | Preset | Score | PSNR | SSIM | ΔE | Edge | Noise | Detail | Artifact |
|---|--------|-------|------|------|----|------|-------|--------|----------|
| 1 | video-enhanced | **52.24** | 20.63 | 0.7346 | 17.79 | 0.656 | **63.8** | 0.828 | 9.01 |
| 2 | edge-preserve | 45.88 | 18.98 | **0.7652** | 24.50 | 0.561 | 356.9 | **0.841** | 13.97 |
| 3 | optimized-fast | 45.77 | **19.99** | 0.7416 | 19.22 | 0.403 | 317.9 | 0.777 | 12.48 |
| 4 | wavelet-denoise | 43.41 | 18.84 | 0.6658 | 23.79 | 0.548 | 213.3 | 0.598 | 14.47 |
| 5 | fast-denoise | 40.39 | 16.76 | 0.4517 | 28.08 | 2.257 | 11017 | 1.773 | 17.66 |

**핵심 발견**:
- video-enhanced가 노이즈 제거율 최고 (Lap.var 63.8 vs degraded 10791 = **170× 개선**)
- edge-preserve의 SSIM 최고 (0.7652) — 구조 보존 우수
- fast-denoise는 노이즈 제거가 거의 안됨 (Lap.var 11017 ≈ degraded 수준)

#### analog_whoop_footage.mp4 (854×480, degrade=none, 실제 아날로그)

| # | Preset | Score | PSNR | SSIM | ΔE | Edge | Noise | Detail | Artifact |
|---|--------|-------|------|------|----|------|-------|--------|----------|
| 1 | fast-denoise | **79.35** | **37.58** | **0.9906** | **3.69** | 0.904 | 271.9 | 0.953 | **2.98** |
| 2 | wavelet-denoise | 78.63 | 36.69 | 0.9837 | 3.80 | 0.875 | 230.7 | 0.930 | 3.07 |
| 3 | video-enhanced | 67.63 | 22.85 | 0.8998 | 7.01 | **0.931** | **104.4** | **1.041** | 8.84 |
| 4 | optimized-fast | 60.56 | 27.10 | 0.7221 | 5.36 | 0.156 | 74.1 | 0.565 | 5.56 |

**핵심 발견**:
- **실제 아날로그 영상**에서는 fast-denoise/wavelet이 압도적 (PSNR 37+)
- video-enhanced는 노이즈를 너무 강하게 제거 (detail 손실로 PSNR 하락)
- 아날로그 영상은 noise level 자체가 낮아 (296 vs 10791), 강한 denoising이 오히려 해로움
- **용도별 최적**: 실시간=fast-denoise, 고품질=wavelet-denoise, 강노이즈=video-enhanced

### 정성 평가 인터페이스

- `run_auto_eval.py` 실행 시 `qualitative_notes_{ts}.md` 자동 생성
- 유저가 직접 보고 1~5점 + 코멘트 입력
- 항목: 색상 / 선명도 / 노이즈 제거 / 전체 인상
- 종합: 최고/최악 preset, 특이사항

### 사용법

```bash
# 전체 preset 정량+정성 평가
python run_auto_eval.py -i input/test_small.jpg --degrade basic --strength 0.5

# 특정 preset만
python run_auto_eval.py -i input/test_small.jpg --presets optimized-fast,video-enhanced

# 실제 아날로그 영상 (degrade 없이)
python run_auto_eval.py -i input/analog_whoop_footage.mp4 --degrade none --sample 3

# 정성 평가 시트만
python run_auto_eval.py -i input/test_small.jpg --qualitative-only
```

### 산출물

```
output/eval/
  auto_eval_{ts}_{name}.csv       — 정량 데이터
  auto_eval_{ts}_{name}.json      — JSON (프로그래밍 접근)
  auto_eval_{ts}_{name}.md        — Markdown 리포트
  auto_eval_{ts}_{name}_radar.png — Radar chart (5축)
  auto_eval_{ts}_{name}_bar.png   — Bar chart (7메트릭)
  auto_eval_{ts}_{name}_grid.png  — 비교 그리드
  auto_eval_{ts}_{name}_qual.png  — 정성 평가 시트
  qualitative_notes_{ts}_{name}.md — 정성 코멘트 템플릿
```

### 패키지 구조 업데이트

```
smu_sig_prossessing/        — 핵심 패키지
  __init__.py
  __main__.py               — python -m 진입점 (process/eval/list-filters)
  config.py                 — PipelineConfig (14개 preset)
  filters.py                — 필터 레지스트리 (23개 필터)
  pipeline.py               — 파이프라인 러너
  degradation.py            — 열화 모듈
  evaluation.py             — 기존 PSNR/SSIM
  auto_evaluation.py        — 🆕 자동 7메트릭 + Composite Score
  eval_viz.py               — 🆕 시각화 (radar/bar/grid/정성시트)
  ntsc_plugin.py            — NTSC 시뮬레이터
```

---

## v3.1 → v4.0 Iteration 7 (2026-06-06)

### 변경: 실제 아날로그 풋샷 기반 필터 개발 + 아날로그 특화 preset

**배경**: Iteration 6까지 합성 노이즈 기반 평가로 진행.  실제 whoop 드론 아날로그 풋샷 24개(251MB)를 확보해 정성 평가 진행.

#### 정성 평가 결과 (VID00002, VID00006 실제 풋샷)

| preset | 장점 | 단점 |
|---|---|---|
| video-enhanced | sharpness 최고 | OSD blur 심함, 비주얼 artifact |
| optimized-fast | artifact 최소 | blur 심함 |
| wavelet-denoise | **영상으로 보기 제일 자연스러움** | 색보정 약간 부족 |
| fast-denoise | 점수는 최고 | 아날로그 특화 아님 |

**공통 미해결 문제**:
- 세로줄 artifact (수직 주기적 패턴)
- 깜빡임 (frame-to-frame brightness flicker)
- 스캔라인 (수평 주기적 라인)
- 울렁임 (undulation)

#### 노이즈 레벨 분석

```
파일                               noise (Laplacian var)
VID00002 seg1 (새)                 5248  ◄ 높음
VID00002 seg2 (새)                 9023  ◄ 최악
VID00006 seg1 (새)                 1995  ◄ 중간
기존 analog_whoop_footage.mp4       296   (너무 깨끗함 — preset 차이 안보임)
```

### 신규 필터 (Phase 7: Analog Video Specific)

**1. `flicker_stabilize`**
- EMA 기반 temporal brightness stabilization
- per-channel mean brightness 추적 → 프레임간 급격한 변화 완화
- strength (0~1): 보정 강도, window: EMA 윈도우 크기
- 기본값: strength=0.7, window=10 (0.33s at 30fps)

**2. `scanline_remove`**
- FFT 기반 수평 scanline 자동 감지 + 제거
- row-mean brightness → FFT → 주기적 peak 탐지 → bad row interpolation
- 두 모드: "detect" (자동) / "fixed" (period 지정)
- blend 파라미터로 원본 대비 대체 비율 조절

**3. `vertical_notch` 개선**
- robust peak detection: 로커스트 moving average로 baseline 추정 후 peak 탐지
- 기존 generic fft_notch와 달리 수직 방향(수평 주파수)만 타겟
- **주의**: 열 전체 zeroing 시 수평 ringing 발생 → narrow band-stop 필요 (추후 개선)

### 신규 Preset

**`analog_clean`** (권장 — analog FPV 기본)
```
scanline_remove (detect, blend=0.5)
flicker_stabilize (strength=0.6, window=10)
wavelet (db4, level=2, soft)
bilateral (d=5, σ=20)
channel_correction (0.85~1.15)
unsharp_mask (0.15, threshold=8)
```

**`analog_heavy`** (강노이즈, noise > 5000)
```
scanline_remove (fixed period=2, blend=0.6)
flicker_stabilize (strength=0.8, window=15)
wavelet (db4, level=3, soft)
bilateral (d=7, σ=40)
channel_correction (0.80~1.20)
unsharp_mask (0.2, threshold=5)
```

### 처리 성능

| preset | VID00002_seg2 (603 frames) | 속도 |
|---|---|---|
| analog-clean | 58.8s | 10.3 fps |
| analog-heavy | 65.1s | 9.3 fps |
| wavelet-denoise | 39.3s | 15.3 fps |

### 기타 변경

- `run_auto_eval.py`: `--zip` 플래그 추가 (결과 전체를 ZIP으로 패키징, Telegram 무압축 전송용)
- `video_enhanced` preset에서 `vertical_notch` 제거 (수평 ringing 문제)

### 패키지 구조 업데이트

```
smu_sig_prossessing/        — 핵심 패키지
  __init__.py
  __main__.py               — python -m 진입점 (process/eval/list-filters)
  config.py                 — PipelineConfig (16개 preset) ← +2
  filters.py                — 필터 레지스트리 (25개 필터) ← +2
  pipeline.py               — 파이프라인 러너
  degradation.py            — 열화 모듈
  evaluation.py             — 기존 PSNR/SSIM
  auto_evaluation.py        — 자동 7메트릭 + Composite Score
  eval_viz.py               — 시각화 (radar/bar/grid/정성시트)
  ntsc_plugin.py            — NTSC 시뮬레이터

run_auto_eval.py            — CLI 평가 도구 (--zip 지원)
```

### 남은 과제

1. **vertical_notch 개선**: narrow band-stop filter로 수평 ringing 방지
2. **Adaptive pipeline**: artifact detector → filter router (검출 후 필요한 필터만 적용)
3. **Temporal denoising 고도화**: 현재 EMA 기반 → motion-compensated 고도화
4. **실제 footage 추가 확보**: 다양한 노이즈 패턴 coverage
5. **Deinterlace**: NTSC 인터레이스 소스 필요시 추가

---

## v3.3 (2026-06-08) — 6개 신규 필터 + 7개 신규 Preset + Benchmark 개선

### 변경 개요

**37개 필터** (+6), **35개 preset** (+7)로 확장.
종합 점수 51.56 → **54.42** (+2.86, **5.5% 향상**).

### 신규 필터 (6개)

| 필터 | 설명 |
|------|------|
| `rolling_guidance` | Rolling Guidance Filter — iterative guided filter로 edge 보존 denoising |
| `cross_bilateral` | Joint/Cross Bilateral Filter — guide image 기반 edge-preserving |
| `detail_boost` | Edge-aware detail enhancement — base/detail 분해 후 detail layer 증폭 |
| `temporal_nlm_multi` | Multi-frame NLM — OpenCV fastNlMeansDenoisingColoredMulti |
| `bm4d_volume` | BM4D spatio-temporal — video volume collaborative filtering |
| `adaptive_equalize` | CLAHE + brightness preservation blend |

### 신규 Preset (7개)

| Preset | Score | PSNR | ΔE | Speed | 설명 |
|--------|-------|------|-----|-------|------|
| 🔵 **temporal-premium** | **54.42** 🥇 | 18.52 | 13.59 | 160ms | Multi-frame NLM 기반 최고 품질 |
| 🔵 **chroma-focus** | **53.59** 🥈 | 17.93 | 13.73 | 34ms | Chroma+luma 분리 집중 |
| 🔵 **rolling-premium** | **51.73** | 18.20 | 14.58 | 35ms | Rolling guidance edge 보존 |
| 🔵 **super-premium-fast** | **51.33** | 18.23 | 14.26 | **25ms** | 가장 빠른 고품질 |
| 🔵 **super-premium** | **50.63** | 18.15 | 14.43 | 156ms | Wavelet+detail boost |
| 🟢 **ultralight** | **41.33** | 16.47 | 15.41 | **3.7ms** | 실시간 270fps 가능 |
| 🔵 **bm4d-temporal** | **47.19** | 17.16 | 13.52 | 6ms | BM4D video volume |

### 벤치마크 Top 10

| # | Preset | Score | PSNR | ΔE | Time |
|---|--------|-------|------|-----|------|
| 1🆕 | temporal-premium | **54.42** | 18.52 | 13.59 | 160ms |
| 2🆕 | chroma-focus | **53.59** | 17.93 | 13.73 | 34ms |
| 3🆕 | rolling-premium | **51.73** | 18.20 | 14.58 | 35ms |
| 4 | video-ultra | 51.55 | 17.77 | 14.18 | 156ms |
| 5 | fast-premium | 51.54 | 17.77 | 14.18 | 138ms |
| 6🆕 | super-premium-fast | **51.33** | 18.23 | 14.26 | **25ms** |
| 7 | video-enhanced | 51.14 | 17.85 | 14.51 | 144ms |
| 8🆕 | super-premium | **50.63** | 18.15 | 14.43 | 156ms |
| 9 | st-video | 47.76 | 17.49 | 17.43 | 7ms |
| 10 | max-quality | 42.21 | 17.82 | 14.55 | 41ms |

### NTSC Heavy 성능

| Preset | Score | PSNR | SSIM |
|--------|-------|------|------|
| wavelet-denoise | 48.13 | 17.88 | 0.6471 |
| temporal-premium | 47.02 | 17.64 | 0.6277 |
| chroma-focus | 46.57 | 17.25 | 0.5959 |
| rolling-premium | 46.55 | 17.44 | 0.6069 |

### Architecture Changes

- `filters.py`: 31 → 37 filters (Phase 8: Rolling Guidance, Phase 9: Temporal Multi-Frame NLM, Phase 10: Adaptive Equalization)
- `config.py`: 26 → 33 static presets (7 new)
- `main.py`: 27 → 34 preset names
- `run_v33_benchmark.py`: New benchmark script with slow-preset skip logic
- 기존 BM3D는 8s/frame으로 실용성 부족 → BM4D가 6ms로 1000× 빠른 대체제

---

## v3.4 (2026-06-09 00:10~01:00) — Grey-Edge Color Constancy + 2h Auto Loop

### 변경 개요

**39개 필터** (+2), **38개 preset** (+3)로 확장.
종합 점수 55.31 → **56.60** (+1.29, **+2.3% 향상**).

### 신규 필터

| 필터 | 설명 | 구현 |
|------|------|------|
| `grey_edge` | Grey-Edge color constancy — edge 기반 조명 추정 white balance | van de Weijer et al., IEEE TIP 2007 |

### 신규 Preset

| Preset | Score | PSNR | ΔE | Speed | 비고 |
|--------|-------|------|-----|-------|------|
| 🔵 **grey-premium** | **56.60** 🥇 | 19.20 | 11.53 | ~400ms | Grey-Edge + temporal-premium |
| 🔵 **grey-fast** | **53.17** | 18.35 | 12.46 | ~200ms | Grey-Edge + guided + wavelet |
| 🟢 **grey-ultralight** | **42.73** | 16.32 | 13.79 | **10ms** | 실시간 100fps + WB |

### 핵심 발견: Grey-Edge Edge Color Constancy

Grey-Edge를 denoising 전단에 pre-processing으로 추가하면:
- **ΔE 13.57 → 11.53** (-15%, 눈에 띄는 색상 개선)
- **PSNR 18.46 → 19.20** (+0.74dB)
- Composite Score 54.18 → 56.60 (+2.42)

강도(strength)=0.22 최적 (0.15~0.35 sweep 결과).
너무 강하면 PSNR 하락, 너무 약하면 효과 미미.

### 자동 개선 루프 구축

- `scripts/iter_loop.py`: 2시간마다 benchmark + experiments + commit
- `scripts/run_experiments.py`: Grey-Edge sweep + 신규 조합 테스트
- Cron job `smu-sig-v34-loop`: 2시간마다, 10회 반복, 01:00 KST 시작

### 아키텍처

- `filters.py`: 37 → 39 filters (Phase 11: Grey-Edge CC)
- `config.py`: 33 → 36 static presets (3 grey-edge presets)
- `main.py`: 34 → 37 preset names
- `scripts/iter_loop.py`: 자동 개선 루프
- `scripts/run_experiments.py`: 파라미터 탐색 실험
