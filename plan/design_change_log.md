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

### 남은 과제

- [ ] 비디오 처리 속도 최적화 (현재 0.3초/프레임, 64초 영상에 ~10분 소요)
- [ ] NLM 파라미터 자동 추정 (h값을 noise level에 따라 자동 조절)
- [ ] 다중 프레임 temporal denoising (VBM3D 등)
- [ ] 실제 아날로그 영상(analog_whoop_footage)에 대한 정량적 평가 지표
