# 아날로그 영상 잡음 및 색 왜곡 완화를 위한 범용 영상처리 파이프라인 설계

**기말 프로젝트 과제계획서**  
**6팀 | 지능IoT융합전공 김승민 · 컴퓨터과학과 김원석**  
**과목: 영상처리 기말 프로젝트**  
**제출일: 2026년 5월 31일 (v2.0)**

---

## 📋 문서 버전 정보
- **버전**: v2.0 (모듈식 리팩토링 + NTSC 통합)
- **작성자**: 김승민, 김원석
- **마지막 업데이트**: 2026-05-31
- **상태**: ✅ v2.0 구현 완료

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

#### 구체적 목표 (v2.0 업데이트)
1. ✅ **NTSC 아날로그 비디오 시뮬레이터(zhuker/ntsc) 통합** — 실제 NTSC 아티팩트(dot crawl, ringing, color bleeding, chroma noise)를 degradation에 활용
2. ✅ **가우시안 로우패스 + 위너 필터 동시 사용 제거** — 위너 필터만 사용 (Gaussian LP는 제거)
3. ✅ **필터 모듈화** — 각 파이프라인 단계를 ON/OFF 가능한 FilterConfig로 분리
4. ✅ **De-blur / blur 최소화** — Unsharp Mask, Wiener Deconvolution, NLM, Bilateral 필터 추가
5. ✅ **추가 연구 방법 적용** — NLM(Non-Local Means), Bilateral Filter, CLAHE, Unsharp Mask, Wiener Deconvolution
6. ✅ **모듈식 파이프라인 아키텍처** — `smu_sig_prossessing/` 패키지로 구조화

---

## 2. 📊 실험 데이터 구성

### 2.1 데이터 유형

#### Type A: 정량 평가용 실험 데이터
- **목적**: PSNR/SSIM을 통한 정량적 성능 평가
- **구성**: 깨끗한 디지털 이미지 5종 + 인위적 열화 적용 (기본 + NTSC)
- **특징**: 원본 영상 보존 → 처리 전후 비교 가능

#### Type B: 실제 적용용 데이터
- **목적**: 현실적인 영상 개선 효과 확인
- **구성**: 실제 아날로그 FPV DVR 영상 프레임 (Google Drive 샘플)
- **특징**: 실제 무선 전송 및 녹화 과정 거침

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

## 3. 🔧 아키텍처 (v2.0)

### 3.1 패키지 구조

```
smu_sig_prossessing/
├── __init__.py          # 패키지 임포트
├── config.py            # PipelineConfig + FilterConfig (모듈식 설정)
├── filters.py           # 필터 레지스트리 (median, wiener, nlm, bilateral, fft_notch, gamma, histeq, deblur, unsharp 등)
├── pipeline.py          # 파이프라인 러너 (설정 기반 순차 실행)
├── degradation.py       # 열화 모듈 (기본 + NTSC)
├── evaluation.py        # 평가 유틸리티 (PSNR/SSIM, 비교 이미지, 히스토그램)
└── ntsc_plugin.py       # zhuker/ntsc 카피 (ringPattern.npy 포함)
```

### 3.2 파이프라인 아키텍처

```
┌─────────────┐    ┌──────────────┐    ┌───────────────────┐    ┌──────────────┐
│  입력 영상   │───→│  열화 (NTSC  │───→│  모듈식 필터       │───→│  평가         │
│ (Clean/DVR) │    │  + 기본)     │    │  PipelineConfig   │    │ PSNR/SSIM    │
└─────────────┘    └──────────────┘    └───────────────────┘    └──────────────┘
                                              │
                                     ┌────────┴────────┐
                                     │  FilterConfig 1  │
                                     │  FilterConfig 2  │
                                     │  FilterConfig N  │
                                     └─────────────────┘
```

### 3.3 PipelineConfig (필터 모듈화)

각 필터는 `FilterConfig(name, enabled, params)`로 독립적으로 제어:

```python
cfg = PipelineConfig(label="Wiener Only")
cfg.add("median", ksize=3)                # ON
cfg.add("wiener", noise_var=625)          # ON
cfg.add("fft_notch", enabled=False)       # OFF (주기적 잡음 없을 때)
cfg.add("channel_correction")
cfg.add("gamma", gamma=1.8)
cfg.add("histogram_eq", mode="yuv")
```

**Preset 파이프라인**:
- `PipelineConfig.wiener_only()` — Wiener 중심 (권장)
- `PipelineConfig.edge_preserving()` — NLM + Wiener (엣지 보존)
- `PipelineConfig.aggressive()` — 강력한 denoising
- `PipelineConfig.research_best()` — 모든 고급 기법 조합

### 3.4 필터 레지스트리 (등록된 모든 필터)

| 필터명 | 설명 | 파라미터 |
|--------|------|----------|
| median | 미디언 필터 (impulse 잡음) | ksize=3 |
| gaussian_lowpass | 가우시안 로우패스 (참고용, 권장 X) | sigma=1.5 |
| **wiener** | **위너 필터 (주파수 도메인, 권장)** | noise_var=625 |
| nlm | Non-Local Means (엣지 보존 탁월) | h=10, template=7, search=21 |
| bilateral | Bilateral Filter (엣지 보존 평활화) | d=9, sigma_color=75 |
| fft_notch | FFT 노치 필터 (주기적 잡음) | threshold=99.5% |
| gamma | 감마 보정 | gamma=1.8 |
| log_transform | 로그 변환 | c=40 |
| histogram_eq | 히스토그램 평활화 (YUV) | mode="yuv" |
| histogram_eq_clahe | CLAHE (적응형 평활화) | clip_limit=2.0 |
| channel_correction | RGB 채널 평균 보정 | clamp=0.7~1.3 |
| **unsharp_mask** | **언샤프 마스크 (블러 복원)** | strength=1.0, radius=1.0 |
| **deblur_wiener** | **위너 디컨볼루션 (디블러)** | kernel_size=5, noise_var=0.01 |

---

## 4. 🎯 개선 사항 (v1.0 → v2.0)

| 항목 | v1.0 | v2.0 | 효과 |
|------|------|------|------|
| **NTSC 열화** | 기본 합성 잡음만 | zhuker/ntsc 통합 | 실제 아날로그 아티팩트 재현 |
| **가우시안 LP + 위너** | 둘 다 사용 | 위너만 사용 | 불필요한 블러 제거 |
| **필터 모듈화** | 하드코딩 | PipelineConfig | ON/OFF 및 파라미터 조절 |
| **디블러/블러 최소화** | 없음 | Unsharp Mask + Wiener Deconv | 복원 영상의 선명도 향상 |
| **추가 연구 방법** | median, wiener, gauss | +NLM, Bilateral, CLAHE, Deblur | 더 다양한 방법 비교 가능 |
| **비디오 평가** | 기본 파이프라인만 | 다중 파이프라인 비교 (Wiener, Edge, Research) | 정량/정성 평가 모두 가능 |
| **코드 구조** | 단일 파일 2개 (717줄) | 모듈식 패키지 (8개 파일) | 유지보수성 향상 |

---

## 5. 📈 평가 계획

### 5.1 정량적 평가 (Quantitative)
- **대상**: Google Photos 샘플 (1분) / 합성 테스트 영상
- **지표**: PSNR, SSIM
- **비교**: 각 파이프라인별 프레임 평균 PSNR/SSIM
- **실행**: `python scripts/run_evaluation.py --quantitative <video>`

### 5.2 정성적 평가 (Qualitative)
- **대상**: Google Drive 샘플 (1분) / 실제 DVR 영상
- **방법**: 처리 전/후 이미지 나란히 비교 (4-way comparison)
- **분석**: NTSC 아티팩트 제거율, 색상 자연스러움, 엣지 보존 정도
- **실행**: `python scripts/run_evaluation.py --qualitative <video>`

---

## 6. 📋 실행 방법

```bash
# 정적 이미지 파이프라인 (기본)
python main.py

# 비디오 파이프라인
python main.py --video

# 필터 목록 확인
python main.py --list-filters

# 정량적 평가 (비디오 필요)
python scripts/run_evaluation.py --quantitative /path/to/video.mp4

# 정성적 평가 (비디오 필요)
python scripts/run_evaluation.py --qualitative /path/to/video.mp4
```

---

## 7. 📚 참고 자료

### 외부 라이브러리
- [NTSC 시뮬레이터 (zhuker/ntsc)](https://github.com/zhuker/ntsc) — 아날로그 비디오 시뮬레이션 (v2.0 통합)
- [Composite Video Simulator](https://github.com/joncampbell123/composite-video-simulator) — 원본 Java 구현

### 추가 연구 기법
- [Non-Local Means Denoising (OpenCV)](https://docs.opencv.org/master/d5/d69/tutorial_py_non_local_means.html)
- [Bilateral Filter (OpenCV)](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html)
- [CLAHE (Contrast Limited Adaptive Histogram Equalization)](https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html)
- [Wiener Deconvolution (Wikipedia)](https://en.wikipedia.org/wiki/Wiener_deconvolution)
