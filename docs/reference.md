# 참고 문헌 및 레퍼런스

본 프로젝트에서 사용한 오픈소스 라이브러리, 외부 코드, 참고 알고리즘 및 논문을 정리한 문서입니다.

---

## 1. 오픈소스 라이브러리 (Dependencies)

### 1.1 OpenCV (`opencv-python-headless`)

- **이름:** OpenCV (Open Source Computer Vision Library)
- **출처:** https://opencv.org / https://github.com/opencv/opencv
- **프로젝트에서의 용도:** 이미지 및 비디오 입출력, 색상 공간 변환, 필터링, 광학 흐름(Farneback), CLAHE 등 다양한 영상 처리 기능 전반
- **라이선스:** Apache 2.0

### 1.2 scikit-image (`scikit-image`)

- **이름:** scikit-image (skimage)
- **출처:** https://scikit-image.org / https://github.com/scikit-image/scikit-image
- **프로젝트에서의 용도:** Total Variation(TV) 디노이징 — `denoise_tv_chambolle` 함수를 활용한 Chambolle 알고리즘 기반 TV 노이즈 제거
- **라이선스:** BSD 3-Clause

### 1.3 SciPy (`scipy`)

- **이름:** SciPy (Scientific Computing Tools for Python)
- **출처:** https://scipy.org / https://github.com/scipy/scipy
- **프로젝트에서의 용도:** `ndimage.median_filter`(미디안 필터), `ndimage.gaussian_filter`(가우시안 필터) 등 과학 계산 및 신호/이미지 처리 유틸리티
- **라이선스:** BSD 3-Clause

### 1.4 NumPy (`numpy`)

- **이름:** NumPy (Numerical Python)
- **출처:** https://numpy.org / https://github.com/numpy/numpy
- **프로젝트에서의 용도:** 다차원 배열 연산, 행렬 계산, 수치 데이터 처리의 기반 라이브러리
- **라이선스:** BSD 3-Clause

### 1.5 PyWavelets (`pywt`)

- **이름:** PyWavelets (pywt)
- **출처:** https://pywavelets.readthedocs.io / https://github.com/PyWavelets/pywt
- **프로젝트에서의 용도:** 이산 웨이블릿 변환(DWT), 다단계 웨이블릿 분해 및 재구성. BayesShrink 임계값 추정과 결합하여 웨이블릿 기반 디노이징 구현
- **라이선스:** MIT

### 1.6 BM3D (`bm3d`)

- **이름:** BM3D (Block-Matching and 3D Filtering)
- **출처:** https://github.com/meric7784/bm3d (Python 구현체)
- **프로젝트에서의 용도:** 블록 매칭 기반 3D 협동 필터링 디노이징 — `bm3d_rgb`(컬러), `bm3d_gray`(그레이스케일) 함수를 통한 고품질 노이즈 제거
- **라이선스:** MIT

### 1.7 BM4D (`bm4d`)

- **이름:** BM4D (Block-Matching and 4D Filtering)
- **출처:** https://github.com/meric7784/bm4d (Python 구현체)
- **프로젝트에서의 용도:** 시공간(Spatio-Temporal) 디노이징 — 다중 프레임에 대한 4D 블록 매칭 필터링. `BM4DProfile2D` 프로파일을 활용한 동적 노이즈 제거
- **라이선스:** MIT

### 1.8 BRISQUE (`brisque`)

- **이름:** BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
- **출처:** https://github.com/bukalapak/pybrisque 및 관련 구현체
- **프로젝트에서의 용도:** 무참조(No-Reference) 이미지 품질 평가 지표 산출. 디노이징 및 강화 처리 후 화질을 정량적으로 평가
- **라이선스:** MIT

---

## 2. 프로젝트에 포함된 외부 코드

### 2.1 NTSC 비디오 시뮬레이터 (`ntsc_plugin.py`)

- **이름:** NTSC — 아날로그 비디오 시뮬레이터
- **출처:** https://github.com/zhuker/ntsc (zhuker/ntsc 저장소)
- **프로젝트에서의 용도:** NTSC 아날로그 비디오 신호 특성(컬러 버스트, 인터레이스, 고스트 등)을 시뮬레이션하여 빈티지/레트로 비디오 효과 생성
- **라이선스:** MIT License
- **비고:** 원본 저장소의 코드를 프로젝트 내 `ntsc_plugin.py`로 통합 포함

---

## 3. 참고 알고리즘 및 논문

### 3.1 디노이징 (Denoising)

#### Non-Local Means (비국소 평균 필터)

- **출처:** Buades, A., Coll, B., & Morel, J.-M. (2005). "A non-local algorithm for image denoising." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
- **프로젝트에서의 용도:** 이미지 내 유사한 패치를 검색하여 평균을 취하는 비국소 디노이징 기법. OpenCV의 `fastNlMeansDenoising` 함수로 구현

#### BM3D (Block-Matching and 3D Filtering)

- **출처:** Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. (2007). "Image denoising by sparse 3-D transform-domain collaborative filtering." *IEEE Transactions on Image Processing*, 16(8), 2080–2095.
- **프로젝트에서의 용도:** 유사 패치를 그룹화하여 3D 변환 도메인에서 협동 필터링을 수행하는 최고 수준의 디노이징 알고리즘

#### BM4D (Block-Matching and 4D Filtering)

- **출처:** Maggioni, M., Boracchi, G., Foi, A., & Egiazarian, K. (2014). "BM4D: A spatio-temporal video denoising algorithm based on block-matching and 4D collaborative filtering." *IEEE Transactions on Image Processing*, 23(8), 3736–3749.
- **프로젝트에서의 용도:** BM3D의 시공간 확장. 복수 프레임에서 유사 볼륨 패치를 수집하여 4D 협동 필터링으로 동영상 노이즈 제거

#### BayesShrink (베이즈 축소 임계값)

- **출처:** Chang, S. G., Yu, B., & Vetterli, M. (2000). "Adaptive wavelet thresholding for image denoising and compression." *IEEE Transactions on Image Processing*, 9(9), 1532–1546.
- **프로젝트에서의 용도:** 웨이블릿 계수의 분산을 추정하여 Bayes 위험을 최소화하는 임계값을 자동 결정. PyWavelets와 결합하여 적응적 웨이블릿 디노이징에 활용

#### Total Variation / ROF Model (전체 변분 모델)

- **출처:**
  - Rudin, L. I., Osher, S., & Fatemi, E. (1992). "Nonlinear total variation based noise removal algorithms." *Physica D: Nonlinear Phenomena*, 60(1–4), 259–268.
  - Chambolle, A. (2004). "An algorithm for total variation minimization and applications." *Journal of Mathematical Imaging and Vision*, 20(1–2), 89–97.
- **프로젝트에서의 용도:** 이미지의 전체 변분(TV)을 최소화하여 에지를 보존하면서 노이즈를 제거하는 최적화 기반 디노이징. Chambolle의 알고리즘으로 해 (scikit-image의 `denoise_tv_chambolle`)

#### Wiener Filter / Wiener Deconvolution (위너 필터)

- **출처:** Wiener, N. (1949). *Extrapolation, Interpolation, and Smoothing of Stationary Time Series*. MIT Press.
- **프로젝트에서의 용도:** 신호와 노이즈의 파워 스펙트럼을 기반으로 최소 평균 제곱 오차(MMSE) 복원 필터. 블러링 제거 및 디컨볼루션에 활용

### 3.2 에지 보존 필터링 (Edge-Preserving Filtering)

#### Bilateral Filter (양방향 필터)

- **출처:** Tomasi, C. & Manduchi, R. (1998). "Bilateral filtering for gray and color images." *IEEE International Conference on Computer Vision (ICCV)*, 839–846.
- **프로젝트에서의 용도:** 공간적 거리와 픽셀 값 차이를 동시에 고려하는 에지 보존 스무딩 필터. 노이즈 제거와 에지 유지를 동시에 달성

#### Guided Filter (가이드 필터)

- **출처:** He, K., Sun, J., & Tang, X. (2010). "Guided image filtering." *European Conference on Computer Vision (ECCV)*, 1–14.
- **프로젝트에서의 용도:** 가이드 이미지를 참조하여 입력 이미지를 필터링하는 에지 보존 필터. 양방향 필터보다 빠르고 그래디언트 왜곡이 적음

#### Rolling Guidance Filter (롤링 가이던스 필터)

- **출처:** Zhang, Q., Shen, X., Xu, L., & Jia, J. (2014). "Rolling guidance filter." *European Conference on Computer Vision (ECCV)*, 815–830.
- **프로젝트에서의 용도:** 반복적 가이던스를 통해 스케일 제어가 가능한 에지 보존 필터링. 다양한 크기의 구조를 선택적으로 보존/제거

#### Domain Transform (도메인 변환)

- **출처:** Gastal, E. S. L. & Oliveira, M. M. (2011). "Domain transform for edge-aware image and video processing." *ACM Transactions on Graphics (TOG)*, 30(4), 69.
- **프로젝트에서의 용도:** 고차원 이미지 데이터를 1차원으로 변환하여 효율적인 에지 인식 필터링 수행. 가우시안, 양방향 등 다양한 에지 보존 필터를 1D 순환으로 근사

#### Perona-Malik Anisotropic Diffusion (비등방성 확산)

- **출처:** Perona, P. & Malik, J. (1990). "Scale-space and edge detection using anisotropic diffusion." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 12(7), 629–639.
- **프로젝트에서의 용도:** 에지에서 확산을 억제하고 균일한 영역에서 확산을 촉진하는 편미분 방정식 기반 스무딩. 에지를 유지하면서 노이즈를 점진적으로 제거

### 3.3 이미지 강화 (Image Enhancement)

#### CLAHE (제한 대비 적응 히스토그램 평활화)

- **출처:** Zuiderveld, K. (1994). "Contrast limited adaptive histogram equalization." In *Graphics Gems IV*, Academic Press, 474–485.
- **프로젝트에서의 용도:** 국소 영역별로 적응적으로 대비를 향상시키는 히스토그램 평활화 기법. 노이즈 증폭을 제한(clip limit)하면서 저대비 영역의 가시성 개선

#### MSRCP / MSRCR Retinex (다중 스케일 Retinex)

- **출처:** Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997). "A multiscale Retinex for bridging the gap between color images and the human observation of scenes." *IEEE Transactions on Image Processing*, 6(7), 965–976.
- **프로젝트에서의 용도:** 다중 스케일 Retinex(MSR) 기반 색상 복원 및 대비 강화. MSRCP(색상 보존 변형) 및 MSRCR(색상 복원 변형)을 활용한 조명 불균일 보정

#### Grey-Edge Color Constancy (그레이 엣지 색 일정성)

- **출처:** van de Weijer, J., Gevers, T., & Gijsenij, A. (2007). "Edge-based color constancy." *IEEE Transactions on Image Processing*, 16(9), 2207–2214.
- **프로젝트에서의 용도:** 이미지 에지의 평균 색상 분포를 기반으로 장면 조명(white balance)을 추정하고 색 보정 수행

### 3.4 영상 품질 평가 (Image Quality Assessment)

#### NIQE (Natural Image Quality Evaluator)

- **출처:** Mittal, A., Soundararajan, R., & Bovik, A. C. (2013). "Making a 'completely blind' image quality analyzer." *IEEE Signal Processing Letters*, 22(3), 209–212.
- **프로젝트에서의 용도:** 자연 이미지의 통계적 특성(MVG 모델)과의 차이를 측정하는 무참조 품질 지표. 처리 전후 화질 변화를 정량 평가

#### BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)

- **출처:** Mittal, A., Moorthy, A. K., & Bovik, A. C. (2012). "No-reference image quality assessment in the spatial domain." *IEEE Transactions on Image Processing*, 21(12), 4695–4708.
- **프로젝트에서의 용도:** 이미지의 국소 정규화된 픽셀 통계(NSS) 기반 무참조 품질 평가. 디노이징 및 강화 파이프라인의 출력 품질 검증에 활용

### 3.5 동영상 처리 (Video Processing)

#### Farneback Optical Flow (파네백 광학 흐름)

- **출처:** Farnebäck, G. (2003). "Two-frame motion estimation based on polynomial expansion." *Scandinavian Conference on Image Analysis (SCIA)*, 363–370.
- **프로젝트에서의 용도:** 연속 프레임 간 밀집 광학 흐름(Dense Optical Flow) 추정. 모션 기반 프레임 정렬, 움직임 감지, 영상 안정화 등에 활용
