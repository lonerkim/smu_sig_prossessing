# smu_sig_prossessing 필터별 상세 설명

본 문서는 `smu_sig_prossessing` 파이프라인에 등록된 40개 필터 각각의 **원리(Principle)**, **작동 방식(Operation)**, **구현 상세(Implementation)**를 한국어로 설명합니다.

필터는 코드베이스의 위상(Phase) 순서에 따라 구성되었으며, 각 위상은 특정 영상 처리 목적(잡음 제거, 선명화, 색상 보정, 아날로그 비디오 복원 등)을 중심으로 그룹화됩니다.

---

## Phase 1–2: 기본 잡음 제거 (Basic Denoising)

### 1. median

**원리**
Median 필터는 각 픽셀을 이웃 픽셀들의 중앙값(median)으로 대체하는 비선형 저역 통과 필터입니다. 임펄스 잡음(salt-and-pepper noise) 제거에 최적화되어 있으며, 평균(mean) 필터와 달리 에지 정보를 상대적으로 잘 보존합니다. 특정 픽셀이 극단적으로 튀는 값(impulse)을 가질 때, 중앙값은 해당 이상치의 영향을 받지 않으므로 효과적으로 제거됩니다.

**작동 방식**
커널 크기 `ksize`(정사각형, 홀수)가 주어지면, 각 픽셀에 대해 `ksize × ksize` 윈도우 내 모든 픽셀 값을 정렬한 후 중앙값을 선택합니다. 컬러 영상의 경우 각 채널(B, G, R)에 독립적으로 적용됩니다.

**구현 상세**
```python
cv2.medianBlur(img, ksize)
```
- **ksize**: 기본값 3 (3×3 커널). 클수록 잡음 제거는 강력하지만 디테일 손실 증가.
- OpenCV 내부 구현은 `cv2.medianBlur()`를 호출하며, SSE/AVX 최적화된 네이티브 코드로 동작.
- 입출력 데이터 타입: `np.ndarray`, uint8 (0–255).

---

### 2. gaussian_lowpass

**원리**
가우시안 저역 통과 필터는 2D 가우시안 함수를 커널로 사용하여 영상을 평탄화(smoothing)합니다. 가우시안 커널은 중심에서 멀어질수록 가중치가 점진적으로 감소하므로, 균일한 블록 커널(box filter)보다 자연스러운 블러를 생성합니다. 주파수 영역에서는 저주파 성분을 통과시키고 고주파 성분(잡음, 에지)을 감쇠시킵니다.

**작동 방식**
커널 크기는 `sigma` 값에 의해 자동 결정되며(`cv2.GaussianBlur`에서 `(0, 0)` 전달 시 sigma에 기반하여 계산됨), 각 픽셀은 커널 범위 내 가중 평균으로 대체됩니다.

**구현 상세**
```python
cv2.GaussianBlur(img, (0, 0), sigma)
```
- **sigma**: 기본값 1.5. σ가 클수록 블러가 강해짐.
- `G(x, y) = (1 / (2πσ²)) · exp(-(x² + y²) / (2σ²))`
- 참고용으로만 유지됨; 실제 파이프라인에서는 `wiener` 필터 사용 권장.

---

### 3. wiener

**원리**
Wiener 필터는 주파수 영역에서 최소 평균 제곱 오차(MMSE) 추정을 수행하는 적응형 잡음 제거 필터입니다. 신호와 잡음의 전력 스펙트럼 밀도(PSD)를 추정하여, 각 주파수 성분에 대해 최적의 이득을 계산합니다. 가우시안 저역 통과 필터와 달리 에지를 보존하면서 잡음을 제거할 수 있습니다.

**작동 방식**
채널별 2D FFT를 계산한 후, Wiener 필터 전달 함수를 적용합니다:
- `H(w) = max(|F|² - N, 0) / max(|F|², N)`
- 여기서 `|F|²`는 관측된 신호의 전력, `N`은 잡음 분산 추정치(`noise_var`)입니다.
- 신호 전력(`|F|² - N`)이 큰 주파수(에지, 텍스처)는 통과시키고, 신호 전력이 작은 주파수(잡음 지배)는 감쇠시킵니다.
- IFFT로 공간 영역 복원 후, `[0, 255]` 클리핑.

**구현 상세**
```python
f = np.fft.fft2(channel.astype(np.float64))
f_shift = np.fft.fftshift(f)
power = np.abs(f_shift) ** 2
signal_est = np.maximum(power - noise_var, 0)
h = signal_est / np.maximum(signal_est + noise_var, 1e-10)
result = f_shift * h
```
- **noise_var**: 기본값 400 (σ≈20에 해당). 낮은 값은 덜 공격적인 잡음 제거.
- 컬러 영상은 채널별 독립 처리.
- `noise_var=625`도 지원 (더 강한 잡음 제거).

---

### 4. nlm

**원리**
비지역 평균(Non-Local Means, NLM) 필터는 공간적으로 인접할 필요 없이 영상 전체에서 유사한 패치(patch)를 찾아 가중 평균하는 방식입니다. Buades et al.(2005)이 제안한 방법으로, 가우시안 블러보다 훨씬 우수한 PSNR을 보이며 에지를 잘 보존합니다. 모든 픽셀 쌍 간의 유사도를 패치 단위로 평가합니다.

**작동 방식**
1. 각 픽셀 주변의 `template_window × template_window` 템플릿 패치를 추출합니다.
2. `search_window × search_window` 탐색 윈도우 내 모든 픽셀에서 유사한 패치를 검색합니다.
3. 패치 간 유클리드 거리에 기반한 가중치로 가중 평균합니다.
4. `h`(필터 강도)가 클수록 더 적극적인 잡음 제거(더 많은 블러).

**구현 상세**
```python
cv2.fastNlMeansDenoisingColored(img, None, h, h, template_window, search_window)
```
- **h**: 기본값 10. 필터 강도 (클수록 잡음 제거 증가).
- **template_window**: 기본값 7. 패치 크기 (홀수).
- **search_window**: 기본값 21. 검색 윈도우 크기 (홀수).
- OpenCV의 `fastNlMeansDenoisingColored` 사용 — YUV 색공간에서 컬러 NLM 수행.

---

### 5. nlm_gray

**원리**
NLM 필터를 휘도(Y) 채널에만 적용하는 변형입니다. 인간 시각이 색차(chroma) 잡음보다 휘도(luma) 잡음에 훨씬 민감하다는 특성을 활용하여, 휘도 채널만 선택적으로 잡음 제거하고 색차 채널은 그대로 유지함으로써 연산량을 줄이고 색상 정보를 보존합니다.

**작동 방식**
1. BGR → YUV 색공간 변환.
2. Y(휘도) 채널에만 `cv2.fastNlMeansDenoising` 적용.
3. U, V(색차) 채널은 변경 없이 그대로 유지.
4. YUV → BGR 역변환.

**구현 상세**
```python
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
yuv[:, :, 0] = cv2.fastNlMeansDenoising(yuv[:, :, 0], None, h, template_window, search_window)
return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
```
- NLM 파라미터는 `nlm`과 동일하나 컬러 버전이 아닌 단일 채널 NLM 사용.
- 연산량이 `nlm` 대비 약 1/3 수준.

---

### 6. bilateral

**원리**
양방향 필터(Bilateral Filter)는 공간적 거리(spatial distance)와 색상 차이(intensity/color difference)를 모두 고려한 가중 평균을 계산하는 에지 보존 평활화 필터입니다. Tomasi와 Manduchi(1998)가 제안했습니다. 공간 가우시안 커널이 인접 픽셀에 높은 가중치를 주고, 색상 범위(range) 가우시안 커널이 색상 차이가 큰 픽셀(에지 건너편)에는 낮은 가중치를 부여합니다.

**작동 방식**
출력 픽셀 `O(p)`는 다음과 같이 계산됩니다:
- `O(p) = (1/W(p)) · Σ_q∈N(p) [ G_s(||p-q||) · G_r(|I(p)-I(q)|) · I(q) ]`
- `G_s`: 공간 가우시안 (σ_space)
- `G_r`: 범위 가우시안 (σ_color)
- `W(p)`: 정규화 상수 (가중치 합)
- `d`: 이웃 반경 (직경)

**구현 상세**
```python
cv2.bilateralFilter(img, d, sigma_color, sigma_space)
```
- **d**: 기본값 9. 픽셀 이웃 직경.
- **sigma_color**: 기본값 75. 색공간 가우시안 σ.
- **sigma_space**: 기본값 75. 좌표공간 가우시안 σ.
- OpenCV 구현은 공간 커널을 1D 분리하여 최적화.

---

### 7. fft_notch

**원리**
노치 필터(Notch Filter)는 주파수 영역에서 특정 주파수 성분(잡음 피크)을 제거하는 필터입니다. 2D FFT로 영상을 주파수 영역으로 변환한 후, 잡음에 해당하는 특정 주파수 대역을 식별하여 제거(zero out)하고 IFFT로 복원합니다. 주기적 잡음(스캔라인, 도트 크롤, 모터 간섭) 제거에 탁월합니다.

**작동 방식**
1. 채널별 2D FFT 수행 → 주파수 스펙트럼 획득.
2. 스펙트럼 `magnitude`의 백분위수 기반 임계값 계산 (`threshold_percentile`).
3. 임계값을 초과하는 피크를 잡음으로 식별.
4. DC 영역(저주파, 중심 7×7)은 보호.
5. 식별된 피크 위치의 주파수 성분을 0으로 설정.
6. IFFT로 공간 영역 복원.

**구현 상세**
```python
threshold = np.percentile(magnitude, threshold_percentile)
mask = magnitude > threshold
mask[crow-3:crow+4, ccol-3:ccol+4] = False  # DC 보호
f_shift[mask] = 0
```
- **threshold_percentile**: 기본값 99.5. 높을수록 극단적인 피크만 제거.
- 3채널 컬러는 각 채널 독립 처리.

---

### 8. vertical_notch

**원리**
수직 방향 노치 필터는 수직선(vertical line) 아티팩트를 주파수 영역에서 제거하는 특수화된 노치 필터입니다. 수직선은 공간 영역에서 특정 수평 주파수(u축)에 집중된 에너지를 생성하며, 이는 모든 수직 주파수(v축)에 걸쳐 일정하게 나타납니다. 따라서 v축 방향으로 평균화된 1D 수평 프로파일을 분석하여 피크를 탐지합니다.

**작동 방식**
1. 2D FFT → 주파수 스펙트럼.
2. v축(수직) 방향으로 magnitude 평균 → 1D 수평 프로파일.
3. 국소 중앙값 필터로 베이스라인 추정, 잔차(residual) 계산.
4. MAD(Median Absolute Deviation) 기반 강건한 이상치 탐지로 피크 주파수 식별.
5. 각 피크 주파수에서 `notch_radius` 대역폭만큼 u축 방향으로 0화(zero out).
6. DC 영역(`protect_dc` 반경)은 보호.
7. IFFT로 복원.

**구현 상세**
```python
h_profile = np.mean(magnitude, axis=0)
baseline = median_filter(h_profile, size=21)
residual = h_profile - baseline
# MAD 기반 임계값
med_res = np.median(full_res)
mad_res = np.median(np.abs(full_res - med_res))
threshold = med_res + sigma_detect * 1.4826 * mad_res
```
- **sigma_detect**: 기본값 3.0 (MAD 스케일). 낮을수록 더 공격적.
- **notch_radius**: 기본값 3. 스펙트럼 누설 처리를 위한 노치 대역 반폭.
- **protect_dc**: 기본값 10 (픽셀). DC 영역 보호 반경.
- 수직선 아티팩트에 특화되어 `fft_notch`보다 정밀.

---

### 9. wavelet

**원리**
웨이블릿 기반 잡음 제거는 영상을 다중 해상도(multi-resolution) 부밴드(subband)로 분해한 후, 각 부밴드에서 적응형 임계값 처리(thresholding)를 수행합니다. BayesShrink 방식은 각 부밴드의 통계적 특성에 기반하여 최적 임계값을 추정합니다. Cycle-spinning(순환 이동)은 웨이블릿 변환의 이동 변위성(shift-variance)으로 인한 아티팩트를 줄이기 위해 여러 번 이동-잡음 제거-복원-평균 과정을 반복합니다.

**작동 방식**
1. PyWavelets의 `wavedec2`로 DWT(이산 웨이블릿 변환) 수행 (레벨 `level`, 웨이블릿 `wavelet`).
2. 각 세부 부밴드(LH, HL, HH)에서:
   - 잡음 표준편차 `σ` 추정: `σ = median(|d|) / 0.6745` (MAD 기반 강건 추정).
   - BayesShrink 임계값: `T = σ² / σ_x` (σ_x = √max(var(d) - σ², 0)).
   - `pywt.threshold()`로 soft 또는 hard 임계값 적용.
3. `waverec2`로 역변환 후 재구성.
4. Cycle-spinning (`n_shifts`): 다중 시프트 → 잡음 제거 → 역시프트 → 평균.

**구현 상세**
- **wavelet**: 기본값 `"bior4.4"` (대칭성 우수, 링잉 감소). `"db4"`, `"sym8"` 지원.
- **level**: 기본값 3 (2–4 권장).
- **threshold_mode**: 기본값 `"soft"` (부드러움), `"hard"` (선명함).
- **n_shifts**: 기본값 3 (0 = cycle-spinning 비활성화).
- `pywt.threshold()`의 mode 파라미터로 soft/hard 제어.
- 3채널 컬러는 각 채널 독립 처리.

---

## Phase 3–4: 고급 필터 (Advanced Filters)

### 10. gamma

**원리**
감마 보정(Gamma Correction)은 각 픽셀 값에 지수 함수를 적용하여 영상의 밝기와 대비를 조정하는 비선형 변환입니다. `O = I^(1/γ)` 형태로, γ > 1이면 어두운 영역이 밝아지고(명암대비 확장), γ < 1이면 밝은 영역이 더 밝아집니다(명암대비 압축). CRT 디스플레이의 비선형 응답을 보정하기 위해 역사적으로 사용되었습니다.

**작동 방식**
0–255 범위의 입력에 대해 LUT(Look-Up Table)를 미리 생성하여 모든 픽셀에 동일한 매핑을 적용합니다: `LUT[i] = ((i/255)^(1/γ)) × 255`

**구현 상세**
```python
table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype("uint8")
return cv2.LUT(img, table)
```
- **gamma**: 기본값 1.5 (어두운 영역 확장). 1.8도 자주 사용.
- `cv2.LUT`를 사용하여 고속 처리 (256개 엔트리 LUT).
- 컬러 영상의 모든 채널에 동일한 LUT 적용.

---

### 11. log_transform

**원리**
로그 변환은 좁은 범위의 어두운 픽셀 값을 넓은 범위로 확장하고, 넓은 범위의 밝은 값을 압축하는 비선형 매핑입니다. `O = c · log(1 + I)` 형태로, 푸리에 스펙트럼 시각화나 저조도 영상 향상에 사용됩니다. 감마 보정과 유사하지만 로그 함수는 낮은 입력값에서 더 급격한 변화를 만듭니다.

**작동 방식**
각 픽셀에 `c * log(1 + pixel_value)`를 적용합니다. `log1p`(자연로그 기반)를 사용하여 수치적 안정성을 확보합니다.

**구현 상세**
```python
result = c * np.log1p(img_f)
```
- **c**: 기본값 40. 출력 스케일 계수.
- `np.log1p(x) = ln(1+x)` — x=0에서 로그 폭발 방지.
- 결과를 `[0, 255]`로 클리핑.

---

### 12. histogram_eq_gray

**원리**
히스토그램 평활화(Histogram Equalization)는 영상의 명암 분포를 균등하게 재분배하여 대비를 향상시키는 방법입니다. 누적 분포 함수(CDF)를 기반으로 픽셀 값을 재매핑합니다. 그레이스케일 버전은 컬러 영상을 먼저 그레이스케일로 변환한 후 적용합니다.

**작동 방식**
1. BGR → Grayscale 변환.
2. `cv2.equalizeHist()`로 글로벌 히스토그램 평활화.
3. Grayscale → BGR 역변환 (모든 채널에 동일한 값).

**구현 상세**
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eq = cv2.equalizeHist(gray)
return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
```
- 결과는 무채색(grayscale) 영상이 컬러 형식으로 반환됨.
- 빠르지만 잡음 증폭 가능성 있음.

---

### 13. histogram_eq

**원리**
YUV 색공간에서 Y(휘도) 채널에만 히스토그램 평활화를 적용하는 방식입니다. 색차 채널(U, V)을 건드리지 않으므로 색상 왜곡 없이 명암대비만 향상됩니다. `histogram_eq_gray`가 영상을 완전히 그레이스케일로 만드는 반면, 이 필터는 원래 색상을 보존합니다.

**작동 방식**
1. BGR → YUV 변환.
2. Y 채널에만 `cv2.equalizeHist()` 적용.
3. U, V 채널은 변경하지 않음.
4. YUV → BGR 역변환.

**구현 상세**
```python
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
```

---

### 14. histogram_eq_clahe

**원리**
CLAHE(Contrast Limited Adaptive Histogram Equalization)는 영상을 작은 타일(tile)로 분할한 후 각 타일에서 국소적으로 히스토그램 평활화를 적용합니다. 글로벌 평활화와 달리 국소 대비 향상이 가능하며, `clip_limit` 파라미터로 대비 증폭을 제한하여 잡음 증폭을 방지합니다. LAB 색공간의 L(명도) 채널에 적용하여 색상 변화를 최소화합니다.

**작동 방식**
1. BGR → LAB 변환.
2. L 채널에 `cv2.createCLAHE(clipLimit, tileGridSize)` 적용.
3. A, B(색상) 채널은 유지.
4. LAB → BGR 역변환.

**구현 상세**
```python
clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
lab[:, :, 0] = clahe.apply(lab[:, :, 0])
```
- **clip_limit**: 기본값 2.0. 대비 제한 임계값 (높을수록 더 강한 대비).
- **tile_size**: 기본값 8. 타일 그리드 크기 (8×8).

---

### 15. channel_correction

**원리**
RGB 채널 평균 보정은 각 색상 채널(R, G, B)의 평균 밝기를 일치시켜 색상 균형(color balance)을 맞추는 방법입니다. Grey-World 가정(영상의 평균 색상이 회색이라는 전제)에 기반하여, 각 채널에 스케일 팩터를 곱해 모든 채널의 평균이 동일해지도록 조정합니다.

**작동 방식**
1. 각 채널의 평균 `mean_c` 계산.
2. 전체 채널 평균 `target = mean(mean_R, mean_G, mean_B)` 계산.
3. 각 채널의 스케일 팩터 `scale_c = target / mean_c`.
4. `scale_c`를 `[clamp_min, clamp_max]` 범위로 클리핑하여 과보정 방지.
5. 각 채널에 `scale_c` 적용 후 `[0, 255]` 클리핑.

**구현 상세**
```python
means = [np.mean(result[:, :, c]) for c in range(3)]
target = np.mean(means)
scale = np.clip(target / means[c], clamp_min, clamp_max)
result[:, :, c] *= scale
```
- **clamp_min**: 기본값 0.7 (사용자 사양: 0.85).
- **clamp_max**: 기본값 1.3 (사용자 사양: 1.25).
- 과보정 방지를 위한 클리핑이 핵심.

---

### 16. unsharp_mask

**원리**
언샤프 마스킹(Unsharp Masking)은 원본 영상에서 블러된(언샤프) 버전을 차감하여 에지 성분(마스크)을 추출한 후, 이를 다시 원본에 더하여 선명도를 높이는 고전적인 샤프닝 기법입니다. 잡음 제거 후 발생한 블러를 보상하여 디테일을 회복하는 데 사용됩니다.

**작동 방식**
1. 가우시안 블러로 원본의 블러 버전 생성.
2. `sharpened = (1 + strength) × original - strength × blurred`
3. `threshold`가 0보다 크면, 원본과 블러 간 차이가 임계값을 초과하는 픽셀에만 샤프닝 적용 (아티팩트 방지).

**구현 상세**
```python
blurred = cv2.GaussianBlur(img, (0, 0), radius)
sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
```
- **strength**: 기본값 1.0. 샤프닝 강도.
- **radius**: 기본값 1.0. 가우시안 블러 반경.
- **threshold**: 기본값 10. 최소 그래디언트 임계값 (높을수록 아티팩트 감소).

---

### 17. deblur_wiener

**원리**
Wiener 디컨볼루션(Deconvolution)은 알려진 또는 추정된 블러 커널(PSF, Point Spread Function)을 역으로 추정하여 영상의 블러를 제거하는 주파수 영역 방법입니다. `F = conj(H) / (|H|² + K) × G` 형태로, 잡음 증폭을 제어하기 위해 정규화 항 `K`(noise_var)를 포함합니다.

**작동 방식**
1. 가우시안 커널을 PSF로 가정 (`cv2.getGaussianKernel`).
2. PSF를 영상 크기로 패딩 후 중심 정렬.
3. FFT로 주파수 영역 변환.
4. Wiener 필터 계수 계산: `H_conj / (|H|² + noise_var)`.
5. 주파수 영역 곱셈 후 IFFT로 복원.

**구현 상세**
```python
kernel = cv2.getGaussianKernel(ksize, -1)
psf = kernel @ kernel.T
psf /= psf.sum()
# Wiener: F = conj(H) / (|H|^2 + K) * G
wiener = psf_conj / (psf_mag + noise_var)
result = img_f * wiener
```
- **kernel_size**: 기본값 5. 가정된 블러 커널 크기.
- **noise_var**: 기본값 0.01. 잡음 전력 추정값 (높을수록 덜 공격적).
- 컬러 영상은 채널별 독립 처리.

---

### 18. guided_filter

**원리**
Guided Filter(가이드 필터)는 로컬 선형 모델(local linear model)을 가정하는 에지 보존 평활화 필터입니다. He et al.(2013)이 제안했습니다. 출력 `q`가 가이드 영상 `I`의 선형 변환으로 표현된다고 가정합니다: `q_i = a_k · I_i + b_k, ∀i∈ω_k`. 가이드 영상과 입력 `p` 사이의 차이를 최소화하는 `(a_k, b_k)`를 각 윈도우에서 최적화합니다. 박스 필터(box filter)만으로 구현되어 O(N)의 빠른 속도를 냅니다.

**작동 방식**
1. 가이드 영상: 입력의 그레이스케일 버전 (정규화 0–1).
2. 각 채널에 대해 guided filter 적용:
   - `mean_I`, `mean_P`, `mean_II`, `mean_IP`를 박스 필터로 계산.
   - `var_I = mean_II - mean_I²`, `cov_IP = mean_IP - mean_I·mean_P`.
   - `a = cov_IP / (var_I + eps)`, `b = mean_P - a·mean_I`.
   - `mean_a`, `mean_b`를 박스 필터로 평활화.
   - `q = mean_a·I + mean_b`.

**구현 상세**
```python
mean_I = cv2.boxFilter(guide, -1, (r, r))
# ... (각 채널에 대해 위 과정 반복)
return mean_a * guide + mean_b
```
- **radius**: 기본값 3. 박스 필터 반경.
- **eps**: 기본값 100.0. 정규화 파라미터 (클수록 더 평탄화).

---

### 19. anisotropic_diffusion

**원리**
Perona-Malik 이방성 확산(Anisotropic Diffusion)은 편미분 방정식(PDE) 기반의 잡음 제거 방법입니다. 열 확산 방정식을 모방하지만, 에지에서 확산 계수를 감소시키는 에지 중단 함수(edge-stopping function)를 사용합니다. `∂I/∂t = div(c(x, y, t) · ∇I)`에서 `c(||∇I||) = exp(-(||∇I||/κ)²)` 형태로, 그래디언트가 큰 곳(에지)에서 확산을 차단합니다.

**작동 방식**
각 반복(iteration)에서:
1. 4방향 그래디언트 계산: 북/남/동/서 (`np.roll` 사용).
2. 지수 함수 기반 에지 중단 계수 계산: `cN = exp(-(n/κ)²)`.
3. 확산 업데이트: `I += γ · (cN·n + cS·s + cE·e + cW·w)`.

**구현 상세**
```python
for _ in range(n_iter):
    n = np.roll(ch, -1, axis=0) - ch
    s = np.roll(ch, 1, axis=0) - ch
    e = np.roll(ch, -1, axis=1) - ch
    w = np.roll(ch, 1, axis=1) - ch
    cN = np.exp(-(n / kappa) ** 2)
    # ...
    ch += gamma * (cN * n + cS * s + cE * e + cW * w)
```
- **n_iter**: 기본값 10. 반복 횟수.
- **kappa**: 기본값 30. 에지 중단 임계값 (낮을수록 에지 보존 강함).
- **gamma**: 기본값 0.25. 확산 속도 (안정성을 위해 ≤ 0.25).

---

### 20. tv_denoise

**원리**
총변동(Total Variation, TV) 잡음 제거는 Rudin-Osher-Fatemi(ROF) 모델로, `min ∫|∇u| + (λ/2)·∫(u-f)²`를 최소화합니다. 첫 번째 항은 출력 영상 `u`의 총변동(에지의 합)을 최소화하고, 두 번째 항은 입력 `f`와의 충실도를 유지합니다. Chambolle의 알고리즘을 사용한 빠른 수렴이 특징으로, 작은 잡음을 제거하면서 선명한 에지를 보존하는 데 탁월합니다.

**작동 방식**
1. 입력을 float32 [0, 1] 범위로 정규화.
2. `skimage.restoration.denoise_tv_chambolle()` 호출.
3. 컬러 영상은 `channel_axis=-1`로 채널 자동 처리.
4. 결과를 `[0, 255]` uint8로 변환.

**구현 상세**
```python
from skimage.restoration import denoise_tv_chambolle
result = denoise_tv_chambolle(img_f, weight=weight, max_num_iter=max_num_iter, eps=eps, channel_axis=-1)
```
- **weight**: 기본값 0.1. λ의 역수 (클수록 더 강한 평활화).
- **eps**: 기본값 2e-4. 수렴 기준.
- **n_iter_max**: 기본값 200. 최대 반복 횟수.

---

## Phase 5–6: 영상 향상 (Enhancement)

### 21. bm3d

**원리**
BM3D(Block-Matching and 3D Filtering)는 현재까지 알려진 가장 성능이 우수한 잡음 제거 알고리즘 중 하나입니다(Dabov et al., 2007). 두 단계로 구성됩니다:
1. **Hard-thresholding 단계**: 영상을 블록으로 나누고 유사한 블록을 그룹화하여 3D 배열을 구성한 후, 3D 변환 도메인에서 hard-thresholding으로 잡음 제거.
2. **Wiener 필터링 단계**: 첫 단계에서 추정된 잡음 제거 결과를 사용하여 Wiener 필터로 최종 잡음 제거.

**작동 방식**
1. BGR → RGB 변환 (`bm3d` 라이브러리가 RGB 기대).
2. `bm3d_rgb()` 호출 (컬러 영상).
3. RGB → BGR 역변환.
4. 결과 float64를 uint8로 반올림 및 클리핑.
5. 그레이스케일의 경우 `bm3d()` 직접 호출 (stage_arg 지원).

**구현 상세**
```python
from bm3d import bm3d_rgb, bm3d
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result_rgb = bm3d_rgb(rgb, sigma_psd=sigma_psd)
return cv2.cvtColor(np.clip(np.round(result_rgb), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
```
- **sigma_psd**: 기본값 15.0. 잡음 표준편차 (5–50).
- **stage_arg**: 기본값 3 (두 단계 모두 사용). 1=hard-thresholding만, 2=Wiener만.

---

### 22. bm3d_denoise

**원리**
BM3D를 각 색상 채널에 독립적으로 적용하는 변형입니다. `bm3d` 필터가 `bm3d_rgb`(공동 컬러 처리)를 사용하는 반면, 이 필터는 각 채널을 별도의 그레이스케일 BM3D로 처리합니다. 채널 간 색상 번짐(color bleeding)을 방지할 수 있습니다.

**작동 방식**
1. 각 채널(R, G, B)을 float64로 변환.
2. 그레이스케일 `bm3d()` 함수 호출 (`BM3DStages.ALL_STAGES`).
3. 모든 채널 결과를 스택하여 최종 영상 생성.

**구현 상세**
```python
denoised = bm3d_gray(channel, sigma_psd=sigma, profile=profile, stage_arg=BM3DStages.ALL_STAGES)
```
- **sigma**: 기본값 25.0 (중간~강한 잡음에 적합).
- **profile**: 기본값 `"np"` (두 단계). `"lc"`(빠름), `"high"`(고품질) 지원.

---

### 23. chroma_denoise

**원리**
색차(Chroma) 전용 잡음 제거는 인간 시각이 휘도(luma) 잡음보다 색차(chroma) 잡음에 훨씬 덜 민감하다는 특성을 활용합니다. YCbCr 색공간으로 변환 후 Cb(청색 차이)와 Cr(적색 차이) 채널에만 강력한 양방향 필터를 적용하고, Y(휘도) 채널은 완전히 보존합니다.

**작동 방식**
1. BGR → YCbCr 변환.
2. Y 채널은 그대로 유지.
3. Cb, Cr 채널에 양방향 필터 적용.
4. `strength` 파라미터에 따라 필터 강도 스케일링.
5. strength > 0.7이면 2차 필터 적용.

**구현 상세**
```python
cb_denoised = cv2.bilateralFilter(cb.astype(np.uint8), d, sigma_color, sigma_space)
cr_denoised = cv2.bilateralFilter(cr.astype(np.uint8), d, sigma_color, sigma_space)
```
- **d**: `max(5, int(9 * strength))`. 필터 직경.
- **sigma_color**: `30 + 70 * strength` (30–100).
- **sigma_space**: `30 + 70 * strength` (30–100).

---

### 24. retinex (MSRCP)

**원리**
다중 스케일 Retinex(MSRCP)는 Jobson et al.(1997)이 제안한 조명 보정 알고리즘입니다. 영상을 반사율(reflectance, R)과 조명(illumination, I) 성분으로 분해한 후, 조명 성분을 정규화하여 일관된 밝기와 대비를 얻습니다. MSRCP 버전은 채도 보존(chromaticity preservation)을 통해 자연스러운 색상을 유지합니다.

**작동 방식**
1. BGR → float32 [0, 1] 변환.
2. 각 픽셀의 최대값 `I = max(R, G, B)`를 강도로 사용.
3. 각 스케일 σ에서: `retinex += weight × (log(I+1) - log(GaussianBlur(I, σ)+1))`
4. Gain/offset 적용.
5. 색도 복원: `out_c = retinex_out × (img_c / I)`.
6. `[0, 1]` 클리핑 후 uint8 변환.

**구현 상세**
```python
I = np.max(img_f, axis=2)
for sigma, weight in zip(sigma_list, weights):
    blurred = cv2.GaussianBlur(I, (0, 0), sigma)
    diff = np.log(I + small_val) - np.log(blurred + small_val)
    retinex_out += weight * diff
ratio = np.divide(img_f, I_3d, where=(I_3d > 0), out=np.zeros_like(img_f))
result = np.expand_dims(retinex_out, axis=2) * ratio
```
- **sigma_list**: 기본값 `[15, 80, 250]`.
- **strength**: 사용자 사양에서 0.5 (gain에 반영).
- **gain**: 기본값 5.0, **offset**: 기본값 0.0.

---

### 25. retinex_msrcr

**원리**
다중 스케일 Retinex with Color Restoration(MSRCR)은 MSR의 단점인 채도 저하(desaturation)를 보완하기 위해 색상 복원 팩터 `C_c`를 도입합니다:
`C_c(x, y) = β · [log(α · I_c(x, y) + 1) - log(Σ_i I_i(x, y) + 1)]`
이 팩터는 원본 색상 비율을 보존하여 바랜(faded) 영상에서도 자연스러운 색상을 복원합니다.

**작동 방식**
1. 각 채널별 MSR 수행 (RGB 각각에 대해 다중 스케일 로그 차분).
2. 색상 복원 팩터 계산 (alpha, beta 파라미터).
3. `R_msrcr_c = gain · C_c · R_c + offset`.
4. `[0, 255]` 클리핑.

**구현 상세**
```python
# Color restoration factor
numerator = alpha * img_f[:, :, c] + 1.0
cr_factor[:, :, c] = np.log(numerator) - np.log(sum_channels)
cr_factor[:, :, c] = np.clip(cr_factor[:, :, c], 1.0, 5.0)
result = gain * cr_factor * msr + offset
```
- **sigma_list**: `[15, 80, 250]`.
- **gain**: 기본값 5.0, **offset**: 기본값 25.0.
- **alpha**: 기본값 125.0, **beta**: 기본값 46.0.

---

### 26. patch_collaborative

**원리**
블록 DCT 잡음 제거는 영상을 고정 크기 블록으로 분할한 후 각 블록에 DCT(이산 코사인 변환)를 적용하고, DCT 도메인에서 hard-thresholding으로 작은 계수(잡음)를 제거한 후 IDCT로 복원하는 방법입니다. BM3D의 간소화된 버전으로, 3D 그룹화 없이 각 블록을 독립적으로 처리하여 훨씬 빠릅니다.

**작동 방식**
1. 입력 영상을 블록 크기(`patch_size`)의 배수로 패딩(reflect 모드).
2. 각 블록에 대해:
   - `cv2.dct()`로 DCT 변환.
   - `|DCT coeff| < h_dct`인 계수를 0으로 설정 (hard threshold).
   - `cv2.idct()`로 역변환.
3. 블록 경계에서 가중치를 누적하여 평균 (중복 블록 처리).
4. 패딩 제거.

**구현 상세**
```python
dct_block = cv2.dct(block)
dct_block[np.abs(dct_block) < h_dct] = 0
recon = cv2.idct(dct_block)
```
- **patch_size**: 기본값 8. 블록 크기.
- **h_dct**: 기본값 30.0. DCT 도메인 hard-threshold (높을수록 더 강한 잡음 제거).

---

### 27. temporal_average

**원리**
시간적 평균화(Temporal Averaging)는 비디오의 연속된 프레임을 누적하여 평균을 내는 가장 기본적인 시간적 잡음 제거 방법입니다. 고정된 장면에서 잡음은 프레임 간 무작위로 변동하므로 평균화를 통해 효과적으로 제거됩니다. 단, 움직임이 있는 영역에서는 고스팅(ghosting) 아티팩트가 발생할 수 있습니다.

**작동 방식**
1. 프레임 누적 버퍼 유지 (모듈 레벨 상태 `_temporal_state`).
2. 각 프레임을 float32 누적기에 추가 (`acc += img`).
3. `n_frames` 프레임 누적 시 평균을 출력하고 버퍼 리셋.
4. 누적이 덜 된 경우 현재까지의 평균을 출력.

**구현 상세**
```python
acc = _temporal_state.get(f"{key}_acc", None)
if acc is None:
    acc = img.astype(np.float32)
    cnt = 1
else:
    acc += img.astype(np.float32)
    cnt += 1
if cnt >= n_frames:
    result = (acc / cnt).astype(np.uint8)
```
- **alpha**: 사용자 사양 0.3 (EMA 가중치). 실제 구현은 슬라이딩 평균.
- **max_frames**: 기본값 5 (n_frames에서 5로 매핑).

---

### 28. temporal_motion

**원리**
움직임 보상 시간적 잡음 제거(Motion-Compensated Temporal Denoising)는 Farneback 광학 흐름(optical flow)을 사용하여 이전 프레임의 잡음 제거 결과를 현재 프레임에 정렬(warp)한 후 블렌딩합니다. 단순 평균화와 달리 움직임을 추적하므로 고스팅 없이 시간적 잡음 제거가 가능합니다.

**작동 방식**
1. 이전 프레임과 현재 프레임 간 Farneback 광학 흐름 계산.
2. 이전 잡음 제거 결과를 흐름 벡터에 따라 `cv2.remap()`으로 워핑.
3. `strength` 비율로 블렌딩: `result = (1-s) × current + s × warped`.
4. 상태 업데이트.

**구현 상세**
```python
flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
    flow_pyr_scale, flow_levels, 15, 3, 5, 1.2, 0)
warped = cv2.remap(prev_denoised, map_x, map_y, cv2.INTER_LINEAR)
result = cv2.addWeighted(img, 1.0 - strength, warped, strength, 0)
```
- **flow_scale**: 기본값 0.5 (피라미드 스케일).
- **strength**: 기본값 0.5. 블렌드 팩터.

---

### 29. temporal_spatial

**원리**
복합 시공간 잡음 제거(Combined Spatio-Temporal Denoising)는 공간적 양방향 필터와 시간적 움직임 보상 블렌딩을 결합한 하이브리드 방식입니다. 공간 필터는 단일 프레임의 잡음을 처리하고, 시간적 필터는 프레임 간 플리커를 줄입니다. 두 결과를 가중 융합하여 각각의 장점을 취합니다.

**작동 방식**
1. 공간 처리: `spatial = bilateralFilter(img)` (약한 양방향 필터).
2. 시간 처리: `temporal = motion_compensated(img)` (위상 28과 동일).
3. 융합: `result = 0.6 × spatial + 0.4 × temporal`.

**구현 상세**
```python
spatial = cv2.bilateralFilter(img, d, spatial_strength, spatial_strength)
temporal = temporal_motion_compensated(img, strength=temporal_strength)
return cv2.addWeighted(spatial, 0.6, temporal, 0.4, 0)
```
- **spatial_strength**: 기본값 5.0. 양방향 필터 sigma.
- **temporal_strength**: 기본값 0.3. 시간적 블렌드 계수.

---

## Phase 7: 아날로그 비디오 복원 (Analog Video)

### 30. flicker_stabilize

**원리**
플리커 안정화(Flicker Stabilization)는 아날로그 비디오에서 프레임 간 밝기 변동(플리커)을 줄이는 시간적 필터입니다. EMA(Exponential Moving Average)로 추적된 기준 밝기와 현재 프레임의 밝기 차이를 보정하여 점진적인 장면 변화(주야간 전환 등)는 허용하면서 갑작스러운 플리커를 억제합니다.

**작동 방식**
1. 현재 프레임의 채널별 평균 밝기 계산.
2. EMA 업데이트: `ema = α × mean + (1-α) × ema` (α = 2/(window+1)).
3. 보정 계수: `correction = ema / mean` (0.85–1.15 클리핑).
4. 원본과 블렌딩: `result = (1 + strength × (correction - 1)) × img`.

**구현 상세**
```python
correction = np.where(means > 1.0, ema / means, 1.0)
correction = np.clip(correction, 0.85, 1.15)
correction = 1.0 + strength * (correction - 1.0)
```
- **strength**: 기본값 0.7. 안정화 강도 (0=효과 없음, 1=EMA에 완전 고정).
- **window**: 기본값 10. EMA 윈도우 크기 (약 0.33초 @ 30fps).

---

### 31. scanline_remove

**원리**
스캔라인 제거는 아날로그 비디오의 수평 스캔라인 아티팩트(주기적인 밝고 어두운 가로 줄무늬)를 제거합니다. 스캔라인은 일반적으로 2행 간격(NTSC 인터레이스 잔류) 또는 다른 규칙적인 간격으로 나타납니다. 행(row)별 평균 밝기의 1D FFT 분석을 통해 스캔라인의 주기를 자동 탐지합니다.

**작동 방식**
1. 각 행의 평균 밝기로 1D 신호 생성.
2. FFT 분석으로 지배적인 주기 탐지:
   - "detect" 모드: 행 평균 FFT → 피크 검출 → 주기 추정.
   - "fixed" 모드: 사전 지정된 `period_hint` 사용.
3. 로컬 평균과의 편차로 "불량 행(bad rows)" 식별.
4. 불량 행을 주변 정상 행의 선형 보간으로 대체.
5. `blend` 비율로 원본과 혼합.

**구현 상세**
```python
row_means = channel.mean(axis=1)
fft = np.abs(np.fft.fft(row_means))
peak_bin = np.argmax(fft[1:len(fft)//2]) + 1
period = max(2, rows // (peak_bin + 1))
# 불량 행 보간
result[r, :] = channel[r, :] * (1 - blend) + interpolated * blend
```
- **mode**: `"detect"` (자동 탐지) 또는 `"fixed"`.
- **threshold**: 기본값 0.7. 탐지 민감도.
- **blend**: 기본값 0.5. 대체 강도 (0=원본 유지, 1=완전 대체).
- **period_hint**: 기본값 0 (고정 모드에서 2행 간격 사용).

---

## Phase 8–10: v3.3 신규 필터 (New Filters)

### 32. rolling_guidance

**원리**
Rolling Guidance Filter(Zhang et al., ECCV 2014)는 반복적(iterative) 에지 보존 평활화 필터입니다. 작은 구조물/텍스처를 제거하면서 주요 에지를 보존합니다. 초기 가이드 영상을 입력(또는 가우시안 블러)으로 시작하여, 각 반복에서 Joint Bilateral Filter(또는 Guided Filter)로 가이드를 점진적으로 개선합니다. 큰 구조물만 남을 때까지 작은 디테일이 반복적으로 제거됩니다.

**작동 방식**
1. 초기 가이드: 가우시안 블러를 적용한 입력.
2. 각 반복에서 가이드된 Guided Filter 수행:
   - `mean_I`, `mean_P`, `mean_II`, `mean_IP` 계산.
   - `a = cov_IP / (var_I + eps)`, `b = mean_P - a·mean_I`.
   - 새 가이드: `guide = mean_a·guide + mean_b`.
3. `n_iter` 반복 후 최종 가이드 반환.

**구현 상세**
```python
guide = gaussian_filter(ch, sigma=sigma_s)
for _ in range(n_iter):
    mean_I = cv2.boxFilter(guide, -1, (radius, radius))
    mean_P = cv2.boxFilter(ch, -1, (radius, radius))
    # ...
    guide = mean_a * guide + mean_b
```
- **sigma_s**: 기본값 3.0. 공간 σ (픽셀).
- **sigma_r**: 기본값 0.1. 범위 σ (0–1 정규화).
- **n_iter**: 기본값 4 (3–5 전형적).

---

### 33. cross_bilateral

**원리**
교차 양방향 필터(Cross/Joint Bilateral Filter)는 가이드 영상의 에지 정보를 활용하여 입력 영상을 평활화합니다. 가이드 영상이 에지를 명확히 보여주면, 입력 영상에서 동일 위치의 에지도 보존하면서 잡음이 많은 평탄 영역을 효과적으로 평활화할 수 있습니다. 기본적으로 가우시안 블러된 자기 자신을 가이드로 사용(self-guiding).

**작동 방식**
1. `use_self=True`이면 가우시안 블러로 자기 가이드 생성.
2. 각 채널에 `cv2.bilateralFilter()` 적용.
3. 가이드는 공간 및 색상 가중치 계산에 사용됨.

**구현 상세**
```python
guide = cv2.GaussianBlur(img, (0, 0), guide_sigma)
for c in range(3):
    result[:, :, c] = cv2.bilateralFilter(img[:, :, c], d, sigma_color, sigma_space)
```
- **guide_sigma**: 기본값 0.5. 자기 가이드 가우시안 σ.
- **d**: 기본값 7. 양방향 필터 직경.
- **sigma_color**: 기본값 50. 색상 σ.
- **sigma_space**: 기본값 20. 공간 σ.

---

### 34. detail_boost

**원리**
에지 인식 디테일 향상(Edge-Aware Detail Enhancement)은 Rolling Guidance Filter로 영상을 베이스(대규모 구조)와 디테일(소규모 세부) 레이어로 분해한 후, 디테일 레이어만 선택적으로 증폭합니다. 임계값(threshold) 이하의 작은 변화(잡음)는 증폭하지 않아 잡음 증폭을 방지하면서 디테일을 향상시킵니다.

**작동 방식**
"layer" 모드:
1. Rolling Guidance Filter로 베이스 레이어 추출.
2. `detail = original - base`.
3. `|detail| > threshold`인 픽셀만 선택 (마스킹).
4. `detail_boosted = detail × (1 + strength)`.
5. `result = base + detail_boosted × mask`.
"local" 모드: 로컬 평균과의 차분을 증폭.

**구현 상세**
```python
base = _rolling_guidance_channel(ch, sigma_s, sigma_r, n_iter=3)
detail = ch - base
mask = np.abs(detail) > threshold
detail_boosted = detail * (1.0 + strength)
result = base + detail_boosted * mask
```
- **strength**: 기본값 0.3. 디테일 증폭 강도.
- **sigma_s**: 기본값 3.0. rolling guidance 공간 σ.
- **sigma_r**: 기본값 0.15 (사용자 사양에 기반, 코드에서는 0.15).

---

### 35. temporal_nlm_multi

**원리**
다중 프레임 NLM(Multi-Frame Non-Local Means)은 공간적 패치 유사도 검색을 시간 축으로 확장한 것입니다. 연속된 여러 프레임에서 유사한 패치를 검색하여 더 많은 후보를 확보함으로써, 단일 프레임 NLM보다 우수한 잡음 제거 성능을 달성합니다. OpenCV의 `fastNlMeansDenoisingColoredMulti`를 사용합니다.

**작동 방식**
1. 링 버퍼에 프레임 누적 (최대 `max_frames`).
2. `temporal_window × 2 + 1` 프레임 이상 누적 시 다중 프레임 NLM 활성화.
3. 대상 프레임과 주변 프레임을 `frames_for_multi` 리스트로 구성.
4. `cv2.fastNlMeansDenoisingColoredMulti()` 호출.
5. 부족 시 단일 프레임 NLM으로 폴백(fallback).

**구현 상세**
```python
buffer.append(img.copy())
if len(buffer) > max_frames:
    buffer.pop(0)
# ...
denoised = cv2.fastNlMeansDenoisingColoredMulti(
    frames_for_multi, target_idx, temporal_window_size,
    None, h, h_color, 7, 21)
```
- **h**: 기본값 10. 휘도 잡음 제거 강도.
- **h_color**: 기본값 10. 색차 잡음 제거 강도.
- **temporal_window**: 기본값 2 (총 5프레임 사용).
- **max_frames**: 기본값 5. 버퍼 최대 크기.

---

### 36. bm4d_volume

**원리**
BM4D는 BM3D의 시공간 확장판으로, 2D 패치 × 2D 검색(공간 및 시간)으로 구성된 4D 그룹에서 협력 필터링을 수행합니다(Maggioni et al., 2012). 비디오 볼륨을 4D 블록(공간 패치 × 시간 패치)으로 그룹화하여 변환 도메인에서 잡음과 신호를 분리합니다. BM3D 대비 5–10배 빠른 비디오 처리와 우수한 품질을 제공합니다.

**작동 방식**
1. 프레임을 링 버퍼에 누적.
2. 최소 8프레임 이상 누적 시 BM4D 처리.
3. 버퍼의 최근 프레임을 `(T, H, W, C)` → `(C, T, H, W)`로 변환.
4. `bm4d.bm4d_multichannel()`로 4D 볼륨 잡음 제거.
5. 잡음 제거된 볼륨의 마지막 프레임(최신) 반환.
6. 프레임 부족 시 BM3D 또는 양방향 필터로 폴백.

**구현 상세**
```python
volume = np.stack(buffer[-n_frames:], axis=0)  # (T, H, W, C)
volume = volume.transpose(3, 0, 1, 2).astype(np.float64)  # (C, T, H, W)
profile = BM4DProfile2D()
denoised_vol = bm4d.bm4d_multichannel(volume, sigma_psd=sigma_psd, profile=profile)
return denoised_vol[:, -1].transpose(1, 2, 0)  # 최신 프레임
```
- **sigma_psd**: 기본값 15.0. 잡음 표준편차.
- **temporal_window**: 기본값 2 (±2 프레임).
- **max_frames**: 기본값 8. 버퍼 최대 크기.

---

### 37. adaptive_equalize

**원리**
적응형 히스토그램 평활화(Adaptive Equalization)는 CLAHE 결과와 원본을 `brightness_preserve` 비율로 블렌딩하여 과도한 평활화와 잡음 증폭을 방지합니다. CLAHE는 국소 대비를 향상시키지만 어두운 영역에서 잡음을 증폭할 수 있습니다. 원본과의 블렌딩으로 자연스러운 결과를 유지합니다.

**작동 방식**
1. LAB 색공간으로 변환 후 L 채널에 CLAHE 적용.
2. CLAHE 결과와 원본을 `alpha = 1 - brightness_preserve` 비율로 블렌딩.
3. LAB → BGR 역변환.

**구현 상세**
```python
clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
lab_eq[:, :, 0] = clahe.apply(lab[:, :, 0])
img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
result = cv2.addWeighted(img_eq, alpha, img, 1.0 - alpha, 0)
```
- **clip_limit**: 기본값 1.5. CLAHE 대비 제한.
- **tile_size**: 기본값 8. 타일 크기.
- **brightness_preserve**: 기본값 0.4. 0=전체 CLAHE, 1=원본 유지.

---

## Phase 11: Grey-Edge (색상 항등성)

### 38. grey_edge

**원리**
Grey-Edge 색상 항등성(Color Constancy)은 van de Weijer et al.(IEEE TIP 2007)이 제안한 방법으로, 영상의 미분(derivative) 정보를 기반으로 조명색을 추정합니다. 단순한 Grey-World 가정(평균이 회색)보다 강건하며, 균일한 영역에서의 과보정을 방지합니다. 영상 미분의 Minkowski 노름(norm)을 사용하여 에지 기반 조명 추정을 수행합니다.

**작동 방식**
1. 가우시안 평활화 (sigma_smooth).
2. 각 채널의 미분 계산 (Sobel 1차 미분 또는 Laplacian 2차 미분).
3. 각 채널의 미분 magnitude에 대한 Minkowski p-norm 계산:
   - p=1: `energy = mean(|derivative|)` (Grey-Edge).
   - p=2: `energy = sqrt(mean(derivative²))` (Shades-of-Grey 변형).
4. Grey-World 정규화: `gain_c = mean(energy) / energy_c`.
5. `[0.5, 2.0]` 클리핑 후 `strength`로 블렌딩.

**구현 상세**
```python
grad_x = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
mag = np.sqrt(grad_x**2 + grad_y**2)
energy = np.mean(mag)  # p=1
gains = np.array([mean_energy / e for e in channel_energy])
gains = np.clip(gains, 0.5, 2.0)
```
- **minkowski_norm**: 기본값 1 (p=1). 2도 지원.
- **sigma_smooth**: 기본값 2.0. 사전 평활화 σ.
- **diff_order**: 기본값 1 (Sobel). 2(Laplacian)도 지원.
- **strength**: 기본값 0.22. 보정 강도.

---

## Phase 12: Domain Transform (에지 보존 필터)

### 39. domain_transform

**원리**
도메인 변환 필터(Domain Transform Filter)는 Gastal & Oliveira(ACM TOG 2011, SIGGRAPH)가 제안한 O(N) 선형 시간 에지 보존 필터입니다. 영상을 1D 도메인으로 변환하여 유클리드 거리가 측지 거리(geodesic distance)를 근사하도록 만든 후, 재귀적(recursive) 평균화를 수행합니다. 매우 빠르며 헤일로(halo) 아티팩트가 없습니다.

**작동 방식**
각 행/열에 대해:
1. 도메인 변환: `ct[i] = Σ_{j=1}^{i} (1 + σ_s/σ_r · 3 · |I'_avg|)`.
   - 픽셀 값 변화가 큰 곳에서 변환이 더 늘어남 → 에지에서 필터링 중단.
2. 순방향 재귀 필터: `v'[i] = (1-a)·v[i] + a·v'[i-1]`
   - `a = exp(-√2 / σ_s · |ct[i] - ct[i-1]|)`.
3. 역방향 재귀 필터: 동일한 방식으로 역방향 처리.
4. 수평 패스 후 수직 패스 (분리 가능).

**구현 상세**
```python
for x in range(1, row.shape[0]):
    diff = np.sum(np.abs(row[x] - row[x-1])) / 3.0
    ct[x] = ct[x-1] + 1.0 + (sigma_s / (sigma_r * 3.0 + 1e-10)) * diff
    a = np.exp(-np.sqrt(2) / (sigma_s + 1e-10) * abs(ct[x] - ct[x-1]))
    result[y, x] = (1 - a) * row[x] + a * result[y, x-1]
```
- **sigma_s**: 기본값 10.0. 공간 표준편차 (픽셀).
- **sigma_r**: 기본값 0.15 (사용자 사양: 30, 정규화 시 0.15). 범위 표준편차.
- **n_iter**: 기본값 3. 반복 횟수 (1–4 권장).

---

## Phase 13: 디인터레이스 (Deinterlace)

### 40. deinterlace

**원리**
디인터레이스(Deinterlacing)는 인터레이스(interlaced) 비디오를 프로그레시브(progressive)로 변환하는 과정입니다. NTSC/PAL 아날로그 비디오는 각 프레임이 서로 다른 시간에 캡처된 두 필드(홀수 행 = 필드 1, 짝수 행 = 필드 2)로 구성됩니다. 디인터레이싱은 이 두 필드를 결합하여 하나의 완전한 프로그레시브 프레임을 재구성합니다.

**작동 방식**
세 가지 모드:

1. **Bob (선형 보간)**: 각 필드를 독립적으로 업샘플링. 누락된 행은 이웃 동일 필드 행의 선형 보간으로 채움. 빠르지만 약간의 블러 발생.

2. **Weave (위브)**: 두 필드를 단순 결합. 정적 장면에 최적이지만 움직임이 있는 영역에서 빗살(combing) 아티팩트 발생.

3. **Motion-Adaptive (움직임 적응)**: 필드 간 픽셀 차이로 움직임 맵 생성. 정적 영역은 weave(높은 해상도), 움직임 영역은 bob(빗살 방지). 움직임 맵 기반 블렌딩.

**구현 상세**
Bob:
```python
result[y, :] = (img[y_above, :] + img[y_below, :]) / 2.0
```

Weave (단일 프레임 폴백):
```python
return _deinterlace_bob(img, is_top_first)  # 빗살 방지를 위해 보간 사용
```

Motion-Adaptive:
```python
motion_map = np.abs(field0 - field1).mean(axis=2)  # 움직임 맵
motion_weight = np.clip(motion_map / 64.0, 0.0, 1.0)
# result = weave × (1 - motion_weight) + bob × motion_weight
```
- **field_order**: 기본값 `"auto"` (필드 순서 자동 탐지).
  - 자동 탐지: 동일 필드 행 간의 상관관계 vs 교차 필드 행 간의 상관관계 비교.
- **method**: `"bob"`, `"weave"`, `"motion-adaptive"` 중 선택.

---

> **참고**: 위 설명에 사용된 파라미터 기본값은 각 필터의 `FILTER_REGISTRY` 등록 시점의 설정을 기준으로 합니다. 실제 파이프라인 실행 시 `pipeline_config.json` 또는 개별 필터 호출 시 전달되는 인자에 의해 재정의될 수 있습니다.
