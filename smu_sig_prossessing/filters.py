"""
Modular filter implementations — each filter is a standalone function
that takes (image: np.ndarray, **params) -> np.ndarray.

Filters are registered in FILTER_REGISTRY so the pipeline runner can
look them up by name.
"""
from __future__ import annotations

import numpy as np
import cv2


# ─── Filter Registry ─────────────────────────────────────────────────

FILTER_REGISTRY: dict[str, callable] = {}


def register(name: str):
    """Decorator to register a filter function."""
    def wrapper(fn):
        FILTER_REGISTRY[name] = fn
        return fn
    return wrapper


# ─── Phase 1: Noise Removal Filters ─────────────────────────────────

@register("median")
def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median filter — best for impulse (salt & pepper) noise."""
    return cv2.medianBlur(img, ksize)


@register("gaussian_lowpass")  # kept for reference, not recommended
def gaussian_lowpass(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian low-pass filter. Use wiener instead for better results."""
    return cv2.GaussianBlur(img, (0, 0), sigma)


@register("wiener")
def wiener_filter(img: np.ndarray, noise_var: float = 400) -> np.ndarray:
    """
    Wiener filter — frequency-domain adaptive denoising.
    H(w) = max(|F|² - N, 0) / max(|F|², N)

    Preserves edges while smoothing noise. Use instead of Gaussian LP.
    Lower noise_var = less aggressive denoising (more detail preserved).
    Default 400 corresponds to σ≈20 noise estimate.
    """
    if len(img.shape) == 3:
        result = np.zeros_like(img)
        for c in range(3):
            result[:, :, c] = _wiener_channel(img[:, :, c], noise_var)
        return result
    return _wiener_channel(img, noise_var)


def _wiener_channel(channel: np.ndarray, noise_var: float) -> np.ndarray:
    f = np.fft.fft2(channel.astype(np.float64))
    f_shift = np.fft.fftshift(f)
    power = np.abs(f_shift) ** 2

    # Correct Wiener: H = max(S, 0) / max(S + N, N)
    # where S = |F|² - N is the estimated signal power
    signal_est = np.maximum(power - noise_var, 0)
    h = signal_est / np.maximum(signal_est + noise_var, 1e-10)

    result = f_shift * h
    result = np.fft.ifftshift(result)
    result = np.fft.ifft2(result)
    return np.clip(np.abs(result), 0, 255).astype(np.uint8)


@register("nlm")
def nlm_filter(img: np.ndarray, h: float = 10,
               template_window: int = 7, search_window: int = 21) -> np.ndarray:
    """
    Non-Local Means Denoising — edge-preserving, excellent PSNR.
    Slower but significantly better than Gaussian LP.

    h : filter strength (higher = more noise removal, more blur)
    template_window : patch size for comparison (odd)
    search_window : window to search for similar patches (odd)
    """
    return cv2.fastNlMeansDenoisingColored(
        img, None, h, h, template_window, search_window
    )


@register("nlm_gray")
def nlm_filter_gray(img: np.ndarray, h: float = 10,
                     template_window: int = 7, search_window: int = 21) -> np.ndarray:
    """Non-Local Means on luminance only (Y channel)."""
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.fastNlMeansDenoising(yuv[:, :, 0], None, h,
                                              template_window, search_window)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


@register("bilateral")
def bilateral_filter(img: np.ndarray, d: int = 9,
                     sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Bilateral filter — edge-preserving smoothing.
    Replaces pixels with weighted average of neighbors, where weights
    depend on both spatial distance AND color/intensity difference.

    d : diameter of pixel neighborhood
    sigma_color : filter sigma in color space
    sigma_space : filter sigma in coordinate space
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


@register("fft_notch")
def fft_notch_filter(img: np.ndarray, threshold_percentile: float = 99.5) -> np.ndarray:
    """
    2D FFT → detect noise peaks → notch removal → IFFT.
    Best for periodic noise (scanlines, dot crawl, motor interference).
    """
    if len(img.shape) == 3:
        result = np.zeros_like(img)
        for c in range(3):
            result[:, :, c] = _notch_channel(img[:, :, c], threshold_percentile)
        return result
    return _notch_channel(img, threshold_percentile)


def _notch_channel(channel: np.ndarray, threshold_percentile: float) -> np.ndarray:
    f = np.fft.fft2(channel.astype(np.float64))
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift)

    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2

    # Detect noise peaks
    threshold = np.percentile(magnitude, threshold_percentile)
    mask = magnitude > threshold
    # Preserve DC and low frequencies
    mask[max(0, crow - 3):crow + 4, max(0, ccol - 3):ccol + 4] = False

    # Remove peaks
    f_shift[mask] = 0

    result = np.fft.ifftshift(f_shift)
    result = np.fft.ifft2(result)
    return np.clip(np.abs(result), 0, 255).astype(np.uint8)


# ─── Phase 2: Wavelet Denoising (Research) ────────────────────────

@register("wavelet")
def wavelet_denoise(img: np.ndarray, wavelet: str = "db4",
                    level: int = 3, threshold_mode: str = "soft") -> np.ndarray:
    """
    Wavelet-based denoising — excellent edge preservation.
    Uses Bayesian thresholding (VisuShrink style).

    wavelet : wavelet family ('db4', 'sym4', 'coif1', 'haar')
    level : decomposition level (2-4)
    threshold_mode : 'soft' (smoother) or 'hard' (sharper)
    """
    import pywt
    import warnings

    if len(img.shape) == 3:
        result = np.zeros_like(img)
        for c in range(3):
            result[:, :, c] = _wavelet_channel(img[:, :, c], wavelet, level, threshold_mode)
        return result
    return _wavelet_channel(img, wavelet, level, threshold_mode)


def _wavelet_channel(channel: np.ndarray, wavelet: str, level: int,
                     threshold_mode: str) -> np.ndarray:
    import pywt
    import numpy as np

    coeffs = pywt.wavedec2(channel.astype(np.float64), wavelet, level=level)
    # Estimate noise sigma from finest-scale diagonal details (HH = index 2 in tuple)
    sigma = 10.0
    if len(coeffs) > 1 and isinstance(coeffs[-1], tuple) and len(coeffs[-1]) > 2:
        hh = coeffs[-1][2]
        if hh.size > 0:
            sigma = np.median(np.abs(hh)) / 0.6745
    # VisuShrink universal threshold
    threshold = sigma * np.sqrt(2 * np.log(channel.size))
    # Apply thresholding to detail coefficients (skip approximation)
    coeffs = list(coeffs)
    for j in range(1, len(coeffs)):
        coeffs[j] = tuple(
            pywt.threshold(d, threshold, mode=threshold_mode)
            for d in coeffs[j]
        )
    # Reconstruct
    reconstructed = pywt.waverec2(coeffs, wavelet)
    # Handle size mismatch (wavelet may round dimensions)
    h, w = channel.shape
    return np.clip(np.abs(reconstructed[:h, :w]), 0, 255).astype(np.uint8)


# ─── Phase 3: Color / Contrast Enhancement ──────────────────────────

@register("gamma")
def gamma_correction(img: np.ndarray, gamma: float = 1.8) -> np.ndarray:
    """
    Gamma correction — O = I^(1/γ).
    γ > 1 brightens dark areas, γ < 1 darkens.
    """
    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


@register("log_transform")
def log_transform(img: np.ndarray, c: float = 40) -> np.ndarray:
    """
    Log transform — O = c * log(1 + I).
    Compresses bright range, expands dark range.
    """
    img_f = img.astype(np.float64)
    result = c * np.log1p(img_f)
    return np.clip(result, 0, 255).astype(np.uint8)


@register("histogram_eq_gray")
def histogram_equalization_gray(img: np.ndarray) -> np.ndarray:
    """Global histogram equalization on grayscale."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


@register("histogram_eq")
def histogram_equalization_yuv(img: np.ndarray) -> np.ndarray:
    """
    Histogram equalization in YUV space — Y channel only.
    Preserves color while improving contrast.
    """
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


@register("histogram_eq_clahe")
def clahe_equalization(img: np.ndarray, clip_limit: float = 2.0,
                        tile_size: int = 8) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Prevents noise amplification better than global equalization.

    clip_limit : threshold for contrast limiting
    tile_size : grid size for local equalization
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@register("channel_correction")
def channel_correction(img: np.ndarray, clamp_min: float = 0.7,
                       clamp_max: float = 1.3) -> np.ndarray:
    """
    RGB channel mean correction — equalizes mean brightness across channels.
    Prevents over-correction with clamping.
    """
    result = img.astype(np.float32)
    means = [np.mean(result[:, :, c]) for c in range(3)]
    target = np.mean(means)
    for c in range(3):
        if means[c] > 0:
            scale = target / means[c]
            scale = np.clip(scale, clamp_min, clamp_max)
            result[:, :, c] *= scale
    return np.clip(result, 0, 255).astype(np.uint8)


@register("unsharp_mask")
def unsharp_mask(img: np.ndarray, strength: float = 1.0,
                 radius: float = 1.0, threshold: int = 0) -> np.ndarray:
    """
    Unsharp masking — edge enhancement to recover detail lost during denoising.
    Helps mitigate denoising-induced blur.

    strength : amount of sharpening
    radius : Gaussian blur radius for the mask
    threshold : minimum gradient to sharpen (higher = fewer artifacts)
    """
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    if threshold > 0:
        # Only sharpen pixels where the difference exceeds threshold
        diff = cv2.subtract(img, blurred)
        mask = cv2.threshold(np.abs(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)),
                             threshold, 1, cv2.THRESH_BINARY)[1][:, :, np.newaxis]
        sharpened = (sharpened * mask + img * (1 - mask)).astype(np.uint8)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


@register("deblur_wiener")
def deblur_wiener(img: np.ndarray, kernel_size: int = 5,
                   noise_var: float = 0.01) -> np.ndarray:
    """
    Frequency-domain deblurring using Wiener deconvolution.
    Estimates and reverses a mild blur kernel.

    kernel_size : assumed blur kernel size
    noise_var : noise power estimate (higher = less aggressive)
    """
    if len(img.shape) == 3:
        result = np.zeros_like(img)
        for c in range(3):
            result[:, :, c] = _deblur_channel(img[:, :, c], kernel_size, noise_var)
        return result
    return _deblur_channel(img, kernel_size, noise_var)


def _deblur_channel(channel: np.ndarray, ksize: int, noise_var: float) -> np.ndarray:
    # Create a mild Gaussian blur kernel as the PSF
    kernel = cv2.getGaussianKernel(ksize, -1)
    psf = kernel @ kernel.T
    psf /= psf.sum()

    # Pad to image size
    rows, cols = channel.shape
    psf_pad = np.zeros((rows, cols))
    psf_pad[:ksize, :ksize] = psf
    # Center the PSF
    psf_pad = np.fft.ifftshift(psf_pad)

    img_f = np.fft.fft2(channel.astype(np.float64))
    psf_f = np.fft.fft2(psf_pad)

    # Wiener deconvolution: F = conj(H) / (|H|^2 + K) * G
    psf_conj = np.conj(psf_f)
    psf_mag = np.abs(psf_f) ** 2
    wiener = psf_conj / (psf_mag + noise_var)

    result = img_f * wiener
    result = np.fft.ifft2(result)
    return np.clip(np.abs(result), 0, 255).astype(np.uint8)


# ─── Phase 4: Advanced Spatial Denoising ──────────────────────────

@register("guided_filter")
def guided_filter(img: np.ndarray, radius: int = 3, eps: float = 100.0) -> np.ndarray:
    """
    Guided filter — edge-preserving smoothing via local linear model.
    Assumes output = a*I + b in each local window.
    Acts as an edge-aware blur — smooths flat regions, preserves edges.
    """
    guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    if len(img.shape) == 3:
        result = np.zeros_like(img, dtype=np.float32)
        for c in range(3):
            result[:, :, c] = _guided_channel(
                img[:, :, c].astype(np.float32) / 255.0, guide, radius, eps
            )
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    gray = img.astype(np.float32) / 255.0
    out = _guided_channel(gray, guide, radius, eps)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def _guided_channel(p: np.ndarray, guide: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Single-channel guided filter (floating-point, 0–1 range)."""
    mean_I = cv2.boxFilter(guide, -1, (r, r))
    mean_P = cv2.boxFilter(p, -1, (r, r))
    mean_II = cv2.boxFilter(guide * guide, -1, (r, r))
    mean_IP = cv2.boxFilter(guide * p, -1, (r, r))
    var_I = mean_II - mean_I * mean_I
    cov_IP = mean_IP - mean_I * mean_P
    a = cov_IP / (var_I + eps)
    b = mean_P - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))
    return mean_a * guide + mean_b


@register("anisotropic_diffusion")
def anisotropic_diffusion(img: np.ndarray, n_iter: int = 10,
                          kappa: float = 50.0, gamma: float = 0.25) -> np.ndarray:
    """
    Perona-Malik anisotropic diffusion.
    Edge-stopping function: c = exp(-(∇I/κ)²).
    Smooths intra-region while preserving edges.
    """
    img_f = img.astype(np.float32)
    if len(img_f.shape) == 3:
        result = np.zeros_like(img_f)
        for c in range(3):
            result[:, :, c] = _aniso_channel(img_f[:, :, c], n_iter, kappa, gamma)
        return np.clip(result, 0, 255).astype(np.uint8)
    return np.clip(_aniso_channel(img_f, n_iter, kappa, gamma), 0, 255).astype(np.uint8)


def _aniso_channel(ch: np.ndarray, n_iter: int, kappa: float, gamma: float) -> np.ndarray:
    """Perona-Malik diffusion on a single channel."""
    for _ in range(n_iter):
        # Four directional gradients
        n = np.roll(ch, -1, axis=0) - ch
        s = np.roll(ch, 1, axis=0) - ch
        e = np.roll(ch, -1, axis=1) - ch
        w = np.roll(ch, 1, axis=1) - ch
        # Edge-stopping (exponential)
        cN = np.exp(-(n / kappa) ** 2)
        cS = np.exp(-(s / kappa) ** 2)
        cE = np.exp(-(e / kappa) ** 2)
        cW = np.exp(-(w / kappa) ** 2)
        ch += gamma * (cN * n + cS * s + cE * e + cW * w)
    return ch


@register("tv_denoise")
def tv_denoise(img: np.ndarray, weight: float = 0.1,
               max_num_iter: int = 100, eps: float = 2e-4) -> np.ndarray:
    """
    Total Variation denoising (ROF model, Chambolle's algorithm).
    Minimises ∫|∇u| + λ/2·∫(u-f)².
    Excellent for removing small noise while preserving sharp edges.
    """
    from skimage.restoration import denoise_tv_chambolle
    img_f = img.astype(np.float32) / 255.0
    result = denoise_tv_chambolle(
        img_f, weight=weight, max_num_iter=max_num_iter, eps=eps,
        channel_axis=-1 if len(img.shape) == 3 else None
    )
    return np.clip(result * 255, 0, 255).astype(np.uint8)


# ─── Phase 5: Patch-based Denoising ───────────────────────────────

@register("patch_collaborative")
def patch_collaborative(img: np.ndarray, patch_size: int = 8,
                         h_dct: float = 30.0) -> np.ndarray:
    """
    Block DCT denoising — divide image into blocks, DCT, hard-threshold, IDCT.
    A simplified (and much faster) BM3D-inspired approach.

    patch_size : block size for DCT decomposition (must divide image evenly-ish)
    h_dct      : DCT domain hard-threshold (higher = stronger denoising)
    """
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        result = np.zeros_like(img, dtype=np.float32)
        for c in range(3):
            result[:, :, c] = _dct_block_denoise(
                img[:, :, c].astype(np.float32), patch_size, h_dct
            )
        return np.clip(result, 0, 255).astype(np.uint8)
    return np.clip(
        _dct_block_denoise(img.astype(np.float32), patch_size, h_dct),
        0, 255
    ).astype(np.uint8)


def _dct_block_denoise(ch: np.ndarray, bs: int, h_dct: float) -> np.ndarray:
    """Block-wise DCT hard-threshold denoising for a single channel."""
    h, w = ch.shape
    # Pad to multiples of bs
    ph = (bs - h % bs) % bs
    pw = (bs - w % bs) % bs
    padded = np.pad(ch, ((0, ph), (0, pw)), mode='reflect')

    out = np.zeros_like(padded)
    weight = np.zeros_like(padded)
    for i in range(0, padded.shape[0], bs):
        for j in range(0, padded.shape[1], bs):
            block = padded[i:i + bs, j:j + bs].copy()
            dct_block = cv2.dct(block)
            # Hard threshold
            dct_block[np.abs(dct_block) < h_dct] = 0
            # Count non-zero coefficients for weighting
            nz = np.count_nonzero(dct_block)
            recon = cv2.idct(dct_block)
            out[i:i + bs, j:j + bs] += recon
            weight[i:i + bs, j:j + bs] += 1.0

    out /= np.maximum(weight, 1e-10)
    return out[:h, :w]


# ─── Phase 6: Temporal (Video) Denoising ─────────────────────────

# Module-level state for temporal filters (reset between videos)
_temporal_state: dict = {}


def reset_temporal_state() -> None:
    """Reset all temporal filter state (call before processing a new video)."""
    global _temporal_state
    _temporal_state = {}


@register("temporal_average")
def temporal_frame_average(img: np.ndarray, n_frames: int = 5,
                            reset: bool = False) -> np.ndarray:
    """
    Sliding-window frame averaging for temporal denoising.
    Accumulates n_frames then outputs the mean.

    n_frames : number of frames to average (higher = smoother, more ghosting)
    reset    : set True to reset accumulator mid-stream
    """
    if reset:
        _temporal_state.pop("tavg_acc", None)
        _temporal_state.pop("tavg_cnt", None)

    key = "tavg"
    acc = _temporal_state.get(f"{key}_acc", None)
    cnt = _temporal_state.get(f"{key}_cnt", 0)

    if acc is None:
        acc = img.astype(np.float32)
        cnt = 1
    else:
        acc += img.astype(np.float32)
        cnt += 1

    if cnt >= n_frames:
        result = (acc / cnt).astype(np.uint8)
        _temporal_state[f"{key}_acc"] = None
        _temporal_state[f"{key}_cnt"] = 0
        return result
    else:
        _temporal_state[f"{key}_acc"] = acc
        _temporal_state[f"{key}_cnt"] = cnt
        return (acc / cnt).astype(np.uint8)


@register("temporal_motion")
def temporal_motion_compensated(img: np.ndarray, strength: float = 0.5,
                                 flow_pyr_scale: float = 0.5,
                                 flow_levels: int = 3,
                                 reset: bool = False) -> np.ndarray:
    """
    Motion-compensated temporal denoising using Farneback optical flow.
    Warps previous denoised frame and blends with current.

    strength          : blend factor (0=current only, 1=warped prev only)
    flow_pyr_scale    : optical flow pyramid scale
    flow_levels       : number of pyramid levels
    reset             : reset motion state mid-stream
    """
    if reset:
        _temporal_state.pop("tmc_prev", None)
        _temporal_state.pop("tmc_denoised", None)

    prev_gray = _temporal_state.get("tmc_prev")
    prev_denoised = _temporal_state.get("tmc_denoised")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        _temporal_state["tmc_prev"] = gray
        _temporal_state["tmc_denoised"] = img.copy()
        return img

    # Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        flow_pyr_scale, flow_levels, 15, 3, 5, 1.2, 0
    )

    # Warp previous denoised
    h, w = flow.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (map_x + flow[:, :, 0]).astype(np.float32)
    map_y = (map_y + flow[:, :, 1]).astype(np.float32)
    warped = cv2.remap(prev_denoised, map_x, map_y, cv2.INTER_LINEAR)

    # Blend
    result = cv2.addWeighted(img, 1.0 - strength, warped, strength, 0)

    _temporal_state["tmc_prev"] = gray
    _temporal_state["tmc_denoised"] = result.copy()
    return result


@register("temporal_spatial")
def temporal_spatial_denoise(img: np.ndarray, spatial_strength: float = 5.0,
                              temporal_strength: float = 0.3,
                              reset: bool = False) -> np.ndarray:
    """
    Combined spatio-temporal denoising.
    Applies mild bilateral spatial denoising + motion-compensated temporal fusion.

    spatial_strength   : sigma for bilateral spatial filter
    temporal_strength  : blend factor for temporal component
    reset              : reset temporal state
    """
    # Spatial: mild bilateral
    d = max(3, int(spatial_strength) | 1)  # ensure odd
    spatial = cv2.bilateralFilter(img, d, spatial_strength, spatial_strength)

    # Temporal: motion-compensated blend
    temporal = temporal_motion_compensated(img, strength=temporal_strength, reset=reset)

    # Fuse: spatial handles noise, temporal handles flicker
    return cv2.addWeighted(spatial, 0.6, temporal, 0.4, 0)


# ─── List registered filters ─────────────────────────────────────────

def list_filters() -> list[str]:
    """Return names of all registered filters."""
    return list(FILTER_REGISTRY.keys())
