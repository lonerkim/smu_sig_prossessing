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
def wiener_filter(img: np.ndarray, noise_var: float = 625) -> np.ndarray:
    """
    Wiener filter — frequency-domain adaptive denoising.
    Only noise removal filter used in the recommended pipeline (no Gaussian LP).

    The Wiener filter estimates the original signal by:
      H(w) = S(w) / (S(w) + N(w))
    where S(w) is the signal power spectrum and N(w) is the noise power.
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
    signal_power = np.mean(power)

    # Wiener: H = S / (S + N)
    h = signal_power / (power + noise_var)
    h = np.clip(h, 0, 1)

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


# ─── Phase 2: Color / Contrast Enhancement ──────────────────────────

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


# ─── List registered filters ─────────────────────────────────────────

def list_filters() -> list[str]:
    """Return names of all registered filters."""
    return list(FILTER_REGISTRY.keys())
