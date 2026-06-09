"""
Fast noise estimation utilities — Laplacian-based noise std, classification,
preset suggestion, and parameter tuning.

Designed as a lightweight complement to smu_sig_prossessing.noise_estimator
(which does full wavelet/MAD noise profiling).  This module provides a
quick per-frame estimate suitable for real-time adaptive pipelines.

Typical usage::

    from smu_sig_prossessing.noise_estimation import (
        estimate_noise_level,
        classify_noise,
        suggest_preset,
        suggest_params,
    )

    noise_std = estimate_noise_level(frame)
    label = classify_noise(noise_std)          # "low", "medium", "high", "extreme"
    preset = suggest_preset(noise_std)         # preset name string
    params = suggest_params(noise_std)         # dict of parameter adjustments
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

# ─── Constants ───────────────────────────────────────────────────────

_PI_OVER_2 = math.sqrt(math.pi / 2.0)

# Thresholds for Laplacian-based noise std (not Laplacian variance)
_NOISE_LOW: float = 10.0       # <10  → low
_NOISE_MEDIUM: float = 25.0    # 10-25 → medium
_NOISE_HIGH: float = 40.0      # 25-40 → high
                                # >40  → extreme

# Preset name mapping based on noise level
_PRESET_MAP: dict[str, str] = {
    "low": "optimal-ultrafast",
    "medium": "grey-guided-chroma",
    "high": "grey-premium",
    "extreme": "video-enhanced",
}


# ─── Core Estimation ─────────────────────────────────────────────────


def estimate_noise_level(img: np.ndarray) -> float:
    """
    Estimate the noise standard deviation from a single image frame.

    Uses the Laplacian-based method:

        1. Convert to grayscale (if colour)
        2. Apply Laplacian operator (CV_64F)
        3. noise_std = std(Laplacian) / sqrt(pi/2)

    The division by sqrt(pi/2) corrects for the fact that the Laplacian
    of Gaussian noise follows a Gaussian distribution whose std relates
    to the underlying image noise std by a factor of sqrt(pi/2) under
    the assumption of a zero-mean Laplacian distribution of the filtered
    output for pure noise regions.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale, uint8).

    Returns
    -------
    float
        Estimated noise standard deviation.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.std(lap) / _PI_OVER_2)


# ─── Classification ──────────────────────────────────────────────────


def classify_noise(noise_std: float) -> str:
    """
    Classify noise level into a human-readable category.

    Parameters
    ----------
    noise_std : float
        Estimated noise standard deviation (from ``estimate_noise_level``).

    Returns
    -------
    str
        One of ``'low'``, ``'medium'``, ``'high'``, ``'extreme'``.
    """
    if noise_std < _NOISE_LOW:
        return "low"
    elif noise_std < _NOISE_MEDIUM:
        return "medium"
    elif noise_std < _NOISE_HIGH:
        return "high"
    else:
        return "extreme"


# ─── Preset Suggestion ───────────────────────────────────────────────


def suggest_preset(noise_std: float, has_motion: bool = False) -> str:
    """
    Return a preset *name* string appropriate for the given noise level
    and optional motion flag.

    The returned string can be mapped to a ``PipelineConfig`` preset
    (e.g. via a lookup dict) or passed directly to a pipeline runner
    that accepts preset names.

    Mapping
    -------
    ``'low'``     → ``'optimal-ultrafast'``
    ``'medium'``  → ``'grey-guided-chroma'``
    ``'high'``    → ``'grey-premium'``
    ``'extreme'`` → ``'video-enhanced'``

    Parameters
    ----------
    noise_std : float
        Estimated noise standard deviation.
    has_motion : bool
        Whether significant motion is detected.  (Currently reserved for
        future use — all presets are motion-aware upstream.)

    Returns
    -------
    str
        Preset name string.
    """
    label = classify_noise(noise_std)

    # Motion override: for high/extreme noise + motion, prefer faster
    # presets that are less prone to ghosting.
    if has_motion:
        if label == "extreme":
            return "optimal-ultrafast"
        elif label == "high":
            return "fast-guided-chroma"
        # medium/low with motion stay at standard mapping

    return _PRESET_MAP.get(label, "optimal-ultrafast")


# ─── Parameter Suggestion ────────────────────────────────────────────


def suggest_params(noise_std: float) -> dict[str, Any]:
    """
    Return suggested filter parameter adjustments based on noise level.

    These are scalar adjustments intended to be merged into a base
    ``PipelineConfig`` (or used directly by a pipeline runner) to adapt
    filter strength to the measured noise.

    Parameters
    ----------
    noise_std : float
        Estimated noise standard deviation.

    Returns
    -------
    dict
        Keys include: ``median_ksize``, ``wavelet_threshold``,
        ``bilateral_sigma``, ``bilateral_d``, ``nlm_h``,
        ``unsharp_strength``, ``guided_eps``, ``chroma_strength``,
        ``detail_boost``, ``flicker_strength``.
    """
    label = classify_noise(noise_std)

    params: dict[str, Any] = {}

    # ── Median kernel size ───────────────────────────────────────
    # Stronger median for higher noise (impulse protection)
    if label == "low":
        params["median_ksize"] = 0  # disabled
    elif label == "medium":
        params["median_ksize"] = 3
    elif label == "high":
        params["median_ksize"] = 5
    else:  # extreme
        params["median_ksize"] = 7

    # ── Wavelet threshold mode ────────────────────────────────────
    # Higher noise → harder thresholding
    if label == "low":
        params["wavelet_threshold_mode"] = "soft"
        params["wavelet_level"] = 1
    elif label == "medium":
        params["wavelet_threshold_mode"] = "soft"
        params["wavelet_level"] = 2
    elif label == "high":
        params["wavelet_threshold_mode"] = "soft"
        params["wavelet_level"] = 3
    else:  # extreme
        params["wavelet_threshold_mode"] = "hard"
        params["wavelet_level"] = 4

    # ── Bilateral filter ─────────────────────────────────────────
    # Scale sigma with noise; increase diameter for heavy noise
    base_sigma = float(np.clip(noise_std * 2.0, 15.0, 150.0))
    params["bilateral_sigma"] = base_sigma

    if label == "low":
        params["bilateral_d"] = 5
    elif label == "medium":
        params["bilateral_d"] = 7
    elif label == "high":
        params["bilateral_d"] = 9
    else:  # extreme
        params["bilateral_d"] = 13

    # ── NLM h-parameter ──────────────────────────────────────────
    # Stronger h for higher noise
    params["nlm_h"] = float(np.clip(noise_std * 0.5, 3.0, 20.0))

    # ── Unsharp mask ─────────────────────────────────────────────
    # More recovery for heavily denoised frames
    if label == "low":
        params["unsharp_strength"] = 0.10
    elif label == "medium":
        params["unsharp_strength"] = 0.20
    elif label == "high":
        params["unsharp_strength"] = 0.30
    else:  # extreme
        params["unsharp_strength"] = 0.40

    # ── Guided filter ────────────────────────────────────────────
    # Eps controls edge-preservation; higher eps = more smoothing
    params["guided_eps"] = float(np.clip(noise_std * 8.0, 50.0, 600.0))

    # ── Chroma denoise ───────────────────────────────────────────
    # Stronger chroma denoise for noisier frames
    params["chroma_strength"] = float(np.clip(noise_std * 0.012, 0.05, 0.8))

    # ── Detail boost ─────────────────────────────────────────────
    # Less detail boost for clean frames (already sharp enough)
    # More for medium frames (needs recovery), less for extreme (risk of noise amp)
    if label == "low":
        params["detail_boost"] = 0.10
    elif label == "medium":
        params["detail_boost"] = 0.30
    elif label == "high":
        params["detail_boost"] = 0.25
    else:  # extreme
        params["detail_boost"] = 0.15

    # ── Flicker stabilization ────────────────────────────────────
    params["flicker_strength"] = float(np.clip(noise_std * 0.015, 0.1, 0.9))

    # ── Temporal filter strength ─────────────────────────────────
    params["temporal_strength"] = float(np.clip(0.5 - noise_std * 0.005, 0.1, 0.5))

    return params
