"""
Degradation module — artificial noise simulation for evaluation.
Includes basic noise types + NTSC analog video emulation.
"""
from __future__ import annotations

import numpy as np
import cv2
import os

# Import NTSC plugin (copy from zhuker/ntsc)
from .ntsc_plugin import Ntsc, random_ntsc, bgr2yiq

_NTSC_RINGPATTERN = os.path.join(os.path.dirname(__file__), "ringPattern.npy")


# ─── Basic Synthetic Degradation ────────────────────────────────────

def add_gaussian_noise(img: np.ndarray, sigma: float = 25) -> np.ndarray:
    """Add Gaussian (AWGN) noise."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_impulse_noise(img: np.ndarray, prob: float = 0.03) -> np.ndarray:
    """Add salt & pepper (impulse) noise."""
    noisy = img.copy()
    mask = np.random.random(img.shape[:2])
    noisy[mask < prob / 2] = 0
    noisy[mask > 1 - prob / 2] = 255
    return noisy


def reduce_brightness(img: np.ndarray, gamma_val: float = 0.4) -> np.ndarray:
    """Simulate low brightness / low contrast via gamma."""
    table = np.array([(i / 255.0) ** (1.0 / gamma_val) * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def add_color_bias(img: np.ndarray,
                   r_gain: float = 1.3, g_gain: float = 0.8,
                   b_gain: float = 1.1) -> np.ndarray:
    """Per-channel color bias."""
    biased = img.astype(np.float32)
    biased[:, :, 2] *= r_gain
    biased[:, :, 1] *= g_gain
    biased[:, :, 0] *= b_gain
    return np.clip(biased, 0, 255).astype(np.uint8)


def add_periodic_noise(img: np.ndarray, freq: float = 30,
                       amplitude: float = 50) -> np.ndarray:
    """Add periodic sine wave noise (diagonal banding).
    NOTE: This is NOT a typical analog artifact. Use add_horizontal_line_noise
    for more realistic analog scan-line interference.
    """
    noisy = img.astype(np.float32)
    rows, cols = img.shape[:2]
    x = np.arange(cols)
    y = np.arange(rows)
    xx, yy = np.meshgrid(x, y)
    pattern = amplitude * np.sin(
        2 * np.pi * freq / cols * xx + 2 * np.pi * freq / rows * yy
    )
    noisy += pattern[:, :, np.newaxis]
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_horizontal_line_noise(img: np.ndarray,
                              intensity: float = 30,
                              density: float = 0.3) -> np.ndarray:
    """
    Realistic analog horizontal scan-line interference.
    Adds random horizontal lines with varying brightness, simulating
    CRT scan-line noise, VHS tracking errors, and interference.

    intensity : max pixel value shift per affected line (0-255)
    density   : fraction of rows affected (0.0-1.0)
    """
    noisy = img.astype(np.float32)
    rows = img.shape[0]
    for r in range(rows):
        if np.random.random() < density:
            shift = np.random.uniform(-intensity, intensity)
            noisy[r, :, :] += shift
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_dropout(img: np.ndarray, streak_count: int = 20,
                max_length_ratio: float = 0.5,
                max_width: int = 3,
                white_prob: float = 0.7) -> np.ndarray:
    """
    Realistic analog tape dropout — random horizontal streaks.
    Common in VHS/Betamax: magnetic tape particles shed, causing
    white or black horizontal streaks.

    streak_count    : number of dropout streaks
    max_length_ratio: max streak length as fraction of image width
    max_width       : max streak height in pixels (1-5 typical for VHS)
    white_prob      : probability of white streak (vs black)
    """
    noisy = img.astype(np.float32)
    h, w = img.shape[:2]
    for _ in range(streak_count):
        row = np.random.randint(0, h - max_width)
        start_col = np.random.randint(0, w - 1)
        length = int(np.random.uniform(0.1, max_length_ratio) * w)
        end_col = min(start_col + length, w)
        val = 255.0 if np.random.random() < white_prob else 0.0
        width = np.random.randint(1, max_width + 1)
        noisy[row:row + width, start_col:end_col, :] = val
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ─── NTSC Analog Video Degradation ──────────────────────────────────

def add_ntsc_noise(img: np.ndarray, seed: int | None = None,
                   intensity: str = "medium") -> np.ndarray:
    """
    Apply realistic NTSC analog video artifacts using zhuker/ntsc.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image.
    seed : int or None
        Random seed for reproducibility.
    intensity : str
        One of "light", "medium", "heavy", "vhs"

    Returns
    -------
    np.ndarray
        Image with simulated NTSC artifacts.
    """
    h, w = img.shape[:2]
    # NTSC expects even height for field-based processing
    if h % 2 != 0:
        img = cv2.resize(img, (w, h - 1))
        h = h - 1

    ntsc = _create_ntsc(intensity, seed)
    dst = np.zeros_like(img, dtype=np.uint8)

    # Process field 0 (even scanlines)
    ntsc.composite_layer(dst, img, field=0, fieldno=0)
    # Process field 1 (odd scanlines)
    ntsc.composite_layer(dst, img, field=1, fieldno=1)

    return dst


def _create_ntsc(intensity: str, seed: int | None) -> Ntsc:
    """Create an Ntsc instance with preset parameters."""
    if intensity == "random":
        return random_ntsc(seed)

    ntsc = Ntsc(random=None)
    if intensity == "light":
        ntsc._video_noise = 1
        ntsc._ringing = 0.85
        ntsc._color_bleed_horiz = 1
        ntsc._freq_noise_size = 0.0
        ntsc._video_chroma_noise = 10
        ntsc._subcarrier_amplitude = 50
        ntsc._subcarrier_amplitude_back = 50
    elif intensity == "medium":
        ntsc._video_noise = 4
        ntsc._ringing = 0.65
        ntsc._color_bleed_horiz = 3
        ntsc._color_bleed_vert = 1
        ntsc._freq_noise_size = 0.3
        ntsc._freq_noise_amplitude = 1.0
        ntsc._video_chroma_noise = 100
        ntsc._video_chroma_phase_noise = 10
        ntsc._subcarrier_amplitude = 50
        ntsc._subcarrier_amplitude_back = 50
    elif intensity == "heavy":
        ntsc._video_noise = 20
        ntsc._ringing = 0.45
        ntsc._color_bleed_horiz = 6
        ntsc._color_bleed_vert = 3
        ntsc._freq_noise_size = 0.6
        ntsc._freq_noise_amplitude = 2.0
        ntsc._video_chroma_noise = 500
        ntsc._video_chroma_phase_noise = 25
        ntsc._video_chroma_loss = 1000
        ntsc._subcarrier_amplitude = 50
        ntsc._subcarrier_amplitude_back = 50
    elif intensity == "vhs":
        ntsc._video_noise = 10
        ntsc._ringing = 0.55
        ntsc._color_bleed_horiz = 5
        ntsc._color_bleed_vert = 2
        ntsc._freq_noise_size = 0.4
        ntsc._freq_noise_amplitude = 1.5
        ntsc._video_chroma_noise = 200
        ntsc._video_chroma_phase_noise = 15
        ntsc._emulating_vhs = True
        ntsc._vhs_head_switching = True
        ntsc._vhs_edge_wave = 3
        ntsc._subcarrier_amplitude = 50
        ntsc._subcarrier_amplitude_back = 50
    else:
        raise ValueError(f"Unknown NTSC intensity: {intensity}")

    return ntsc


# ─── Composite Degradation ──────────────────────────────────────────

def degrade_image(img: np.ndarray, use_ntsc: bool = False,
                  ntsc_intensity: str = "medium",
                  strength: float = 0.5,
                  sigma: float | None = None,
                  impulse_prob: float | None = None,
                  color_r: float | None = None, color_g: float | None = None,
                  color_b: float | None = None,
                  brightness_gamma: float | None = None,
                  periodic_freq: float = 25,
                  periodic_amp: float | None = None) -> np.ndarray:
    """
    Composite degradation — stack multiple noise sources scaled by strength.

    Parameters
    ----------
    img : clean image
    use_ntsc : if True, use NTSC emulation instead of basic noise
    ntsc_intensity : NTSC artifact intensity
    strength : float 0.0–1.0 — scales ALL noise parameters linearly.
               0.0 = no noise, 0.5 = moderate (default), 1.0 = extreme.
               Individual params override the scaled value when explicitly set.
    sigma, impulse_prob, etc. : explicit overrides (take precedence over strength)
    """
    s = np.clip(strength, 0.0, 1.0)
    d = img.copy()

    if use_ntsc:
        d = add_ntsc_noise(d, intensity=ntsc_intensity)
    else:
        # Scale defaults by strength
        _sigma = sigma if sigma is not None else s * 30
        _impulse = impulse_prob if impulse_prob is not None else s * 0.05
        _r = color_r if color_r is not None else 1.0 + s * 0.35
        _g = color_g if color_g is not None else 1.0 - s * 0.35
        _b = color_b if color_b is not None else 1.0 + s * 0.15
        _gamma = brightness_gamma if brightness_gamma is not None else 1.0 - s * 0.55
        _hline_intensity = s * 25          # horizontal scan-line noise
        _hline_density = s * 0.25
        _dropout_count = int(s * 15)

        d = add_gaussian_noise(d, sigma=_sigma)
        d = add_impulse_noise(d, prob=_impulse)
        d = add_color_bias(d, r_gain=_r, g_gain=_g, b_gain=_b)
        d = reduce_brightness(d, gamma_val=max(0.1, _gamma))
        # Replace diagonal periodic noise with realistic analog artifacts
        if _hline_intensity > 0:
            d = add_horizontal_line_noise(d, intensity=_hline_intensity,
                                           density=_hline_density)
        if _dropout_count > 0:
            d = add_dropout(d, streak_count=max(1, _dropout_count))

    return d
