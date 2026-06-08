"""
Adaptive noise estimation module.

Analyzes image frames to estimate noise characteristics and return a NoiseProfile
with level, type, and recommended processing parameters.

Methods:
  - MAD (Median Absolute Deviation) of wavelet coefficients for robust noise sigma
  - Laplacian variance for spatial noise assessment
  - Noise type classification (Gaussian / impulse / periodic / mixed)
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pywt


# ─── Noise Level Enum ────────────────────────────────────────────────

class NoiseLevel(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class NoiseType(enum.Enum):
    GAUSSIAN = "gaussian"
    IMPULSE = "impulse"
    PERIODIC = "periodic"
    MIXED = "mixed"
    CLEAN = "clean"


# ─── Noise Profile ───────────────────────────────────────────────────

@dataclass
class NoiseProfile:
    """Complete noise characterisation for a single frame."""
    level: NoiseLevel
    noise_type: NoiseType
    sigma: float                # estimated noise standard deviation (0-255 scale)
    laplacian_var: float        # spatial noise indicator (Laplacian variance)
    impulse_ratio: float        # fraction of pixels suspected as impulse noise
    periodic_score: float       # strength of periodic component (0-1)
    recommended_params: dict[str, Any] = field(default_factory=dict)

    # handy thresholds
    _SIGMA_THRESHOLDS = (10.0, 25.0, 50.0)   # low/medium/high/extreme

    def __repr__(self) -> str:
        return (f"NoiseProfile(level={self.level.value}, type={self.noise_type.value}, "
                f"sigma={self.sigma:.1f}, lap_var={self.laplacian_var:.1f}, "
                f"impulse={self.impulse_ratio:.4f}, periodic={self.periodic_score:.3f})")


# ─── Noise Estimator ─────────────────────────────────────────────────

class NoiseEstimator:
    """
    Estimate noise characteristics from a single BGR frame.

    Usage::

        estimator = NoiseEstimator()
        profile = estimator.estimate(frame)
        print(profile.level, profile.noise_type)
    """

    def __init__(
        self,
        wavelet: str = "db4",
        decomp_level: int = 3,
        impulse_threshold: float = 3.0,
        periodic_fft_bins: int = 10,
    ) -> None:
        self.wavelet = wavelet
        self.decomp_level = decomp_level
        self.impulse_threshold = impulse_threshold
        self.periodic_fft_bins = periodic_fft_bins

    # ── Public API ──────────────────────────────────────────────────

    def estimate(self, frame: np.ndarray) -> NoiseProfile:
        """Analyse *frame* and return a full NoiseProfile."""
        gray = self._to_gray(frame)

        sigma = self._mad_wavelet_sigma(gray)
        lap_var = self._laplacian_variance(gray)
        impulse_ratio = self._impulse_score(gray)
        periodic_score = self._periodic_score(gray)

        level = self._classify_level(sigma)
        noise_type = self._classify_type(impulse_ratio, periodic_score, sigma)

        params = self._recommended_params(level, noise_type, sigma)

        return NoiseProfile(
            level=level,
            noise_type=noise_type,
            sigma=sigma,
            laplacian_var=lap_var,
            impulse_ratio=impulse_ratio,
            periodic_score=periodic_score,
            recommended_params=params,
        )

    # ── Quick helpers (no object needed) ────────────────────────────

    @staticmethod
    def quick_sigma(frame: np.ndarray) -> float:
        """Return just the MAD-based noise sigma for a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        coeffs = pywt.wavedec2(gray.astype(np.float64), "db4", level=3)
        if isinstance(coeffs[-1], tuple) and len(coeffs[-1]) > 2:
            hh = coeffs[-1][2]
            if hh.size > 0:
                return float(np.median(np.abs(hh)) / 0.6745)
        return 10.0

    # ── Internal ────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 2:
            return frame.astype(np.float64)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

    def _mad_wavelet_sigma(self, gray: np.ndarray) -> float:
        """Robust noise sigma via MAD of finest-scale wavelet coefficients."""
        try:
            coeffs = pywt.wavedec2(gray, self.wavelet, level=self.decomp_level)
        except ValueError:
            # fallback to haar if image too small
            coeffs = pywt.wavedec2(gray, "haar", level=min(2, self.decomp_level))

        # finest detail coefficients (HH sub-band)
        if isinstance(coeffs[-1], tuple) and len(coeffs[-1]) > 2:
            hh = coeffs[-1][2]
            if hh.size > 0:
                return float(np.median(np.abs(hh)) / 0.6745)
        return 10.0

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        """Spatial noise assessment via Laplacian variance."""
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())

    def _impulse_score(self, gray: np.ndarray) -> float:
        """
        Estimate impulse (salt & pepper) noise fraction.

        Compare each pixel to its 3×3 median.  Pixels that differ by more
        than *impulse_threshold* × local MAD are counted as impulse outliers.
        """
        median_filtered = cv2.medianBlur(gray.astype(np.float32), 3).astype(np.float64)
        diff = np.abs(gray - median_filtered)

        # local MAD as robust scale estimator
        local_mad = np.median(np.abs(diff)) / 0.6745
        local_mad = max(local_mad, 1.0)  # avoid zero-division for clean images

        outlier_mask = diff > self.impulse_threshold * local_mad
        return float(np.mean(outlier_mask))

    def _periodic_score(self, gray: np.ndarray) -> float:
        """
        Detect periodic noise via row-mean FFT analysis.

        Returns a score in [0, 1] where higher means stronger periodic component.
        """
        rows, cols = gray.shape
        # Row means → 1-D signal
        row_means = gray.mean(axis=1)
        row_means -= row_means.mean()

        if len(row_means) < 8:
            return 0.0

        spectrum = np.abs(np.fft.fft(row_means))
        spectrum = spectrum[1:len(spectrum) // 2]  # remove DC and mirror

        if len(spectrum) < 2:
            return 0.0

        # Compare top-N peak energy to total energy
        top_n = min(self.periodic_fft_bins, len(spectrum))
        peak_energy = np.sum(np.sort(spectrum)[-top_n:] ** 2)
        total_energy = np.sum(spectrum ** 2)

        if total_energy < 1e-10:
            return 0.0

        raw_score = peak_energy / total_energy
        # Normalise: for pure Gaussian noise this ratio is ~top_n/len(spectrum)
        baseline = top_n / len(spectrum)
        return float(np.clip((raw_score - baseline) / (1.0 - baseline + 1e-10), 0, 1))

    # ── Classification ──────────────────────────────────────────────

    @staticmethod
    def _classify_level(sigma: float) -> NoiseLevel:
        if sigma < NoiseProfile._SIGMA_THRESHOLDS[0]:
            return NoiseLevel.LOW
        elif sigma < NoiseProfile._SIGMA_THRESHOLDS[1]:
            return NoiseLevel.MEDIUM
        elif sigma < NoiseProfile._SIGMA_THRESHOLDS[2]:
            return NoiseLevel.HIGH
        else:
            return NoiseLevel.EXTREME

    @staticmethod
    def _classify_type(impulse_ratio: float, periodic_score: float,
                       sigma: float) -> NoiseType:
        has_impulse = impulse_ratio > 0.02
        has_periodic = periodic_score > 0.3
        has_gaussian = sigma > 10.0

        types = []
        if has_gaussian:
            types.append("gaussian")
        if has_impulse:
            types.append("impulse")
        if has_periodic:
            types.append("periodic")

        if len(types) == 0:
            return NoiseType.CLEAN
        if len(types) == 1:
            return NoiseType(types[0])
        return NoiseType.MIXED

    @staticmethod
    def _recommended_params(level: NoiseLevel, noise_type: NoiseType,
                            sigma: float) -> dict[str, Any]:
        """
        Return recommended filter parameters based on noise analysis.

        The adaptive pipeline can use these as a starting point and then
        further refine with temporal smoothing.
        """
        # Scale bilateral sigma with estimated noise
        bilateral_sigma = float(np.clip(sigma * 2.5, 15, 150))
        bilateral_d = 5 if level in (NoiseLevel.LOW, NoiseLevel.MEDIUM) else 9

        params: dict[str, Any] = {
            "bilateral_sigma": bilateral_sigma,
            "bilateral_d": bilateral_d,
            "wavelet_level": 2 if level in (NoiseLevel.LOW, NoiseLevel.MEDIUM) else 3,
            "median_ksize": 3 if has_impulse_type(noise_type) else 0,
            "unsharp_strength": 0.15 if level == NoiseLevel.LOW else 0.25,
            "use_fft_notch": noise_type in (NoiseType.PERIODIC, NoiseType.MIXED),
            "use_scanline": noise_type in (NoiseType.PERIODIC, NoiseType.MIXED),
            "flicker_strength": 0.6 if level in (NoiseLevel.HIGH, NoiseLevel.EXTREME) else 0.3,
        }

        # For extreme noise, be more aggressive
        if level == NoiseLevel.EXTREME:
            params["bilateral_sigma"] = min(150, sigma * 3.0)
            params["bilateral_d"] = 13
            params["wavelet_level"] = 3
            params["unsharp_strength"] = 0.3
            params["flicker_strength"] = 0.8

        # For clean / low noise, be gentle
        if level == NoiseLevel.LOW:
            params["bilateral_sigma"] = max(15, sigma * 1.5)
            params["bilateral_d"] = 5

        return params


def has_impulse_type(noise_type: NoiseType) -> bool:
    return noise_type in (NoiseType.IMPULSE, NoiseType.MIXED)
