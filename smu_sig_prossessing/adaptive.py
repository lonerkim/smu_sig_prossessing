"""
Content-aware adaptive pipeline.

Uses NoiseEstimator to analyse each frame and dynamically selects the best
preset + parameter adjustments.  Temporal smoothing prevents jarring parameter
changes between consecutive frames.
"""
from __future__ import annotations

import copy
from typing import Any

import cv2
import numpy as np

from .config import PipelineConfig, FilterConfig
from .noise_estimator import (
    NoiseEstimator,
    NoiseLevel,
    NoiseType,
    NoiseProfile,
)
from .pipeline import apply_pipeline


# ─── Temporal smoother ───────────────────────────────────────────────

class _TemporalSmoother:
    """
    Exponential moving average (EMA) for scalar parameters across frames.

    Prevents abrupt parameter jumps that would cause visible flickering.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """
        Parameters
        ----------
        alpha : float
            EMA blending factor.  Higher = faster adaptation (more jitter).
            Lower = smoother (more lag).  Default 0.3 is a good compromise.
        """
        self.alpha = alpha
        self._state: dict[str, float] = {}

    def smooth(self, key: str, value: float) -> float:
        if key not in self._state:
            self._state[key] = value
        else:
            self._state[key] = self.alpha * value + (1.0 - self.alpha) * self._state[key]
        return self._state[key]

    def reset(self) -> None:
        self._state.clear()


# ─── Adaptive Pipeline ───────────────────────────────────────────────

class AdaptivePipeline:
    """
    Content-aware pipeline that auto-tunes per-frame.

    Usage::

        ap = AdaptivePipeline()
        for frame in video:
            result = ap.process(frame)
    """

    def __init__(
        self,
        estimator: NoiseEstimator | None = None,
        ema_alpha: float = 0.3,
        verbose: bool = False,
    ) -> None:
        self.estimator = estimator or NoiseEstimator()
        self.smoother = _TemporalSmoother(alpha=ema_alpha)
        self.verbose = verbose
        self._last_profile: NoiseProfile | None = None
        self._last_config: PipelineConfig | None = None

    def reset(self) -> None:
        """Reset temporal state (call before processing a new video clip)."""
        self.smoother.reset()
        self._last_profile = None
        self._last_config = None

    # ── Public API ──────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Analyse *frame*, build a tuned PipelineConfig, and run the pipeline.
        """
        profile = self.estimator.estimate(frame)
        self._last_profile = profile

        cfg = self._build_config(profile)
        self._last_config = cfg

        if self.verbose:
            print(f"  [adaptive] {profile}  stages={len(cfg.stages)}")

        return apply_pipeline(frame, cfg)

    def process_with_profile(self, frame: np.ndarray) -> tuple[np.ndarray, NoiseProfile]:
        """Like ``process`` but also returns the NoiseProfile."""
        result = self.process(frame)
        return result, self._last_profile  # type: ignore[return-value]

    @property
    def last_profile(self) -> NoiseProfile | None:
        return self._last_profile

    @property
    def last_config(self) -> PipelineConfig | None:
        return self._last_config

    # ── Config builder ──────────────────────────────────────────────

    def _build_config(self, profile: NoiseProfile) -> PipelineConfig:
        """
        Select a base preset and tune its parameters based on *profile*.
        """
        # 1. Choose base preset
        base = self._select_base_preset(profile)

        # 2. Deep-copy so we don't mutate cached presets
        cfg = base.copy()

        # 3. Adjust parameters with temporal smoothing
        self._tune_config(cfg, profile)

        return cfg

    def _select_base_preset(self, profile: NoiseProfile) -> PipelineConfig:
        """
        Pick the best starting preset for a given noise profile.

        Logic:
          - CLEAN / LOW noise  → fast_denoise (lightweight)
          - MEDIUM + gaussian  → edge_preserve (NLM + bilateral)
          - MEDIUM + impulse   → wavelet_denoise (handles impulse well)
          - HIGH noise         → video_enhanced (stronger, guided + wavelet)
          - EXTREME noise      → aggressive (maximum denoising)
          - periodic component → analog_clean (scanline + flicker)
        """
        ntype = profile.noise_type
        nlevel = profile.level

        # Periodic / analog artifact patterns
        if ntype in (NoiseType.PERIODIC, NoiseType.MIXED):
            if nlevel in (NoiseLevel.HIGH, NoiseLevel.EXTREME):
                return PipelineConfig.analog_heavy()
            return PipelineConfig.analog_clean()

        # Clean / low noise → keep it fast and light
        if ntype == NoiseType.CLEAN or nlevel == NoiseLevel.LOW:
            return PipelineConfig.fast_denoise()

        # Medium noise
        if nlevel == NoiseLevel.MEDIUM:
            if ntype == NoiseType.IMPULSE:
                return PipelineConfig.wavelet_denoise()
            return PipelineConfig.edge_preserve()

        # High noise
        if nlevel == NoiseLevel.HIGH:
            return PipelineConfig.video_enhanced()

        # Extreme
        return PipelineConfig.aggressive()

    def _tune_config(self, cfg: PipelineConfig, profile: NoiseProfile) -> None:
        """
        Mutate *cfg* in-place to adapt its filter parameters to the current
        noise profile, with temporal smoothing.
        """
        params = profile.recommended_params

        for stage in cfg.stages:
            if not stage.enabled:
                continue

            if stage.name == "bilateral":
                sig = self.smoother.smooth("bilateral_sigma",
                                           params.get("bilateral_sigma", 50.0))
                d = int(self.smoother.smooth("bilateral_d",
                                             float(params.get("bilateral_d", 7))))
                d = d if d % 2 == 1 else d + 1  # bilateral d must be odd
                stage.params["sigma_color"] = sig
                stage.params["sigma_space"] = sig
                stage.params["d"] = d

            elif stage.name == "wavelet":
                lvl = int(self.smoother.smooth("wavelet_level",
                                               float(params.get("wavelet_level", 2))))
                stage.params["level"] = max(1, lvl)

            elif stage.name == "median":
                ksz = params.get("median_ksize", 0)
                if ksz > 0:
                    stage.enabled = True
                    stage.params["ksize"] = int(ksz)
                else:
                    stage.enabled = False

            elif stage.name == "nlm":
                # Scale NLM h with noise sigma
                h_val = float(np.clip(profile.sigma * 0.6, 3, 15))
                h_val = self.smoother.smooth("nlm_h", h_val)
                stage.params["h"] = h_val

            elif stage.name == "unsharp_mask":
                us_str = self.smoother.smooth("unsharp_strength",
                                              params.get("unsharp_strength", 0.2))
                stage.params["strength"] = us_str

            elif stage.name == "flicker_stabilize":
                flick_s = self.smoother.smooth("flicker_strength",
                                               params.get("flicker_strength", 0.5))
                stage.params["strength"] = flick_s

            elif stage.name == "fft_notch":
                stage.enabled = bool(params.get("use_fft_notch", False))

            elif stage.name == "scanline_remove":
                stage.enabled = bool(params.get("use_scanline", False))

            elif stage.name == "guided_filter":
                # Scale eps with noise — more noise = larger eps (more smoothing)
                eps_val = float(np.clip(profile.sigma * 10.0, 50.0, 500.0))
                eps_val = self.smoother.smooth("guided_eps", eps_val)
                stage.params["eps"] = eps_val

        # Inject missing critical stages that the base preset may not have
        self._ensure_median(cfg, profile)
        self._ensure_unsharp(cfg, profile)

    @staticmethod
    def _ensure_median(cfg: PipelineConfig, profile: NoiseProfile) -> None:
        """Add a median pre-filter if impulse noise is detected but no median stage exists."""
        if profile.noise_type not in (NoiseType.IMPULSE, NoiseType.MIXED):
            return
        for s in cfg.stages:
            if s.name == "median" and s.enabled:
                return
        # Insert median as first stage
        med = FilterConfig(name="median", enabled=True, params={"ksize": 3})
        cfg.stages.insert(0, med)

    @staticmethod
    def _ensure_unsharp(cfg: PipelineConfig, profile: NoiseProfile) -> None:
        """Ensure an unsharp_mask recovery stage exists."""
        for s in cfg.stages:
            if s.name == "unsharp_mask" and s.enabled:
                return
        strength = 0.15 if profile.level in (NoiseLevel.LOW,) else 0.25
        us = FilterConfig(name="unsharp_mask", enabled=True,
                          params={"strength": strength, "radius": 0.5, "threshold": 8})
        cfg.stages.append(us)
