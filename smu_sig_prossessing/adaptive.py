"""
Content-aware adaptive pipeline.

Uses NoiseEstimator to analyse each frame and dynamically selects the best
preset + parameter adjustments.  Temporal smoothing prevents jarring parameter
changes between consecutive frames.

v3.6 improvements:
  - Uses v3.5 optimal presets for better quality/speed balance
  - Motion detection to adjust temporal filter strength
  - BRISQUE-aware quality validation
  - Chroma denoise auto-tuning
  - Better noise-type specific tuning
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
from .noise_estimation import (
    estimate_noise_level,
    classify_noise,
    suggest_preset,
    suggest_params,
)


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
        # Motion tracking
        self._prev_gray: np.ndarray | None = None
        self._motion_level: float = 0.0

    def reset(self) -> None:
        """Reset temporal state (call before processing a new video clip)."""
        self.smoother.reset()
        self._last_profile = None
        self._last_config = None
        self._prev_gray = None
        self._motion_level = 0.0

    # ── Public API ──────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Analyse *frame*, build a tuned PipelineConfig, and run the pipeline.
        """
        profile = self.estimator.estimate(frame)
        self._last_profile = profile

        # Detect motion between frames
        motion = self._detect_motion(frame)
        self._motion_level = self.smoother.smooth("motion", motion)

        cfg = self._build_config(profile)
        self._last_config = cfg

        if self.verbose:
            print(f"  [adaptive] {profile}  motion={motion:.3f}  stages={len(cfg.stages)}")

        return apply_pipeline(frame, cfg)

    def process_with_profile(self, frame: np.ndarray) -> tuple[np.ndarray, NoiseProfile]:
        """Like ``process`` but also returns the NoiseProfile."""
        result = self.process(frame)
        return result, self._last_profile  # type: ignore[return-value]

    def noise_aware_process(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Analyse *frame*, estimate noise + motion, select a preset, adjust
        parameters, apply the pipeline, and return (processed, metadata).

        This is a self-contained convenience method that does not depend on
        the internal ``NoiseEstimator`` — it uses the lightweight Laplacian-
        based ``noise_estimation`` module for a fast per-frame estimate.

        Parameters
        ----------
        frame : np.ndarray
            Current BGR frame (uint8).
        prev_frame : np.ndarray | None
            Previous frame for motion detection.  ``None`` on the first call.

        Returns
        -------
        processed : np.ndarray
            Pipeline-processed frame.
        metadata : dict
            Diagnostic dictionary with keys:

            - ``noise_std`` : float — estimated noise std dev
            - ``noise_label`` : str — ``'low'/'medium'/'high'/'extreme'``
            - ``motion`` : float — motion level (0-1)
            - ``preset_name`` : str — selected preset name
            - ``params`` : dict — parameter adjustments applied
            - ``config_label`` : str — human-readable pipeline label
        """
        # ── 1. Estimate noise level ────────────────────────────────────
        noise_std = estimate_noise_level(frame)
        noise_label = classify_noise(noise_std)

        # ── 2. Detect motion ───────────────────────────────────────────
        if prev_frame is not None:
            motion = self._detect_motion_from_pair(prev_frame, frame)
        else:
            motion = 0.0
            # Initialise internal tracking for consistent state
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            self._prev_gray = gray

        has_motion = motion > 0.4

        # ── 3. Select preset name ──────────────────────────────────────
        preset_name = suggest_preset(noise_std, has_motion=has_motion)

        # ── 4. Get parameter adjustments ───────────────────────────────
        param_adjustments = suggest_params(noise_std)

        # ── 5. Build config from preset name ───────────────────────────
        cfg = self._preset_from_name(preset_name)
        if cfg is None:
            # Fallback: use the regular estimator-based build
            from .noise_estimator import NoiseEstimator

            estimator = NoiseEstimator()
            profile = estimator.estimate(frame)
            cfg = self._build_config(profile)
            preset_name = "estimator-fallback"

        # Apply noise-based parameter overrides
        self._apply_noise_params(cfg, param_adjustments)

        # ── 6. Process and return ──────────────────────────────────────
        processed = apply_pipeline(frame, cfg)

        metadata: dict = {
            "noise_std": round(noise_std, 2),
            "noise_label": noise_label,
            "motion": round(motion, 3),
            "preset_name": preset_name,
            "params": param_adjustments,
            "config_label": cfg.label,
        }

        return processed, metadata

    # ── Internal helpers for noise_aware_process ──────────────────────

    @staticmethod
    def _detect_motion_from_pair(
        prev: np.ndarray,
        curr: np.ndarray,
    ) -> float:
        """
        Compute motion level (0-1) from two frames.

        Pure static method — does not touch instance state so it can be
        called before ``_prev_gray`` is initialised.
        """
        gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.mean(np.abs(gray_curr - gray_prev))
        return float(np.clip(diff / 10.0, 0.0, 1.0))

    @staticmethod
    def _preset_from_name(name: str) -> PipelineConfig | None:
        """
        Resolve a preset *name* string to a ``PipelineConfig`` instance.

        Returns ``None`` if the name is unknown (caller should fall back).
        """
        # Import here to avoid circular dependency at module level
        from .config import PipelineConfig as PC

        registry: dict[str, Any] = {
            "optimal-ultrafast": PC.optimal_ultrafast,
            "grey-guided-chroma": PC.grey_guided_chroma,
            "grey-premium": PC.grey_premium,
            "video-enhanced": PC.video_enhanced,
            "fast-guided-chroma": PC.fast_guided_chroma,
            "optimal-bior4": PC.optimal_bior4,
            "nlm-chroma": PC.nlm_chroma_preset,
            "cross-chroma-detail": PC.cross_chroma_detail,
            "aggressive": PC.aggressive,
            "fast-guided-chroma": PC.fast_guided_chroma,
        }
        builder = registry.get(name)
        if builder is not None:
            return builder()
        return None

    @staticmethod
    def _apply_noise_params(
        cfg: PipelineConfig,
        params: dict,
    ) -> None:
        """
        Mutate *cfg* in-place to apply noise-based parameter overrides.

        The *params* dict is expected to come from ``suggest_params``.
        Only known stage names are modified; unknown keys are silently
        ignored.
        """
        for stage in cfg.stages:
            if not stage.enabled:
                continue

            if stage.name == "median":
                ksz = params.get("median_ksize", 0)
                if ksz and ksz > 0:
                    stage.enabled = True
                    stage.params["ksize"] = int(ksz)
                else:
                    stage.enabled = False

            elif stage.name == "wavelet":
                if "wavelet_level" in params:
                    stage.params["level"] = int(params["wavelet_level"])
                if "wavelet_threshold_mode" in params:
                    stage.params["threshold_mode"] = params["wavelet_threshold_mode"]

            elif stage.name == "bilateral":
                if "bilateral_sigma" in params:
                    stage.params["sigma_color"] = params["bilateral_sigma"]
                    stage.params["sigma_space"] = params["bilateral_sigma"]
                if "bilateral_d" in params:
                    d = int(params["bilateral_d"])
                    stage.params["d"] = d if d % 2 == 1 else d + 1

            elif stage.name == "cross_bilateral":
                if "bilateral_sigma" in params:
                    stage.params["sigma_color"] = params["bilateral_sigma"]
                    stage.params["sigma_space"] = params["bilateral_sigma"]
                if "bilateral_d" in params:
                    d = int(params["bilateral_d"])
                    stage.params["d"] = d if d % 2 == 1 else d + 1

            elif stage.name == "nlm":
                if "nlm_h" in params:
                    stage.params["h"] = float(params["nlm_h"])

            elif stage.name == "unsharp_mask":
                if "unsharp_strength" in params:
                    stage.params["strength"] = params["unsharp_strength"]

            elif stage.name == "guided_filter":
                if "guided_eps" in params:
                    stage.params["eps"] = params["guided_eps"]

            elif stage.name == "chroma_denoise":
                if "chroma_strength" in params:
                    stage.params["strength"] = params["chroma_strength"]

            elif stage.name == "detail_boost":
                if "detail_boost" in params:
                    stage.params["strength"] = params["detail_boost"]

            elif stage.name == "flicker_stabilize":
                if "flicker_strength" in params:
                    stage.params["strength"] = params["flicker_strength"]

            elif stage.name == "temporal_motion":
                if "temporal_strength" in params:
                    stage.params["strength"] = params["temporal_strength"]

            elif stage.name == "temporal_nlm_multi":
                if "nlm_h" in params:
                    h = params["nlm_h"]
                    stage.params["h"] = h
                    stage.params["h_color"] = h * 0.8

    @property
    def last_profile(self) -> NoiseProfile | None:
        return self._last_profile

    @property
    def last_config(self) -> PipelineConfig | None:
        return self._last_config

    def _detect_motion(self, frame: np.ndarray) -> float:
        """
        Estimate scene motion from frame-to-frame differences.
        Returns 0 (static) to 1 (high motion).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0

        diff = np.mean(np.abs(gray - self._prev_gray))
        self._prev_gray = gray
        # Normalize: diff of ~5.0 is moderate motion for 8-bit
        return float(np.clip(diff / 10.0, 0.0, 1.0))

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

        v3.6.2: Updated to use v3.6 optimized presets (optimal-bior4 NIQE=7.26,
        nlm-chroma BRISQUE=41.28, fast-guided-chroma BRISQUE=55.42@49ms,
        cross-chroma-detail NIQE=7.35@169ms).  Motion-aware selection
        switches between fast and quality branches.
        """
        ntype = profile.noise_type
        nlevel = profile.level
        motion = self._motion_level

        # Periodic / analog artifact patterns — need scanline+flicker removal
        if ntype in (NoiseType.PERIODIC, NoiseType.MIXED):
            if nlevel in (NoiseLevel.HIGH, NoiseLevel.EXTREME):
                return PipelineConfig.analog_heavy()
            # For moderate periodic noise, use analog-clean (most consistent)
            if motion > 0.5:
                return PipelineConfig.analog_clean()
            return PipelineConfig.analog_clean()

        # Clean / low noise → keep it fast
        if ntype == NoiseType.CLEAN or nlevel == NoiseLevel.LOW:
            if motion > 0.5:
                return PipelineConfig.optimal_ultrafast()  # 10ms, 80fps+
            return PipelineConfig.fast_guided_chroma()  # 49ms, best BRISQUE balance

        # Gaussian noise — DCT and wavelet handle this well
        if ntype == NoiseType.GAUSSIAN:
            if nlevel == NoiseLevel.MEDIUM:
                # optimal-bior4 has best NIQE (7.26) for general use
                if motion > 0.5:
                    return PipelineConfig.cross_chroma_detail()  # 169ms
                return PipelineConfig.optimal_bior4()
            elif nlevel == NoiseLevel.HIGH:
                if motion > 0.5:
                    return PipelineConfig.fast_guided_chroma()  # 49ms
                return PipelineConfig.nlm_chroma_preset()  # NIQE=7.33, BRISQUE=41.28
            else:  # EXTREME
                if motion > 0.5:
                    return PipelineConfig.optimal_ultrafast()
                return PipelineConfig.aggressive()

        # Impulse noise — median + wavelet
        if ntype == NoiseType.IMPULSE:
            if nlevel == NoiseLevel.MEDIUM:
                if motion > 0.5:
                    return PipelineConfig.cross_chroma_detail()  # 169ms
                return PipelineConfig.optimal_bior4()  # NIQE=7.26
            elif nlevel == NoiseLevel.HIGH:
                if motion > 0.5:
                    return PipelineConfig.fast_guided_chroma()
                return PipelineConfig.wavelet_denoise()
            else:
                if motion > 0.5:
                    return PipelineConfig.optimal_ultrafast()
                return PipelineConfig.aggressive()

        # General fallback by noise level
        if nlevel == NoiseLevel.MEDIUM:
            # optimal-bior4: best NIQE overall
            if motion > 0.5:
                return PipelineConfig.cross_chroma_detail()
            return PipelineConfig.optimal_bior4()

        # High noise
        if nlevel == NoiseLevel.HIGH:
            if motion > 0.5:
                return PipelineConfig.fast_guided_chroma()
            return PipelineConfig.nlm_chroma_preset()

        # Extreme
        if motion > 0.5:
            return PipelineConfig.optimal_ultrafast()
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

            elif stage.name == "cross_bilateral":
                # Scale with noise
                sig = self.smoother.smooth("xbilateral_sigma",
                                           params.get("bilateral_sigma", 30.0))
                d = int(self.smoother.smooth("xbilateral_d",
                                             float(params.get("bilateral_d", 5))))
                d = d if d % 2 == 1 else d + 1
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
                eps_val = float(np.clip(profile.sigma * 10.0, 50.0, 500.0))
                eps_val = self.smoother.smooth("guided_eps", eps_val)
                stage.params["eps"] = eps_val

            elif stage.name == "chroma_denoise":
                # Scale chroma denoise with noise + motion
                strength = float(np.clip(profile.sigma * 0.015, 0.1, 0.8))
                # Reduce chroma denoise on high motion to avoid color trails
                motion_factor = 1.0 - self._motion_level * 0.5
                strength *= motion_factor
                strength = self.smoother.smooth("chroma_strength",
                                                float(np.clip(strength, 0.05, 0.9)))
                stage.params["strength"] = strength

            elif stage.name == "detail_boost":
                # More detail boost for cleaner frames, less for noisy
                boost = float(np.clip(0.4 - profile.sigma * 0.01, 0.05, 0.5))
                boost = self.smoother.smooth("detail_boost", boost)
                stage.params["strength"] = boost

            elif stage.name == "temporal_motion":
                # Adjust temporal blending with motion + noise
                t_str = float(np.clip(0.5 - self._motion_level * 0.4, 0.1, 0.5))
                t_str = self.smoother.smooth("temporal_strength", t_str)
                stage.params["strength"] = t_str

            elif stage.name == "temporal_nlm_multi":
                # Adjust temporal window based on motion
                # High motion → smaller window to avoid ghosting
                if self._motion_level > 0.4:
                    stage.params["temporal_window"] = 1  # just ±1
                    stage.params["max_frames"] = 3
                else:
                    stage.params["temporal_window"] = 2
                    stage.params["max_frames"] = 5
                # Scale h with noise
                h_val = float(np.clip(profile.sigma * 0.5, 3, 15))
                h_val = self.smoother.smooth("tnlm_h", h_val)
                stage.params["h"] = h_val
                stage.params["h_color"] = h_val * 0.8

        # Inject missing critical stages
        self._ensure_median(cfg, profile)
        self._ensure_unsharp(cfg, profile)
        self._ensure_chroma(cfg, profile)

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

    @staticmethod
    def _ensure_chroma(cfg: PipelineConfig, profile: NoiseProfile) -> None:
        """Add chroma_denoise if noise level is moderate+ but no chroma stage exists."""
        if profile.level in (NoiseLevel.LOW,):
            return
        for s in cfg.stages:
            if s.name == "chroma_denoise" and s.enabled:
                return
        strength = 0.3 if profile.level in (NoiseLevel.MEDIUM,) else 0.5
        cd = FilterConfig(name="chroma_denoise", enabled=True,
                          params={"strength": strength})
        # Insert before unsharp
        insert_before = len(cfg.stages)
        for i, s in enumerate(cfg.stages):
            if s.name == "unsharp_mask":
                insert_before = i
                break
        cfg.stages.insert(insert_before, cd)

