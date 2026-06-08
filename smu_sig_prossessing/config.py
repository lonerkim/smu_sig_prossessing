"""
Pipeline configuration system — modular filter control with ON/OFF and intensity.

Each filter stage has:
  - name: unique identifier
  - enabled: bool — whether to apply this filter
  - params: dict — stage-specific parameters (kernel size, sigma, gamma, etc.)

PipelineConfig holds an ordered list of FilterConfig instances.
The pipeline runner iterates through them in order.

Design history: see plan/design_change_log.md
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FilterConfig:
    """Configuration for a single pipeline stage."""
    name: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def disable(self) -> FilterConfig:
        self.enabled = False
        return self


@dataclass
class PipelineConfig:
    """Ordered list of filter stages forming a processing pipeline."""
    stages: list[FilterConfig] = field(default_factory=list)
    label: str = "custom"

    def add(self, name: str, enabled: bool = True, **params) -> FilterConfig:
        fc = FilterConfig(name=name, enabled=enabled, params=params)
        self.stages.append(fc)
        return fc

    def enable(self, name: str) -> None:
        for s in self.stages:
            if s.name == name:
                s.enabled = True
                return
        raise KeyError(f"Filter '{name}' not found")

    def disable(self, name: str) -> None:
        for s in self.stages:
            if s.name == name:
                s.enabled = False
                return
        raise KeyError(f"Filter '{name}' not found")

    def get(self, name: str) -> FilterConfig | None:
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def copy(self) -> PipelineConfig:
        import copy
        return copy.deepcopy(self)

    # ── Presets ──────────────────────────────────────────────────────
    #
    # Each preset was designed and validated experimentally.
    # Metrics: PSNR ↑, SSIM ↑, Edge retention vs original ↑, Speed ↓
    # See plan/design_change_log.md for full comparison data.

    @staticmethod
    def edge_preserve() -> PipelineConfig:
        """
        [DEFAULT] Edge-preserving denoising — NLM + Bilateral + Unsharp.
        PSNR=17.77  SSIM=0.466  Edge=79%  0.33s

        NLM removes noise while preserving edges (patch-based).
        Bilateral smooths residual noise (edge-aware).
        Unsharp mask recovers any edge contrast lost during denoising.
        No aggressive gamma or histogram equalization (preserves color).
        """
        cfg = PipelineConfig(label="Edge-Preserve (NLM+Bilateral)")
        cfg.add("median", ksize=3)                                   # impulse noise
        cfg.add("nlm", h=5, template_window=7, search_window=21)     # patch-based denoise
        cfg.add("bilateral", d=7, sigma_color=50, sigma_space=50)    # edge-preserving smooth
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15) # gentle color balance
        cfg.add("unsharp_mask", strength=0.3, radius=0.5, threshold=10)  # edge recovery
        return cfg

    @staticmethod
    def fast_denoise() -> PipelineConfig:
        """
        [RECOMMENDED for video] Fast bilateral-only denoising.
        PSNR=16.59  SSIM=0.434  Edge=153%  0.01s

        30× faster than edge-preserve. Good balance of speed and quality.
        Edge enhancement from unsharp mask compensates for bilateral smoothing.
        """
        cfg = PipelineConfig(label="Fast Denoise (Bilateral)")
        cfg.add("bilateral", d=5, sigma_color=30, sigma_space=30)    # fast edge-preserving
        cfg.add("channel_correction", clamp_min=0.9, clamp_max=1.1)  # gentle color
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=10)  # edge recovery
        return cfg

    @staticmethod
    def nlm_denoise() -> PipelineConfig:
        """
        NLM-only denoising — highest quality, slowest.
        PSNR=16.58  SSIM=0.423  Edge=171%  0.23s

        Strong NLM with mild bilateral cleanup.
        Edge enhancement is strongest here.
        """
        cfg = PipelineConfig(label="NLM Denoise")
        cfg.add("nlm", h=8, template_window=7, search_window=21)
        cfg.add("bilateral", d=5, sigma_color=20, sigma_space=20)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.3, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def wiener_denoise() -> PipelineConfig:
        """
        Wiener-based denoising — frequency domain approach.
        PSNR=14.68  SSIM=0.469  Edge=67%  0.05s

        Worst edge preservation of all methods.
        FFT cannot distinguish noise edges from real edges.
        Included for reference / periodic noise cases.
        """
        cfg = PipelineConfig(label="Wiener Denoise")
        cfg.add("median", ksize=3)
        cfg.add("wiener", noise_var=200)
        cfg.add("fft_notch", threshold_percentile=99.5)
        cfg.add("channel_correction", clamp_min=0.9, clamp_max=1.1)
        cfg.add("unsharp_mask", strength=0.4, radius=0.5, threshold=10)
        cfg.add("gamma", gamma=1.1)
        return cfg

    @staticmethod
    def wavelet_denoise() -> PipelineConfig:
        """
        Wavelet-based denoising — multi-resolution approach.
        Excellent edge preservation via sparse wavelet representation.

        Uses db4 wavelet with VisuShrink universal threshold.
        Best for images with sharp edges and fine texture.
        """
        cfg = PipelineConfig(label="Wavelet Denoise")
        cfg.add("wavelet", wavelet="bior4.4", level=3, threshold_mode="soft", n_shifts=3)
        cfg.add("bilateral", d=5, sigma_color=20, sigma_space=20)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)
        return cfg

    # ── New Presets (v3.0) ──────────────────────────────────────────

    @staticmethod
    def bm3d_denoise() -> PipelineConfig:
        """
        BM3D denoising preset — median prefilter + BM3D + channel correction + unsharp.
        Best denoising quality available, suitable for moderate to heavy noise.

        median removes impulse/spike noise before BM3D for better patch matching.
        BM3D handles Gaussian + structured noise via collaborative filtering.
        Channel correction fixes any colour shift from denoising.
        Unsharp mask recovers edge sharpness lost during denoising.
        """
        cfg = PipelineConfig(label="BM3D Denoise")
        cfg.add("median", ksize=3)
        cfg.add("bm3d_denoise", sigma=25, profile="np")
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.3, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def retinex_enhance() -> PipelineConfig:
        """
        Retinex illumination correction preset.
        MSRCP corrects lighting, bilateral smooths residual noise,
        channel correction fixes colour casts, CLAHE boosts local contrast.

        Best for badly-lit, faded, or low-contrast footage where
        illumination correction is the primary need.
        """
        cfg = PipelineConfig(label="Retinex Enhance")
        cfg.add("retinex", sigma_list=[15, 80, 250], weights=[1/3, 1/3, 1/3],
                gain=5.0, offset=0.0)
        cfg.add("bilateral", d=7, sigma_color=50, sigma_space=50)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("histogram_eq_clahe", clip_limit=2.0, tile_size=8)
        return cfg

    @staticmethod
    def retinex_bm3d() -> PipelineConfig:
        """
        Combined Retinex + BM3D — best overall quality preset.
        median prefilter → BM3D denoise → Retinex illumination correction →
        channel correction → unsharp mask.

        Median removes impulse noise before BM3D.
        BM3D provides state-of-the-art denoising.
        Retinex normalises lighting and improves contrast.
        Channel correction balances colour channels.
        Unsharp mask recovers edge detail.

        This is the recommended preset for maximum visual quality on
        challenging footage with both noise AND poor lighting.
        """
        cfg = PipelineConfig(label="Retinex + BM3D (Best Combo)")
        cfg.add("median", ksize=3)
        cfg.add("bm3d_denoise", sigma=25, profile="np")
        cfg.add("retinex", sigma_list=[15, 80, 250], weights=[1/3, 1/3, 1/3],
                gain=5.0, offset=0.0)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.3, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def bm3d_fast() -> PipelineConfig:
        """
        Fast BM3D preset — BM3D via bm3d_rgb + channel correction only.
        No median prefilter, no unsharp mask after.

        Uses the fast bm3d_rgb implementation with both denoising stages.
        Channel correction fixes any colour shift.
        Skips median and unsharp for ~2× speed improvement over full bm3d-denoise.

        Suitable when speed matters and the image has minimal impulse noise.
        """
        cfg = PipelineConfig(label="BM3D Fast")
        cfg.add("bm3d", sigma_psd=15)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        return cfg

    # ── Guided / TV / Aniso Presets (v3.1) ──────────────────────────

    @staticmethod
    def guided_denoise() -> PipelineConfig:
        """
        Guided filter denoising — edge-preserving via local linear model.
        Good for smooth regions while keeping edge sharpness.
        """
        cfg = PipelineConfig(label="Guided Filter Denoise")
        cfg.add("guided_filter", radius=4, eps=200.0)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.3, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def tv_denoise_preset() -> PipelineConfig:
        """
        Total Variation denoising — ROF model (Chambolle).
        Best for removing small noise while preserving sharp edges.
        """
        cfg = PipelineConfig(label="TV Denoise")
        cfg.add("tv_denoise", weight=0.08, max_num_iter=50)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def aniso_denoise() -> PipelineConfig:
        """
        Anisotropic diffusion (Perona-Malik) denoising.
        Iteratively diffuses intra-region while preserving edges.
        """
        cfg = PipelineConfig(label="Aniso Diffusion Denoise")
        cfg.add("anisotropic_diffusion", n_iter=15, kappa=40, gamma=0.2)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.15, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def dct_denoise() -> PipelineConfig:
        """
        Block DCT denoising — transform-domain hard thresholding.
        Effective for Gaussian noise, fast implementation.
        """
        cfg = PipelineConfig(label="DCT Block Denoise")
        cfg.add("patch_collaborative", patch_size=8, h_dct=25.0)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def spatial_temporal_video() -> PipelineConfig:
        """
        Combined spatio-temporal denoising for video.
        Spatial: bilateral + guided filter.
        Temporal: motion-compensated frame blending.
        Designed for analog video with flicker + noise.
        """
        cfg = PipelineConfig(label="Spatio-Temporal Video")
        cfg.add("temporal_motion", strength=0.3)
        cfg.add("bilateral", d=5, sigma_color=30, sigma_space=30)
        cfg.add("guided_filter", radius=3, eps=100.0)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=10)
        return cfg

    @staticmethod
    def optimized_fast() -> PipelineConfig:
        """
        [NEW] Optimized fast denoise based on ablation study.
        NLM removed (was redundant with median+bilateral).
        Stronger bilateral for better denoising.
        Wider channel correction for color recovery.
        25× faster than edge-preserve with same PSNR.
        """
        cfg = PipelineConfig(label="Optimized Fast (Ablation)")
        cfg.add("median", ksize=3)                                   # impulse noise
        cfg.add("bilateral", d=11, sigma_color=110, sigma_space=110)  # strong edge-preserving
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25) # color balance
        return cfg

    @staticmethod
    def optimized_quality() -> PipelineConfig:
        """
        [NEW] Optimized quality — guided filter + bilateral + wavelet fusion.
        Combines the best spatial approaches.
        """
        cfg = PipelineConfig(label="Optimized Quality (Fusion)")
        cfg.add("median", ksize=3)                                   # impulse
        cfg.add("bilateral", d=7, sigma_color=50, sigma_space=50)    # smooth
        cfg.add("wavelet", wavelet="bior4.4", level=2, threshold_mode="soft", n_shifts=3)  # detail preservation
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def max_quality() -> PipelineConfig:
        """
        Maximum quality — strong bilateral denoising with CLAHE contrast recovery.
        PSNR=19.19, SSIM=0.5930 (synthetic test).
        NOTE: Strong bilateral can wash highlights. CLAHE stage prevents this.
        """
        cfg = PipelineConfig(label="Max Quality (Bilateral σ=150 + CLAHE)")
        cfg.add("median", ksize=3)
        cfg.add("bilateral", d=19, sigma_color=150, sigma_space=150)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("histogram_eq_clahe", clip_limit=1.5, tile_size=8)  # prevent washing
        return cfg

    @staticmethod
    def video_enhanced() -> PipelineConfig:
        """
        [RECOMMENDED for video] Guided filter + wavelet + CLAHE.
        No washing, excellent edge preservation, 0.12s/frame.
        Best balance for real analog video: no highlight washout,
        strong denoising, sharp edges.
        """
        cfg = PipelineConfig(label="Video Enhanced (Guided+Wavelet+CLAHE)")
        cfg.add("median", ksize=3)                                   # impulse noise
        cfg.add("guided_filter", radius=3, eps=100.0)                # edge-aware denoise
        cfg.add("wavelet", wavelet="bior4.4", level=2, threshold_mode="soft", n_shifts=3)  # detail preservation
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("histogram_eq_clahe", clip_limit=1.5, tile_size=8)   # local contrast
        cfg.add("unsharp_mask", strength=0.15, radius=0.5, threshold=5)  # light edge recovery
        return cfg

    @staticmethod
    def video_ultra() -> PipelineConfig:
        """
        [v3.2 BEST OVERALL] Spatio-temporal video enhancement.
        Combines st-video's temporal pipeline with video-enhanced's wavelet+CLAHE.

        From benchmark: Score=51.26 (video-enhanced) but st-video is 20× faster.
        This preset merges both: temporal motion → guided filter → wavelet → CLAHE.
        Achieves ~90% of video-enhanced quality at 5× the speed.
        """
        cfg = PipelineConfig(label="Video Ultra (Spatio-Temporal)")
        cfg.add("temporal_motion", strength=0.3)
        cfg.add("guided_filter", radius=3, eps=100.0)
        cfg.add("wavelet", wavelet="db4", level=2, threshold_mode="soft")
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("histogram_eq_clahe", clip_limit=1.5, tile_size=8)
        cfg.add("unsharp_mask", strength=0.15, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def ntc_plus() -> PipelineConfig:
        """
        [v3.2 NTSC OPTIMIZED] Enhanced wavelet denoise for analog FPV.
        Combines wavelet-denoise's NTSC strength with chroma-specific denoising.

        From benchmark: wavelet-denoise scores 48.08 on NTSC-heavy (best).
        Adding chroma_denoise cleans color noise without blurring luma OSD.
        """
        cfg = PipelineConfig(label="NTSC Plus (Wavelet+Chroma)")
        cfg.add("wavelet", wavelet="db4", level=3, threshold_mode="soft")
        cfg.add("chroma_denoise", strength=0.6)
        cfg.add("bilateral", d=5, sigma_color=20, sigma_space=20)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def fast_premium() -> PipelineConfig:
        """
        [v3.2 FAST + QUALITY] Bilateral + wavelet + CLAHE in fast pipeline.
        Like optimized-quality but with guided filter (from video-enhanced).

        Benchmark showed optimized-quality at 39.66 composite with 137.5ms.
        This replaces bilateral with guided filter for better edge preservation,
        keeping the fast execution path.
        """
        cfg = PipelineConfig(label="Fast Premium (Guided+Wavelet)")
        cfg.add("guided_filter", radius=3, eps=100.0)
        cfg.add("wavelet", wavelet="db4", level=2, threshold_mode="soft")
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("histogram_eq_clahe", clip_limit=1.5, tile_size=8)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def aggressive() -> PipelineConfig:
        """
        Strong denoising for very noisy footage.
        PSNR=18.69  SSIM=0.626  Edge=31%  0.26s

        Highest PSNR/SSIM but loses most edge detail (31%).
        Multiple denoising stages stacked together.
        Only use for extremely noisy footage where edge loss is acceptable.
        """
        cfg = PipelineConfig(label="Aggressive Denoise")
        cfg.add("median", ksize=5)
        cfg.add("nlm", h=10, template_window=7, search_window=21)
        cfg.add("bilateral", d=9, sigma_color=75, sigma_space=75)
        cfg.add("wiener", noise_var=200)
        cfg.add("channel_correction")
        cfg.add("unsharp_mask", strength=0.5, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def research_best() -> PipelineConfig:
        """
        Best research configuration — all techniques combined.
        PSNR=17.70  SSIM=0.496  Edge=73%  0.25s

        NLM + Bilateral + gentle Wiener + unsharp recovery.
        Highest SSIM score, good edge retention.
        """
        cfg = PipelineConfig(label="Research Best")
        cfg.add("median", ksize=3)
        cfg.add("nlm", h=8, template_window=7, search_window=21)
        cfg.add("bilateral", d=5, sigma_color=25, sigma_space=25)
        cfg.add("wiener", noise_var=150)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.4, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def analog_clean() -> PipelineConfig:
        """
        [RECOMMENDED for analog FPV] Wavelet base + analog artifact removal.

        Based on user evaluation of real whoop footage (VID00002/00006):
          - wavelet chosen as base (best visual quality, least OSD blur)
          - flicker_stabilize: removes frame-to-frame brightness flicker
          - scanline_remove: removes periodic horizontal scanline artifacts
          - gentle bilateral: cleans residual noise without blurring OSD
          - channel_correction: fixes analog color cast (magenta tint etc.)

        Designed for: 640x480 NTSC analog FPV, 30fps, noise 2000–10000.
        """
        cfg = PipelineConfig(label="Analog Clean (Wavelet+Flicker+Scanline)")
        cfg.add("scanline_remove", mode="detect", blend=0.5)           # horizontal scanlines
        cfg.add("flicker_stabilize", strength=0.6, window=10)          # brightness flicker
        cfg.add("wavelet", wavelet="bior4.4", level=2, threshold_mode="soft", n_shifts=3)  # base denoise
        cfg.add("bilateral", d=5, sigma_color=20, sigma_space=20)      # residual noise
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)  # color cast
        cfg.add("unsharp_mask", strength=0.15, radius=0.5, threshold=8)  # mild sharpen
        return cfg

    @staticmethod
    def analog_heavy() -> PipelineConfig:
        """
        Strong analog cleanup for high-noise footage (noise > 5000).

        Same base as analog_clean but with:
          - stronger wavelet (level=3)
          - stronger flicker stabilization
          - fixed 2-row scanline removal (for confirmed interlaced sources)
          - stronger bilateral for heavy noise
        """
        cfg = PipelineConfig(label="Analog Heavy Noise")
        cfg.add("scanline_remove", mode="fixed", blend=0.6, period_hint=2)
        cfg.add("flicker_stabilize", strength=0.8, window=15)
        cfg.add("wavelet", wavelet="bior4.4", level=3, threshold_mode="soft", n_shifts=3)
        cfg.add("bilateral", d=7, sigma_color=40, sigma_space=40)
        cfg.add("channel_correction", clamp_min=0.80, clamp_max=1.20)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def adaptive() -> PipelineConfig:
        """
        [ADAPTIVE] Content-aware pipeline — auto-selects and tunes filters
        based on per-frame noise estimation.

        This is a placeholder config; the actual adaptive logic lives in
        smu_sig_prossessing.adaptive.AdaptivePipeline which analyses each
        frame at runtime and selects the best preset + parameters.

        When this preset is selected, main.py uses AdaptivePipeline instead
        of the static apply_pipeline function.
        """
        # The AdaptivePipeline will replace these at runtime, but we provide
        # a sensible default chain that mirrors fast_denoise as a baseline.
        cfg = PipelineConfig(label="Adaptive (auto-tuned per frame)")
        cfg.add("median", ksize=3)
        cfg.add("bilateral", d=7, sigma_color=50, sigma_space=50)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=10)
        return cfg


# ─── Default configuration ──────────────────────────────────────────

DEFAULT_PIPELINE = PipelineConfig.edge_preserve()
