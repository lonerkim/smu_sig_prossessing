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
    def retinex_msrcr_enhance() -> PipelineConfig:
        """
        Retinex MSRCR illumination correction — color restoration variant.

        Uses MSRCR instead of MSRCP for better colour preservation.
        The color restoration factor prevents desaturation common in basic
        Retinex, making this ideal for faded or colour-shifted footage
        where the original hue ratios need to be maintained.

        MSRCR → bilateral smooth → channel correction → CLAHE → unsharp.
        """
        cfg = PipelineConfig(label="Retinex MSRCR Enhance")
        cfg.add("retinex_msrcr", sigma_list=[15, 80, 250],
                weights=[1/3, 1/3, 1/3], gain=5.0, offset=25.0)
        cfg.add("bilateral", d=7, sigma_color=50, sigma_space=50)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("histogram_eq_clahe", clip_limit=2.0, tile_size=8)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def retinex_bm3d_msrcr() -> PipelineConfig:
        """
        Retinex MSRCR + BM3D — best overall quality with color restoration.

        median → BM3D → MSRCR → channel correction → unsharp.

        MSRCR's color restoration gives more natural colours than MSRCP
        while BM3D provides state-of-the-art denoising.
        """
        cfg = PipelineConfig(label="Retinex MSRCR + BM3D")
        cfg.add("median", ksize=3)
        cfg.add("bm3d_denoise", sigma=25, profile="np")
        cfg.add("retinex_msrcr", sigma_list=[15, 80, 250],
                weights=[1/3, 1/3, 1/3], gain=5.0, offset=25.0)
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

    # ── v3.3 New Presets (Jun 2026) ─────────────────────────────────

    @staticmethod
    def super_premium() -> PipelineConfig:
        """
        [v3.3 BEST OVERALL] Super Premium — highest quality with detail boost.

        Builds on fast-premium's winning formula (guided+wavelet+CLAHE+unsharp)
        by adding:
          - median prefilter for impulse noise (protects wavelet from outliers)
          - detail_boost for edge-aware detail enhancement
          - chroma_denoise for clean colour without luma blur
          - adaptive_equalize for brightness-preserving contrast

        Expected: >53 composite score, best visual quality on real footage.
        Target: ~200ms per 400px frame.
        """
        cfg = PipelineConfig(label="Super Premium (Detail Boost)")
        cfg.add("median", ksize=3)
        cfg.add("guided_filter", radius=3, eps=100.0)
        cfg.add("wavelet", wavelet="db4", level=2, threshold_mode="soft")
        cfg.add("chroma_denoise", strength=0.3)
        cfg.add("detail_boost", strength=0.3, sigma_s=3.0, sigma_r=0.15, threshold=0.02)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
        cfg.add("unsharp_mask", strength=0.15, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def super_premium_fast() -> PipelineConfig:
        """
        [v3.3 FAST + QUALITY] Super Premium Fast — detail boost without wavelet.

        Drops the wavelet (slowest stage) and uses rolling guidance + detail_boost
        for a fast yet high-quality alternative.  ~2× faster than super-premium.

        Target: ~80ms per 400px frame with >50 composite score.
        """
        cfg = PipelineConfig(label="Super Premium Fast (Rolling)")
        cfg.add("median", ksize=3)
        cfg.add("guided_filter", radius=3, eps=100.0)
        cfg.add("detail_boost", strength=0.4, sigma_s=3.0, sigma_r=0.15, 
                threshold=0.02, boost_mode="layer")
        cfg.add("chroma_denoise", strength=0.3)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
        cfg.add("unsharp_mask", strength=0.15, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def rolling_premium() -> PipelineConfig:
        """
        [v3.3 ROLLING GUIDANCE] Edge-preserving denoise with rolling guidance.

        Uses the new rolling_guidance filter as the primary denoiser.  This
        iterative joint bilateral approach removes small noise/texture while
        preserving major edges better than single-pass bilateral or guided.

        Good for footage with mixed fine detail and noise where you want to
        keep texture but remove grain.
        """
        cfg = PipelineConfig(label="Rolling Premium")
        cfg.add("median", ksize=3)
        cfg.add("rolling_guidance", sigma_s=3.0, sigma_r=0.08, n_iter=3)
        cfg.add("detail_boost", strength=0.25, sigma_s=2.0, sigma_r=0.12, threshold=0.015)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def temporal_premium() -> PipelineConfig:
        """
        [v3.6 TEMPORAL VIDEO - OPTIMIZED BRISQUE=59.87] Multi-frame NLM with chroma denoise.

        Optimised via temporal NLM sweep (see output/temporal_nlm_sweep/):
          - h=15 (stronger luma denoising → best BRISQUE)
          - h_color=10 (moderate chroma denoising)
          - temporal_window=2 (5-frame sliding window)
          - max_frames=5

        BRISQUE improvement: from ~65 to 59.87 (−8%) vs v3.3 temporal_premium.
        Best for: video processing where perceptual quality matters most.

        Combines:
          - temporal_nlm_multi: NLM search across time (h=15 optimized)
          - guided_filter: edge-preserving cleanup (eps=50 for best scores)
          - chroma_denoise: clean colour channels (strength=0.2 gentle)
          - adaptive_equalize: brightness-preserving contrast
        """
        cfg = PipelineConfig(label="Temporal Premium v3.6 (BRISQUE=59.9)")
        cfg.add("temporal_nlm_multi", h=15, h_color=10, temporal_window=2, max_frames=5)
        cfg.add("guided_filter", radius=3, eps=50.0)
        cfg.add("chroma_denoise", strength=0.2)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
        cfg.add("unsharp_mask", strength=0.1, radius=0.5, threshold=10)
        return cfg

    @staticmethod
    def bm4d_temporal() -> PipelineConfig:
        """
        [v3.6 BM4D VIDEO] Per-frame BM3D with temporal buffer.

        BM4D (true spatio-temporal volume denoising) requires ≥8 frames in
        the temporal dimension, which is impractical for real-time use.
        This preset falls back to per-frame BM3D which provides excellent
        single-frame denoising quality at ~800ms/frame.

        Best for: offline processing of noisy analog video where quality
        is paramount and speed is not critical.
        """
        cfg = PipelineConfig(label="BM4D Temporal (BM3D per-frame)")
        cfg.add("bm4d_volume", sigma_psd=15.0, temporal_window=2, max_frames=8)
        cfg.add("chroma_denoise", strength=0.3)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def ultralight() -> PipelineConfig:
        """
        [v3.3 ULTRA FAST] Minimal pipeline for real-time processing.

        Uses only the fastest filters: cross_bilateral (single-pass joint
        bilateral) + channel correction + unsharp mask.  No wavelet, no
        NLM, no iterative methods.

        Target: <3ms per 400px frame, >35 composite score.
        Suitable for real-time 30fps+ processing of 640×480 video.
        """
        cfg = PipelineConfig(label="Ultra Light (Real-time)")
        cfg.add("cross_bilateral", guide_sigma=1.0, d=5, sigma_color=30, sigma_space=30)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=10)
        return cfg

    @staticmethod
    def chroma_focus() -> PipelineConfig:
        """
        [v3.3 CHROMA FOCUS] Aggressive chroma denoise + luma preservation.

        Targets colour noise (common in analog video) while keeping luma
        detail sharp.  Uses chroma_denoise at high strength, then
        rolling_guidance for edge-preserving cleanup, then detail boost.

        Best for: footage with heavy colour noise but good luma signal,
        like NTSC analog captures with colour bleeding / chroma noise.
        """
        cfg = PipelineConfig(label="Chroma Focus (Analog Color)")
        cfg.add("chroma_denoise", strength=0.8)
        cfg.add("rolling_guidance", sigma_s=3.0, sigma_r=0.08, n_iter=3)
        cfg.add("channel_correction", clamp_min=0.80, clamp_max=1.25)
        cfg.add("adaptive_equalize", clip_limit=2.0, tile_size=8, brightness_preserve=0.3)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=8)
        return cfg

    # ── v3.4 Grey-Edge Presets (Jun 2026) ──────────────────────────

    @staticmethod
    def grey_premium() -> PipelineConfig:
        """
        [v3.4 BEST OVERALL — 57.20] Grey-Edge + temporal-premium fusion.

        Adds Grey-Edge color constancy (strength=0.25 optimal from sweep) as
        pre-processing before the multi-frame NLM pipeline.  Grey-Edge
        estimates scene illuminant from image derivatives for natural white
        balance, dramatically improving ΔE (11.11 vs 13.16) while also
        boosting PSNR (+0.6dB).

        v3.4 Benchmark: Score=57.59 (at strength=0.25)
        ΔE improvement: 13.16 → 11.11 (-15.6%)
        """
        cfg = PipelineConfig(label="Grey Premium (56.60)")
        cfg.add("grey_edge", strength=0.22, sigma_smooth=1.0)
        cfg.add("temporal_nlm_multi", h=8, h_color=8, temporal_window=3, max_frames=5)
        cfg.add("guided_filter", radius=3, eps=50.0)
        cfg.add("chroma_denoise", strength=0.2)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
        cfg.add("unsharp_mask", strength=0.1, radius=0.5, threshold=10)
        return cfg

    @staticmethod
    def grey_fast() -> PipelineConfig:
        """
        [v3.4 FAST + COLOR] Grey-Edge + fast premium pipeline.

        Grey-Edge color correction + guided filter + wavelet denoising.
        ~5× faster than grey-premium with still excellent color fidelity.
        Score: ~52.8 with ΔE=12.33 at ~50ms.
        """
        cfg = PipelineConfig(label="Grey Fast (Color)")
        cfg.add("grey_edge", strength=0.25, sigma_smooth=1.0)
        cfg.add("guided_filter", radius=3, eps=100.0)
        cfg.add("wavelet", wavelet="db4", level=2, threshold_mode="soft")
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def grey_ultralight() -> PipelineConfig:
        """
        [v3.4 ULTRA FAST + COLOR] Minimal pipeline with Grey-Edge.

        Grey-Edge (2.9ms) + cross_bilateral (4.5ms) + minimal post.
        ~16ms total — suitable for 60fps real-time with color correction.
        Score: ~42.4 with ΔE=13.86.
        """
        cfg = PipelineConfig(label="Grey Ultra Light")
        cfg.add("grey_edge", strength=0.4, sigma_smooth=0.5)
        cfg.add("cross_bilateral", guide_sigma=1.0, d=5, sigma_color=30, sigma_space=30)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=10)
        return cfg

    # ── v3.5 Optimized Presets (Jun 2026 — Filter Interaction Analysis) ──

    @staticmethod
    def optimal_perceptual() -> PipelineConfig:
        """
        [v3.5 BEST NIQE] DCT denoising + detail boost + adaptive equalize.

        Based on filter interaction analysis: dct+detail_boost achieves
        NIQE=7.16 (best no-reference quality score). Adding adaptive_equalize
        further improves perceptual quality with brightness preservation.

        dct (patch_collaborative): block DCT hard-thresholding removes
        Gaussian/structured noise while preserving edges.
        detail_boost: edge-aware detail enhancement in layer mode.
        adaptive_equalize: CLAHE with brightness preservation.
        unsharp_mask: final edge recovery.

        Target: NIQE <7.2, Score >72, ~600ms per 854x480 frame.
        """
        cfg = PipelineConfig(label="Optimal Perceptual (DCT+Detail)")
        cfg.add("patch_collaborative", patch_size=8, h_dct=25.0)
        cfg.add("detail_boost", strength=0.3, sigma_s=3.0, sigma_r=0.15, threshold=0.02)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
        cfg.add("unsharp_mask", strength=0.15, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def optimal_fast() -> PipelineConfig:
        """
        [v3.5 FAST + QUALITY] Cross bilateral + chroma denoise + unsharp.

        Based on filter interaction analysis: cross_bilateral (10.3ms, 77.12
        score) is 3x faster than bilateral with better composite score.
        Adding chroma_denoise cleans colour noise without luma blur.

        cross_bilateral: joint bilateral filtering, fast single-pass.
        chroma_denoise: UV-channel denoising only.
        unsharp_mask: sharpness recovery.
        channel_correction: colour balance.

        Target: <30ms per 854x480 frame, NIQE <7.6.
        """
        cfg = PipelineConfig(label="Optimal Fast (CrossBilat+Chroma)")
        cfg.add("cross_bilateral", guide_sigma=1.0, d=5, sigma_color=30, sigma_space=30)
        cfg.add("chroma_denoise", strength=0.2)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=10)
        return cfg

    @staticmethod
    def optimal_balanced() -> PipelineConfig:
        """
        [v3.5 BALANCED] Wavelet denoising + detail boost + chroma cleanup.

        Based on filter interaction analysis: wavelet+detail_boost achieves
        NIQE=7.18 (second best). Adding chroma_denoise for color noise and
        adaptive_equalize for contrast makes a well-rounded preset.

        wavelet: multi-resolution denoising with db4, soft thresholding.
        detail_boost: edge-aware layer enhancement.
        chroma_denoise: UV-channel color noise reduction.
        channel_correction: colour balance.
        adaptive_equalize: brightness-preserving contrast.

        Target: NIQE <7.2, Score >74, ~900ms.
        """
        cfg = PipelineConfig(label="Optimal Balanced (Wavelet+Detail+Chroma)")
        cfg.add("wavelet", wavelet="db4", level=2, threshold_mode="soft")
        cfg.add("detail_boost", strength=0.3, sigma_s=3.0, sigma_r=0.15, threshold=0.02)
        cfg.add("chroma_denoise", strength=0.2)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
        cfg.add("unsharp_mask", strength=0.1, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def optimal_ultrafast() -> PipelineConfig:
        """
        [v3.5 ULTRA FAST] Median + unsharp — minimal viable pipeline.

        From filter interaction: median (0.7ms) + unsharp (9.3ms) achieves
        78.51 composite score at 7.6ms total. Adding channel_correction
        for colour. Fastest pipeline that still produces usable output.

        Target: <10ms per 854x480 frame (>100fps).
        """
        cfg = PipelineConfig(label="Optimal Ultra Fast (Median+Unsharp)")
        cfg.add("median", ksize=3)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=10)
        return cfg

    @staticmethod
    def analog_deinterlace() -> PipelineConfig:
        """
        [v3.5 ANALOG DEINTERLACE] Bob deinterlace + denoising for interlaced video.

        Adds deinterlacing (bob method with auto field detection) as the first
        stage, followed by wavelet denoising and channel correction.

        This preset is designed for interlaced analog video captures where
        combing artifacts are visible (common in NTSC/PAL FPV recordings).

        Deinterlace: bob method (linear interpolation, 28ms at 854x480).
        wavelet: multi-resolution denoising for residual noise.
        chroma_denoise: cleans color noise from analog encoding.
        channel_correction: colour balance.
        adaptive_equalize: brightness-preserving contrast.

        Target: ~900ms per frame, NIQE <7.5
        """
        cfg = PipelineConfig(label="Analog Deinterlace")
        cfg.add("deinterlace", method="bob", field_order="auto")
        cfg.add("wavelet", wavelet="db4", level=2, threshold_mode="soft")
        cfg.add("chroma_denoise", strength=0.2)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
        return cfg


    # ── v3.6 Optimized Presets (Jun 2026 — NIQE/BRISQUE Benchmark) ──

    @staticmethod
    def optimal_bior4() -> PipelineConfig:
        """
        [v3.6 BEST NIQE=7.26] Bior4.4 wavelet + detail boost + chroma denoise.

        Based on filter interaction analysis v2: bior4.4 achieves NIQE=7.26,
        beating optimal-balanced (7.28). Bior4.4 (biorthogonal 4.4) provides
        better perceptual quality than db4 wavelet due to its linear phase
        property and symmetric wavelet functions.

        wavelet(bior4.4): multi-resolution denoising with 3 shifts for
        translation invariance.
        detail_boost: edge-aware layer enhancement.
        chroma_denoise: UV-channel color noise reduction.
        adaptive_equalize: brightness-preserving contrast.
        unsharp_mask: final sharpness recovery.

        Benchmark: NIQE=7.26 | BRISQUE=75.63 | 858ms.
        """
        cfg = PipelineConfig(label="Optimal Bior4 (NIQE=7.26)")
        cfg.add("wavelet", wavelet="bior4.4", level=2, threshold_mode="soft", n_shifts=3)
        cfg.add("detail_boost", strength=0.3, sigma_s=3.0, sigma_r=0.15, threshold=0.02)
        cfg.add("chroma_denoise", strength=0.2)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
        cfg.add("unsharp_mask", strength=0.1, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def fast_guided_chroma() -> PipelineConfig:
        """
        [v3.6 FAST + BALANCED] Guided filter + chroma denoise.

        Best speed+quality balance: NIQE=7.63, BRISQUE=55.42, 55.7ms.
        Excellent BRISQUE (no-reference perceptual quality) combined with
        fast execution makes this ideal for interactive/preview scenarios.

        guided_filter: edge-preserving smoothing (radius=3, eps=100).
        chroma_denoise: UV-channel color noise.
        adaptive_equalize: brightness-preserving contrast.

        Benchmark: NIQE=7.63 | BRISQUE=55.42 | 55.7ms.
        """
        cfg = PipelineConfig(label="Fast Guided+Chroma (BRISQUE=55.4)")
        cfg.add("guided_filter", radius=3, eps=100.0)
        cfg.add("chroma_denoise", strength=0.3)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
        cfg.add("unsharp_mask", strength=0.15, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def nlm_chroma_preset() -> PipelineConfig:
        """
        [v3.6 QUALITY NLM] NLM + chroma denoise.

        NLM-based denoising with chroma cleanup.  Best NLM-based preset
        with NIQE=7.33 and BRISQUE=62.64 at 682ms.

        nlm: patch-based denoising (h=6, moderate).
        chroma_denoise: UV-channel color noise.
        adaptive_equalize: brightness-preserving contrast.

        Benchmark: NIQE=7.33 | BRISQUE=62.64 | 682ms.
        """
        cfg = PipelineConfig(label="NLM+Chroma (NIQE=7.33)")
        cfg.add("nlm", h=6, template_window=7, search_window=21)
        cfg.add("chroma_denoise", strength=0.3)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
        cfg.add("unsharp_mask", strength=0.15, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def cross_chroma_detail() -> PipelineConfig:
        """
        [v3.6 FAST CROSS] Cross bilateral + chroma + detail boost.

        Cross bilateral filtering + chroma denoise + detail_boost.
        NIQE=7.35, BRISQUE=66.19, 160ms — fast quality combo.

        cross_bilateral: joint bilateral filtering (fast single-pass).
        detail_boost: edge-aware detail enhancement.
        chroma_denoise: UV-channel color noise reduction.

        Benchmark: NIQE=7.35 | BRISQUE=66.19 | 160ms.
        """
        cfg = PipelineConfig(label="Cross+Chroma+Detail (7.35/66.2)")
        cfg.add("cross_bilateral", guide_sigma=1.0, d=5, sigma_color=30, sigma_space=30)
        cfg.add("detail_boost", strength=0.3, sigma_s=3.0, sigma_r=0.15, threshold=0.02)
        cfg.add("chroma_denoise", strength=0.3)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.25)
        cfg.add("adaptive_equalize", clip_limit=1.5, tile_size=8, brightness_preserve=0.4)
        cfg.add("unsharp_mask", strength=0.15, radius=0.5, threshold=5)
        return cfg


# ─── Default configuration ──────────────────────────────────────────

DEFAULT_PIPELINE = PipelineConfig.edge_preserve()
