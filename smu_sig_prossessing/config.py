"""
Pipeline configuration system — modular filter control with ON/OFF and intensity.

Each filter stage has:
  - name: unique identifier
  - enabled: bool — whether to apply this filter
  - params: dict — stage-specific parameters (kernel size, sigma, gamma, etc.)

PipelineConfig holds an ordered list of FilterConfig instances.
The pipeline runner iterates through them in order.
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

    @staticmethod
    def edge_preserve() -> PipelineConfig:
        """
        NEW DEFAULT — Edge-preserving denoising.
        NLM + Bilateral = preserves edges while removing noise.
        Unsharp mask recovers edge contrast lost during denoising.
        Gentle gamma (1.1) only slightly brightens, no color washout.
        """
        cfg = PipelineConfig(label="Edge-Preserve (NLM+Bilateral)")
        cfg.add("median", ksize=3)                                   # impulse noise removal
        cfg.add("nlm", h=5, template_window=7, search_window=21)     # NLM (mild, edge-preserving)
        cfg.add("bilateral", d=5, sigma_color=30, sigma_space=30)    # bilateral (edge-preserving smooth)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15) # gentle color balance
        cfg.add("unsharp_mask", strength=0.3, radius=0.5, threshold=10)  # edge recovery
        return cfg

    @staticmethod
    def nlm_denoise() -> PipelineConfig:
        """
        NLM-only denoising — slow but highest quality.
        Best detail retention, no frequency-domain artifacts.
        """
        cfg = PipelineConfig(label="NLM Denoise")
        cfg.add("nlm", h=8, template_window=7, search_window=21)
        cfg.add("bilateral", d=5, sigma_color=20, sigma_space=20)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.3, radius=0.5, threshold=5)
        return cfg

    @staticmethod
    def fast_denoise() -> PipelineConfig:
        """
        Fast denoising — bilateral only (no NLM).
        Good for quick previews or large images.
        """
        cfg = PipelineConfig(label="Fast Denoise (Bilateral)")
        cfg.add("bilateral", d=7, sigma_color=50, sigma_space=50)
        cfg.add("channel_correction", clamp_min=0.9, clamp_max=1.1)
        cfg.add("unsharp_mask", strength=0.2, radius=0.5, threshold=10)
        return cfg

    @staticmethod
    def wiener_denoise() -> PipelineConfig:
        """
        Wiener-based denoising — frequency domain.
        Can cause blur/ringing. Best for periodic/gaussian noise.
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
    def aggressive() -> PipelineConfig:
        """
        Strong denoising for very noisy footage.
        Multiple passes but with unsharp to prevent blur.
        """
        cfg = PipelineConfig(label="Aggressive Denoise")
        cfg.add("median", ksize=5)
        cfg.add("nlm", h=10, template_window=7, search_window=21)
        cfg.add("bilateral", d=9, sigma_color=75, sigma_space=75)
        cfg.add("wiener", noise_var=300)
        cfg.add("channel_correction")
        cfg.add("unsharp_mask", strength=0.5, radius=0.5, threshold=5)
        cfg.add("gamma", gamma=1.15)
        return cfg

    @staticmethod
    def research_best() -> PipelineConfig:
        """
        Best research configuration — multi-stage edge-preserving denoise.
        NLM + Bilateral + gentle Wiener + unsharp recovery.
        """
        cfg = PipelineConfig(label="Research Best")
        cfg.add("median", ksize=3)
        cfg.add("nlm", h=8, template_window=7, search_window=21)
        cfg.add("bilateral", d=5, sigma_color=25, sigma_space=25)
        cfg.add("wiener", noise_var=150)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.4, radius=0.5, threshold=5)
        return cfg


# ─── Default configuration ──────────────────────────────────────────

DEFAULT_PIPELINE = PipelineConfig.edge_preserve()
