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
    def wiener_only() -> PipelineConfig:
        """
        Recommended default — clean denoising with minimal color/contrast change.
        No aggressive gamma or histogram equalization that would alter colors.
        """
        cfg = PipelineConfig(label="Wiener Denoise")
        cfg.add("median", ksize=3)                                   # impulse noise
        cfg.add("wiener", noise_var=400)                             # frequency denoise
        cfg.add("fft_notch", threshold_percentile=99.5)              # periodic noise
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15) # gentle color balance
        cfg.add("gamma", gamma=1.15)                                 # very mild brightening
        return cfg

    @staticmethod
    def edge_preserving() -> PipelineConfig:
        """
        Edge-preserving denoising using NLM + gentle Wiener.
        Best detail retention, no aggressive enhancement.
        """
        cfg = PipelineConfig(label="Edge-Preserving Denoise")
        cfg.add("median", ksize=3)
        cfg.add("nlm", h=8, template_window=7, search_window=21)
        cfg.add("wiener", noise_var=300)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.5, radius=1.0, threshold=5)
        return cfg

    @staticmethod
    def light_denoise() -> PipelineConfig:
        """Light denoising — minimal change, for already decent footage."""
        cfg = PipelineConfig(label="Light Denoise")
        cfg.add("median", ksize=3)
        cfg.add("wiener", noise_var=200)
        cfg.add("channel_correction", clamp_min=0.9, clamp_max=1.1)
        return cfg

    @staticmethod
    def aggressive() -> PipelineConfig:
        """
        Strong denoising for very noisy footage.
        Bilateral + Wiener combination, still avoids aggressive color shift.
        """
        cfg = PipelineConfig(label="Aggressive Denoise")
        cfg.add("median", ksize=5)
        cfg.add("bilateral", d=7, sigma_color=50, sigma_space=50)
        cfg.add("wiener", noise_var=600)
        cfg.add("fft_notch", threshold_percentile=99.0)
        cfg.add("channel_correction")
        cfg.add("gamma", gamma=1.2)
        return cfg

    @staticmethod
    def research_best() -> PipelineConfig:
        """
        Best research configuration — NLM + Wiener + mild sharpening.
        Avoids histogram equalization which causes color/contrast artifacts.
        """
        cfg = PipelineConfig(label="Research Best")
        cfg.add("median", ksize=3)
        cfg.add("nlm", h=10, template_window=7, search_window=21)
        cfg.add("wiener", noise_var=400)
        cfg.add("fft_notch", threshold_percentile=99.0)
        cfg.add("bilateral", d=5, sigma_color=30, sigma_space=30)
        cfg.add("channel_correction", clamp_min=0.85, clamp_max=1.15)
        cfg.add("unsharp_mask", strength=0.5, radius=1.0, threshold=5)
        return cfg


# ─── Default configuration for backward compatibility ───────────────

DEFAULT_PIPELINE = PipelineConfig.wiener_only()
