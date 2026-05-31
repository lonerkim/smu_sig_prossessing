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
        """Recommended: Wiener filter only (no Gaussian LP), modular stages."""
        cfg = PipelineConfig(label="Wiener Only — Optimal")
        cfg.add("median", ksize=3)
        cfg.add("wiener", noise_var=625)
        cfg.add("fft_notch", threshold_percentile=99.5)
        cfg.add("channel_correction")
        cfg.add("gamma", gamma=1.8)
        cfg.add("histogram_eq")
        return cfg

    @staticmethod
    def edge_preserving() -> PipelineConfig:
        """Edge-preserving pipeline using NLM + Wiener."""
        cfg = PipelineConfig(label="Edge-Preserving Pipeline")
        cfg.add("median", ksize=3)
        cfg.add("nlm", h=10, template_window=7, search_window=21)
        cfg.add("wiener", noise_var=400)
        cfg.add("fft_notch", threshold_percentile=99.0)
        cfg.add("channel_correction")
        cfg.add("gamma", gamma=1.8)
        cfg.add("histogram_eq")
        return cfg

    @staticmethod
    def light_denoise() -> PipelineConfig:
        """Light denoising — minimal blur, good for already decent footage."""
        cfg = PipelineConfig(label="Light Denoise")
        cfg.add("median", ksize=3)
        cfg.add("wiener", noise_var=200)
        cfg.add("channel_correction")
        cfg.add("gamma", gamma=1.5)
        return cfg

    @staticmethod
    def aggressive() -> PipelineConfig:
        """Aggressive denoising for very noisy footage."""
        cfg = PipelineConfig(label="Aggressive Denoise")
        cfg.add("median", ksize=5)
        cfg.add("bilateral", d=9, sigma_color=75, sigma_space=75)
        cfg.add("wiener", noise_var=900)
        cfg.add("fft_notch", threshold_percentile=99.0)
        cfg.add("channel_correction")
        cfg.add("gamma", gamma=2.0)
        cfg.add("histogram_eq")
        return cfg

    @staticmethod
    def research_best() -> PipelineConfig:
        """Best research configuration — combination of all advanced methods."""
        cfg = PipelineConfig(label="Research Best")
        cfg.add("median", ksize=3)
        cfg.add("nlm", h=12, template_window=7, search_window=21)
        cfg.add("wiener", noise_var=500)
        cfg.add("fft_notch", threshold_percentile=99.0)
        cfg.add("bilateral", d=5, sigma_color=50, sigma_space=50)
        cfg.add("channel_correction")
        cfg.add("gamma", gamma=1.8)
        cfg.add("histogram_eq")
        return cfg


# ─── Default configuration for backward compatibility ───────────────

DEFAULT_PIPELINE = PipelineConfig.wiener_only()
