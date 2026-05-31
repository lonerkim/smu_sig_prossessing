"""
Pipeline runner — applies a PipelineConfig to one or more images.
"""
from __future__ import annotations

import numpy as np
from .config import PipelineConfig, FilterConfig
from .filters import FILTER_REGISTRY


def apply_pipeline(img: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """
    Run all enabled filter stages in order on the input image.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image.
    config : PipelineConfig
        Ordered list of filter stages to apply.

    Returns
    -------
    np.ndarray
        Processed image.
    """
    result = img.copy()
    for stage in config.stages:
        if not stage.enabled:
            continue
        filter_fn = FILTER_REGISTRY.get(stage.name)
        if filter_fn is None:
            raise KeyError(f"Filter '{stage.name}' not registered. "
                           f"Available: {list(FILTER_REGISTRY.keys())}")
        result = filter_fn(result, **stage.params)
    return result


def apply_pipeline_with_info(img: np.ndarray, config: PipelineConfig
                              ) -> tuple[np.ndarray, list[dict]]:
    """
    Like apply_pipeline but also returns intermediate results.

    Returns
    -------
    (final_result, stage_outputs)
    stage_outputs : list of dicts with keys: name, enabled, params, output
    """
    result = img.copy()
    stages = []
    for stage in config.stages:
        entry = {
            "name": stage.name,
            "enabled": stage.enabled,
            "params": stage.params,
        }
        if not stage.enabled:
            entry["output"] = result
            stages.append(entry)
            continue
        filter_fn = FILTER_REGISTRY.get(stage.name)
        if filter_fn is None:
            raise KeyError(f"Filter '{stage.name}' not registered.")
        result = filter_fn(result, **stage.params)
        entry["output"] = result
        stages.append(entry)
    return result, stages


def list_available_filters() -> str:
    """Print all registered filters with their details."""
    lines = []
    for name in sorted(FILTER_REGISTRY.keys()):
        fn = FILTER_REGISTRY[name]
        lines.append(f"  {name:25s} — {fn.__doc__.strip().split(chr(10))[0] if fn.__doc__ else '(no doc)'}")
    return "\n".join(lines)
