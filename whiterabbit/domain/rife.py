# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""RIFE model metadata and interpolation timing configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass

RIFE_SCALE_FACTOR_MINIMUM = 0.25
RIFE_SCALE_FACTOR_MAXIMUM = 4.0
RIFE_SCALE_FACTOR_STEP = 0.05
RIFE_TIMING_MODES = [
    "linear",
    "gamma_in",
    "gamma_out",
    "gamma_in_out",
    "bounded_linear",
    "custom_list",
]


@dataclass(frozen=True)
class RifeModelSpec:
    """One trusted downloadable RIFE checkpoint."""

    filename: str
    version: str
    architecture: str
    source_url: str
    sha256: str


@dataclass(frozen=True)
class FpsResampleOptions:
    """RIFE inference and stabilization controls for FPS conversion."""

    model_name: str
    input_fps: float
    output_fps: float
    scale_factor: float = 1.0
    ensemble: bool = True
    linearize: bool = False
    low_frequency_guardrail: bool = False
    low_frequency_sigma: float = 13.0
    source_pair_match: bool = False
    match_scale_cap: float = 0.02
    match_offset_cap: float = 2.0 / 255.0
    edge_band_lock: bool = False
    edge_low_threshold: float = 1.5 / 255.0
    edge_high_threshold: float = 6.0 / 255.0
    edge_band_radius: int = 4
    edge_band_sigma: float = 2.0
    clear_cache_interval: int = 10


RIFE_MODELS = (
    RifeModelSpec(
        "rife47.pth",
        "4.7",
        "legacy47",
        "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/"
        "releases/download/models/rife47.pth",
        "6a8a825ab2750558bdd20dcced386fd82b7222c7ba58c11d3b611d9c44f1be63",
    ),
    RifeModelSpec(
        "rife49.pth",
        "4.9",
        "legacy47",
        "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/"
        "releases/download/models/rife49.pth",
        "e55fd00f3cc184e3c65961f4bb827a9da022e78eed36b055242c0ac30000d533",
    ),
    RifeModelSpec(
        "rife_v4.25.safetensors",
        "4.25",
        "core",
        "https://huggingface.co/Comfy-Org/frame_interpolation/resolve/main/"
        "frame_interpolation/rife_v4.25.safetensors",
        "1505884b9bdae956795430d2a70f7e2317b2abd8f130f8cfdb35a5759f909481",
    ),
    RifeModelSpec(
        "rife_v4.26.safetensors",
        "4.26",
        "core",
        "https://huggingface.co/Comfy-Org/frame_interpolation/resolve/main/"
        "frame_interpolation/rife_v4.26.safetensors",
        "151874592c877740e5db11522f4514df569eeafb0a0fcb2696f16e9e8d317c94",
    ),
)
RIFE_MODEL_NAMES = [model.filename for model in RIFE_MODELS]


def get_rife_model_spec(filename: str) -> RifeModelSpec:
    """Resolve a trusted model name without accepting arbitrary download URLs."""

    for model in RIFE_MODELS:
        if model.filename == filename:
            return model
    raise ValueError(
        f"Unsupported RIFE model '{filename}'. Choose one of: "
        f"{', '.join(RIFE_MODEL_NAMES)}."
    )


def scale_pyramid(scale_factor: float, block_count: int) -> list[float]:
    """Return the legacy quality/speed pyramid for four- or five-block IFNet."""

    factor = float(scale_factor)
    if factor <= 0:
        raise ValueError("scale_factor must be greater than zero.")
    base = [8.0, 4.0, 2.0, 1.0] if block_count == 4 else [16.0, 8.0, 4.0, 2.0, 1.0]
    return [value / factor for value in base]


def parse_custom_timings(value: str) -> list[float]:
    """Parse, clamp, and sort comma- or whitespace-separated blend positions."""

    tokens = [token for token in re.split(r"[,\s]+", value.strip()) if token]
    return sorted(max(0.0, min(1.0, float(token))) for token in tokens)


def map_timing(
    timestep: float,
    mode: str,
    gamma: float,
    minimum: float,
    maximum: float,
    custom_timings: list[float],
    multiplier: int,
) -> float:
    """Map a linear interpolation position through WhiteRabbit timing controls."""

    timing = max(0.0, min(1.0, timestep))
    if mode == "custom_list" and custom_timings:
        index = round(timing * (multiplier + 1)) - 1
        return custom_timings[max(0, min(len(custom_timings) - 1, index))]
    exponent = max(1e-6, gamma)
    if mode == "gamma_in":
        timing = timing**exponent
    elif mode == "gamma_out":
        timing = 1 - (1 - timing) ** exponent
    elif mode == "gamma_in_out":
        timing = (
            0.5 * (2 * timing) ** exponent
            if timing < 0.5
            else 1 - 0.5 * (2 * (1 - timing)) ** exponent
        )
    return max(0.0, min(1.0, minimum + (maximum - minimum) * timing))


__all__ = [
    "RIFE_MODELS",
    "RIFE_MODEL_NAMES",
    "RIFE_SCALE_FACTOR_MAXIMUM",
    "RIFE_SCALE_FACTOR_MINIMUM",
    "RIFE_SCALE_FACTOR_STEP",
    "RIFE_TIMING_MODES",
    "FpsResampleOptions",
    "RifeModelSpec",
    "get_rife_model_spec",
    "map_timing",
    "parse_custom_timings",
    "scale_pyramid",
]
