# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Configuration model for reference-based pixel stabilization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PixelHoldOptions:
    """Validated controls used by the Pixel Hold runtime."""

    reference_source: str = "external"
    reference_index: int = 0
    linearize: bool = True
    automatic_luma: bool = True
    automatic_strength: float = 2.5
    luma_threshold: float = 1.5 / 255.0
    gradient_threshold: float = 0.02
    mode: str = "tile"
    tile_size: int = 32
    score_mode: str = "l1_tile"
    edge_band: bool = True
    band_radius: int = 4
    edge_low_threshold: float = 1.5 / 255.0
    edge_high_threshold: float = 6.0 / 255.0
    apply_mode: str = "all"
    dilation: int = 1
    feather_sigma: float = 2.0
    processing_device: str = "auto"
    gpu_clear_interval: int = 0


__all__ = ["PixelHoldOptions"]
