# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Application service for reference-based pixel stabilization."""

from __future__ import annotations

import torch

from ..domain.pixel_hold import PixelHoldOptions
from ..runtime.pixel_hold import PixelHoldRuntime


class PixelHoldService:
    """Translate workflow controls into the Pixel Hold domain configuration."""

    def __init__(self, runtime: PixelHoldRuntime | None = None) -> None:
        """Create the service with an injectable tensor runtime."""

        self._runtime = runtime or PixelHoldRuntime()

    def apply(
        self,
        frames: torch.Tensor,
        ref_source: str = "external",
        ref_index: int = 0,
        reference: torch.Tensor | None = None,
        linearize: bool = True,
        auto_luma: bool = True,
        auto_k: float = 2.5,
        tau_luma: float = 1.5 / 255.0,
        tau_grad: float = 0.02,
        mode: str = "tile",
        tile_size: int = 32,
        score_mode: str = "l1_tile",
        edge_band: bool = True,
        band_radius: int = 4,
        tau_edge_low: float = 1.5 / 255.0,
        tau_edge_high: float = 6.0 / 255.0,
        apply: str = "all",
        dilate: int = 1,
        feather_sigma: float = 2.0,
        process_on: str = "auto",
        gpu_clear_every: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stabilize a frame batch while retaining the public node signature."""

        options = PixelHoldOptions(
            reference_source=ref_source,
            reference_index=ref_index,
            linearize=linearize,
            automatic_luma=auto_luma,
            automatic_strength=auto_k,
            luma_threshold=tau_luma,
            gradient_threshold=tau_grad,
            mode=mode,
            tile_size=tile_size,
            score_mode=score_mode,
            edge_band=edge_band,
            band_radius=band_radius,
            edge_low_threshold=tau_edge_low,
            edge_high_threshold=tau_edge_high,
            apply_mode=apply,
            dilation=dilate,
            feather_sigma=feather_sigma,
            processing_device=process_on,
            gpu_clear_interval=gpu_clear_every,
        )
        return self._runtime.apply(frames, reference, options)


__all__ = ["PixelHoldService"]
