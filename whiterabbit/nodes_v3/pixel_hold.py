# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Comfy v3 node for Pixel Hold stabilization."""

from __future__ import annotations

from typing import Any

import torch

from ..services.pixel_hold import PixelHoldService
from ._api import ComfyNodeBase, io

_SERVICE = PixelHoldService()


class PixelHoldV3(ComfyNodeBase):
    """Expose reference-based pixel stabilization to ComfyUI."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the stable Pixel Hold workflow schema."""

        return io.Schema(
            node_id="PixelHold",
            display_name="🐇 Pixel Hold",
            category="video utils",
            description=(
                "Locks parts of each frame to a chosen reference (external image "
                "or a frame from the clip) whenever changes are small—useful for "
                "stabilizing flat areas or backgrounds while leaving motion to "
                "pass through."
            ),
            inputs=_pixel_hold_inputs(),
            outputs=[
                io.Image.Output("images"),
                io.Image.Output("mask_preview"),
            ],
        )

    @classmethod
    def execute(
        cls,
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
        """Apply Pixel Hold through the typed application service."""

        return _SERVICE.apply(
            frames,
            ref_source,
            ref_index,
            reference,
            linearize,
            auto_luma,
            auto_k,
            tau_luma,
            tau_grad,
            mode,
            tile_size,
            score_mode,
            edge_band,
            band_radius,
            tau_edge_low,
            tau_edge_high,
            apply,
            dilate,
            feather_sigma,
            process_on,
            gpu_clear_every,
        )


def _pixel_hold_inputs() -> list[Any]:
    """Build ordered Pixel Hold inputs without mixing schema and execution logic."""

    return [
        io.Image.Input("frames", tooltip="Your clip (frames×H×W×C, values 0–1)."),
        io.Combo.Input(
            "ref_source",
            options=["external", "batch_index"],
            default="external",
            tooltip="Pick the reference: an external image or a frame from this clip.",
        ),
        io.Int.Input(
            "ref_index",
            default=0,
            min=0,
            max=999999,
            tooltip=(
                "If using a frame from this clip, which frame to use as the reference."
            ),
        ),
        io.Image.Input(
            "reference",
            optional=True,
            tooltip=(
                "Optional external reference (1×H×W×C). If sizes differ, it will "
                "be resized to match."
            ),
        ),
        io.Boolean.Input(
            "linearize",
            default=True,
            tooltip="Work in linear color for steadier results on flat areas.",
        ),
        io.Boolean.Input(
            "auto_luma",
            default=True,
            tooltip="Auto sensitivity for brightness changes (adapts per frame).",
        ),
        io.Float.Input(
            "auto_k",
            default=2.5,
            min=0.5,
            max=6.0,
            step=0.1,
            tooltip=(
                "Auto strength. Higher = lock more to the reference (2–3 is typical)."
            ),
        ),
        io.Float.Input(
            "tau_luma",
            default=1.5 / 255.0,
            min=0.0,
            max=4.0 / 255.0,
            step=0.0005,
            tooltip=(
                "Manual brightness threshold when Auto is OFF. Lower = stricter "
                "(more locking)."
            ),
        ),
        io.Float.Input(
            "tau_grad",
            default=0.02,
            min=0.0,
            max=1.0,
            step=0.001,
            tooltip="How much edge change to allow. Lower protects edges more.",
        ),
        io.Combo.Input(
            "mode",
            options=["tile", "pixel"],
            default="tile",
            tooltip="Tile: fast & robust. Pixel: finer but noisier.",
        ),
        io.Int.Input(
            "tile_size",
            default=32,
            min=8,
            max=256,
            step=8,
            tooltip="Tile size when using Tile mode.",
        ),
        io.Combo.Input(
            "score_mode",
            options=["l1_tile", "mad_tile"],
            default="l1_tile",
            tooltip=(
                "How tiles measure change: mean abs diff (fast) or median abs dev "
                "(robust)."
            ),
        ),
        io.Boolean.Input(
            "edge_band",
            default=True,
            tooltip="Protect a belt around strong edges to avoid wobble/stretch.",
        ),
        io.Int.Input(
            "band_radius",
            default=4,
            min=0,
            max=64,
            tooltip="Width of the protected belt (pixels).",
        ),
        io.Float.Input(
            "tau_edge_low",
            default=1.5 / 255.0,
            min=0.0,
            max=0.25,
            step=0.0005,
            tooltip="Treat as low-motion below this level (edge belt).",
        ),
        io.Float.Input(
            "tau_edge_high",
            default=6.0 / 255.0,
            min=0.0,
            max=0.5,
            step=0.0005,
            tooltip="Treat as high-motion above this level (edge belt).",
        ),
        io.Combo.Input(
            "apply",
            options=["all", "lowfreq"],
            default="all",
            tooltip="Hold the whole image (All) or only its smooth part (Low-freq).",
        ),
        io.Int.Input(
            "dilate",
            default=1,
            min=0,
            max=16,
            tooltip="Expand the mask (pixels).",
        ),
        io.Float.Input(
            "feather_sigma",
            default=2.0,
            min=0.0,
            max=16.0,
            step=0.5,
            tooltip="Soften mask edges (pixels).",
        ),
        io.Combo.Input(
            "process_on",
            options=["auto", "cpu", "gpu"],
            default="auto",
            tooltip="Choose CPU/GPU. Auto switches to GPU on very large frames.",
        ),
        io.Int.Input(
            "gpu_clear_every",
            default=0,
            min=0,
            max=1000,
            tooltip="If >0 and using GPU, free memory every N frames.",
        ),
    ]


PIXEL_HOLD_NODES = [PixelHoldV3]

__all__ = ["PIXEL_HOLD_NODES", "PixelHoldV3"]
