# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Comfy v3 node for batch watermark compositing."""

from __future__ import annotations

from typing import Any

import torch

from ..services.watermark import WatermarkService
from ._api import ComfyNodeBase, io

_SERVICE = WatermarkService()


class BatchWatermarkSingleV3(ComfyNodeBase):
    """Expose GPU-aware single-watermark batch compositing."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the stable Watermark workflow schema."""

        return io.Schema(
            node_id="BatchWatermarkSingle",
            display_name="🐇 Watermark",
            category="image/post",
            description=(
                "GPU accelerated watermark overlay. TorchLanc resize for quality "
                "and speed. Works for single images, but efficient for batches, too!"
            ),
            inputs=_watermark_inputs(),
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        watermark: str,
        position: str,
        scale: int,
        transparency: int,
        rotation: int,
        padding_x: int,
        padding_y: int,
        optical_padding: bool,
        optical_strength: int,
        max_batch_size: int,
        sinc_window: int,
        precision: str,
    ) -> tuple[torch.Tensor]:
        """Apply the selected watermark through the application service."""

        return _SERVICE.apply(
            image,
            watermark,
            position,
            scale,
            transparency,
            rotation,
            padding_x,
            padding_y,
            optical_padding,
            optical_strength,
            max_batch_size,
            sinc_window,
            precision,
        )


def _watermark_inputs() -> list[Any]:
    """Build ordered watermark inputs including Comfy's dynamic upload choices."""

    return [
        io.Image.Input(
            "image",
            tooltip=(
                "Images to watermark. Accepts (H,W,C) or (B,H,W,C) with values "
                "in [0–1]. Processed on GPU."
            ),
        ),
        io.Combo.Input(
            "watermark",
            options=_SERVICE.choices(),
            extra_dict={"image_upload": True},
            tooltip=(
                "Select or upload the watermark image (PNG recommended). The "
                "file’s transparency is preserved."
            ),
        ),
        io.Combo.Input(
            "position",
            options=["bottom-right", "bottom-left", "top-right", "top-left", "center"],
            default="bottom-right",
            tooltip=(
                "Where to place the watermark. Padding is ignored when 'center' "
                "is selected. Rotation expands the watermark canvas."
            ),
        ),
        io.Int.Input(
            "scale",
            default=70,
            min=1,
            max=100,
            step=1,
            tooltip=(
                "Width-based scaling. Target watermark width = image width × "
                "(scale/100). Aspect ratio preserved."
            ),
        ),
        io.Int.Input(
            "transparency",
            default=100,
            min=0,
            max=100,
            step=1,
            tooltip=(
                "Alpha multiplier for the watermark: 100 = unchanged, 0 = fully "
                "transparent."
            ),
        ),
        io.Int.Input(
            "rotation",
            default=0,
            min=0,
            max=359,
            step=1,
            tooltip=(
                "Rotate the watermark (degrees) with bicubic resampling. Canvas "
                "expands so nothing is clipped (PIL-style)."
            ),
        ),
        io.Int.Input(
            "padding_x",
            default=0,
            min=0,
            max=16384,
            step=1,
            tooltip=(
                "Extra horizontal padding in pixels from the chosen edge (ignored "
                "when position='center')."
            ),
        ),
        io.Int.Input(
            "padding_y",
            default=0,
            min=0,
            max=16384,
            step=1,
            tooltip=(
                "Extra vertical padding in pixels from the chosen edge (ignored "
                "when position='center')."
            ),
        ),
        io.Boolean.Input(
            "optical_padding",
            default=False,
            tooltip=(
                "Adjust placement by the watermark’s visual center so equal "
                "padding looks right (optical alignment). Affects corner "
                "positions; ignored when position='center'."
            ),
        ),
        io.Int.Input(
            "optical_strength",
            default=40,
            min=0,
            max=100,
            step=5,
            tooltip=(
                "How strongly to nudge toward visual centering (0–100). 0 = off. "
                "Higher values shift more for wide/rotated marks."
            ),
        ),
        io.Int.Input(
            "max_batch_size",
            default=0,
            min=0,
            max=4096,
            step=1,
            tooltip=(
                "Process images in chunks to control VRAM. 0 = process the whole "
                "batch at once."
            ),
        ),
        io.Int.Input(
            "sinc_window",
            default=3,
            min=1,
            max=8,
            step=1,
            tooltip=(
                "Lanczos window size (a) used when resizing the watermark. Higher "
                "= sharper (but more ringing)."
            ),
        ),
        io.Combo.Input(
            "precision",
            options=["fp32", "fp16", "bf16"],
            default="fp32",
            tooltip=(
                "Resampling compute dtype. fp32 = safest quality; fp16/bf16 can be "
                "faster on many GPUs."
            ),
        ),
    ]


WATERMARK_NODES = [BatchWatermarkSingleV3]

__all__ = ["BatchWatermarkSingleV3", "WATERMARK_NODES"]
