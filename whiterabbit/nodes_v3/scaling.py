# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Comfy v3 nodes for image resizing and model upscaling."""

from __future__ import annotations

from typing import Any

import torch

from ..domain.image_geometry import ResizeMode
from ..services.image_resize import ImageResizeService
from ..services.model_upscale import ModelUpscaleService
from ._api import ComfyNodeBase, io

_RESIZE_SERVICE = ImageResizeService()
_UPSCALE_SERVICE = ModelUpscaleService()
_PRECISIONS = ["fp32", "fp16", "bf16"]
_POSITIONS = [
    "center",
    "top-left",
    "top",
    "top-right",
    "left",
    "right",
    "bottom-left",
    "bottom",
    "bottom-right",
]


class UpscaleWithModelAdvancedV3(ComfyNodeBase):
    """Expose configurable tiled upscale-model execution."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the stable advanced upscale schema."""

        return io.Schema(
            node_id="UpscaleWithModelAdvanced",
            display_name="🐇 Upscale w/ Model (Advanced)",
            category="image/upscaling",
            description=(
                'Based on Comfy\'s native "Upscale Image (using Model)", with '
                "controls exposed to tune for large batches, avoid slow\nOOM "
                "fallbacks, and create opportunities to optimize for speed.\n\n"
                "Defaults\n- Behaves about the same as the original node.\n\n"
                "Controls\n- max_batch_size > 0: process images in chunks to "
                "keep VRAM steady and reduce fallback slowdowns.\n- tile_size: "
                "choose a starting tile; original node defaults to 512. 0 = auto "
                "(falls back 512 → 256 → 128 on OOM).\n- channels_last: try ON "
                "for a speedup on some systems.\n- precision: lower (fp16/bf16) "
                "can be faster; may impact quality depending on the model.\n"
            ),
            inputs=[
                io.UpscaleModel.Input(
                    "upscale_model",
                    tooltip="Pick your ESRGAN model (e.g. 2× / 4×).",
                ),
                io.Image.Input(
                    "image",
                    tooltip=(
                        "Images to upscale. Accepts a batch: frames×H×W×C with "
                        "values in [0–1]."
                    ),
                ),
                io.Int.Input(
                    "max_batch_size",
                    default=0,
                    min=0,
                    max=4096,
                    step=1,
                    optional=True,
                    tooltip=(
                        "How many images to process at once. 0 = all at once. Set "
                        ">0 if you hit OOM."
                    ),
                ),
                io.Int.Input(
                    "tile_size",
                    default=0,
                    min=0,
                    max=2048,
                    step=32,
                    optional=True,
                    tooltip=(
                        "How big each tile is. 0 = auto (starts at 512 and halves "
                        "on OOM). Bigger is faster; smaller is safer."
                    ),
                ),
                io.Boolean.Input(
                    "channels_last",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Try this ON for a small speed boost on some GPUs. If you "
                        "see no gain, leave it OFF."
                    ),
                ),
                io.Combo.Input(
                    "precision",
                    options=_PRECISIONS,
                    default="fp32",
                    optional=True,
                    tooltip=(
                        "Math mode. fp32 = safest. fp16/bf16 can be faster on many "
                        "GPUs, may impact image quality."
                    ),
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        upscale_model: Any,
        image: torch.Tensor,
        max_batch_size: int = 0,
        tile_size: int = 0,
        channels_last: bool = False,
        precision: str = "fp32",
    ) -> tuple[torch.Tensor]:
        """Upscale through the typed application service."""

        return _UPSCALE_SERVICE.upscale(
            upscale_model,
            image,
            max_batch_size,
            tile_size,
            channels_last,
            precision,
        )


class BatchResizeWithLanczosV3(ComfyNodeBase):
    """Expose batch Lanczos resizing with aligned masks."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the stable batch resize schema."""

        return io.Schema(
            node_id="BatchResizeWithLanczos",
            display_name="🐇 Batch Resize w/ Lanczos",
            category="image/resize",
            description=(
                "CUDA-accelerated, gamma-correct Lanczos resizer (TorchLanc).\n\n"
                "Modes:\n• Keep AR\n• Stretch\n• Crop (Cover + Crop)\n"
                "• Pad (Fit + Pad)\n• AR Scale + Divisible Crop\n\n"
                "Node functionality based on Resize nodes by Kijai\n\n"
                "More from me!: https://artificialsweetener.ai"
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input batch (B,H,W,C) in [0,1] float.\nProcessed on GPU.",
                ),
                io.Int.Input(
                    "width",
                    default=1024,
                    min=1,
                    max=16384,
                    step=1,
                    tooltip=(
                        "Target width (pixels).\n\nNotes:\n• Keep AR / Pad: maximum "
                        "width for the fit\n• Crop: final output width\n• AR Scale + "
                        "Divisible Crop: requested width before divisibility"
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=576,
                    min=1,
                    max=16384,
                    step=1,
                    tooltip=(
                        "Target height (pixels).\n\nNotes:\n• Keep AR / Pad: maximum "
                        "height for the fit\n• Crop: final output height\n• AR Scale + "
                        "Divisible Crop: requested height before divisibility"
                    ),
                ),
                io.Combo.Input(
                    "resize_mode",
                    options=[mode.value for mode in ResizeMode],
                    default=ResizeMode.KEEP_ASPECT.value,
                    tooltip=(
                        "Modes:\n- Keep AR: Fit inside width×height (preserve "
                        "aspect)\n- Stretch: Force to width×height (may distort)\n"
                        "- Crop (Cover + Crop): Scale to cover, then crop to "
                        "width×height\n- Pad (Fit + Pad): Fit inside, then pad to "
                        "width×height\n- AR Scale + Divisible Crop: Scale by SOURCE "
                        "long side to ≤ requested divisible; crop ONLY the short "
                        "side to its divisible"
                    ),
                ),
                io.Int.Input(
                    "divisible_by",
                    default=1,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip=(
                        "Force output dimensions to multiples of N.\n\nDetails:\n"
                        "• Keep AR: Fit → then step down to the largest size ≤ "
                        "requested that keeps AR AND makes both sides divisible\n"
                        "• AR Scale + Divisible Crop: Lock the scaled LONG side to "
                        "its divisible target; crop ONLY the short side to its "
                        "divisible\nSet to 1 (or 0 in UI) to disable"
                    ),
                ),
                io.Int.Input(
                    "max_batch_size",
                    default=0,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip=(
                        "0 = process whole batch\n>0 = chunk the batch to this size"
                    ),
                ),
                io.Int.Input(
                    "sinc_window",
                    default=3,
                    min=1,
                    max=8,
                    step=1,
                    tooltip="Lanczos window size (a). Higher = sharper (more ringing).",
                ),
                io.String.Input(
                    "pad_color",
                    default="0, 0, 0",
                    tooltip="Pad mode only. RGB as 'r, g, b' (0-255).",
                ),
                io.Combo.Input(
                    "crop_position",
                    options=_POSITIONS,
                    default="center",
                    tooltip=(
                        "Where to crop/pad from.\nChoose which edges are preserved "
                        "for cropping, or where padding is added."
                    ),
                ),
                io.Combo.Input(
                    "precision",
                    options=_PRECISIONS,
                    default="fp32",
                    tooltip="Resampling compute dtype.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Optional mask (B,H,W) in [0,1].\nResized with nearest.\n"
                        "Follows the same crop/pad as the image."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Mask.Output("mask"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        width: int,
        height: int,
        resize_mode: str,
        divisible_by: int,
        max_batch_size: int,
        sinc_window: int,
        pad_color: str,
        crop_position: str,
        precision: str,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        """Resize through the typed application service."""

        return _RESIZE_SERVICE.resize(
            image,
            width,
            height,
            resize_mode,
            divisible_by,
            max_batch_size,
            sinc_window,
            pad_color,
            crop_position,
            precision,
            mask,
        )


SCALING_NODES = [UpscaleWithModelAdvancedV3, BatchResizeWithLanczosV3]

__all__ = [
    "BatchResizeWithLanczosV3",
    "SCALING_NODES",
    "UpscaleWithModelAdvancedV3",
]
