# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Application service for batch image and mask resizing."""

from __future__ import annotations

import torch

from ..domain.image_geometry import build_resize_plan
from ..runtime.image_resampling import LanczosResampler
from ..shared.tensor_validation import validate_image_batch


class ImageResizeService:
    """Validate resize requests, plan geometry, and execute tensor resampling."""

    def __init__(self, resampler: LanczosResampler | None = None) -> None:
        """Create the service with an injectable resampling runtime."""

        self._resampler = resampler or LanczosResampler()

    def resize(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        resize_mode: str,
        divisible_by: int,
        maximum_batch_size: int,
        sinc_window: int,
        pad_color: str,
        crop_position: str,
        precision: str,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        """Resize a Comfy image batch and an optional aligned mask."""

        shape = validate_image_batch(image, name="image")
        plan = build_resize_plan(
            shape.width,
            shape.height,
            width,
            height,
            resize_mode,
            divisible_by,
            crop_position,
        )
        return self._resampler.resize(
            image,
            plan,
            maximum_batch_size=maximum_batch_size,
            sinc_window=sinc_window,
            pad_color=pad_color,
            precision=precision,
            mask=mask,
        )


__all__ = ["ImageResizeService"]
