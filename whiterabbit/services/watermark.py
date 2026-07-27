# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Application service for file-backed batch watermark compositing."""

from __future__ import annotations

import torch

from ..domain.watermark import WatermarkOptions
from ..runtime.watermark_composite import WatermarkCompositor
from ..runtime.watermark_files import WatermarkFileResolver


class WatermarkService:
    """Resolve trusted input files and execute watermark compositing."""

    def __init__(
        self,
        file_resolver: WatermarkFileResolver | None = None,
        compositor: WatermarkCompositor | None = None,
    ) -> None:
        """Create the service with injectable file and tensor runtimes."""

        self._file_resolver = file_resolver or WatermarkFileResolver()
        self._compositor = compositor or WatermarkCompositor()

    def choices(self) -> list[str]:
        """Return selectable Comfy input images."""

        return self._file_resolver.choices()

    def apply(
        self,
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
        """Apply one selected watermark across an image batch."""

        path = self._file_resolver.resolve(watermark)
        options = WatermarkOptions(
            position=position,
            scale_percent=scale,
            transparency_percent=transparency,
            rotation_degrees=rotation % 360,
            padding_x=padding_x,
            padding_y=padding_y,
            optical_padding=optical_padding,
            optical_strength=optical_strength,
            maximum_batch_size=max_batch_size,
            sinc_window=sinc_window,
            precision=precision,
        )
        return (self._compositor.apply(image, path, options),)


__all__ = ["WatermarkService"]
